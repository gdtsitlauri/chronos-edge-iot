"""Multi-Agent PPO (MAPPO) baseline.

Standard MAPPO with centralized critic and decentralized actors.
No causal reasoning, no hypergraph, no SNN — standard MLP policies.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chronos.agents.base_agent import BaseAgent


class MAPPOActor(nn.Module):
    """Decentralized actor with MLP policy."""

    def __init__(self, obs_dim: int, num_nodes: int, num_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.offload_head = nn.Linear(hidden_dim, num_nodes + 1)
        self.channel_head = nn.Linear(hidden_dim, num_channels)
        self.resource_head = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, obs: torch.Tensor) -> dict:
        h = self.backbone(obs)
        return {
            "offload_logits": self.offload_head(h),
            "channel_logits": self.channel_head(h),
            "resource_params": self.resource_head(h),
        }

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        out = self.forward(obs)
        offload_dist = torch.distributions.Categorical(logits=out["offload_logits"])
        channel_dist = torch.distributions.Categorical(logits=out["channel_logits"])

        if deterministic:
            offload = offload_dist.probs.argmax(dim=-1)
            channel = channel_dist.probs.argmax(dim=-1)
        else:
            offload = offload_dist.sample()
            channel = channel_dist.sample()

        log_prob = offload_dist.log_prob(offload) + channel_dist.log_prob(channel)
        entropy = offload_dist.entropy() + channel_dist.entropy()

        return {
            "offload": offload,
            "channel": channel,
            "resource": out["resource_params"],
            "log_prob": log_prob,
            "entropy": entropy,
        }


class MAPPOCritic(nn.Module):
    """Centralized value function."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class MAPPOAgent(BaseAgent):
    """Multi-Agent PPO baseline with centralized critic, decentralized actors."""

    def __init__(self, agent_id: int, config: dict,
                 device: torch.device = torch.device("cpu")):
        super().__init__(agent_id, config)
        self.device = device
        self.num_edge_nodes = config["system"]["num_edge_nodes"]
        self.num_iot_devices = config["system"]["num_iot_devices"]
        self.num_channels = config["system"]["num_channels"]

        obs_dim = self.num_edge_nodes * 10 + 20
        state_dim = obs_dim  # Centralized = full state

        self.actor = MAPPOActor(obs_dim, self.num_edge_nodes, self.num_channels).to(device)
        self.critic = MAPPOCritic(state_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coeff = 0.01
        self.buffer = []

    def _obs_to_tensor(self, obs: dict) -> torch.Tensor:
        parts = []
        node_feats = obs.get("node_features", torch.zeros(self.num_edge_nodes, 10))
        parts.append(node_feats.flatten()[:self.num_edge_nodes * 10])
        parts.append(torch.tensor([
            obs.get("num_active_tasks", 0) / 100.0,
            obs.get("current_time", 0) / 10000.0,
        ]))
        state = torch.cat(parts)
        target_dim = self.num_edge_nodes * 10 + 20
        if state.shape[0] < target_dim:
            state = torch.cat([state, torch.zeros(target_dim - state.shape[0])])
        return state[:target_dim].to(self.device)

    def select_action(self, observation: dict, deterministic: bool = False) -> dict:
        state = self._obs_to_tensor(observation)
        with torch.no_grad():
            action_out = self.actor.get_action(state.unsqueeze(0), deterministic)

        offload_idx = action_out["offload"].item()
        channel_idx = action_out["channel"].item()
        resource = action_out["resource"].cpu().numpy().flatten()

        num_tasks = observation.get("num_active_tasks", 1)

        self._last_log_prob = action_out["log_prob"].item()

        return {
            "offloading": np.full(max(num_tasks, 1), offload_idx, dtype=int),
            "resource_alloc": np.full(self.num_edge_nodes, float(resource[0])),
            "power_control": np.full(self.num_iot_devices, float(resource[1] if len(resource) > 1 else 0.5)),
            "channel_assign": np.full(self.num_iot_devices, channel_idx, dtype=int),
            "fl_participate": np.ones(self.num_edge_nodes, dtype=int),
            "fl_local_steps": np.full(self.num_edge_nodes, 5),
        }

    def store_transition(self, obs: dict, action: dict, reward: float,
                         next_obs: dict, done: bool):
        self.buffer.append({
            "state": self._obs_to_tensor(obs),
            "reward": reward,
            "done": done,
            "log_prob": self._last_log_prob,
        })

    def update(self, batch: dict = None) -> dict[str, float]:
        if len(self.buffer) < 32:
            return {}

        states = torch.stack([t["state"] for t in self.buffer])
        rewards = torch.tensor([t["reward"] for t in self.buffer], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t["done"] for t in self.buffer], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor([t["log_prob"] for t in self.buffer], dtype=torch.float32, device=self.device)

        # Compute values and GAE
        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = torch.cat([values[1:], torch.zeros(1, device=self.device)])

        # GAE
        T = len(self.buffer)
        advantages = torch.zeros(T, device=self.device)
        last_gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(4):  # update epochs
            action_out = self.actor.get_action(states)
            new_log_probs = action_out["log_prob"]
            entropy = action_out["entropy"]

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            critic_loss = F.mse_loss(self.critic(states).squeeze(-1), returns)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

        self.buffer.clear()
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def get_state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
