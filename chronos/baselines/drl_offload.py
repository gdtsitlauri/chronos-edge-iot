"""DRL-Offload baseline — single-agent DQN for task offloading.

Standard Deep Q-Network with experience replay, no multi-agent, no causal reasoning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chronos.agents.base_agent import BaseAgent


class DQNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DRLOffloadAgent(BaseAgent):
    """Single-agent DQN for task offloading decisions."""

    def __init__(self, agent_id: int, config: dict,
                 device: torch.device = torch.device("cpu")):
        super().__init__(agent_id, config)
        self.device = device
        self.num_edge_nodes = config["system"]["num_edge_nodes"]
        self.num_iot_devices = config["system"]["num_iot_devices"]
        self.num_channels = config["system"]["num_channels"]

        state_dim = self.num_edge_nodes * 10 + 20
        self.num_actions = self.num_edge_nodes + 1

        self.q_net = DQNetwork(state_dim, self.num_actions).to(device)
        self.target_net = DQNetwork(state_dim, self.num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 100
        self.step_count = 0
        self.buffer = []
        self.buffer_max = 10000

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

        if not deterministic and np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.num_actions)
        else:
            with torch.no_grad():
                q_values = self.q_net(state.unsqueeze(0))
                action_idx = q_values.argmax(dim=-1).item()

        num_tasks = observation.get("num_active_tasks", 1)

        return {
            "offloading": np.full(max(num_tasks, 1), action_idx, dtype=int),
            "resource_alloc": np.full(self.num_edge_nodes, 0.5),
            "power_control": np.full(self.num_iot_devices, 0.5),
            "channel_assign": np.zeros(self.num_iot_devices, dtype=int),
            "fl_participate": np.ones(self.num_edge_nodes, dtype=int),
            "fl_local_steps": np.full(self.num_edge_nodes, 5),
        }

    def store_transition(self, obs: dict, action: dict, reward: float,
                         next_obs: dict, done: bool):
        self.buffer.append({
            "state": self._obs_to_tensor(obs).cpu(),
            "action": action["offloading"][0],
            "reward": reward,
            "next_state": self._obs_to_tensor(next_obs).cpu(),
            "done": done,
        })
        if len(self.buffer) > self.buffer_max:
            self.buffer.pop(0)

    def update(self, batch: dict = None) -> dict[str, float]:
        if len(self.buffer) < 64:
            return {}

        self.step_count += 1
        indices = np.random.choice(len(self.buffer), 64, replace=False)
        batch_data = [self.buffer[i] for i in indices]

        states = torch.stack([t["state"] for t in batch_data]).to(self.device)
        actions = torch.tensor([t["action"] for t in batch_data], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t["reward"] for t in batch_data], dtype=torch.float32, device=self.device)
        next_states = torch.stack([t["next_state"] for t in batch_data]).to(self.device)
        dones = torch.tensor([t["done"] for t in batch_data], dtype=torch.float32, device=self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=-1)[0]
            targets = rewards + self.gamma * (1 - dones) * next_q

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"dqn_loss": loss.item(), "epsilon": self.epsilon}

    def get_state_dict(self) -> dict:
        return {"q_net": self.q_net.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self.q_net.load_state_dict(state_dict["q_net"])
