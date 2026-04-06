"""QMIX baseline — value decomposition for cooperative multi-agent RL.

Implements QMIX with individual agent Q-networks and a mixing network
that enforces monotonicity for decentralized execution.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chronos.agents.base_agent import BaseAgent


class AgentQNetwork(nn.Module):
    """Individual agent Q-network with GRU for partial observability."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)
        self.hidden_dim = hidden_dim

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        h = self.gru(x, hidden)
        q = self.fc2(h)
        return q, h

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim)


class MixingNetwork(nn.Module):
    """QMIX hypernetwork-based mixing network.

    Ensures Q_tot is monotonic in individual Q_i values.
    """

    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim

        # Hypernetworks for mixing weights (constrained to be positive)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Mix individual Q-values into Q_tot.

        Args:
            agent_qs: (batch, num_agents) individual Q-values
            state: (batch, state_dim) global state

        Returns:
            q_total: (batch, 1)
        """
        batch = agent_qs.shape[0]

        # First layer
        w1 = torch.abs(self.hyper_w1(state)).view(batch, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # Second layer
        w2 = torch.abs(self.hyper_w2(state)).view(batch, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2

        return q_total.squeeze(-1).squeeze(-1)


class QMIXAgent(BaseAgent):
    """QMIX cooperative MARL baseline."""

    def __init__(self, agent_id: int, config: dict,
                 device: torch.device = torch.device("cpu")):
        super().__init__(agent_id, config)
        self.device = device
        self.num_edge_nodes = config["system"]["num_edge_nodes"]
        self.num_iot_devices = config["system"]["num_iot_devices"]
        self.num_channels = config["system"]["num_channels"]
        self.num_agents = min(self.num_edge_nodes, 10)  # Cap for tractability

        obs_dim = 10 + 20  # Per-agent obs (node features + global stats)
        self.num_actions = self.num_edge_nodes + 1  # Offloading target
        state_dim = self.num_edge_nodes * 10 + 20

        # Agent Q-networks (shared parameters)
        self.agent_q = AgentQNetwork(obs_dim, self.num_actions).to(device)
        self.target_agent_q = AgentQNetwork(obs_dim, self.num_actions).to(device)
        self.target_agent_q.load_state_dict(self.agent_q.state_dict())

        # Mixing network
        self.mixer = MixingNetwork(self.num_agents, state_dim).to(device)
        self.target_mixer = MixingNetwork(self.num_agents, state_dim).to(device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.optimizer = torch.optim.Adam(
            list(self.agent_q.parameters()) + list(self.mixer.parameters()),
            lr=5e-4,
        )

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.target_update_interval = 200
        self.step_count = 0

        self.buffer = []
        self.hidden_states = None

    def _get_agent_obs(self, obs: dict, agent_idx: int) -> torch.Tensor:
        """Extract local observation for a specific agent."""
        node_feats = obs.get("node_features", torch.zeros(self.num_edge_nodes, 10))
        if agent_idx < node_feats.shape[0]:
            local = node_feats[agent_idx]
        else:
            local = torch.zeros(10)

        global_stats = torch.tensor([
            obs.get("num_active_tasks", 0) / 100.0,
            obs.get("current_time", 0) / 10000.0,
        ])

        # Pad to obs_dim = 30
        combined = torch.cat([local, global_stats])
        target_dim = 30
        if combined.shape[0] < target_dim:
            combined = torch.cat([combined, torch.zeros(target_dim - combined.shape[0])])
        return combined[:target_dim].to(self.device)

    def _get_state(self, obs: dict) -> torch.Tensor:
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
        if self.hidden_states is None:
            self.hidden_states = [self.agent_q.init_hidden(1).to(self.device)
                                  for _ in range(self.num_agents)]

        actions = []
        for i in range(self.num_agents):
            obs_i = self._get_agent_obs(observation, i).unsqueeze(0)
            q_values, self.hidden_states[i] = self.agent_q(obs_i, self.hidden_states[i])

            if deterministic or np.random.random() > self.epsilon:
                action = q_values.argmax(dim=-1).item()
            else:
                action = np.random.randint(0, self.num_actions)
            actions.append(action)

        # Use first agent's action as the global offloading decision
        primary_action = actions[0]
        num_tasks = observation.get("num_active_tasks", 1)

        self._last_actions = actions
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {
            "offloading": np.full(max(num_tasks, 1), primary_action, dtype=int),
            "resource_alloc": np.full(self.num_edge_nodes, 0.5),
            "power_control": np.full(self.num_iot_devices, 0.5),
            "channel_assign": np.zeros(self.num_iot_devices, dtype=int),
            "fl_participate": np.ones(self.num_edge_nodes, dtype=int),
            "fl_local_steps": np.full(self.num_edge_nodes, 5),
        }

    def store_transition(self, obs: dict, action: dict, reward: float,
                         next_obs: dict, done: bool):
        self.buffer.append({
            "state": self._get_state(obs),
            "next_state": self._get_state(next_obs),
            "actions": torch.tensor(self._last_actions, dtype=torch.long),
            "reward": reward,
            "done": done,
        })

    def update(self, batch: dict = None) -> dict[str, float]:
        if len(self.buffer) < 64:
            return {}

        self.step_count += 1

        # Sample batch
        indices = np.random.choice(len(self.buffer), size=min(64, len(self.buffer)), replace=False)
        batch_data = [self.buffer[i] for i in indices]

        states = torch.stack([t["state"] for t in batch_data])
        next_states = torch.stack([t["next_state"] for t in batch_data])
        actions = torch.stack([t["actions"] for t in batch_data]).to(self.device)
        rewards = torch.tensor([t["reward"] for t in batch_data], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t["done"] for t in batch_data], dtype=torch.float32, device=self.device)

        B = states.shape[0]

        # Get agent Q-values for chosen actions
        chosen_qs = []
        h = self.agent_q.init_hidden(B).to(self.device)
        for i in range(self.num_agents):
            obs_i = states[:, i * 10:(i + 1) * 10] if i * 10 < states.shape[1] else torch.zeros(B, 10, device=self.device)
            # Pad obs
            pad_dim = 30 - obs_i.shape[1]
            if pad_dim > 0:
                obs_i = torch.cat([obs_i, torch.zeros(B, pad_dim, device=self.device)], dim=1)

            q, h = self.agent_q(obs_i, h)
            agent_action = actions[:, i] if i < actions.shape[1] else torch.zeros(B, dtype=torch.long, device=self.device)
            chosen_q = q.gather(1, agent_action.unsqueeze(1)).squeeze(1)
            chosen_qs.append(chosen_q)

        chosen_qs = torch.stack(chosen_qs, dim=1)  # (B, num_agents)
        q_total = self.mixer(chosen_qs, states)

        # Target Q-values
        with torch.no_grad():
            target_qs = []
            h_t = self.target_agent_q.init_hidden(B).to(self.device)
            for i in range(self.num_agents):
                obs_i = next_states[:, i * 10:(i + 1) * 10] if i * 10 < next_states.shape[1] else torch.zeros(B, 10, device=self.device)
                pad_dim = 30 - obs_i.shape[1]
                if pad_dim > 0:
                    obs_i = torch.cat([obs_i, torch.zeros(B, pad_dim, device=self.device)], dim=1)

                q_t, h_t = self.target_agent_q(obs_i, h_t)
                target_qs.append(q_t.max(dim=-1)[0])

            target_qs = torch.stack(target_qs, dim=1)
            q_total_target = self.target_mixer(target_qs, next_states)
            targets = rewards + self.gamma * (1 - dones) * q_total_target

        loss = F.mse_loss(q_total, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent_q.parameters()) + list(self.mixer.parameters()), 10.0
        )
        self.optimizer.step()

        # Target update
        if self.step_count % self.target_update_interval == 0:
            self.target_agent_q.load_state_dict(self.agent_q.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        # Trim buffer
        if len(self.buffer) > 10000:
            self.buffer = self.buffer[-5000:]

        return {"qmix_loss": loss.item(), "epsilon": self.epsilon}

    def get_state_dict(self) -> dict:
        return {
            "agent_q": self.agent_q.state_dict(),
            "mixer": self.mixer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        self.agent_q.load_state_dict(state_dict["agent_q"])
        self.mixer.load_state_dict(state_dict["mixer"])
