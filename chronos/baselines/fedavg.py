"""FedAvg + Greedy Offloading baseline.

Standard federated averaging for the learning component with
greedy nearest-node task offloading.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from chronos.agents.base_agent import BaseAgent


class SimpleMLPPolicy(nn.Module):
    """Simple MLP policy for task offloading decisions."""

    def __init__(self, input_dim: int, num_nodes: int, num_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.offload_head = nn.Linear(hidden_dim, num_nodes + 1)
        self.channel_head = nn.Linear(hidden_dim, num_channels)
        self.resource_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict:
        features = self.net(x)
        return {
            "offload_logits": self.offload_head(features),
            "channel_logits": self.channel_head(features),
            "resource": torch.sigmoid(self.resource_head(features)),
        }


class FedAvgAgent(BaseAgent):
    """FedAvg + greedy offloading baseline.

    - Federated learning uses standard FedAvg aggregation
    - Task offloading uses nearest node or learned MLP policy
    - No causal reasoning, no hypergraph, no SNN
    """

    def __init__(self, agent_id: int, config: dict,
                 device: torch.device = torch.device("cpu")):
        super().__init__(agent_id, config)
        self.device = device
        self.num_edge_nodes = config["system"]["num_edge_nodes"]
        self.num_iot_devices = config["system"]["num_iot_devices"]
        self.num_channels = config["system"]["num_channels"]

        # Simple state representation (flatten node + device features)
        self.state_dim = self.num_edge_nodes * 10 + 20  # Simplified
        self.policy = SimpleMLPPolicy(
            self.state_dim, self.num_edge_nodes, self.num_channels
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

        # Experience buffer
        self.buffer = []

    def _obs_to_tensor(self, obs: dict) -> torch.Tensor:
        """Flatten observation to a fixed-size tensor."""
        parts = []
        node_feats = obs.get("node_features", torch.zeros(self.num_edge_nodes, 10))
        parts.append(node_feats.flatten()[:self.num_edge_nodes * 10])

        # Global statistics
        parts.append(torch.tensor([
            obs.get("num_active_tasks", 0) / 100.0,
            obs.get("current_time", 0) / 10000.0,
        ]))

        # Pad to state_dim
        state = torch.cat(parts)
        if state.shape[0] < self.state_dim:
            state = torch.cat([state, torch.zeros(self.state_dim - state.shape[0])])
        return state[:self.state_dim].to(self.device)

    def select_action(self, observation: dict, deterministic: bool = False) -> dict:
        state = self._obs_to_tensor(observation)
        with torch.no_grad():
            outputs = self.policy(state.unsqueeze(0))

        # Offloading
        if deterministic:
            offload_idx = outputs["offload_logits"].argmax(dim=-1).item()
        else:
            probs = torch.softmax(outputs["offload_logits"], dim=-1)
            offload_idx = torch.multinomial(probs, 1).item()

        num_tasks = observation.get("num_active_tasks", 1)
        offloading = np.full(max(num_tasks, 1), offload_idx, dtype=int)

        # Channel
        if deterministic:
            channel_idx = outputs["channel_logits"].argmax(dim=-1).item()
        else:
            ch_probs = torch.softmax(outputs["channel_logits"], dim=-1)
            channel_idx = torch.multinomial(ch_probs, 1).item()

        resource = float(outputs["resource"].item())

        return {
            "offloading": offloading,
            "resource_alloc": np.full(self.num_edge_nodes, resource),
            "power_control": np.full(self.num_iot_devices, 0.5),
            "channel_assign": np.full(self.num_iot_devices, channel_idx, dtype=int),
            "fl_participate": np.ones(self.num_edge_nodes, dtype=int),
            "fl_local_steps": np.full(self.num_edge_nodes, 5),
        }

    def update(self, batch: dict = None) -> dict[str, float]:
        if len(self.buffer) < 32:
            return {}

        # Simple REINFORCE update
        states = torch.stack([t["state"] for t in self.buffer]).to(self.device)
        rewards = torch.tensor([t["reward"] for t in self.buffer], dtype=torch.float32).to(self.device)
        actions = torch.tensor([t["offload_action"] for t in self.buffer], dtype=torch.long).to(self.device)

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        outputs = self.policy(states)
        log_probs = torch.log_softmax(outputs["offload_logits"], dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = -(selected_log_probs * rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.clear()
        return {"policy_loss": loss.item()}

    def store_transition(self, obs: dict, action: dict, reward: float):
        self.buffer.append({
            "state": self._obs_to_tensor(obs),
            "offload_action": action["offloading"][0],
            "reward": reward,
        })

    def get_state_dict(self) -> dict:
        return {"policy": self.policy.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self.policy.load_state_dict(state_dict["policy"])
