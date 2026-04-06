"""GNN-based Task Scheduler baseline.

Uses a standard Graph Neural Network (pairwise edges, no hypergraph, no causal reasoning)
for task scheduling decisions. Represents edge-IoT system as a bipartite graph.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chronos.agents.base_agent import BaseAgent


class GraphAttentionLayer(nn.Module):
    """Standard Graph Attention Network (GAT) layer."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            adj: (N, N) adjacency matrix

        Returns:
            (N, out_dim) updated features
        """
        N = x.shape[0]
        Wh = self.W(x).view(N, self.num_heads, self.head_dim)

        outputs = []
        for head in range(self.num_heads):
            wh = Wh[:, head, :]  # (N, d_h)

            # Compute attention for all pairs
            a_input = torch.cat([
                wh.unsqueeze(1).expand(N, N, self.head_dim),
                wh.unsqueeze(0).expand(N, N, self.head_dim),
            ], dim=-1)  # (N, N, 2*d_h)

            e = self.leaky_relu(self.attn(a_input).squeeze(-1))  # (N, N)

            # Mask non-adjacent
            e = e.masked_fill(adj == 0, float('-inf'))
            alpha = F.softmax(e, dim=-1)
            alpha = torch.nan_to_num(alpha, nan=0.0)
            alpha = self.dropout(alpha)

            out = torch.mm(alpha, wh)  # (N, d_h)
            outputs.append(out)

        return torch.cat(outputs, dim=-1)  # (N, out_dim)


class GNNScheduler(nn.Module):
    """GNN-based scheduler that processes the system graph and outputs scheduling decisions."""

    def __init__(self, node_feat_dim: int, hidden_dim: int = 64, output_dim: int = 32,
                 num_layers: int = 2, num_actions: int = 21):
        super().__init__()
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(output_dim, num_actions)
        self.value_head = nn.Linear(output_dim, 1)

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> dict:
        x = F.relu(self.input_proj(node_features))

        for layer in self.gat_layers:
            x = F.relu(layer(x, adj))

        # Global pooling
        graph_embed = x.mean(dim=0)
        features = self.readout(graph_embed)

        return {
            "action_logits": self.action_head(features),
            "value": self.value_head(features),
        }


class GNNSchedulerAgent(BaseAgent):
    """GNN-based scheduling baseline.

    Represents the system as a pairwise graph (no hyperedges, no causal reasoning).
    """

    def __init__(self, agent_id: int, config: dict,
                 device: torch.device = torch.device("cpu")):
        super().__init__(agent_id, config)
        self.device = device
        self.num_edge_nodes = config["system"]["num_edge_nodes"]
        self.num_iot_devices = config["system"]["num_iot_devices"]
        self.num_channels = config["system"]["num_channels"]

        node_feat_dim = 10
        num_actions = self.num_edge_nodes + 1

        self.gnn = GNNScheduler(
            node_feat_dim=node_feat_dim,
            hidden_dim=64,
            output_dim=32,
            num_layers=2,
            num_actions=num_actions,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=3e-4)
        self.buffer = []
        self.gamma = 0.99

    def _build_graph(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Build adjacency matrix from observation (proximity-based pairwise graph)."""
        node_feats = obs.get("node_features", torch.zeros(self.num_edge_nodes, 10))
        node_pos = obs.get("node_positions", torch.zeros(self.num_edge_nodes, 2))

        N = node_feats.shape[0]
        adj = torch.zeros(N, N)

        if node_pos is not None and node_pos.shape[0] == N:
            # Connect nodes within distance threshold
            for i in range(N):
                for j in range(i + 1, N):
                    dist = torch.norm(node_pos[i] - node_pos[j]).item()
                    if dist < 200.0:
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0

        # Self-loops
        adj += torch.eye(N)

        return node_feats.to(self.device), adj.to(self.device)

    def select_action(self, observation: dict, deterministic: bool = False) -> dict:
        node_feats, adj = self._build_graph(observation)

        with torch.no_grad():
            outputs = self.gnn(node_feats, adj)

        if deterministic:
            action_idx = outputs["action_logits"].argmax().item()
        else:
            probs = F.softmax(outputs["action_logits"], dim=-1)
            action_idx = torch.multinomial(probs.unsqueeze(0), 1).item()

        num_tasks = observation.get("num_active_tasks", 1)
        self._last_value = outputs["value"].item()

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
        node_feats, adj = self._build_graph(obs)
        self.buffer.append({
            "node_feats": node_feats.cpu(),
            "adj": adj.cpu(),
            "action": action["offloading"][0],
            "reward": reward,
            "done": done,
            "value": self._last_value,
        })

    def update(self, batch: dict = None) -> dict[str, float]:
        if len(self.buffer) < 32:
            return {}

        rewards = torch.tensor([t["reward"] for t in self.buffer], dtype=torch.float32)
        values = torch.tensor([t["value"] for t in self.buffer], dtype=torch.float32)
        actions = torch.tensor([t["action"] for t in self.buffer], dtype=torch.long)

        # Compute returns
        returns = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(self.buffer))):
            R = rewards[t] + self.gamma * R * (1 - float(self.buffer[t]["done"]))
            returns[t] = R

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        for i, t in enumerate(self.buffer):
            node_feats = t["node_feats"].to(self.device)
            adj = t["adj"].to(self.device)

            outputs = self.gnn(node_feats, adj)
            log_probs = F.log_softmax(outputs["action_logits"], dim=-1)
            action_log_prob = log_probs[actions[i]]

            policy_loss = -action_log_prob * advantages[i]
            value_loss = F.mse_loss(outputs["value"].squeeze(), returns[i].to(self.device))

            loss = policy_loss + 0.5 * value_loss
            total_loss += loss

        self.optimizer.zero_grad()
        (total_loss / len(self.buffer)).backward()
        nn.utils.clip_grad_norm_(self.gnn.parameters(), 0.5)
        self.optimizer.step()

        self.buffer.clear()
        return {"gnn_loss": (total_loss / len(self.buffer)).item()}

    def get_state_dict(self) -> dict:
        return {"gnn": self.gnn.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self.gnn.load_state_dict(state_dict["gnn"])
