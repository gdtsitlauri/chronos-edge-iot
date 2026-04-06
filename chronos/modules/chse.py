"""Module 1: Causal Hypergraph State Encoder (CHSE).

Implements the Causal Hypergraph Attention Network (CHAN) with:
- Type-specific linear projections for heterogeneous vertices
- Hypergraph attention with causal encoding vectors
- Causal gates modulated by ACE/NDE from the SCM
- Set2Set readout for final state embedding
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TypeSpecificProjection(nn.Module):
    """Projects heterogeneous vertex features to a common embedding space.

    Each NodeType gets its own linear projection W_{tp(v)}.
    """

    def __init__(self, type_dims: dict[int, int], output_dim: int, num_types: int = 4):
        super().__init__()
        self.projections = nn.ModuleList()
        for t in range(num_types):
            in_dim = type_dims.get(t, 16)
            self.projections.append(nn.Linear(in_dim, output_dim))
        self.num_types = num_types

    def forward(self, features: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        """Project each vertex using its type-specific projection.

        Args:
            features: (|V|, max_feat_dim) padded vertex features
            type_ids: (|V|,) integer type indices

        Returns:
            (|V|, output_dim) projected features
        """
        output = torch.zeros(features.shape[0], self.projections[0].out_features,
                             device=features.device)
        for t in range(self.num_types):
            mask = type_ids == t
            if mask.any():
                in_dim = self.projections[t].in_features
                output[mask] = self.projections[t](features[mask, :in_dim])
        return output


class CausalGate(nn.Module):
    """Causal gate psi(G, v, u) that modulates message passing by causal strength.

    psi = sigma(w_psi^T * [ACE(v->u) || NDE(v->u)])
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, ace: torch.Tensor, nde: torch.Tensor) -> torch.Tensor:
        """Compute causal gate values.

        Args:
            ace: (num_pairs,) average causal effect estimates
            nde: (num_pairs,) natural direct effect estimates

        Returns:
            (num_pairs, 1) gate values in [0, 1]
        """
        x = torch.stack([ace, nde], dim=-1)
        return self.net(x)


class CausalHypergraphAttentionLayer(nn.Module):
    """One layer of the Causal Hypergraph Attention Network (CHAN).

    h_v^{l+1} = sigma( sum_{e in E(v)} alpha_{v,e} * sum_{u in e\\{v}} W * h_u * psi(G,v,u) )

    Attention: alpha_{v,e} = softmax( LeakyReLU( a^T [Wh_v || W*h_bar_e || c_{v,e}] ) )
    """

    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4,
                 causal_gate_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        # Linear transformations
        self.W = nn.Linear(input_dim, output_dim, bias=False)
        # Attention vector: a^T [Wh_v || Wh_bar_e || c_{v,e}]
        # c_{v,e} has dim = causal_encoding_dim (we use 8)
        self.causal_encoding_dim = 8
        self.attention = nn.Linear(2 * self.head_dim + self.causal_encoding_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Causal gate
        self.causal_gate = CausalGate(causal_gate_hidden)

        # Causal encoding projection
        self.causal_encoder = nn.Linear(2, self.causal_encoding_dim)

        # Output
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, incidence: torch.Tensor,
                causal_effects: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of one CHAN layer.

        Args:
            h: (|V|, input_dim) vertex features
            incidence: (|V|, |E|) incidence matrix
            causal_effects: (|V|, |V|, 2) tensor of [ACE, NDE] between vertex pairs
                           or None (defaults to ones = no causal modulation)

        Returns:
            h_new: (|V|, output_dim) updated vertex features
        """
        num_v, num_e = incidence.shape
        Wh = self.W(h)  # (|V|, output_dim)

        # Reshape for multi-head
        Wh_heads = Wh.view(num_v, self.num_heads, self.head_dim)  # (|V|, H, d_h)

        # Compute hyperedge mean representations: h_bar_e = mean of Wh for vertices in e
        # Using incidence matrix: h_bar_e = (H^T @ Wh) / degree_e
        degree_e = incidence.sum(dim=0, keepdim=True).clamp(min=1)  # (1, |E|)
        h_bar_e = (incidence.t() @ Wh) / degree_e.t()  # (|E|, output_dim)
        h_bar_e_heads = h_bar_e.view(num_e, self.num_heads, self.head_dim)

        # Compute attention for each vertex-edge pair
        # For each v and each e incident to v, compute attention score
        output = torch.zeros_like(Wh)

        for head in range(self.num_heads):
            head_output = torch.zeros(num_v, self.head_dim, device=h.device)

            for v in range(num_v):
                # Get incident edges
                incident_mask = incidence[v] > 0  # (|E|,)
                if not incident_mask.any():
                    continue

                incident_indices = incident_mask.nonzero(as_tuple=True)[0]
                num_incident = incident_indices.shape[0]

                # Vertex feature for this head
                wh_v = Wh_heads[v, head]  # (d_h,)

                # Hyperedge features for incident edges
                wh_bar_es = h_bar_e_heads[incident_indices, head]  # (num_incident, d_h)

                # Causal encoding: c_{v,e} for each incident edge
                if causal_effects is not None:
                    # Average causal effect from v to other vertices in each edge
                    causal_vecs = []
                    for e_idx in incident_indices:
                        members = (incidence[:, e_idx] > 0).nonzero(as_tuple=True)[0]
                        others = members[members != v]
                        if len(others) > 0:
                            ce = causal_effects[v, others].mean(dim=0)  # (2,)
                        else:
                            ce = torch.zeros(2, device=h.device)
                        causal_vecs.append(ce)
                    causal_vecs = torch.stack(causal_vecs)  # (num_incident, 2)
                else:
                    causal_vecs = torch.zeros(num_incident, 2, device=h.device)

                c_ve = self.causal_encoder(causal_vecs)  # (num_incident, causal_encoding_dim)

                # Attention scores
                wh_v_expanded = wh_v.unsqueeze(0).expand(num_incident, -1)
                attn_input = torch.cat([wh_v_expanded, wh_bar_es, c_ve], dim=-1)
                attn_scores = self.leaky_relu(self.attention(attn_input)).squeeze(-1)
                attn_weights = F.softmax(attn_scores, dim=0)  # (num_incident,)

                # Aggregate messages from each incident edge
                edge_messages = torch.zeros(num_incident, self.head_dim, device=h.device)
                for idx, e_idx in enumerate(incident_indices):
                    members = (incidence[:, e_idx] > 0).nonzero(as_tuple=True)[0]
                    others = members[members != v]
                    if len(others) == 0:
                        continue

                    # Message from neighbors in this edge
                    neighbor_features = Wh_heads[others, head]  # (num_others, d_h)

                    # Apply causal gate
                    if causal_effects is not None:
                        ace_vals = causal_effects[v, others, 0]
                        nde_vals = causal_effects[v, others, 1]
                        gates = self.causal_gate(ace_vals, nde_vals)  # (num_others, 1)
                        neighbor_features = neighbor_features * gates
                    edge_messages[idx] = neighbor_features.mean(dim=0)

                # Weighted sum over incident edges
                head_output[v] = (attn_weights.unsqueeze(-1) * edge_messages).sum(dim=0)

            output[:, head * self.head_dim:(head + 1) * self.head_dim] = head_output

        # Residual + LayerNorm
        if self.input_dim == self.output_dim:
            output = self.layer_norm(output + Wh)
        else:
            output = self.layer_norm(output)
        output = self.dropout(output)

        return output


class Set2SetReadout(nn.Module):
    """Set2Set pooling for graph-level readout from vertex embeddings."""

    def __init__(self, input_dim: int, num_steps: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.lstm = nn.LSTMCell(2 * input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool vertex embeddings into a single graph-level vector.

        Args:
            x: (|V|, input_dim) vertex embeddings

        Returns:
            (2 * input_dim,) graph-level embedding
        """
        n = x.shape[0]
        h = torch.zeros(1, self.input_dim, device=x.device)
        c = torch.zeros(1, self.input_dim, device=x.device)
        q_star = torch.zeros(1, 2 * self.input_dim, device=x.device)

        for _ in range(self.num_steps):
            h, c = self.lstm(q_star, (h, c))
            # Attention over vertices
            e = (x * h.expand(n, -1)).sum(dim=-1)  # (|V|,)
            a = F.softmax(e, dim=0)  # (|V|,)
            r = (a.unsqueeze(-1) * x).sum(dim=0, keepdim=True)  # (1, input_dim)
            q_star = torch.cat([h, r], dim=-1)

        return q_star.squeeze(0)


class CausalHypergraphStateEncoder(nn.Module):
    """Complete CHSE module: encodes system state via causal hypergraph attention.

    Pipeline:
    1. Type-specific projection of heterogeneous vertex features
    2. L layers of Causal Hypergraph Attention (CHAN)
    3. Set2Set readout to produce state embedding z(t)
    """

    def __init__(
        self,
        type_dims: dict[int, int],
        node_embedding_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 3,
        num_attention_heads: int = 4,
        causal_gate_hidden: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Type-specific input projection
        self.type_projection = TypeSpecificProjection(type_dims, node_embedding_dim)

        # CHAN layers
        self.layers = nn.ModuleList()
        dims = [node_embedding_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for l in range(num_layers):
            self.layers.append(CausalHypergraphAttentionLayer(
                input_dim=dims[l],
                output_dim=dims[l + 1],
                num_heads=num_attention_heads,
                causal_gate_hidden=causal_gate_hidden,
                dropout=dropout,
            ))

        # Readout
        self.readout = Set2SetReadout(output_dim, num_steps=3)

        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, hg_data: dict,
                causal_effects: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode hypergraph state.

        Args:
            hg_data: dict from CausalHypergraph.to_pyg_data()
            causal_effects: (|V|, |V|, 2) causal effect tensor or None

        Returns:
            z: (output_dim,) global state embedding
            vertex_embeddings: (|V|, output_dim) per-vertex embeddings
        """
        x = hg_data["x"]
        type_ids = hg_data["vertex_types"]
        incidence = hg_data["incidence"]

        # Step 1: Type-specific projection
        h = self.type_projection(x, type_ids)

        # Step 2: CHAN layers
        for layer in self.layers:
            h = layer(h, incidence, causal_effects)

        # Step 3: Set2Set readout
        z_pooled = self.readout(h)

        # Step 4: Final projection
        z = self.final_proj(z_pooled)

        return z, h

    def encode_local(self, hg_data: dict, agent_vertex_id: int,
                     causal_effects: torch.Tensor | None = None) -> torch.Tensor:
        """Encode local observation for a specific agent.

        Returns the embedding of the agent's vertex after message passing.
        """
        z, vertex_embeddings = self.forward(hg_data, causal_effects)
        if agent_vertex_id < vertex_embeddings.shape[0]:
            return vertex_embeddings[agent_vertex_id]
        return z  # Fallback to global embedding
