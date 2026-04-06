"""Dynamic Causal Hypergraph data structure.

Models the edge-IoT system as H(t) = (V(t), E(t), W(t), G(t)) where vertices are
heterogeneous (devices, nodes, tasks, channels) and hyperedges encode multi-way
causal interactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from chronos.core.types import NodeType


@dataclass
class HypergraphVertex:
    """A vertex in the causal hypergraph."""
    vertex_id: int
    node_type: NodeType
    features: torch.Tensor = field(default_factory=lambda: torch.zeros(1))
    entity_id: int = 0  # Original ID in the system (edge node id, device id, etc.)


@dataclass
class Hyperedge:
    """A hyperedge connecting an arbitrary subset of vertices.

    Encodes a causal interaction, e.g., {device_3, node_1, task_5, channel_2} means
    'device 3 offloads task 5 to node 1 via channel 2'.
    """
    edge_id: int
    vertex_ids: set[int]
    weight: float = 1.0
    causal_strength: float = 0.0
    edge_type: str = "interaction"  # interaction, offloading, communication, learning
    metadata: dict = field(default_factory=dict)


class CausalHypergraph:
    """Dynamic causal hypergraph with support for heterogeneous vertices,
    weighted hyperedges, and incidence-matrix representations for neural processing.
    """

    def __init__(self, max_hyperedge_size: int = 6):
        self.max_hyperedge_size = max_hyperedge_size

        # Core data
        self.vertices: dict[int, HypergraphVertex] = {}
        self.hyperedges: dict[int, Hyperedge] = {}
        self._next_vertex_id = 0
        self._next_edge_id = 0

        # Indices for fast lookup
        self._vertex_to_edges: dict[int, set[int]] = {}  # vertex_id -> set of edge_ids
        self._type_to_vertices: dict[NodeType, set[int]] = {t: set() for t in NodeType}

        # Causal structure (adjacency among hyperedges for causal DAG)
        self.causal_parents: dict[int, set[int]] = {}  # edge_id -> parent edge_ids
        self.causal_children: dict[int, set[int]] = {}  # edge_id -> child edge_ids

    def add_vertex(self, node_type: NodeType, features: torch.Tensor,
                   entity_id: int = 0) -> int:
        """Add a vertex. Returns vertex_id."""
        vid = self._next_vertex_id
        self._next_vertex_id += 1
        self.vertices[vid] = HypergraphVertex(
            vertex_id=vid, node_type=node_type,
            features=features, entity_id=entity_id,
        )
        self._vertex_to_edges[vid] = set()
        self._type_to_vertices[node_type].add(vid)
        return vid

    def add_hyperedge(self, vertex_ids: set[int], weight: float = 1.0,
                      causal_strength: float = 0.0, edge_type: str = "interaction",
                      metadata: dict | None = None) -> int:
        """Add a hyperedge connecting the given vertices. Returns edge_id."""
        # Validate
        vertex_ids = {v for v in vertex_ids if v in self.vertices}
        if len(vertex_ids) < 2:
            return -1
        if len(vertex_ids) > self.max_hyperedge_size:
            # Truncate to max size (keep highest-degree vertices)
            degrees = {v: len(self._vertex_to_edges[v]) for v in vertex_ids}
            sorted_v = sorted(vertex_ids, key=lambda v: degrees[v], reverse=True)
            vertex_ids = set(sorted_v[:self.max_hyperedge_size])

        eid = self._next_edge_id
        self._next_edge_id += 1
        self.hyperedges[eid] = Hyperedge(
            edge_id=eid, vertex_ids=vertex_ids, weight=weight,
            causal_strength=causal_strength, edge_type=edge_type,
            metadata=metadata or {},
        )

        for vid in vertex_ids:
            self._vertex_to_edges[vid].add(eid)

        self.causal_parents[eid] = set()
        self.causal_children[eid] = set()
        return eid

    def remove_hyperedge(self, edge_id: int):
        """Remove a hyperedge."""
        if edge_id not in self.hyperedges:
            return
        edge = self.hyperedges[edge_id]
        for vid in edge.vertex_ids:
            self._vertex_to_edges[vid].discard(edge_id)

        # Remove from causal structure
        for parent_id in self.causal_parents.get(edge_id, set()):
            self.causal_children[parent_id].discard(edge_id)
        for child_id in self.causal_children.get(edge_id, set()):
            self.causal_parents[child_id].discard(edge_id)

        del self.hyperedges[edge_id]
        self.causal_parents.pop(edge_id, None)
        self.causal_children.pop(edge_id, None)

    def remove_vertex(self, vertex_id: int):
        """Remove a vertex and all incident hyperedges."""
        if vertex_id not in self.vertices:
            return
        edges_to_remove = list(self._vertex_to_edges.get(vertex_id, set()))
        for eid in edges_to_remove:
            self.remove_hyperedge(eid)

        v = self.vertices[vertex_id]
        self._type_to_vertices[v.node_type].discard(vertex_id)
        del self.vertices[vertex_id]
        del self._vertex_to_edges[vertex_id]

    def add_causal_edge(self, parent_edge_id: int, child_edge_id: int):
        """Add a directed causal link between two hyperedges."""
        if parent_edge_id in self.hyperedges and child_edge_id in self.hyperedges:
            self.causal_children.setdefault(parent_edge_id, set()).add(child_edge_id)
            self.causal_parents.setdefault(child_edge_id, set()).add(parent_edge_id)

    def get_incident_edges(self, vertex_id: int) -> list[Hyperedge]:
        """Get all hyperedges incident to a vertex: E(v) = {e in E : v in e}."""
        edge_ids = self._vertex_to_edges.get(vertex_id, set())
        return [self.hyperedges[eid] for eid in edge_ids if eid in self.hyperedges]

    def get_neighbors(self, vertex_id: int) -> set[int]:
        """Get all vertices connected to vertex_id via any hyperedge."""
        neighbors = set()
        for edge in self.get_incident_edges(vertex_id):
            neighbors.update(edge.vertex_ids)
        neighbors.discard(vertex_id)
        return neighbors

    def get_vertices_by_type(self, node_type: NodeType) -> list[HypergraphVertex]:
        """Get all vertices of a given type."""
        return [self.vertices[vid] for vid in self._type_to_vertices[node_type]
                if vid in self.vertices]

    def get_incidence_matrix(self) -> tuple[torch.Tensor, list[int], list[int]]:
        """Compute incidence matrix H where H[v,e] = 1 if v in e.

        Returns:
            H: (|V|, |E|) sparse tensor
            vertex_order: list mapping matrix row to vertex_id
            edge_order: list mapping matrix column to edge_id
        """
        vertex_order = sorted(self.vertices.keys())
        edge_order = sorted(self.hyperedges.keys())
        v_map = {vid: i for i, vid in enumerate(vertex_order)}
        e_map = {eid: j for j, eid in enumerate(edge_order)}

        indices = [[], []]
        values = []
        for eid, edge in self.hyperedges.items():
            for vid in edge.vertex_ids:
                indices[0].append(v_map[vid])
                indices[1].append(e_map[eid])
                values.append(edge.weight)

        if not values:
            H = torch.zeros(len(vertex_order), max(len(edge_order), 1))
        else:
            H = torch.sparse_coo_tensor(
                torch.tensor(indices, dtype=torch.long),
                torch.tensor(values, dtype=torch.float32),
                size=(len(vertex_order), len(edge_order)),
            ).to_dense()

        return H, vertex_order, edge_order

    def get_vertex_feature_matrix(self) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Stack all vertex features and type indicators.

        Returns:
            features: (|V|, feat_dim) tensor
            type_ids: (|V|,) tensor of NodeType values
            vertex_order: list mapping rows to vertex_ids
        """
        vertex_order = sorted(self.vertices.keys())
        if not vertex_order:
            return torch.zeros(1, 1), torch.zeros(1, dtype=torch.long), []

        features = []
        type_ids = []
        max_dim = max(v.features.shape[0] for v in self.vertices.values())

        for vid in vertex_order:
            v = self.vertices[vid]
            feat = v.features
            # Pad to max_dim
            if feat.shape[0] < max_dim:
                feat = torch.cat([feat, torch.zeros(max_dim - feat.shape[0])])
            features.append(feat)
            type_ids.append(v.node_type.value)

        return torch.stack(features), torch.tensor(type_ids, dtype=torch.long), vertex_order

    def get_edge_weight_vector(self) -> tuple[torch.Tensor, list[int]]:
        """Get hyperedge weights and causal strengths.

        Returns:
            weights: (|E|, 2) tensor — [weight, causal_strength]
            edge_order: list mapping rows to edge_ids
        """
        edge_order = sorted(self.hyperedges.keys())
        if not edge_order:
            return torch.zeros(1, 2), []

        weights = []
        for eid in edge_order:
            e = self.hyperedges[eid]
            weights.append([e.weight, e.causal_strength])
        return torch.tensor(weights, dtype=torch.float32), edge_order

    def get_causal_adjacency(self) -> tuple[torch.Tensor, list[int]]:
        """Get directed causal adjacency matrix among hyperedges.

        Returns:
            A_causal: (|E|, |E|) tensor, A[i,j]=1 means edge i causes edge j
            edge_order: list mapping indices to edge_ids
        """
        edge_order = sorted(self.hyperedges.keys())
        e_map = {eid: i for i, eid in enumerate(edge_order)}
        n = len(edge_order)
        A = torch.zeros(n, n)

        for eid in edge_order:
            for child_id in self.causal_children.get(eid, set()):
                if child_id in e_map:
                    A[e_map[eid], e_map[child_id]] = 1.0

        return A, edge_order

    def to_pyg_data(self) -> dict:
        """Convert to PyTorch Geometric compatible format.

        Returns dict with keys needed for hypergraph neural network processing.
        """
        features, type_ids, v_order = self.get_vertex_feature_matrix()
        H, v_order2, e_order = self.get_incidence_matrix()
        weights, _ = self.get_edge_weight_vector()
        causal_adj, _ = self.get_causal_adjacency()

        return {
            "x": features,                  # (|V|, feat_dim) vertex features
            "vertex_types": type_ids,        # (|V|,) type indices
            "incidence": H,                  # (|V|, |E|) incidence matrix
            "edge_weights": weights,         # (|E|, 2) weights & causal strength
            "causal_adj": causal_adj,        # (|E|, |E|) causal DAG
            "num_vertices": len(v_order),
            "num_edges": len(e_order),
            "vertex_order": v_order,
            "edge_order": e_order,
        }

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_hyperedges(self) -> int:
        return len(self.hyperedges)

    def clear(self):
        """Remove all vertices and edges."""
        self.vertices.clear()
        self.hyperedges.clear()
        self._vertex_to_edges.clear()
        for t in NodeType:
            self._type_to_vertices[t] = set()
        self.causal_parents.clear()
        self.causal_children.clear()
        self._next_vertex_id = 0
        self._next_edge_id = 0


class HypergraphBuilder:
    """Constructs a CausalHypergraph from environment observations.

    Builds vertices for all system entities and creates hyperedges representing
    offloading relationships, communication links, and task-device-node tuples.
    """

    def __init__(self, max_hyperedge_size: int = 6):
        self.max_hyperedge_size = max_hyperedge_size

    def build_from_observation(self, obs: dict, env_config: dict) -> CausalHypergraph:
        """Construct hypergraph from environment observation dict."""
        hg = CausalHypergraph(max_hyperedge_size=self.max_hyperedge_size)

        num_nodes = obs["node_features"].shape[0]
        num_devices = obs["device_features"].shape[0]
        num_channels = obs["channel_features"].shape[0]
        num_tasks = obs["task_features"].shape[0]

        # --- Add vertices ---
        node_vids = []
        for i in range(num_nodes):
            vid = hg.add_vertex(NodeType.EDGE_NODE, obs["node_features"][i], entity_id=i)
            node_vids.append(vid)

        device_vids = []
        for i in range(num_devices):
            vid = hg.add_vertex(NodeType.IOT_DEVICE, obs["device_features"][i], entity_id=i)
            device_vids.append(vid)

        channel_vids = []
        for i in range(num_channels):
            vid = hg.add_vertex(NodeType.CHANNEL, obs["channel_features"][i], entity_id=i)
            channel_vids.append(vid)

        task_vids = []
        for i in range(num_tasks):
            vid = hg.add_vertex(NodeType.TASK, obs["task_features"][i], entity_id=i)
            task_vids.append(vid)

        # --- Build hyperedges ---
        # 1. Proximity-based offloading candidates: {device, nearest nodes, best channel}
        if "device_positions" in obs and "node_positions" in obs:
            dev_pos = obs["device_positions"].numpy()
            node_pos = obs["node_positions"].numpy()

            for d_idx in range(num_devices):
                dists = np.linalg.norm(dev_pos[d_idx] - node_pos, axis=1)
                nearest_nodes = np.argsort(dists)[:3]  # Top-3 nearest

                for n_idx in nearest_nodes:
                    # Find best channel for this device-node pair
                    if "channel_gains" in obs and obs["channel_gains"].shape[0] > 0:
                        gains = obs["channel_gains"][:, d_idx, n_idx].numpy()
                        best_ch = int(np.argmax(gains))
                    else:
                        best_ch = 0

                    # Offloading hyperedge: {device, node, channel}
                    vertex_set = {device_vids[d_idx], node_vids[n_idx]}
                    if best_ch < len(channel_vids):
                        vertex_set.add(channel_vids[best_ch])

                    gain_val = float(gains[best_ch]) if "channel_gains" in obs else 1.0
                    hg.add_hyperedge(
                        vertex_set, weight=gain_val,
                        edge_type="offloading",
                    )

        # 2. Task-device hyperedges
        for t_idx in range(num_tasks):
            if t_idx < len(task_vids):
                # Source device from task features (last feature is normalized device id)
                source_frac = float(obs["task_features"][t_idx, -1])
                source_dev = int(source_frac * num_devices) % num_devices

                vertex_set = {task_vids[t_idx], device_vids[source_dev]}
                hg.add_hyperedge(vertex_set, edge_type="task_source")

        # 3. Co-located device clusters (devices within communication range)
        if "device_positions" in obs and num_devices > 1:
            dev_pos = obs["device_positions"].numpy()
            comm_range = 100.0  # meters

            for i in range(min(num_devices, 50)):  # Cap for efficiency
                nearby = []
                for j in range(i + 1, min(num_devices, 50)):
                    dist = np.linalg.norm(dev_pos[i] - dev_pos[j])
                    if dist < comm_range:
                        nearby.append(j)

                if nearby:
                    cluster = {device_vids[i]} | {device_vids[j] for j in nearby[:4]}
                    hg.add_hyperedge(cluster, edge_type="device_cluster")

        # 4. Node cooperation hyperedges (nodes sharing load)
        if num_nodes > 1 and "node_positions" in obs:
            node_pos_np = obs["node_positions"].numpy()
            for i in range(num_nodes):
                dists = np.linalg.norm(node_pos_np[i] - node_pos_np, axis=1)
                nearby = np.where((dists > 0) & (dists < 200.0))[0][:3]
                if len(nearby) > 0:
                    vertex_set = {node_vids[i]} | {node_vids[j] for j in nearby}
                    hg.add_hyperedge(vertex_set, edge_type="node_cooperation")

        return hg
