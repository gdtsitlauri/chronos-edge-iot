"""Module 4: Hypergraph-Federated Aggregation (HFA).

Implements federated learning with:
- Causal contribution weighting: alpha_i^causal based on ACE
- Hyperedge-aware aggregation: aggregate within hyperedges first, then across
- Non-IID correction via causal optimal transport
"""

from __future__ import annotations

from typing import Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_knopp(cost_matrix: torch.Tensor, reg: float = 0.1,
                   max_iter: int = 100, threshold: float = 1e-5) -> torch.Tensor:
    """Sinkhorn-Knopp algorithm for entropic optimal transport.

    Solves: min_Pi <Pi, C> + reg * KL(Pi || a x b)
    where a, b are uniform marginals.

    Args:
        cost_matrix: (n, m) cost matrix
        reg: entropic regularization
        max_iter: maximum Sinkhorn iterations
        threshold: convergence threshold

    Returns:
        transport_plan: (n, m) optimal transport matrix
    """
    n, m = cost_matrix.shape
    a = torch.ones(n, device=cost_matrix.device) / n
    b = torch.ones(m, device=cost_matrix.device) / m

    K = torch.exp(-cost_matrix / reg)
    K = K.clamp(min=1e-10)

    u = torch.ones(n, device=cost_matrix.device)
    for _ in range(max_iter):
        u_prev = u.clone()
        v = b / (K.t() @ u)
        u = a / (K @ v)

        if (u - u_prev).abs().max() < threshold:
            break

    transport_plan = torch.diag(u) @ K @ torch.diag(v)
    return transport_plan


class FederatedModel:
    """Wrapper for a federated learning model with local training capabilities."""

    def __init__(self, model: nn.Module, lr: float = 0.01, momentum: float = 0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    def get_parameters(self) -> OrderedDict:
        """Get model parameters as an ordered dict of tensors."""
        return OrderedDict(
            (name, param.data.clone())
            for name, param in self.model.named_parameters()
        )

    def set_parameters(self, params: OrderedDict):
        """Set model parameters from an ordered dict."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.copy_(params[name])

    def get_gradient(self) -> OrderedDict:
        """Get current gradients as ordered dict."""
        return OrderedDict(
            (name, param.grad.clone() if param.grad is not None else torch.zeros_like(param))
            for name, param in self.model.named_parameters()
        )

    def local_train(self, data_loader, num_steps: int = 5,
                    loss_fn: nn.Module | None = None) -> float:
        """Perform local SGD steps.

        Args:
            data_loader: iterable of (x, y) batches
            num_steps: number of local gradient steps
            loss_fn: loss function (default: CrossEntropy)

        Returns:
            average loss
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        total_loss = 0.0
        steps_done = 0
        data_iter = iter(data_loader)

        for step in range(num_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    break

            self.optimizer.zero_grad()
            output = self.model(x)
            loss = loss_fn(output, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps_done += 1

        return total_loss / max(steps_done, 1)


class HypergraphFederatedAggregation:
    """Federated aggregation with causal hypergraph structure.

    Three-phase aggregation:
    1. Causal contribution weighting (alpha_i^causal based on ACE)
    2. Hyperedge-aware aggregation (within-hyperedge first, then across)
    3. Non-IID correction via causal optimal transport
    """

    def __init__(
        self,
        num_clients: int,
        aggregation_interval: int = 5,
        ot_regularization: float = 0.1,
        ot_max_iter: int = 100,
        compression_ratio: float = 0.1,
        min_participation: float = 0.3,
    ):
        self.num_clients = num_clients
        self.aggregation_interval = aggregation_interval
        self.ot_reg = ot_regularization
        self.ot_max_iter = ot_max_iter
        self.compression_ratio = compression_ratio
        self.min_participation = min_participation

        # Track client contributions
        self.client_losses: dict[int, list[float]] = {i: [] for i in range(num_clients)}
        self.causal_weights = torch.ones(num_clients) / num_clients
        self.round_count = 0

    def select_participants(self, num_clients: int | None = None) -> list[int]:
        """Select participating clients for this round.

        Biases selection toward clients with higher causal contribution.
        """
        if num_clients is None:
            num_clients = max(int(self.num_clients * self.min_participation), 1)

        probs = F.softmax(self.causal_weights, dim=0).numpy()
        selected = np.random.choice(
            self.num_clients, size=min(num_clients, self.num_clients),
            replace=False, p=probs,
        )
        return sorted(selected.tolist())

    def compress_gradient(self, gradient: OrderedDict) -> OrderedDict:
        """Top-K gradient sparsification.

        Keep only the top compression_ratio fraction of gradient elements.
        """
        compressed = OrderedDict()
        for name, grad in gradient.items():
            flat = grad.flatten()
            k = max(1, int(flat.numel() * self.compression_ratio))
            topk_vals, topk_idx = torch.topk(flat.abs(), k)

            sparse_grad = torch.zeros_like(flat)
            sparse_grad[topk_idx] = flat[topk_idx]
            compressed[name] = sparse_grad.reshape(grad.shape)

        return compressed

    def compute_causal_weights(
        self,
        client_params: dict[int, OrderedDict],
        global_params: OrderedDict,
        client_losses: dict[int, float],
        causal_effects: dict[int, float] | None = None,
    ) -> torch.Tensor:
        """Compute causal contribution weights for each client.

        alpha_i^causal = ACE(w_i -> F(w_global)) / sum_j ACE(w_j -> F(w_global))

        When ACE estimates are unavailable, falls back to loss-based importance.

        Args:
            client_params: {client_id: parameters}
            global_params: current global parameters
            client_losses: {client_id: local loss}
            causal_effects: {client_id: estimated ACE} (optional)

        Returns:
            weights: (num_participating_clients,) normalized weights
        """
        participating = sorted(client_params.keys())
        n = len(participating)

        if causal_effects:
            # Use ACE-based weights
            raw_weights = torch.tensor([
                max(causal_effects.get(i, 0.0), 1e-6) for i in participating
            ])
        else:
            # Fallback: inverse loss weighting (lower loss = higher weight)
            losses = torch.tensor([
                max(client_losses.get(i, 1.0), 1e-6) for i in participating
            ])
            raw_weights = 1.0 / losses

        weights = F.softmax(raw_weights, dim=0)
        return weights

    def hyperedge_aggregate(
        self,
        client_params: dict[int, OrderedDict],
        hyperedge_groups: list[set[int]],
        causal_weights: torch.Tensor,
        participating: list[int],
    ) -> OrderedDict:
        """Two-level hyperedge-aware aggregation.

        Level 1: Average within each hyperedge
        Level 2: Weighted average across hyperedges

        Args:
            client_params: {client_id: parameters}
            hyperedge_groups: list of sets of client ids sharing a hyperedge
            causal_weights: per-client causal weights
            participating: list of participating client ids

        Returns:
            aggregated_params: global model parameters
        """
        client_idx_map = {cid: i for i, cid in enumerate(participating)}

        # Level 1: Within-hyperedge aggregation
        hyperedge_params: list[tuple[OrderedDict, float]] = []

        for group in hyperedge_groups:
            members = [c for c in group if c in client_params]
            if not members:
                continue

            # Average parameters within hyperedge
            param_names = list(client_params[members[0]].keys())
            avg_params = OrderedDict()
            total_weight = 0.0

            for name in param_names:
                weighted_sum = torch.zeros_like(client_params[members[0]][name])
                for cid in members:
                    idx = client_idx_map.get(cid, 0)
                    w = float(causal_weights[idx]) if idx < len(causal_weights) else 1.0 / len(members)
                    weighted_sum += w * client_params[cid][name]
                    total_weight += w
                avg_params[name] = weighted_sum

            # Hyperedge-level weight is sum of member causal weights
            he_weight = sum(
                float(causal_weights[client_idx_map[c]])
                for c in members if c in client_idx_map
            )
            hyperedge_params.append((avg_params, he_weight))

        if not hyperedge_params:
            # Fallback to simple weighted average
            return self._simple_weighted_avg(client_params, causal_weights, participating)

        # Level 2: Across-hyperedge aggregation
        total_weight = sum(w for _, w in hyperedge_params)
        if total_weight < 1e-10:
            total_weight = 1.0

        param_names = list(hyperedge_params[0][0].keys())
        global_params = OrderedDict()

        for name in param_names:
            weighted_sum = torch.zeros_like(hyperedge_params[0][0][name])
            for he_params, he_weight in hyperedge_params:
                weighted_sum += (he_weight / total_weight) * he_params[name]
            global_params[name] = weighted_sum

        return global_params

    def _simple_weighted_avg(
        self, client_params: dict[int, OrderedDict],
        weights: torch.Tensor, participating: list[int],
    ) -> OrderedDict:
        """Simple weighted average (fallback)."""
        param_names = list(client_params[participating[0]].keys())
        result = OrderedDict()

        for name in param_names:
            weighted_sum = torch.zeros_like(client_params[participating[0]][name])
            for i, cid in enumerate(participating):
                w = float(weights[i]) if i < len(weights) else 1.0 / len(participating)
                weighted_sum += w * client_params[cid][name]
            result[name] = weighted_sum

        return result

    def causal_ot_correction(
        self,
        client_gradients: dict[int, OrderedDict],
        global_gradient: OrderedDict,
        causal_structure: torch.Tensor | None = None,
    ) -> dict[int, OrderedDict]:
        """Non-IID correction via causal optimal transport.

        Computes OT plan between each client's gradient distribution and global gradient,
        constrained by causal structure.

        Args:
            client_gradients: {client_id: gradient_dict}
            global_gradient: global average gradient
            causal_structure: (num_param_groups, num_param_groups) causal adjacency

        Returns:
            corrected_gradients: {client_id: corrected_gradient}
        """
        corrected = {}

        for cid, client_grad in client_gradients.items():
            corrected_grad = OrderedDict()

            for name in client_grad:
                local_flat = client_grad[name].flatten()
                global_flat = global_gradient[name].flatten()

                n = local_flat.shape[0]
                if n <= 1:
                    corrected_grad[name] = client_grad[name]
                    continue

                # Subsample for efficiency
                max_samples = 500
                if n > max_samples:
                    idx = torch.randperm(n)[:max_samples]
                    local_sub = local_flat[idx].unsqueeze(1)
                    global_sub = global_flat[idx].unsqueeze(1)
                else:
                    local_sub = local_flat.unsqueeze(1)
                    global_sub = global_flat.unsqueeze(1)

                # Cost matrix: squared distance between gradient elements
                cost = torch.cdist(local_sub, global_sub, p=2).pow(2)

                # Apply causal mask if available
                if causal_structure is not None and causal_structure.shape[0] == cost.shape[0]:
                    # Zero out cost for causally independent parameter pairs
                    causal_mask = (causal_structure > 0).float()
                    cost = cost * causal_mask + cost.max() * (1 - causal_mask)

                # Solve OT
                transport = sinkhorn_knopp(cost, reg=self.ot_reg, max_iter=self.ot_max_iter)

                # Apply correction: shift local gradient towards global via transport plan
                correction = transport.sum(dim=1) * (global_sub.squeeze() - local_sub.squeeze())

                if n > max_samples:
                    full_correction = torch.zeros_like(local_flat)
                    full_correction[idx] = correction
                else:
                    full_correction = correction

                corrected_grad[name] = (
                    client_grad[name] + 0.5 * full_correction.reshape(client_grad[name].shape)
                )

            corrected[cid] = corrected_grad

        return corrected

    def aggregate(
        self,
        client_models: dict[int, FederatedModel],
        global_model: FederatedModel,
        hyperedge_groups: list[set[int]],
        client_data_sizes: dict[int, int] | None = None,
        causal_effects: dict[int, float] | None = None,
    ) -> OrderedDict:
        """Full HFA aggregation pipeline.

        1. Collect client parameters and gradients
        2. Compute causal contribution weights
        3. Compress gradients
        4. Apply causal OT correction for non-IID
        5. Hyperedge-aware aggregation

        Returns:
            new_global_params: updated global model parameters
        """
        self.round_count += 1

        # Select participants
        participating = self.select_participants()

        # Collect parameters
        client_params = {}
        client_losses = {}
        client_gradients = {}

        global_params = global_model.get_parameters()

        for cid in participating:
            if cid not in client_models:
                continue
            client_params[cid] = client_models[cid].get_parameters()
            client_losses[cid] = self.client_losses.get(cid, [1.0])[-1] if self.client_losses.get(cid) else 1.0

            # Compute gradient as difference from global
            grad = OrderedDict()
            for name in global_params:
                grad[name] = client_params[cid][name] - global_params[name]
            client_gradients[cid] = grad

        if not client_params:
            return global_params

        participating = sorted(client_params.keys())

        # Compute causal weights
        causal_weights = self.compute_causal_weights(
            client_params, global_params, client_losses, causal_effects
        )
        self.causal_weights[participating] = causal_weights

        # Compress gradients
        compressed_gradients = {
            cid: self.compress_gradient(grad)
            for cid, grad in client_gradients.items()
        }

        # Compute global average gradient
        global_grad = OrderedDict()
        for name in global_params:
            global_grad[name] = torch.zeros_like(global_params[name])
            for i, cid in enumerate(participating):
                w = float(causal_weights[i])
                global_grad[name] += w * compressed_gradients[cid][name]

        # Causal OT correction
        corrected_gradients = self.causal_ot_correction(
            compressed_gradients, global_grad
        )

        # Apply corrected gradients to get client params
        corrected_params = {}
        for cid in participating:
            corrected_params[cid] = OrderedDict()
            for name in global_params:
                corrected_params[cid][name] = global_params[name] + corrected_gradients[cid][name]

        # Hyperedge-aware aggregation
        # Filter hyperedge groups to participating clients
        active_groups = []
        for group in hyperedge_groups:
            active = group & set(participating)
            if len(active) >= 2:
                active_groups.append(active)

        if active_groups:
            new_params = self.hyperedge_aggregate(
                corrected_params, active_groups, causal_weights, participating
            )
        else:
            new_params = self._simple_weighted_avg(
                corrected_params, causal_weights, participating
            )

        return new_params

    def record_client_loss(self, client_id: int, loss: float):
        """Record a client's training loss for tracking."""
        if client_id not in self.client_losses:
            self.client_losses[client_id] = []
        self.client_losses[client_id].append(loss)

    def get_communication_cost(self, client_params: OrderedDict) -> int:
        """Estimate communication cost in bits after compression."""
        total_elements = sum(p.numel() for p in client_params.values())
        transmitted = int(total_elements * self.compression_ratio)
        bits_per_element = 32  # float32
        return transmitted * bits_per_element
