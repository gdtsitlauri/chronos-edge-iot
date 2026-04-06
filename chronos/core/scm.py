"""Structural Causal Model (SCM) for the edge-IoT system.

Implements G(t) = (U, V_H, F, P(U)) for the causal hypergraph,
supporting interventions (do-calculus) and counterfactual reasoning.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


class StructuralEquation(nn.Module):
    """A learnable structural equation: V_i = f_i(Pa(V_i), U_i).

    Parameterized as a small MLP that takes parent values and noise as input.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + output_dim, hidden_dim),  # +output_dim for noise U_i
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, parent_values: torch.Tensor,
                noise: torch.Tensor | None = None) -> torch.Tensor:
        """Compute V_i = f_i(Pa(V_i), U_i).

        Args:
            parent_values: (batch, parent_dim)
            noise: (batch, output_dim) exogenous noise. If None, sample from N(0,1).
        """
        if noise is None:
            noise = torch.randn(parent_values.shape[0], self.output_dim,
                                device=parent_values.device)
        x = torch.cat([parent_values, noise], dim=-1)
        return self.net(x)


class StructuralCausalModel(nn.Module):
    """Structural Causal Model for the edge-IoT system.

    Maintains a set of endogenous variables V_H (associated with hypergraph elements)
    and structural equations F that define causal mechanisms.

    Supports:
    - Forward sampling: generate V given U
    - Intervention: do(V_i = v) — fix V_i and propagate
    - Counterfactual: given observed (V, U), compute what V' would be under do(V_i = v')
    """

    def __init__(self, variable_dims: dict[str, int], hidden_dim: int = 32):
        """
        Args:
            variable_dims: {variable_name: dimension} for each endogenous variable
        """
        super().__init__()
        self.variable_dims = variable_dims
        self.variable_names = sorted(variable_dims.keys())
        self.hidden_dim = hidden_dim

        # Structural equations: one per endogenous variable
        self.equations = nn.ModuleDict()
        # Parent mapping: variable -> list of parent variable names
        self.parents: dict[str, list[str]] = {v: [] for v in self.variable_names}
        # Topological order (updated when parents change)
        self._topo_order: list[str] = list(self.variable_names)

        # Initialize equations (initially no parents, just noise-driven)
        for var_name in self.variable_names:
            dim = variable_dims[var_name]
            self.equations[var_name] = StructuralEquation(
                input_dim=0, output_dim=dim, hidden_dim=hidden_dim
            )

    def set_causal_structure(self, parent_map: dict[str, list[str]]):
        """Update the causal graph structure and rebuild structural equations.

        Args:
            parent_map: {variable: [list of parent variables]}
        """
        self.parents = {v: list(parents) for v, parents in parent_map.items()}

        # Rebuild equations with correct input dimensions
        for var_name in self.variable_names:
            parent_dim = sum(self.variable_dims[p] for p in self.parents.get(var_name, []))
            dim = self.variable_dims[var_name]
            self.equations[var_name] = StructuralEquation(
                input_dim=parent_dim, output_dim=dim, hidden_dim=self.hidden_dim
            )

        self._update_topo_order()

    def _update_topo_order(self):
        """Compute topological ordering of variables via Kahn's algorithm."""
        in_degree = {v: 0 for v in self.variable_names}
        children: dict[str, list[str]] = {v: [] for v in self.variable_names}

        for var, parents in self.parents.items():
            for p in parents:
                if p in children:
                    children[p].append(var)
                    in_degree[var] += 1

        queue = [v for v in self.variable_names if in_degree[v] == 0]
        order = []

        while queue:
            v = queue.pop(0)
            order.append(v)
            for child in children.get(v, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.variable_names):
            # Cycle detected — fall back to default ordering
            self._topo_order = list(self.variable_names)
        else:
            self._topo_order = order

    def forward_sample(self, batch_size: int = 1,
                       noise: dict[str, torch.Tensor] | None = None,
                       device: torch.device | None = None) -> dict[str, torch.Tensor]:
        """Sample from the SCM by forward propagation through the DAG.

        Returns: {variable_name: (batch_size, dim) tensor}
        """
        if device is None:
            device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')

        values: dict[str, torch.Tensor] = {}

        for var_name in self._topo_order:
            parent_names = self.parents.get(var_name, [])
            if parent_names:
                parent_vals = torch.cat([values[p] for p in parent_names], dim=-1)
            else:
                parent_vals = torch.zeros(batch_size, 0, device=device)

            u = noise.get(var_name) if noise else None
            values[var_name] = self.equations[var_name](parent_vals, u)

        return values

    def intervene(self, intervention: dict[str, torch.Tensor],
                  batch_size: int = 1,
                  noise: dict[str, torch.Tensor] | None = None,
                  device: torch.device | None = None) -> dict[str, torch.Tensor]:
        """Perform do-intervention: do(V_i = v_i) for variables in intervention dict.

        Intervened variables are set to fixed values (severing parent edges).
        Non-intervened variables follow their structural equations.

        Args:
            intervention: {variable_name: fixed_value_tensor}

        Returns: {variable_name: (batch_size, dim) tensor}
        """
        if device is None:
            device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')

        values: dict[str, torch.Tensor] = {}

        for var_name in self._topo_order:
            if var_name in intervention:
                # Intervention: set to fixed value, ignoring parents
                val = intervention[var_name]
                if val.dim() == 1:
                    val = val.unsqueeze(0).expand(batch_size, -1)
                values[var_name] = val
            else:
                parent_names = self.parents.get(var_name, [])
                if parent_names:
                    parent_vals = torch.cat([values[p] for p in parent_names], dim=-1)
                else:
                    parent_vals = torch.zeros(batch_size, 0, device=device)

                u = noise.get(var_name) if noise else None
                values[var_name] = self.equations[var_name](parent_vals, u)

        return values

    def counterfactual(
        self,
        observed: dict[str, torch.Tensor],
        intervention: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute counterfactual: given observed V, what would V' be under do(V_i = v')?

        Steps (three-step counterfactual):
        1. Abduction: infer noise U from observed V
        2. Action: apply intervention do(V_i = v')
        3. Prediction: propagate with inferred U and intervention

        Args:
            observed: {variable_name: observed_value}
            intervention: {variable_name: counterfactual_value}
        """
        batch_size = list(observed.values())[0].shape[0]

        # Step 1: Abduction — infer noise (approximate by residual)
        inferred_noise: dict[str, torch.Tensor] = {}
        for var_name in self._topo_order:
            parent_names = self.parents.get(var_name, [])
            if parent_names and all(p in observed for p in parent_names):
                parent_vals = torch.cat([observed[p] for p in parent_names], dim=-1)
                # f(Pa, U) = V => U ≈ V - f(Pa, 0)
                predicted = self.equations[var_name](
                    parent_vals, torch.zeros(batch_size, self.variable_dims[var_name])
                )
                if var_name in observed:
                    inferred_noise[var_name] = observed[var_name] - predicted
                else:
                    inferred_noise[var_name] = torch.zeros(batch_size, self.variable_dims[var_name])
            else:
                inferred_noise[var_name] = torch.zeros(batch_size, self.variable_dims[var_name])

        # Steps 2 & 3: Action + Prediction
        return self.intervene(intervention, batch_size, inferred_noise)

    def compute_ace(self, cause_var: str, effect_var: str,
                    num_samples: int = 100, delta: float = 1.0) -> float:
        """Estimate Average Causal Effect: ACE = E[Y | do(X=x+d)] - E[Y | do(X=x)].

        Uses Monte Carlo estimation.
        """
        dim_x = self.variable_dims[cause_var]

        with torch.no_grad():
            # Baseline: do(X = 0)
            intervention_base = {cause_var: torch.zeros(num_samples, dim_x)}
            values_base = self.intervene(intervention_base, num_samples)

            # Treatment: do(X = delta)
            intervention_treat = {cause_var: torch.full((num_samples, dim_x), delta)}
            values_treat = self.intervene(intervention_treat, num_samples)

            if effect_var in values_base and effect_var in values_treat:
                ace = (values_treat[effect_var] - values_base[effect_var]).mean().item()
            else:
                ace = 0.0

        return ace

    def compute_nde(self, cause_var: str, effect_var: str, mediator_var: str,
                    num_samples: int = 100, delta: float = 1.0) -> float:
        """Estimate Natural Direct Effect: NDE = E[Y(x+d, M(x))] - E[Y(x, M(x))].

        The effect of X on Y *not* through M.
        """
        dim_x = self.variable_dims[cause_var]

        with torch.no_grad():
            # Get mediator value under baseline: M(x=0)
            intervention_base = {cause_var: torch.zeros(num_samples, dim_x)}
            values_base = self.intervene(intervention_base, num_samples)
            M_baseline = values_base.get(mediator_var,
                                          torch.zeros(num_samples, self.variable_dims.get(mediator_var, 1)))

            # Y under treatment X=delta but mediator fixed to M(x=0)
            intervention_nde = {
                cause_var: torch.full((num_samples, dim_x), delta),
                mediator_var: M_baseline,
            }
            values_nde = self.intervene(intervention_nde, num_samples)

            if effect_var in values_base and effect_var in values_nde:
                nde = (values_nde[effect_var] - values_base[effect_var]).mean().item()
            else:
                nde = 0.0

        return nde

    def fit_from_data(self, data: dict[str, torch.Tensor],
                      num_epochs: int = 100, lr: float = 1e-3):
        """Fit structural equations from observational data.

        Args:
            data: {variable_name: (n_samples, dim) tensor}
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        n = list(data.values())[0].shape[0]

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            total_loss = 0.0

            for var_name in self._topo_order:
                parent_names = self.parents.get(var_name, [])
                target = data[var_name]

                if parent_names:
                    parent_vals = torch.cat([data[p] for p in parent_names], dim=-1)
                else:
                    parent_vals = torch.zeros(n, 0)

                # Predict with zero noise (learn the deterministic part)
                predicted = self.equations[var_name](
                    parent_vals, torch.zeros(n, self.variable_dims[var_name])
                )
                loss = nn.functional.mse_loss(predicted, target)
                total_loss += loss

            total_loss.backward()
            optimizer.step()

        return total_loss.item()
