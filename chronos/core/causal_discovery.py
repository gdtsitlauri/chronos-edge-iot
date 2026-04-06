"""Online causal discovery on dynamic hypergraphs.

Extends conditional independence testing to hypergraph structures using
kernel-based methods with sliding window updates.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy import stats


class KernelCITest:
    """Kernel-based Conditional Independence Test (KCIT) for hypergraph signals.

    Tests H0: X _||_ Y | Z using a kernel-based approach with HSIC
    (Hilbert-Schmidt Independence Criterion).
    """

    def __init__(self, kernel_width: float = 1.0, significance: float = 0.05):
        self.kernel_width = kernel_width
        self.significance = significance

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix."""
        sq_dists = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
        return np.exp(-sq_dists / (2 * self.kernel_width ** 2))

    def _hsic(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Hilbert-Schmidt Independence Criterion."""
        n = X.shape[0]
        if n < 4:
            return 0.0

        K = self._rbf_kernel(X, X)
        L = self._rbf_kernel(Y, Y)
        H = np.eye(n) - np.ones((n, n)) / n

        # HSIC = trace(KHLH) / (n-1)^2
        return float(np.trace(K @ H @ L @ H) / (n - 1) ** 2)

    def test(self, X: np.ndarray, Y: np.ndarray,
             Z: np.ndarray | None = None) -> tuple[float, bool]:
        """Test X _||_ Y | Z.

        Args:
            X: (n, dx) samples of variable X
            Y: (n, dy) samples of variable Y
            Z: (n, dz) samples of conditioning variable Z (optional)

        Returns:
            (p_value, is_independent): p-value and independence decision
        """
        n = X.shape[0]
        if n < 10:
            return 1.0, True

        if Z is not None and Z.shape[1] > 0:
            # Residualize X and Y on Z using kernel ridge regression
            X_res = self._residualize(X, Z)
            Y_res = self._residualize(Y, Z)
        else:
            X_res = X
            Y_res = Y

        # Compute HSIC statistic
        hsic_val = self._hsic(X_res, Y_res)

        # Permutation test for p-value
        n_perm = 200
        perm_stats = np.zeros(n_perm)
        for i in range(n_perm):
            perm_idx = np.random.permutation(n)
            perm_stats[i] = self._hsic(X_res[perm_idx], Y_res)

        p_value = float(np.mean(perm_stats >= hsic_val))
        return p_value, p_value > self.significance

    def _residualize(self, X: np.ndarray, Z: np.ndarray,
                     reg: float = 1e-3) -> np.ndarray:
        """Remove effect of Z from X via kernel ridge regression."""
        n = X.shape[0]
        K_z = self._rbf_kernel(Z, Z)
        alpha = np.linalg.solve(K_z + reg * n * np.eye(n), X)
        X_hat = K_z @ alpha
        return X - X_hat


class OnlineCausalDiscovery:
    """Online causal discovery for dynamic hypergraphs.

    Maintains a sliding window of observations and performs incremental
    causal structure updates using constraint-based methods.
    """

    def __init__(
        self,
        window_size: int = 50,
        significance: float = 0.05,
        max_conditioning_set: int = 3,
        update_interval: int = 10,
        kernel_width: float = 1.0,
    ):
        self.window_size = window_size
        self.significance = significance
        self.max_conditioning_set = max_conditioning_set
        self.update_interval = update_interval

        self.ci_test = KernelCITest(kernel_width=kernel_width, significance=significance)

        # Sliding window buffers for each variable
        self._observation_buffer: deque[dict[str, np.ndarray]] = deque(maxlen=window_size)
        self._step_count = 0

        # Discovered causal structure: (var_i, var_j) -> causal_strength
        self.causal_edges: dict[tuple[str, str], float] = {}
        # Skeleton: undirected edges found by CI tests
        self._skeleton: set[tuple[str, str]] = set()

    def observe(self, variables: dict[str, np.ndarray]):
        """Record one observation of system variables.

        Args:
            variables: {variable_name: value_array} e.g.,
                {"node_0_load": [0.5], "node_1_load": [0.3], "latency": [120.0]}
        """
        self._observation_buffer.append(variables)
        self._step_count += 1

    def should_update(self) -> bool:
        """Check if we have enough data and it's time to update."""
        return (
            self._step_count % self.update_interval == 0
            and len(self._observation_buffer) >= 20
        )

    def discover(self, variable_names: list[str] | None = None) -> dict[tuple[str, str], float]:
        """Run causal discovery on buffered observations.

        Returns updated causal_edges dict: {(cause, effect): strength}.
        """
        if len(self._observation_buffer) < 20:
            return self.causal_edges

        # Build data matrix
        all_vars = list(self._observation_buffer[0].keys())
        if variable_names:
            all_vars = [v for v in all_vars if v in variable_names]

        n = len(self._observation_buffer)
        data = {}
        for var in all_vars:
            vals = []
            for obs in self._observation_buffer:
                v = obs.get(var, np.array([0.0]))
                vals.append(v.flatten())
            data[var] = np.array(vals)

        # Phase 1: Skeleton discovery (find undirected edges via CI tests)
        self._skeleton = set()
        for i, var_i in enumerate(all_vars):
            for j, var_j in enumerate(all_vars):
                if i >= j:
                    continue

                X = data[var_i]
                Y = data[var_j]

                # Test marginal independence
                _, independent = self.ci_test.test(X, Y)
                if independent:
                    continue

                # Test conditional independence with other variables
                found_independent = False
                other_vars = [v for v in all_vars if v != var_i and v != var_j]

                for cond_size in range(1, min(self.max_conditioning_set + 1, len(other_vars) + 1)):
                    if found_independent:
                        break
                    # Test with subsets of size cond_size
                    for k in range(min(len(other_vars), 10)):  # Cap iterations
                        cond_idx = np.random.choice(len(other_vars),
                                                     size=min(cond_size, len(other_vars)),
                                                     replace=False)
                        Z_vars = [other_vars[idx] for idx in cond_idx]
                        Z = np.column_stack([data[z] for z in Z_vars])
                        _, independent = self.ci_test.test(X, Y, Z)
                        if independent:
                            found_independent = True
                            break

                if not found_independent:
                    self._skeleton.add((var_i, var_j))

        # Phase 2: Orient edges using collider detection
        self.causal_edges = {}
        for var_i, var_j in self._skeleton:
            # Simple orientation: use temporal ordering (Granger-like)
            # and variance of residuals
            X = data[var_i]
            Y = data[var_j]

            # Test if X Granger-causes Y (lagged correlation)
            if X.shape[0] > 2:
                X_lagged = X[:-1]
                Y_current = Y[1:]
                Y_lagged = Y[:-1]

                # Correlation of lagged X with current Y (controlling for lagged Y)
                if Y_lagged.shape[1] > 0 and X_lagged.shape[1] > 0:
                    try:
                        corr_xy, _ = stats.pearsonr(X_lagged[:, 0], Y_current[:, 0])
                        corr_yx, _ = stats.pearsonr(Y_lagged[:, 0], X.flatten()[1:])
                    except (ValueError, IndexError):
                        corr_xy = corr_yx = 0.0

                    strength = abs(corr_xy) - abs(corr_yx)
                    if strength > 0.05:
                        self.causal_edges[(var_i, var_j)] = abs(corr_xy)
                    elif strength < -0.05:
                        self.causal_edges[(var_j, var_i)] = abs(corr_yx)
                    else:
                        # Bidirectional / unknown
                        avg = (abs(corr_xy) + abs(corr_yx)) / 2
                        self.causal_edges[(var_i, var_j)] = avg
                        self.causal_edges[(var_j, var_i)] = avg

        return self.causal_edges

    def get_causal_strength(self, cause: str, effect: str) -> float:
        """Get estimated causal strength from cause to effect."""
        return self.causal_edges.get((cause, effect), 0.0)

    def get_causal_parents(self, variable: str) -> dict[str, float]:
        """Get all causal parents of a variable with strengths."""
        parents = {}
        for (cause, effect), strength in self.causal_edges.items():
            if effect == variable:
                parents[cause] = strength
        return parents

    def get_ace_estimate(self, cause_var: str, effect_var: str) -> float:
        """Estimate Average Causal Effect (ACE) from buffered data.

        Uses a simple linear regression-based estimator as an approximation
        to the true interventional ACE.
        """
        if len(self._observation_buffer) < 10:
            return 0.0

        data_cause = []
        data_effect = []
        for obs in self._observation_buffer:
            if cause_var in obs and effect_var in obs:
                data_cause.append(obs[cause_var].flatten()[0])
                data_effect.append(obs[effect_var].flatten()[0])

        if len(data_cause) < 10:
            return 0.0

        X = np.array(data_cause)
        Y = np.array(data_effect)

        # Simple OLS estimate of ACE (valid under linearity + no confounding)
        X_centered = X - X.mean()
        if np.std(X_centered) < 1e-8:
            return 0.0

        ace = float(np.cov(X_centered, Y - Y.mean())[0, 1] / np.var(X_centered))
        return ace

    def get_nde_estimate(self, cause_var: str, effect_var: str,
                         mediator_var: str) -> float:
        """Estimate Natural Direct Effect (NDE).

        NDE = E[Y(1, M(0))] - E[Y(0, M(0))]
        Approximated via regression adjustment.
        """
        if len(self._observation_buffer) < 10:
            return 0.0

        data = {"cause": [], "mediator": [], "effect": []}
        for obs in self._observation_buffer:
            if all(v in obs for v in [cause_var, mediator_var, effect_var]):
                data["cause"].append(obs[cause_var].flatten()[0])
                data["mediator"].append(obs[mediator_var].flatten()[0])
                data["effect"].append(obs[effect_var].flatten()[0])

        if len(data["cause"]) < 10:
            return 0.0

        X = np.array(data["cause"])
        M = np.array(data["mediator"])
        Y = np.array(data["effect"])

        # Regression: Y = a + b*X + c*M + noise
        # NDE ≈ b (direct effect coefficient)
        A = np.column_stack([np.ones_like(X), X, M])
        try:
            coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
            return float(coeffs[1])  # Direct effect of X on Y
        except np.linalg.LinAlgError:
            return 0.0

    def apply_to_hypergraph(self, hg, variable_to_edge_map: dict[str, int]):
        """Apply discovered causal edges to the hypergraph structure.

        Args:
            hg: CausalHypergraph instance
            variable_to_edge_map: {variable_name: hyperedge_id}
        """
        for (cause, effect), strength in self.causal_edges.items():
            cause_eid = variable_to_edge_map.get(cause)
            effect_eid = variable_to_edge_map.get(effect)
            if cause_eid is not None and effect_eid is not None:
                hg.add_causal_edge(cause_eid, effect_eid)
                if effect_eid in hg.hyperedges:
                    hg.hyperedges[effect_eid].causal_strength = max(
                        hg.hyperedges[effect_eid].causal_strength, strength
                    )
