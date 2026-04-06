"""Pareto front analysis for multi-objective evaluation."""

from __future__ import annotations

from typing import Optional

import numpy as np


class ParetoAnalyzer:
    """Analyzes multi-objective results and computes Pareto metrics."""

    def __init__(self, objective_names: list[str] | None = None,
                 maximize: list[bool] | None = None):
        """
        Args:
            objective_names: names for each objective
            maximize: True if objective should be maximized, False for minimize.
                      Default: [True, False, False, False] (accuracy max, rest min)
        """
        self.objective_names = objective_names or ["accuracy", "latency", "energy", "communication"]
        self.maximize = maximize or [True, False, False, False]
        self.num_objectives = len(self.objective_names)

        # Results storage: {method_name: (num_runs, num_objectives)}
        self.results: dict[str, np.ndarray] = {}

    def add_results(self, method_name: str, objective_values: np.ndarray):
        """Add results for a method.

        Args:
            method_name: e.g., "CHRONOS", "FedAvg", "MAPPO"
            objective_values: (num_runs, num_objectives) array
        """
        self.results[method_name] = np.array(objective_values)

    def is_dominated(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check if point a is dominated by point b.

        b dominates a if b is at least as good in all objectives and strictly
        better in at least one.
        """
        at_least_as_good = True
        strictly_better = False

        for i in range(self.num_objectives):
            if self.maximize[i]:
                if b[i] < a[i]:
                    at_least_as_good = False
                elif b[i] > a[i]:
                    strictly_better = True
            else:
                if b[i] > a[i]:
                    at_least_as_good = False
                elif b[i] < a[i]:
                    strictly_better = True

        return at_least_as_good and strictly_better

    def compute_pareto_front(self, points: np.ndarray) -> np.ndarray:
        """Find Pareto-optimal points.

        Args:
            points: (n, num_objectives) array

        Returns:
            pareto_mask: (n,) boolean array, True for Pareto-optimal points
        """
        n = points.shape[0]
        pareto_mask = np.ones(n, dtype=bool)

        for i in range(n):
            if not pareto_mask[i]:
                continue
            for j in range(n):
                if i == j or not pareto_mask[j]:
                    continue
                if self.is_dominated(points[i], points[j]):
                    pareto_mask[i] = False
                    break

        return pareto_mask

    def compute_hypervolume(self, points: np.ndarray,
                            reference_point: np.ndarray) -> float:
        """Compute hypervolume indicator (2D or approximation for higher dimensions).

        For 2D: exact computation via sweepline.
        For >2D: Monte Carlo approximation.

        Args:
            points: (n, num_objectives) Pareto-front points
            reference_point: reference point for hypervolume computation

        Returns:
            hypervolume: scalar
        """
        if points.shape[0] == 0:
            return 0.0

        # Normalize directions: convert all to "maximize" for computation
        norm_points = points.copy()
        norm_ref = reference_point.copy()
        for i in range(self.num_objectives):
            if not self.maximize[i]:
                norm_points[:, i] = -norm_points[:, i]
                norm_ref[i] = -norm_ref[i]

        if self.num_objectives == 2:
            return self._hypervolume_2d(norm_points, norm_ref)
        else:
            return self._hypervolume_mc(norm_points, norm_ref, num_samples=10000)

    def _hypervolume_2d(self, points: np.ndarray, ref: np.ndarray) -> float:
        """Exact 2D hypervolume via sweepline."""
        # Filter points that dominate reference
        valid = (points[:, 0] > ref[0]) & (points[:, 1] > ref[1])
        pts = points[valid]

        if len(pts) == 0:
            return 0.0

        # Sort by first objective descending
        sorted_idx = np.argsort(-pts[:, 0])
        pts = pts[sorted_idx]

        hv = 0.0
        prev_y = ref[1]
        for p in pts:
            if p[1] > prev_y:
                hv += (p[0] - ref[0]) * (p[1] - prev_y)
                prev_y = p[1]

        return float(hv)

    def _hypervolume_mc(self, points: np.ndarray, ref: np.ndarray,
                        num_samples: int = 10000) -> float:
        """Monte Carlo hypervolume approximation for >2D."""
        # Find bounding box
        upper = np.max(points, axis=0)

        # Volume of bounding box
        box_volume = np.prod(upper - ref)
        if box_volume <= 0:
            return 0.0

        # Sample uniformly in bounding box and count dominated points
        samples = np.random.uniform(ref, upper, size=(num_samples, self.num_objectives))

        dominated_count = 0
        for s in samples:
            # Check if sample is dominated by any point in the front
            for p in points:
                if np.all(p >= s):
                    dominated_count += 1
                    break

        return float(box_volume * dominated_count / num_samples)

    def compare_methods(self, reference_point: np.ndarray | None = None) -> dict:
        """Compare all registered methods using Pareto metrics.

        Returns:
            comparison: dict with hypervolume, Pareto dominance, etc.
        """
        if reference_point is None:
            reference_point = np.array([0.0, 1000.0, 10000.0, 1e8])

        comparison = {}
        all_points = []
        method_labels = []

        for method, values in self.results.items():
            mean_values = values.mean(axis=0) if values.ndim > 1 else values

            # Pareto front of this method's runs
            if values.ndim > 1 and values.shape[0] > 1:
                pareto_mask = self.compute_pareto_front(values)
                pareto_points = values[pareto_mask]
            else:
                pareto_points = values.reshape(1, -1)

            hv = self.compute_hypervolume(pareto_points, reference_point)

            comparison[method] = {
                "mean_objectives": {
                    name: float(mean_values[i])
                    for i, name in enumerate(self.objective_names)
                    if i < len(mean_values)
                },
                "hypervolume": hv,
                "pareto_front_size": int(pareto_points.shape[0]),
                "num_runs": int(values.shape[0]) if values.ndim > 1 else 1,
            }

            all_points.append(mean_values)
            method_labels.append(method)

        # Check pairwise dominance
        all_pts = np.array(all_points)
        for i, m1 in enumerate(method_labels):
            dominated_by = []
            for j, m2 in enumerate(method_labels):
                if i != j and self.is_dominated(all_pts[i], all_pts[j]):
                    dominated_by.append(m2)
            comparison[m1]["dominated_by"] = dominated_by

        return comparison

    def generate_report(self) -> str:
        """Generate a text report comparing all methods."""
        comparison = self.compare_methods()

        lines = ["=" * 60, "PARETO ANALYSIS REPORT", "=" * 60, ""]

        # Rank by hypervolume
        ranked = sorted(comparison.items(), key=lambda x: x[1]["hypervolume"], reverse=True)

        lines.append("RANKING BY HYPERVOLUME:")
        for rank, (method, data) in enumerate(ranked, 1):
            lines.append(f"  {rank}. {method}: HV = {data['hypervolume']:.6f}")

        lines.append("")
        lines.append("DETAILED RESULTS:")
        for method, data in ranked:
            lines.append(f"\n  {method}:")
            for obj, val in data["mean_objectives"].items():
                lines.append(f"    {obj}: {val:.4f}")
            lines.append(f"    Hypervolume: {data['hypervolume']:.6f}")
            lines.append(f"    Pareto front size: {data['pareto_front_size']}")
            if data["dominated_by"]:
                lines.append(f"    Dominated by: {', '.join(data['dominated_by'])}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
