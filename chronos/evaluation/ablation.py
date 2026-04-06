"""Ablation study framework for CHRONOS modules."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from chronos.evaluation.metrics import MetricsTracker
from chronos.evaluation.pareto import ParetoAnalyzer


ABLATION_VARIANTS = {
    "CHRONOS-full": {
        "description": "Full CHRONOS with all modules",
        "disable": [],
    },
    "CHRONOS-noSNN": {
        "description": "Replace SNN with MLP policies",
        "disable": ["snn"],
    },
    "CHRONOS-noHG": {
        "description": "Replace hypergraph with standard pairwise graph",
        "disable": ["hypergraph"],
    },
    "CHRONOS-noCausal": {
        "description": "Remove causal components (use correlational)",
        "disable": ["causal"],
    },
    "CHRONOS-noFed": {
        "description": "Centralized learning (no federated)",
        "disable": ["federated"],
    },
    "CHRONOS-noDT": {
        "description": "No Digital Twin (learn from real system only)",
        "disable": ["digital_twin"],
    },
}


class AblationStudy:
    """Framework for systematic ablation experiments."""

    def __init__(self, base_config: dict, variants: dict | None = None):
        self.base_config = base_config
        self.variants = variants or ABLATION_VARIANTS
        self.results: dict[str, dict] = {}

    def create_variant_config(self, variant_name: str) -> dict:
        """Create a modified config for a specific ablation variant."""
        import copy
        config = copy.deepcopy(self.base_config)
        variant = self.variants[variant_name]

        for module in variant["disable"]:
            if module == "snn":
                config["spn"]["time_steps"] = 1  # Effectively reduces to MLP
                config["spn"]["num_lif_layers"] = 0
            elif module == "hypergraph":
                config["hypergraph"]["max_hyperedge_size"] = 2  # Pairwise only
            elif module == "causal":
                config["hypergraph"]["causal_significance"] = 1.0  # Never discover causal edges
                config["chse"]["causal_gate_hidden"] = 0
            elif module == "federated":
                config["hfa"]["min_participation"] = 1.0  # All clients always participate
                config["hfa"]["aggregation_interval"] = 1  # Aggregate every round
            elif module == "digital_twin":
                config["dtcs"]["sync_interval"] = 999999  # Never sync = effectively disabled

        return config

    def record_variant_results(self, variant_name: str, metrics: dict):
        """Record results from running a variant."""
        self.results[variant_name] = metrics

    def compute_ablation_impacts(self) -> dict:
        """Compute the impact of each module by comparing with full model.

        Returns:
            impacts: {variant_name: {metric: delta_from_full}}
        """
        if "CHRONOS-full" not in self.results:
            return {}

        full_results = self.results["CHRONOS-full"]
        impacts = {}

        for variant_name, variant_results in self.results.items():
            if variant_name == "CHRONOS-full":
                continue

            impact = {}
            for metric, full_val in full_results.items():
                if isinstance(full_val, (int, float)) and metric in variant_results:
                    variant_val = variant_results[metric]
                    if isinstance(variant_val, (int, float)):
                        delta = variant_val - full_val
                        denom = abs(full_val) + abs(variant_val) + 1e-8
                        # Bounded in [-200, 200], avoids exploding percentages near zero baselines.
                        relative_change_pct = 200.0 * delta / denom
                        impact[metric] = {
                            "delta": delta,
                            "relative_change_pct": relative_change_pct,
                        }

            impacts[variant_name] = {
                "description": self.variants[variant_name]["description"],
                "disabled": self.variants[variant_name]["disable"],
                "impacts": impact,
            }

        return impacts

    def rank_module_importance(self) -> list[tuple[str, float]]:
        """Rank modules by their importance (largest performance drop when removed).

        Returns:
            [(module_name, importance_score)] sorted by importance descending
        """
        impacts = self.compute_ablation_impacts()

        module_importance = {}
        for variant_name, data in impacts.items():
            disabled = data["disabled"]
            # Use combined reward as the key metric
            combined_impact = data["impacts"].get("avg_combined_reward", {})
            delta = abs(combined_impact.get("delta", 0.0))

            for module in disabled:
                module_importance[module] = module_importance.get(module, 0.0) + delta

        ranked = sorted(module_importance.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def generate_report(self) -> str:
        """Generate ablation study report."""
        lines = ["=" * 60, "ABLATION STUDY REPORT", "=" * 60, ""]

        if "CHRONOS-full" in self.results:
            lines.append("FULL MODEL PERFORMANCE:")
            for k, v in self.results["CHRONOS-full"].items():
                if isinstance(v, (int, float)):
                    lines.append(f"  {k}: {v:.4f}")
            lines.append("")

        impacts = self.compute_ablation_impacts()
        lines.append("ABLATION IMPACTS:")
        lines.append("  Note: percentages use symmetric relative change (bounded in [-200%, 200%]).")
        for variant, data in impacts.items():
            lines.append(f"\n  {variant}: {data['description']}")
            lines.append(f"  Disabled: {', '.join(data['disabled'])}")
            for metric, impact in data["impacts"].items():
                delta = impact["delta"]
                pct = impact["relative_change_pct"]
                direction = "+" if delta > 0 else ""
                lines.append(f"    {metric}: {direction}{delta:.4f} ({direction}{pct:.1f}%)")

        lines.append("\nMODULE IMPORTANCE RANKING:")
        for module, importance in self.rank_module_importance():
            lines.append(f"  {module}: {importance:.4f}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
