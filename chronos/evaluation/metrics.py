"""Comprehensive evaluation metrics for CHRONOS experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np


class MetricsTracker:
    """Tracks and computes all evaluation metrics across experiments."""

    def __init__(self):
        self.episode_metrics: list[dict] = []
        self.step_metrics: list[dict] = []

    def record_episode(self, metrics: dict):
        """Record metrics from one episode."""
        self.episode_metrics.append(metrics)

    def record_step(self, metrics: dict):
        """Record metrics from one step."""
        self.step_metrics.append(metrics)

    def compute_summary(self) -> dict:
        """Compute summary statistics over all recorded episodes."""
        if not self.episode_metrics:
            return {}

        summary = {}
        numeric_keys = [k for k in self.episode_metrics[0]
                        if isinstance(self.episode_metrics[0][k], (int, float, np.floating))]

        for key in numeric_keys:
            values = [m[key] for m in self.episode_metrics if key in m]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
                summary[f"{key}_min"] = float(np.min(values))
                summary[f"{key}_max"] = float(np.max(values))
                summary[f"{key}_median"] = float(np.median(values))
                summary[f"{key}_p95"] = float(np.percentile(values, 95))

        return summary

    def compute_objective_metrics(self) -> dict:
        """Compute per-objective performance metrics."""
        objectives = ["accuracy", "latency", "energy", "communication"]
        result = {}

        for obj in objectives:
            key = f"avg_{obj}_reward"
            values = [m.get(key, 0.0) for m in self.episode_metrics]
            if values:
                result[f"{obj}_mean"] = float(np.mean(values))
                result[f"{obj}_std"] = float(np.std(values))

        return result

    def compute_deadline_violation_rate(self) -> float:
        """Compute overall deadline violation rate."""
        total_violations = sum(m.get("deadline_violations", 0) for m in self.episode_metrics)
        total_tasks = sum(
            m.get("tasks_completed", 0) + m.get("tasks_failed", 0)
            for m in self.episode_metrics
        )
        return total_violations / max(total_tasks, 1)

    def compute_jains_fairness(self, per_agent_rewards: list[list[float]]) -> float:
        """Compute Jain's fairness index across agents.

        J = (sum x_i)^2 / (n * sum x_i^2)
        """
        if not per_agent_rewards:
            return 1.0

        agent_means = [np.mean(rewards) for rewards in per_agent_rewards if rewards]
        if not agent_means:
            return 1.0

        n = len(agent_means)
        x = np.array(agent_means)
        numerator = x.sum() ** 2
        denominator = n * (x ** 2).sum()

        return float(numerator / max(denominator, 1e-10))

    def compute_convergence_round(self, target_reward: float,
                                   window: int = 20) -> int:
        """Find the round at which the agent first reaches target reward consistently."""
        if not self.episode_metrics:
            return -1

        rewards = [m.get("avg_combined_reward", m.get("combined_reward", 0.0))
                    for m in self.episode_metrics]

        for i in range(len(rewards) - window):
            window_mean = np.mean(rewards[i:i + window])
            if window_mean >= target_reward:
                return i

        return -1

    def compute_communication_efficiency(self) -> dict:
        """Compute communication cost metrics."""
        total_bits = sum(m.get("total_comm_bits", 0) for m in self.episode_metrics)
        total_rounds = len(self.episode_metrics)

        return {
            "total_bits": total_bits,
            "avg_bits_per_round": total_bits / max(total_rounds, 1),
            "total_mbits": total_bits / 1e6,
        }

    def compute_energy_efficiency(self) -> dict:
        """Compute energy consumption metrics."""
        total_energy = sum(m.get("total_energy_j", 0) for m in self.episode_metrics)
        total_tasks = sum(m.get("tasks_completed", 0) for m in self.episode_metrics)

        return {
            "total_energy_j": total_energy,
            "energy_per_task_j": total_energy / max(total_tasks, 1),
            "avg_energy_per_episode": total_energy / max(len(self.episode_metrics), 1),
        }

    def to_dataframe(self):
        """Convert episode metrics to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.episode_metrics)

    def clear(self):
        self.episode_metrics.clear()
        self.step_metrics.clear()
