"""Utilities for normalizing experiment outputs to a common schema."""

from __future__ import annotations

from typing import Any


CANONICAL_KEYS = [
    "avg_accuracy_reward",
    "std_accuracy_reward",
    "avg_latency_reward",
    "std_latency_reward",
    "avg_energy_reward",
    "std_energy_reward",
    "avg_communication_reward",
    "std_communication_reward",
    "avg_combined_reward",
    "std_combined_reward",
    "avg_steps",
    "deadline_violation_rate",
    "snn_energy_ratio",
]


def _to_float(value: Any) -> float | None:
    """Best-effort conversion to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_metric(metrics: dict[str, Any], key: str) -> float | None:
    """Pick a metric from direct or summary-style keys."""
    direct = _to_float(metrics.get(key))
    if direct is not None:
        return direct

    mean_key = f"{key}_mean"
    return _to_float(metrics.get(mean_key))


def canonicalize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Normalize metrics from training/baseline/ablation outputs.

    Supports both direct keys (e.g., ``avg_accuracy_reward``) and
    summary keys (e.g., ``avg_accuracy_reward_mean``).
    """
    canonical: dict[str, float] = {}

    avg_std_pairs = [
        "accuracy",
        "latency",
        "energy",
        "communication",
        "combined",
    ]

    for name in avg_std_pairs:
        avg_key = f"avg_{name}_reward"
        std_key = f"std_{name}_reward"

        avg_val = _pick_metric(metrics, avg_key)
        if avg_val is not None:
            canonical[avg_key] = avg_val

        std_val = _pick_metric(metrics, std_key)
        if std_val is not None:
            canonical[std_key] = std_val

    avg_steps = _pick_metric(metrics, "avg_steps")
    if avg_steps is None:
        avg_steps = _to_float(metrics.get("steps_mean"))
    if avg_steps is not None:
        canonical["avg_steps"] = avg_steps

    deadline_rate = _to_float(metrics.get("deadline_violation_rate"))
    if deadline_rate is None:
        deadline_violations = _to_float(metrics.get("deadline_violations_mean"))
        tasks_completed = _to_float(metrics.get("tasks_completed_mean"))
        tasks_failed = _to_float(metrics.get("tasks_failed_mean"))
        if deadline_violations is not None and tasks_completed is not None and tasks_failed is not None:
            total_tasks = max(tasks_completed + tasks_failed, 1.0)
            deadline_rate = deadline_violations / total_tasks
    if deadline_rate is not None:
        canonical["deadline_violation_rate"] = deadline_rate

    snn_ratio = _to_float(metrics.get("snn_energy_ratio"))
    if snn_ratio is not None:
        canonical["snn_energy_ratio"] = snn_ratio

    return canonical


def canonicalize_method_results(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Canonicalize metrics for all methods in a result map."""
    normalized: dict[str, dict[str, float]] = {}
    for method, metrics in results.items():
        normalized[method] = canonicalize_metrics(metrics)
    return normalized
