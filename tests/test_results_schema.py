from __future__ import annotations

from chronos.evaluation.results_schema import (
    canonicalize_metrics,
    canonicalize_method_results,
)


def test_canonicalize_from_direct_metrics():
    metrics = {
        "avg_accuracy_reward": 0.7,
        "std_accuracy_reward": 0.1,
        "avg_latency_reward": -0.02,
        "avg_energy_reward": -0.001,
        "avg_communication_reward": 0.0,
        "avg_combined_reward": 0.25,
        "avg_steps": 30,
        "deadline_violation_rate": 0.0,
        "snn_energy_ratio": 0.02,
    }

    canonical = canonicalize_metrics(metrics)

    assert canonical["avg_accuracy_reward"] == 0.7
    assert canonical["avg_combined_reward"] == 0.25
    assert canonical["avg_steps"] == 30.0
    assert canonical["deadline_violation_rate"] == 0.0


def test_canonicalize_from_summary_metrics_and_deadline_rate():
    summary = {
        "avg_accuracy_reward_mean": 0.4,
        "avg_latency_reward_mean": -0.3,
        "avg_energy_reward_mean": -0.01,
        "avg_communication_reward_mean": -5.0,
        "avg_combined_reward_mean": -1.2,
        "steps_mean": 25.0,
        "deadline_violations_mean": 5.0,
        "tasks_completed_mean": 15.0,
        "tasks_failed_mean": 5.0,
    }

    canonical = canonicalize_metrics(summary)

    assert canonical["avg_accuracy_reward"] == 0.4
    assert canonical["avg_combined_reward"] == -1.2
    assert canonical["avg_steps"] == 25.0
    # 5 / (15 + 5) = 0.25
    assert abs(canonical["deadline_violation_rate"] - 0.25) < 1e-9


def test_canonicalize_method_results_keeps_method_names():
    methods = {
        "Random": {"avg_combined_reward_mean": -2.0},
        "CHRONOS": {"avg_combined_reward": 0.3},
    }

    canonical = canonicalize_method_results(methods)

    assert set(canonical.keys()) == {"Random", "CHRONOS"}
    assert canonical["Random"]["avg_combined_reward"] == -2.0
    assert canonical["CHRONOS"]["avg_combined_reward"] == 0.3
