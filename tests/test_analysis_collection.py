from __future__ import annotations

import json
from pathlib import Path

from experiments.analyze_results import collect_results, load_chronos_results


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def test_collect_results_prefers_chronos_final_results(tmp_path: Path):
    baselines_dir = tmp_path / "baselines"
    chronos_dir = tmp_path / "chronos"
    ablation_dir = tmp_path / "ablation"

    _write_json(
        baselines_dir / "all_baselines_summary.json",
        {
            "Random": {
                "avg_accuracy_reward": 0.2,
                "avg_latency_reward": -0.2,
                "avg_energy_reward": -0.01,
                "avg_communication_reward": -10.0,
                "avg_combined_reward": -1.0,
            }
        },
    )

    _write_json(
        chronos_dir / "final_results.json",
        {
            "avg_accuracy_reward": 0.7,
            "avg_latency_reward": -0.02,
            "avg_energy_reward": -0.001,
            "avg_communication_reward": 0.0,
            "avg_combined_reward": 0.3,
        },
    )

    # Should be ignored because final_results.json exists.
    _write_json(ablation_dir / "CHRONOS-full_results.json", {"avg_combined_reward": -99.0})

    results = collect_results(baselines_dir, chronos_dir, ablation_dir)

    assert "CHRONOS" in results
    assert results["CHRONOS"]["avg_combined_reward"] == 0.3


def test_load_chronos_results_falls_back_to_ablation(tmp_path: Path):
    chronos_dir = tmp_path / "chronos"
    ablation_dir = tmp_path / "ablation"

    _write_json(
        ablation_dir / "CHRONOS-full_results.json",
        {
            "avg_accuracy_reward": 0.6,
            "avg_latency_reward": -0.03,
            "avg_energy_reward": -0.002,
            "avg_communication_reward": 0.0,
            "avg_combined_reward": 0.2,
        },
    )

    result = load_chronos_results(chronos_dir, ablation_dir)

    assert result["avg_accuracy_reward"] == 0.6
    assert result["avg_combined_reward"] == 0.2
