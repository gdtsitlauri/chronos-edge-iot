"""Run multi-seed reproducibility experiments and significance tests."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from scipy import stats
except Exception:
    stats = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos.evaluation.results_schema import canonicalize_metrics


def parse_seeds(seed_arg: str) -> list[int]:
    """Parse a comma-separated seed list."""
    seeds = []
    for part in seed_arg.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def run_command(command: list[str]):
    """Run a command and fail fast on error."""
    print("$", " ".join(command))
    subprocess.run(command, check=True)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def cohen_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d effect size."""
    if len(a) < 2 or len(b) < 2:
        return 0.0

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / max(len(a) + len(b) - 2, 1)
    if pooled <= 0:
        return 0.0
    return (mean_a - mean_b) / float(np.sqrt(pooled))


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(statistics.median(values)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run CHRONOS reproducibility experiments")
    parser.add_argument("--config", type=str, default="configs/fast_experiment.yaml")
    parser.add_argument("--output", type=str, default="outputs/reproducibility")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--chronos-rounds", type=int, default=30)
    parser.add_argument("--baseline-rounds", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--baselines", nargs="+", default=None,
                        help="Optional subset, e.g. Random MAPPO QMIX")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed = []

    for seed in seeds:
        print("\n" + "=" * 72)
        print(f"SEED {seed}")
        print("=" * 72)

        seed_dir = output_dir / f"seed_{seed}"
        chronos_out = seed_dir / "chronos"
        baselines_out = seed_dir / "baselines"

        run_command([
            sys.executable,
            "experiments/run_chronos.py",
            "--config", args.config,
            "--output", str(chronos_out),
            "--rounds", str(args.chronos_rounds),
            "--seed", str(seed),
            "--device", args.device,
        ])

        if not args.skip_baselines:
            cmd = [
                sys.executable,
                "experiments/run_baselines.py",
                "--config", args.config,
                "--output", str(baselines_out),
                "--train-rounds", str(args.baseline_rounds),
                "--eval-episodes", str(args.eval_episodes),
                "--seed", str(seed),
            ]
            if args.baselines:
                cmd += ["--baselines", *args.baselines]
            run_command(cmd)

        seed_results = {"seed": seed, "methods": {}}

        chronos_file = chronos_out / "final_results.json"
        if chronos_file.exists():
            chronos_data = load_json(chronos_file)
            source = chronos_data.get("raw_metrics", chronos_data)
            seed_results["methods"]["CHRONOS"] = canonicalize_metrics(source)

        baselines_summary = baselines_out / "all_baselines_summary.json"
        if baselines_summary.exists():
            baseline_data = load_json(baselines_summary)
            for method, metrics in baseline_data.items():
                seed_results["methods"][method] = canonicalize_metrics(metrics)

        per_seed.append(seed_results)

    with open(output_dir / "per_seed_results.json", "w") as f:
        json.dump(per_seed, f, indent=2)

    # Aggregate by method
    method_values: dict[str, list[float]] = {}
    for run in per_seed:
        for method, metrics in run.get("methods", {}).items():
            if "avg_combined_reward" in metrics:
                method_values.setdefault(method, []).append(float(metrics["avg_combined_reward"]))

    summary = {method: summarize(vals) for method, vals in method_values.items()}
    with open(output_dir / "reproducibility_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    significance = {}
    chronos_vals = method_values.get("CHRONOS", [])
    if stats is not None and len(chronos_vals) >= 2:
        for method, vals in method_values.items():
            if method == "CHRONOS" or len(vals) < 2:
                continue
            t_stat, p_value = stats.ttest_ind(chronos_vals, vals, equal_var=False)
            significance[method] = {
                "chronos_mean": float(np.mean(chronos_vals)),
                "baseline_mean": float(np.mean(vals)),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "cohen_d": float(cohen_d(chronos_vals, vals)),
            }

    with open(output_dir / "significance_tests.json", "w") as f:
        json.dump(significance, f, indent=2)

    # Text report
    lines = [
        "=" * 72,
        "REPRODUCIBILITY REPORT",
        "=" * 72,
        "",
        "SEEDS: " + ", ".join(str(s) for s in seeds),
        "",
        "SUMMARY (avg_combined_reward):",
    ]

    for method, data in sorted(summary.items()):
        if data.get("n", 0) == 0:
            continue
        lines.append(
            f"  {method}: n={data['n']}, mean={data['mean']:.4f}, std={data['std']:.4f}, "
            f"min={data['min']:.4f}, max={data['max']:.4f}"
        )

    lines.append("")
    lines.append("SIGNIFICANCE TESTS (CHRONOS vs baseline):")
    if not significance:
        if stats is None:
            lines.append("  Skipped: scipy.stats unavailable")
        else:
            lines.append("  Skipped: need at least 2 seeds per method")
    else:
        for method, data in sorted(significance.items()):
            lines.append(
                f"  {method}: p={data['p_value']:.6f}, t={data['t_stat']:.4f}, "
                f"Cohen_d={data['cohen_d']:.4f}"
            )

    report = "\n".join(lines)
    print("\n" + report)

    with open(output_dir / "reproducibility_report.txt", "w") as f:
        f.write(report)

    print(f"\nSaved reproducibility outputs to: {output_dir}")


if __name__ == "__main__":
    main()
