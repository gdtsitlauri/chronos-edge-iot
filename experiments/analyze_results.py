"""Analyze and visualize experimental results.

Generates:
1. Convergence curves (all methods)
2. Pareto front visualization
3. Ablation impact bar chart
4. Per-objective comparison tables
5. Scalability analysis
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos.evaluation.pareto import ParetoAnalyzer
from chronos.evaluation.results_schema import canonicalize_metrics, canonicalize_method_results


def load_results(results_dir: str) -> dict:
    """Load all results JSON files from a directory."""
    results = {}
    results_path = Path(results_dir)

    for f in results_path.glob("*.json"):
        with open(f) as fh:
            results[f.stem] = json.load(fh)

    return results


def _load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def load_baseline_results(baselines_dir: Path) -> dict:
    """Load baseline results and normalize them to the canonical schema."""
    if not baselines_dir.exists():
        return {}

    summary_file = baselines_dir / "all_baselines_summary.json"
    if summary_file.exists():
        return canonicalize_method_results(_load_json(summary_file))

    combined_file = baselines_dir / "all_baselines_results.json"
    if combined_file.exists():
        return canonicalize_method_results(_load_json(combined_file))

    # Fallback: load individual files if a combined file does not exist.
    results = {}
    for f in baselines_dir.glob("*_results.json"):
        if f.name.startswith("all_baselines"):
            continue
        method = f.stem.replace("_results", "").replace("_", "-")
        results[method] = canonicalize_metrics(_load_json(f))
    return results


def load_chronos_results(chronos_dir: Path, ablation_dir: Path) -> dict:
    """Load CHRONOS metrics with clear fallback priority.

    Priority:
    1) outputs/chronos/final_results.json
    2) outputs/ablation/CHRONOS-full_results.json
    3) outputs/chronos/training_log.json (last eval block)
    """
    final_file = chronos_dir / "final_results.json"
    if final_file.exists():
        data = _load_json(final_file)
        if isinstance(data.get("raw_metrics"), dict):
            merged = canonicalize_metrics(data["raw_metrics"])
            merged.update(canonicalize_metrics(data))
            return merged
        return canonicalize_metrics(data)

    ablation_full = ablation_dir / "CHRONOS-full_results.json"
    if ablation_full.exists():
        return canonicalize_metrics(_load_json(ablation_full))

    log_file = chronos_dir / "training_log.json"
    if log_file.exists():
        log = _load_json(log_file)
        if log:
            last = log[-1]
            return canonicalize_metrics(last.get("eval", last))

    return {}


def collect_results(baselines_dir: Path, chronos_dir: Path, ablation_dir: Path) -> dict:
    """Collect and normalize all experiment results for analysis."""
    all_results = load_baseline_results(baselines_dir)
    chronos_metrics = load_chronos_results(chronos_dir, ablation_dir)
    if chronos_metrics:
        all_results["CHRONOS"] = chronos_metrics
    return all_results


def generate_comparison_table(results: dict, output_dir: Path):
    """Generate a formatted comparison table."""
    objectives = ["accuracy", "latency", "energy", "communication"]

    lines = []
    lines.append("=" * 100)
    lines.append("EXPERIMENTAL RESULTS COMPARISON")
    lines.append("=" * 100)

    header = f"{'Method':<25}"
    for obj in objectives:
        header += f"{'  ' + obj.capitalize():>15}"
    header += f"{'  Combined':>15}"
    lines.append(header)
    lines.append("-" * 100)

    for method_name, data in sorted(results.items()):
        row = f"{method_name:<25}"
        for obj in objectives:
            val = data.get(f"avg_{obj}_reward", 0.0)
            row += f"{val:>15.4f}"
        combined = data.get("avg_combined_reward", 0.0)
        row += f"{combined:>15.4f}"
        lines.append(row)

    lines.append("=" * 100)

    table_str = "\n".join(lines)
    print(table_str)

    with open(output_dir / "comparison_table.txt", "w") as f:
        f.write(table_str)


def run_pareto_analysis(results: dict, output_dir: Path):
    """Run Pareto front analysis on all methods."""
    analyzer = ParetoAnalyzer()

    for method_name, data in results.items():
        objectives = []
        for obj in ["accuracy", "latency", "energy", "communication"]:
            val = data.get(f"avg_{obj}_reward", 0.0)
            objectives.append(val)

        if objectives:
            analyzer.add_results(method_name, np.array([objectives]))

    report = analyzer.generate_report()
    print(report)

    with open(output_dir / "pareto_report.txt", "w") as f:
        f.write(report)


def generate_latex_table(results: dict, output_dir: Path):
    """Generate LaTeX table for paper inclusion."""
    objectives = ["accuracy", "latency", "energy", "communication"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Performance comparison across all methods.}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{l" + "r" * (len(objectives) + 1) + "}",
        r"\toprule",
        r"Method & " + " & ".join([o.capitalize() for o in objectives]) + r" & Combined \\",
        r"\midrule",
    ]

    for method_name, data in sorted(results.items()):
        row_parts = [method_name.replace("_", r"\_")]
        for obj in objectives:
            val = data.get(f"avg_{obj}_reward", 0.0)
            row_parts.append(f"{val:.4f}")
        combined = data.get("avg_combined_reward", 0.0)
        row_parts.append(f"{combined:.4f}")
        lines.append(" & ".join(row_parts) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_str = "\n".join(lines)
    with open(output_dir / "comparison_table.tex", "w") as f:
        f.write(latex_str)

    print(f"\nLaTeX table saved to {output_dir / 'comparison_table.tex'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze CHRONOS experimental results")
    parser.add_argument("--baselines-dir", type=str, default="outputs/baselines")
    parser.add_argument("--chronos-dir", type=str, default="outputs/chronos")
    parser.add_argument("--ablation-dir", type=str, default="outputs/ablation")
    parser.add_argument("--output", type=str, default="outputs/analysis")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and normalize results
    all_results = collect_results(
        baselines_dir=Path(args.baselines_dir),
        chronos_dir=Path(args.chronos_dir),
        ablation_dir=Path(args.ablation_dir),
    )

    if not all_results:
        print("No results found. Run experiments first:")
        print("  python experiments/run_chronos.py")
        print("  python experiments/run_baselines.py")
        print("  python experiments/run_ablation.py")
        return

    # Generate analyses
    print("\n" + "=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)

    generate_comparison_table(all_results, output_dir)
    run_pareto_analysis(all_results, output_dir)
    generate_latex_table(all_results, output_dir)

    with open(output_dir / "analysis_input_snapshot.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Load and report ablation
    ablation_path = Path(args.ablation_dir)
    if ablation_path.exists():
        report_file = ablation_path / "ablation_report.txt"
        if report_file.exists():
            print(f"\n{report_file.read_text()}")

    print(f"\nAll analysis outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
