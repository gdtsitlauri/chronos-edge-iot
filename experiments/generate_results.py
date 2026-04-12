"""Generate results/ directory: comparison table, ablation report, and README snippet.

Usage:
    python experiments/generate_results.py \
        --repro outputs/reproducibility \
        --ablation outputs/ablation \
        --results results/

Reads the per_seed_results.json from the reproducibility run and the
ablation_impacts.json + per-variant result JSONs from the ablation run,
then writes:
  results/comparison_table.md
  results/comparison_table.tex
  results/ablation_report.md
  results/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def fmt(v, decimals=3):
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def pct_change(new, ref):
    if ref is None or ref == 0:
        return None
    return (new - ref) / abs(ref) * 100


# ─── aggregate per-seed results ───────────────────────────────────────────────

def aggregate_per_seed(per_seed: list[dict]) -> dict[str, dict]:
    """Return {method: {metric: {mean, std, n}}}."""
    method_vals: dict[str, dict[str, list[float]]] = {}
    for run in per_seed:
        for method, metrics in run.get("methods", {}).items():
            method_vals.setdefault(method, {})
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    method_vals[method].setdefault(k, []).append(float(v))

    out: dict[str, dict] = {}
    for method, metric_lists in method_vals.items():
        out[method] = {}
        for metric, vals in metric_lists.items():
            out[method][metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
                "values": vals,
            }
    return out


# ─── comparison table ─────────────────────────────────────────────────────────

BASELINE_ORDER = [
    "CHRONOS",
    "GNN-Sched",
    "DRL-Offload",
    "MAPPO",
    "QMIX",
    "FedAvg+Greedy",
    "Random",
]

DISPLAY_METRICS = [
    ("avg_combined_reward",       "Combined Reward ↑"),
    ("avg_latency_reward",        "Latency Reward ↑"),
    ("avg_energy_reward",         "Energy Reward ↑"),
    ("avg_accuracy_reward",       "Accuracy Reward ↑"),
    ("avg_communication_reward",  "Comm. Reward ↑"),
    ("deadline_violation_rate",   "Deadline Viol. ↓"),
]


def build_comparison_table_md(agg: dict[str, dict], seeds: list[int]) -> str:
    seed_str = ", ".join(str(s) for s in seeds)
    lines = [
        f"# CHRONOS vs Baselines — Comparison Table",
        f"",
        f"**Seeds:** {seed_str}  |  **Config:** fast_experiment (30 rounds, 3 ep/round, 30 steps/ep)",
        f"",
        f"Mean ± std across {len(seeds)} seeds. Best value in each column is **bold**.",
        f"",
    ]

    # Header
    col_heads = ["Method"] + [label for _, label in DISPLAY_METRICS]
    sep = [":---"] + [":---:" for _ in DISPLAY_METRICS]
    lines.append("| " + " | ".join(col_heads) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    # Determine best values (for bolding)
    best: dict[str, float] = {}
    for metric_key, _ in DISPLAY_METRICS:
        vals = []
        for method in BASELINE_ORDER:
            if method in agg and metric_key in agg[method]:
                vals.append(agg[method][metric_key]["mean"])
        if not vals:
            continue
        # Higher is better for rewards, lower for violation rate
        if "violation" in metric_key:
            best[metric_key] = min(vals)
        else:
            best[metric_key] = max(vals)

    for method in BASELINE_ORDER:
        if method not in agg:
            continue
        row = [method]
        for metric_key, _ in DISPLAY_METRICS:
            if metric_key not in agg[method]:
                row.append("—")
                continue
            m = agg[method][metric_key]["mean"]
            s = agg[method][metric_key]["std"]
            cell = f"{m:.3f} ± {s:.3f}"
            if metric_key in best and abs(m - best[metric_key]) < 1e-9:
                cell = f"**{cell}**"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")

    # Improvement row (CHRONOS vs best baseline)
    if "CHRONOS" in agg:
        row = ["*CHRONOS Δ vs best baseline*"]
        chronos_combined = agg["CHRONOS"].get("avg_combined_reward", {}).get("mean")
        for metric_key, _ in DISPLAY_METRICS:
            if metric_key not in agg.get("CHRONOS", {}):
                row.append("—")
                continue
            chronos_val = agg["CHRONOS"][metric_key]["mean"]
            baseline_vals = []
            for m in BASELINE_ORDER:
                if m == "CHRONOS":
                    continue
                if m in agg and metric_key in agg[m]:
                    baseline_vals.append(agg[m][metric_key]["mean"])
            if not baseline_vals:
                row.append("—")
                continue
            if "violation" in metric_key:
                ref = min(baseline_vals)
                delta = chronos_val - ref
                sign = "+" if delta > 0 else ""
                row.append(f"{sign}{delta:.3f}")
            else:
                ref = max(baseline_vals)
                delta = chronos_val - ref
                sign = "+" if delta > 0 else ""
                row.append(f"{sign}{delta:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def build_comparison_table_tex(agg: dict[str, dict], seeds: list[int]) -> str:
    seed_str = ", ".join(str(s) for s in seeds)
    col_spec = "l" + "r" * len(DISPLAY_METRICS)
    header = " & ".join(
        ["Method"] + [label.replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$")
                      for _, label in DISPLAY_METRICS]
    )

    rows = []
    for method in BASELINE_ORDER:
        if method not in agg:
            continue
        cells = [method.replace("_", r"\_").replace("+", r"\texttt{+}")]
        for metric_key, _ in DISPLAY_METRICS:
            if metric_key not in agg[method]:
                cells.append("—")
                continue
            m = agg[method][metric_key]["mean"]
            s = agg[method][metric_key]["std"]
            cells.append(f"{m:.3f} $\\pm$ {s:.3f}")
        if method == "CHRONOS":
            rows.append(r"\midrule")
            rows.append(" & ".join(cells) + r" \\")
            rows.append(r"\midrule")
        else:
            rows.append(" & ".join(cells) + r" \\")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{CHRONOS vs Baselines (" + f"seeds: {seed_str}" + r")}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ] + rows + [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─── ablation report ──────────────────────────────────────────────────────────

ABLATION_VARIANT_LABELS = {
    "CHRONOS-noSNN":    "w/o SNN (→ MLP)",
    "CHRONOS-noHG":     "w/o Hypergraph (→ graph)",
    "CHRONOS-noCausal": "w/o Causal reasoning",
    "CHRONOS-noFed":    "w/o Federated learning",
    "CHRONOS-noDT":     "w/o Digital Twin",
}

ABLATION_KEY_METRICS = [
    ("avg_combined_reward", "Combined Reward"),
    ("deadline_violation_rate", "Deadline Viol."),
    ("avg_latency_reward", "Latency Reward"),
    ("avg_energy_reward", "Energy Reward"),
    ("snn_energy_ratio", "SNN Energy Ratio"),
]


def build_ablation_report(ablation_dir: Path, full_chronos: dict | None) -> str:
    lines = [
        "# CHRONOS Ablation Study",
        "",
        "Each variant removes one component of CHRONOS. Δ is relative to CHRONOS-full.",
        "",
    ]

    # Header
    col_heads = ["Variant"] + [label for _, label in ABLATION_KEY_METRICS] + ["Δ Combined"]
    sep = [":---"] + [":---:" for _ in col_heads[1:]]
    lines.append("| " + " | ".join(col_heads) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    full_combined = None
    if full_chronos:
        full_combined = full_chronos.get("avg_combined_reward")

    # Full CHRONOS row
    if full_chronos:
        row = ["**CHRONOS-full**"]
        for metric_key, _ in ABLATION_KEY_METRICS:
            v = full_chronos.get(metric_key)
            row.append(fmt(v))
        row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    # Variant rows
    for variant_key, label in ABLATION_VARIANT_LABELS.items():
        result_file = ablation_dir / f"{variant_key}_results.json"
        if not result_file.exists():
            continue
        data = load_json(result_file)
        row = [label]
        for metric_key, _ in ABLATION_KEY_METRICS:
            v = data.get(metric_key)
            row.append(fmt(v))
        # Delta combined reward
        v_combined = data.get("avg_combined_reward")
        if v_combined is not None and full_combined is not None:
            delta = v_combined - full_combined
            sign = "+" if delta > 0 else ""
            row.append(f"{sign}{delta:.3f}")
        else:
            row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Component Importance (by impact on Combined Reward)",
        "",
        "| Component Removed | Δ Combined Reward | Relative Change |",
        "| :--- | :---: | :---: |",
    ]

    impacts = []
    for variant_key, label in ABLATION_VARIANT_LABELS.items():
        result_file = ablation_dir / f"{variant_key}_results.json"
        if not result_file.exists():
            continue
        data = load_json(result_file)
        v = data.get("avg_combined_reward")
        if v is not None and full_combined is not None:
            delta = v - full_combined
            rel = pct_change(v, full_combined)
            impacts.append((label, delta, rel))

    impacts.sort(key=lambda x: x[1])  # most negative first

    for label, delta, rel in impacts:
        sign = "+" if delta > 0 else ""
        rel_str = f"{rel:+.1f}%" if rel is not None else "—"
        lines.append(f"| {label} | {sign}{delta:.3f} | {rel_str} |")

    lines += [
        "",
        "> Larger negative Δ = that component contributes more to performance.",
        "",
    ]

    return "\n".join(lines)


# ─── summary JSON ─────────────────────────────────────────────────────────────

def build_summary_json(agg: dict[str, dict], seeds: list[int], ablation_dir: Path) -> dict:
    out: dict = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "methods": {},
        "ablation": {},
    }

    for method in BASELINE_ORDER:
        if method not in agg:
            continue
        metrics = {}
        for metric_key, _ in DISPLAY_METRICS:
            if metric_key in agg[method]:
                metrics[metric_key] = {
                    "mean": agg[method][metric_key]["mean"],
                    "std": agg[method][metric_key]["std"],
                    "n": agg[method][metric_key]["n"],
                }
        out["methods"][method] = metrics

    for variant_key in ABLATION_VARIANT_LABELS:
        result_file = ablation_dir / f"{variant_key}_results.json"
        if result_file.exists():
            out["ablation"][variant_key] = load_json(result_file)

    full_file = ablation_dir / "CHRONOS-full_results.json"
    if full_file.exists():
        out["ablation"]["CHRONOS-full"] = load_json(full_file)

    return out


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repro", type=str, default="outputs/reproducibility",
                        help="Reproducibility output directory")
    parser.add_argument("--ablation", type=str, default="outputs/ablation",
                        help="Ablation output directory")
    parser.add_argument("--results", type=str, default="results",
                        help="Output directory for results artifacts")
    args = parser.parse_args()

    repro_dir = Path(args.repro)
    ablation_dir = Path(args.ablation)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load per-seed results
    per_seed_file = repro_dir / "per_seed_results.json"
    if not per_seed_file.exists():
        print(f"ERROR: {per_seed_file} not found. Run the reproducibility experiment first.")
        sys.exit(1)

    per_seed = load_json(per_seed_file)
    seeds = [run["seed"] for run in per_seed]
    print(f"Loaded results for seeds: {seeds}")

    agg = aggregate_per_seed(per_seed)
    print(f"Methods found: {list(agg.keys())}")

    # Load CHRONOS-full ablation baseline
    full_chronos_file = ablation_dir / "CHRONOS-full_results.json"
    full_chronos = load_json(full_chronos_file) if full_chronos_file.exists() else None

    # Comparison table
    comp_md = build_comparison_table_md(agg, seeds)
    comp_tex = build_comparison_table_tex(agg, seeds)
    (results_dir / "comparison_table.md").write_text(comp_md, encoding="utf-8")
    (results_dir / "comparison_table.tex").write_text(comp_tex, encoding="utf-8")
    print(f"Written: {results_dir}/comparison_table.md")
    print(f"Written: {results_dir}/comparison_table.tex")

    # Ablation report
    ablation_md = build_ablation_report(ablation_dir, full_chronos)
    (results_dir / "ablation_report.md").write_text(ablation_md, encoding="utf-8")
    print(f"Written: {results_dir}/ablation_report.md")

    # Summary JSON
    summary = build_summary_json(agg, seeds, ablation_dir)
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Written: {results_dir}/summary.json")

    # Print comparison table preview
    print("\n" + "=" * 72)
    print("COMPARISON TABLE PREVIEW")
    print("=" * 72)
    print(comp_md[:3000])

    print("\n" + "=" * 72)
    print("ABLATION REPORT PREVIEW")
    print("=" * 72)
    print(ablation_md[:2000])


if __name__ == "__main__":
    main()
