"""Visualization utilities for CHRONOS experimental results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def plot_convergence_curves(training_logs: dict[str, list[dict]],
                            metric: str = "avg_episode_reward",
                            output_path: str | None = None):
    """Plot convergence curves for multiple methods.

    Args:
        training_logs: {method_name: list of per-round metric dicts}
        metric: which metric to plot
        output_path: save path (if None, displays interactively)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("matplotlib/seaborn not available, skipping plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for method_name, logs in training_logs.items():
        rounds = [entry.get("round", i) for i, entry in enumerate(logs)]
        values = [entry.get(metric, 0.0) for entry in logs]

        # Smooth with moving average
        window = max(1, len(values) // 50)
        if len(values) > window:
            smoothed = np.convolve(values, np.ones(window) / window, mode='valid')
            ax.plot(rounds[:len(smoothed)], smoothed, label=method_name, linewidth=2)
            ax.fill_between(
                rounds[:len(smoothed)],
                smoothed - np.std(values[:len(smoothed)]) * 0.5,
                smoothed + np.std(values[:len(smoothed)]) * 0.5,
                alpha=0.1,
            )
        else:
            ax.plot(rounds, values, label=method_name, linewidth=2)

    ax.set_xlabel("Training Round", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title("Convergence Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved convergence plot to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_pareto_front(method_results: dict[str, np.ndarray],
                      obj_x: int = 0, obj_y: int = 1,
                      obj_names: list[str] | None = None,
                      output_path: str | None = None):
    """Plot 2D Pareto front projection.

    Args:
        method_results: {method_name: (num_runs, num_objectives) array}
        obj_x: index of x-axis objective
        obj_y: index of y-axis objective
        obj_names: names for objectives
        output_path: save path
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("matplotlib/seaborn not available, skipping plot")
        return

    if obj_names is None:
        obj_names = ["Accuracy", "Latency", "Energy", "Communication"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i, (method, results) in enumerate(method_results.items()):
        if results.ndim == 1:
            results = results.reshape(1, -1)
        x_vals = results[:, obj_x]
        y_vals = results[:, obj_y]
        ax.scatter(x_vals, y_vals, label=method, marker=markers[i % len(markers)],
                   s=80, alpha=0.8, edgecolors='black', linewidth=0.5)

    ax.set_xlabel(obj_names[obj_x], fontsize=12)
    ax.set_ylabel(obj_names[obj_y], fontsize=12)
    ax.set_title(f"Pareto Front: {obj_names[obj_x]} vs {obj_names[obj_y]}", fontsize=14)
    ax.legend(fontsize=10)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_ablation_impacts(impacts: dict, output_path: str | None = None):
    """Plot ablation study impact bar chart.

    Args:
        impacts: output from AblationStudy.compute_ablation_impacts()
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("matplotlib/seaborn not available, skipping plot")
        return

    variant_names = []
    deltas = []
    metric_key = "avg_combined_reward"

    for variant, data in impacts.items():
        if metric_key in data.get("impacts", {}):
            variant_names.append(variant.replace("CHRONOS-", ""))
            deltas.append(data["impacts"][metric_key]["relative_change_pct"])

    if not variant_names:
        print("No ablation data to plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    colors = ['#e74c3c' if d < 0 else '#2ecc71' for d in deltas]
    bars = ax.barh(variant_names, deltas, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Relative Change in Combined Reward (%)", fontsize=12)
    ax.set_title("Ablation Study: Module Impact", fontsize=14)
    ax.axvline(x=0, color='black', linewidth=0.8)

    for bar, delta in zip(bars, deltas):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{delta:+.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_multi_objective_radar(method_results: dict[str, dict[str, float]],
                                output_path: str | None = None):
    """Radar/spider chart comparing methods across objectives.

    Args:
        method_results: {method_name: {objective_name: value}}
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    objectives = ["accuracy", "latency", "energy", "communication"]
    N = len(objectives)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))

    for method_name, values in method_results.items():
        vals = [values.get(obj, 0.0) for obj in objectives]

        # Normalize to [0, 1] range across methods
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, linewidth=2, label=method_name)
        ax.fill(angles, vals_plot, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([o.capitalize() for o in objectives], fontsize=12)
    ax.set_title("Multi-Objective Comparison", fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
