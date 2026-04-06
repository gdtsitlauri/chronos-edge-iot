"""Rebuild ablation report/impacts from existing ablation result files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos.evaluation.ablation import AblationStudy, ABLATION_VARIANTS
from chronos.evaluation.results_schema import canonicalize_metrics


def main():
    parser = argparse.ArgumentParser(description="Rebuild ablation report from existing result JSON files")
    parser.add_argument("--ablation-dir", type=str, default="outputs/ablation")
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    if not ablation_dir.exists():
        raise FileNotFoundError(f"Ablation directory not found: {ablation_dir}")

    study = AblationStudy(base_config={})

    loaded = 0
    for variant_name in ABLATION_VARIANTS:
        path = ablation_dir / f"{variant_name}_results.json"
        if not path.exists():
            continue

        with open(path) as f:
            metrics = json.load(f)
        study.record_variant_results(variant_name, canonicalize_metrics(metrics))
        loaded += 1

    if loaded == 0:
        raise RuntimeError("No ablation result files were found.")

    report = study.generate_report()
    impacts = study.compute_ablation_impacts()

    with open(ablation_dir / "ablation_report.txt", "w") as f:
        f.write(report)

    with open(ablation_dir / "ablation_impacts.json", "w") as f:
        json.dump(impacts, f, indent=2, default=str)

    print(f"Loaded {loaded} ablation result files")
    print(f"Updated: {ablation_dir / 'ablation_report.txt'}")
    print(f"Updated: {ablation_dir / 'ablation_impacts.json'}")


if __name__ == "__main__":
    main()
