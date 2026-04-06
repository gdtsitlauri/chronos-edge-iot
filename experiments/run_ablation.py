"""Run ablation study for CHRONOS modules."""

import argparse
import json
import sys
import os
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos.training.trainer import ChronosTrainer
from chronos.evaluation.ablation import AblationStudy, ABLATION_VARIANTS
from chronos.evaluation.results_schema import canonicalize_metrics
from chronos.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run CHRONOS ablation study")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="outputs/ablation")
    parser.add_argument("--train-rounds", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Specific variants to run (default: all)")
    args = parser.parse_args()

    base_config = load_config(args.config)
    if args.seed is not None:
        base_config.setdefault("system", {})["seed"] = args.seed

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation = AblationStudy(base_config)

    variants_to_run = args.variants or list(ABLATION_VARIANTS.keys())

    for variant_name in variants_to_run:
        print(f"\n{'='*60}")
        print(f"Ablation variant: {variant_name}")
        print(f"  {ABLATION_VARIANTS[variant_name]['description']}")
        print(f"  Disabled: {ABLATION_VARIANTS[variant_name]['disable']}")
        print(f"{'='*60}")

        # Create variant config
        variant_config = ablation.create_variant_config(variant_name)
        variant_config.setdefault("training", {})["total_rounds"] = args.train_rounds

        # Train and evaluate
        variant_output = str(output_dir / variant_name.replace("-", "_"))
        trainer = ChronosTrainer(variant_config, output_dir=variant_output)

        try:
            trainer.train()
            final_metrics = trainer.evaluate(num_episodes=args.eval_episodes)
        except Exception as e:
            print(f"  Error training {variant_name}: {e}")
            final_metrics = {"avg_combined_reward": 0.0, "error": str(e)}

        canonical_metrics = canonicalize_metrics(final_metrics)
        if "avg_combined_reward" not in canonical_metrics:
            canonical_metrics["avg_combined_reward"] = 0.0
        if "error" in final_metrics:
            canonical_metrics["error"] = final_metrics["error"]

        ablation.record_variant_results(variant_name, canonical_metrics)

        # Save canonical variant results
        with open(output_dir / f"{variant_name}_results.json", "w") as f:
            json.dump(canonical_metrics, f, indent=2, default=str)

        # Save raw variant results for debugging/reproducibility
        with open(output_dir / f"{variant_name}_results_raw.json", "w") as f:
            json.dump(final_metrics, f, indent=2, default=str)

        print(f"\n  Results:")
        for k, v in sorted(canonical_metrics.items()):
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")

    # Generate ablation report
    report = ablation.generate_report()
    print(f"\n{report}")

    with open(output_dir / "ablation_report.txt", "w") as f:
        f.write(report)

    # Save combined results
    impacts = ablation.compute_ablation_impacts()
    with open(output_dir / "ablation_impacts.json", "w") as f:
        json.dump(impacts, f, indent=2, default=str)

    ranking = ablation.rank_module_importance()
    print("\nModule Importance Ranking:")
    for module, score in ranking:
        print(f"  {module}: {score:.4f}")


if __name__ == "__main__":
    main()
