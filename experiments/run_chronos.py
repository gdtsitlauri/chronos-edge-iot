"""Run CHRONOS training on a specified scenario."""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos.training.trainer import ChronosTrainer
from chronos.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train CHRONOS")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--output", type=str, default="outputs/chronos",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override total training rounds")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.rounds:
        config.setdefault("training", {})["total_rounds"] = args.rounds
    if args.seed:
        config["system"]["seed"] = args.seed

    trainer = ChronosTrainer(config, output_dir=args.output, device=args.device)
    final_metrics = trainer.train()

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    for k, v in sorted(final_metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    final_results_path = Path(args.output) / "final_results.json"
    if final_results_path.exists():
        print(f"\nSaved final results to: {final_results_path}")


if __name__ == "__main__":
    main()
