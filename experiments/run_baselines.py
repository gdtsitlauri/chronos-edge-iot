"""Run all baselines on a specified scenario for comparison."""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos.simulator.environment import EdgeIoTEnvironment
from chronos.baselines.random_agent import RandomAgent
from chronos.baselines.fedavg import FedAvgAgent
from chronos.baselines.mappo import MAPPOAgent
from chronos.baselines.qmix import QMIXAgent
from chronos.baselines.gnn_scheduler import GNNSchedulerAgent
from chronos.baselines.drl_offload import DRLOffloadAgent
from chronos.evaluation.metrics import MetricsTracker
from chronos.evaluation.results_schema import canonicalize_metrics
from chronos.utils.config import load_config


BASELINES = {
    "Random": RandomAgent,
    "FedAvg+Greedy": FedAvgAgent,
    "MAPPO": MAPPOAgent,
    "QMIX": QMIXAgent,
    "GNN-Sched": GNNSchedulerAgent,
    "DRL-Offload": DRLOffloadAgent,
}


def train_and_evaluate(agent, env, config, num_train_rounds=500,
                       num_eval_episodes=50, max_steps=200):
    """Train a baseline agent and evaluate it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training phase
    print(f"  Training for {num_train_rounds} rounds...")
    for round_idx in tqdm(range(num_train_rounds), desc="  Training", leave=False):
        obs = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.select_action(obs)
            result = env.step(action)
            total_reward += result.rewards.get("combined", 0.0)

            # Store experience (different interfaces per baseline)
            if hasattr(agent, "store_transition"):
                if isinstance(agent, (MAPPOAgent, QMIXAgent, GNNSchedulerAgent, DRLOffloadAgent)):
                    agent.store_transition(obs, action, result.rewards.get("combined", 0.0),
                                           result.next_state, result.done)
                elif isinstance(agent, FedAvgAgent):
                    agent.store_transition(obs, action, result.rewards.get("combined", 0.0))

            obs = result.next_state
            if result.done:
                break

        agent.update()

    # Evaluation phase
    print(f"  Evaluating for {num_eval_episodes} episodes...")
    tracker = MetricsTracker()

    for ep in tqdm(range(num_eval_episodes), desc="  Evaluating", leave=False):
        obs = env.reset()
        ep_rewards = {"accuracy": 0.0, "latency": 0.0, "energy": 0.0, "communication": 0.0}
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.select_action(obs, deterministic=True)
            result = env.step(action)

            for k in ep_rewards:
                ep_rewards[k] += result.rewards.get(k, 0.0)
            total_reward += result.rewards.get("combined", 0.0)

            obs = result.next_state
            if result.done:
                break

        summary = env.get_system_summary()
        tracker.record_episode({
            "combined_reward": total_reward,
            "avg_combined_reward": total_reward / (step + 1),
            **{f"avg_{k}_reward": v / (step + 1) for k, v in ep_rewards.items()},
            "steps": step + 1,
            "deadline_violations": summary["episode_metrics"]["deadline_violations"],
            "tasks_completed": summary["episode_metrics"]["tasks_completed"],
            "tasks_failed": summary["episode_metrics"]["tasks_failed"],
            "total_energy_j": summary["episode_metrics"]["total_energy_j"],
            "total_comm_bits": summary["episode_metrics"]["total_comm_bits"],
        })

    return tracker.compute_summary()


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparisons")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="outputs/baselines")
    parser.add_argument("--train-rounds", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--baselines", nargs="+", default=None,
                        help="Specific baselines to run (default: all)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.setdefault("system", {})["seed"] = args.seed

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = EdgeIoTEnvironment(config)

    baselines_to_run = args.baselines or list(BASELINES.keys())
    all_results = {}
    all_summaries = {}

    for name in baselines_to_run:
        if name not in BASELINES:
            print(f"Unknown baseline: {name}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Running baseline: {name}")
        print(f"{'='*60}")

        AgentClass = BASELINES[name]
        if name == "Random":
            agent = AgentClass(0, config)
        else:
            agent = AgentClass(0, config, device=device)

        results = train_and_evaluate(
            agent, env, config,
            num_train_rounds=args.train_rounds,
            num_eval_episodes=args.eval_episodes,
        )
        summary = canonicalize_metrics(results)

        all_results[name] = results
        all_summaries[name] = summary
        print(f"\n  Results for {name}:")
        for k, v in sorted(summary.items()):
            if k.startswith("avg_") or k.startswith("std_") or k == "deadline_violation_rate":
                print(f"    {k}: {v:.4f}")

        # Save individual raw results
        with open(output_dir / f"{name.replace('+', '_').replace('-', '_')}_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save individual canonical summary
        with open(output_dir / f"{name.replace('+', '_').replace('-', '_')}_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    # Save combined results
    with open(output_dir / "all_baselines_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save standardized summary for downstream analysis
    with open(output_dir / "all_baselines_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Reward':>10} {'Accuracy':>10} {'Latency':>10} {'Energy':>10}")
    print("-" * 80)
    for name, results in all_summaries.items():
        print(f"{name:<20} "
              f"{results.get('avg_combined_reward', 0):>10.4f} "
              f"{results.get('avg_accuracy_reward', 0):>10.4f} "
              f"{results.get('avg_latency_reward', 0):>10.4f} "
              f"{results.get('avg_energy_reward', 0):>10.4f}")


if __name__ == "__main__":
    main()
