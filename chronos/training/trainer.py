"""Main CHRONOS training loop.

Orchestrates the full Algorithm 1 from the proposal:
- Phase 1: Causal Hypergraph Update
- Phase 2: State Encoding
- Phase 3: Spiking Policy Execution
- Phase 4: Digital Twin Simulation
- Phase 5: Environment Execution
- Phase 6: Causal Policy Gradient Update
- Phase 7: Hypergraph-Federated Aggregation
"""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from chronos.agents.chronos_agent import ChronosAgent
from chronos.evaluation.results_schema import canonicalize_metrics
from chronos.simulator.environment import EdgeIoTEnvironment
from chronos.utils.config import load_config


class ChronosTrainer:
    """Full training loop for the CHRONOS framework."""

    def __init__(self, config: dict, output_dir: str = "outputs",
                 device: str = "auto"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Training config
        train_cfg = config.get("training", {})
        self.total_rounds = train_cfg.get("total_rounds", 1000)
        self.episodes_per_round = train_cfg.get("episodes_per_round", 10)
        self.max_steps = train_cfg.get("max_steps_per_episode", 200)
        self.eval_interval = train_cfg.get("eval_interval", 10)
        self.save_interval = train_cfg.get("save_interval", 50)
        self.log_interval = train_cfg.get("log_interval", 5)

        # Build environment
        self.env = EdgeIoTEnvironment(config)

        # Build agent
        self.agent = ChronosAgent(
            agent_id=0,
            config=config,
            num_agents=config["system"]["num_edge_nodes"],
            device=self.device,
        )

        # Logging
        self.training_log: list[dict] = []
        self.best_reward = -float("inf")

    def train(self) -> dict:
        """Run the full CHRONOS training algorithm.

        Returns:
            final_metrics: dict of final training metrics
        """
        print(f"Starting CHRONOS training on {self.device}")
        print(f"  Rounds: {self.total_rounds}")
        print(f"  Episodes/round: {self.episodes_per_round}")
        print(f"  Max steps/episode: {self.max_steps}")
        print(f"  Edge nodes: {self.config['system']['num_edge_nodes']}")
        print(f"  IoT devices: {self.config['system']['num_iot_devices']}")
        print()

        for round_idx in tqdm(range(self.total_rounds), desc="Training"):
            round_metrics = self._train_round(round_idx)

            # Logging
            if round_idx % self.log_interval == 0:
                self.training_log.append(round_metrics)
                self._log_metrics(round_idx, round_metrics)

            # Evaluation
            if round_idx % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                round_metrics["eval"] = eval_metrics

                # Track best
                combined = eval_metrics.get("avg_combined_reward", -float("inf"))
                if combined > self.best_reward:
                    self.best_reward = combined
                    self._save_checkpoint(round_idx, is_best=True)

            # Periodic save
            if round_idx % self.save_interval == 0:
                self._save_checkpoint(round_idx)

        # Final evaluation
        final_metrics = self.evaluate()
        self._save_checkpoint(self.total_rounds, is_best=False)
        self._save_training_log()
        self._save_final_results(final_metrics)

        return final_metrics

    def _train_round(self, round_idx: int) -> dict:
        """Execute one training round (multiple episodes)."""
        round_rewards = []
        round_metrics = {}

        for ep in range(self.episodes_per_round):
            ep_reward, ep_info = self._run_episode(training=True)
            round_rewards.append(ep_reward)

        # Update agent from collected experience
        update_metrics = self.agent.update()
        round_metrics.update(update_metrics)

        round_metrics["round"] = round_idx
        round_metrics["avg_episode_reward"] = np.mean(round_rewards)
        round_metrics["std_episode_reward"] = np.std(round_rewards)

        return round_metrics

    def _run_episode(self, training: bool = True) -> tuple[float, dict]:
        """Run a single episode.

        Returns:
            total_reward: scalar combined reward
            info: dict of episode info
        """
        obs = self.env.reset()
        total_reward = 0.0
        episode_rewards = {
            "accuracy": 0.0, "latency": 0.0,
            "energy": 0.0, "communication": 0.0,
        }

        for step in range(self.max_steps):
            # Agent selects action
            action = self.agent.select_action(obs, deterministic=not training)

            # Environment step
            result = self.env.step(action)

            # Accumulate rewards
            for k in episode_rewards:
                episode_rewards[k] += result.rewards.get(k, 0.0)
            total_reward += result.rewards.get("combined", 0.0)

            # Store transition for training
            if training:
                # Get next state embedding
                next_hg = self.agent.hypergraph_builder.build_from_observation(
                    result.next_state, self.config
                )
                next_hg_data = next_hg.to_pyg_data()
                for k, v in next_hg_data.items():
                    if isinstance(v, torch.Tensor):
                        next_hg_data[k] = v.to(self.device)

                with torch.no_grad():
                    next_z, _ = self.agent.chse(next_hg_data)

                self.agent.store_transition(
                    self.agent._last_state_embedding,
                    action, result.rewards, next_z,
                    result.done, result.constraint_violations,
                )

            obs = result.next_state

            if result.done:
                break

        info = {
            "episode_rewards": episode_rewards,
            "steps": step + 1,
            "system_summary": self.env.get_system_summary(),
        }

        return total_reward, info

    def evaluate(self, num_episodes: int | None = None) -> dict:
        """Evaluate current policy without training.

        Returns:
            metrics: dict of evaluation metrics
        """
        if num_episodes is None:
            num_episodes = self.config.get("evaluation", {}).get("num_eval_episodes", 50)

        all_rewards = []
        all_objective_rewards = {
            "accuracy": [], "latency": [], "energy": [], "communication": [],
        }
        all_steps = []
        deadline_violations = 0
        total_tasks = 0

        for _ in range(num_episodes):
            ep_reward, ep_info = self._run_episode(training=False)
            all_rewards.append(ep_reward)
            all_steps.append(ep_info["steps"])

            for k in all_objective_rewards:
                all_objective_rewards[k].append(ep_info["episode_rewards"][k])

            summary = ep_info["system_summary"]
            deadline_violations += summary["episode_metrics"]["deadline_violations"]
            total_tasks += summary["episode_metrics"]["tasks_completed"] + summary["episode_metrics"]["tasks_failed"]

        metrics = {
            "avg_combined_reward": float(np.mean(all_rewards)),
            "std_combined_reward": float(np.std(all_rewards)),
            "avg_steps": float(np.mean(all_steps)),
            "deadline_violation_rate": deadline_violations / max(total_tasks, 1),
        }

        for k, vals in all_objective_rewards.items():
            metrics[f"avg_{k}_reward"] = float(np.mean(vals))
            metrics[f"std_{k}_reward"] = float(np.std(vals))

        # SNN energy stats
        energy_stats = self.agent.spn.compute_spike_energy()
        metrics["snn_energy_ratio"] = energy_stats["energy_ratio"]

        return metrics

    def _log_metrics(self, round_idx: int, metrics: dict):
        """Print training metrics."""
        parts = [f"Round {round_idx}"]
        if "avg_episode_reward" in metrics:
            parts.append(f"Reward: {metrics['avg_episode_reward']:.4f}")
        if "policy_loss" in metrics:
            parts.append(f"Policy Loss: {metrics['policy_loss']:.4f}")
        if "critic_loss" in metrics:
            parts.append(f"Critic Loss: {metrics['critic_loss']:.4f}")
        if "causal_edges_found" in metrics:
            parts.append(f"Causal Edges: {metrics['causal_edges_found']}")
        if "eval" in metrics:
            parts.append(f"Eval: {metrics['eval'].get('avg_combined_reward', 0):.4f}")
        tqdm.write(" | ".join(parts))

    def _save_checkpoint(self, round_idx: int, is_best: bool = False):
        """Save model checkpoint."""
        ckpt = {
            "round": round_idx,
            "agent_state": self.agent.get_state_dict(),
            "config": self.config,
            "best_reward": self.best_reward,
        }
        path = self.output_dir / f"checkpoint_round{round_idx}.pt"
        torch.save(ckpt, path)

        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(ckpt, best_path)

    def _save_training_log(self):
        """Save training log to JSON."""
        log_path = self.output_dir / "training_log.json"
        # Convert numpy/torch values to Python types
        serializable = []
        for entry in self.training_log:
            clean = {}
            for k, v in entry.items():
                if isinstance(v, (np.floating, np.integer)):
                    clean[k] = float(v)
                elif isinstance(v, dict):
                    clean[k] = {
                        sk: float(sv) if isinstance(sv, (np.floating, np.integer, torch.Tensor)) else sv
                        for sk, sv in v.items()
                    }
                elif isinstance(v, (list, tuple)):
                    clean[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                else:
                    clean[k] = v
            serializable.append(clean)

        with open(log_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    def _save_final_results(self, metrics: dict):
        """Save final evaluation metrics with a stable schema."""
        canonical = canonicalize_metrics(metrics)
        raw_metrics = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in metrics.items()
        }
        payload = canonical.copy()
        payload["metadata"] = {
            "rounds_completed": int(self.total_rounds),
            "episodes_per_round": int(self.episodes_per_round),
            "max_steps_per_episode": int(self.max_steps),
            "device": str(self.device),
        }
        payload["raw_metrics"] = raw_metrics

        with open(self.output_dir / "final_results.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)

    @classmethod
    def from_config_file(cls, config_path: str, **kwargs) -> ChronosTrainer:
        """Create trainer from a YAML config file."""
        config = load_config(config_path)
        return cls(config, **kwargs)
