"""CHRONOS Agent: integrates all 5 modules into a single decision-making agent.

Pipeline per step:
1. CHSE encodes hypergraph state -> z(t)
2. SPN produces spike-coded actions
3. DTCS validates action safety
4. CCPG computes advantages and updates policy
5. HFA handles federated aggregation (periodic)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from chronos.agents.base_agent import BaseAgent
from chronos.core.hypergraph import CausalHypergraph, HypergraphBuilder
from chronos.core.causal_discovery import OnlineCausalDiscovery
from chronos.core.scm import StructuralCausalModel
from chronos.modules.chse import CausalHypergraphStateEncoder
from chronos.modules.spn import SpikingPolicyNetwork
from chronos.modules.ccpg import CausalCounterfactualPolicyGradient
from chronos.modules.hfa import HypergraphFederatedAggregation
from chronos.modules.dtcs import DigitalTwinCausalSimulator


class ChronosAgent(BaseAgent):
    """Full CHRONOS agent integrating all modules."""

    def __init__(self, agent_id: int, config: dict, num_agents: int,
                 device: torch.device = torch.device("cpu")):
        super().__init__(agent_id, config)
        self.num_agents = num_agents
        self.device = device

        # Extract dimensions from config
        sys_cfg = config["system"]
        chse_cfg = config.get("chse", {})
        spn_cfg = config.get("spn", {})
        ccpg_cfg = config.get("ccpg", {})
        hfa_cfg = config.get("hfa", {})
        dtcs_cfg = config.get("dtcs", {})

        self.num_edge_nodes = sys_cfg["num_edge_nodes"]
        self.num_channels = sys_cfg["num_channels"]
        self.state_dim = chse_cfg.get("output_dim", 64)
        self.action_dim = self.num_edge_nodes + 1 + 2 + self.num_channels + 2  # offload + resource/power + channel + FL

        # Type dims for heterogeneous vertices
        type_dims = {0: 10, 1: 8, 2: 7, 3: 6}  # edge_node, iot_device, task, channel

        # Module 1: CHSE
        self.chse = CausalHypergraphStateEncoder(
            type_dims=type_dims,
            node_embedding_dim=chse_cfg.get("node_embedding_dim", 64),
            hidden_dim=chse_cfg.get("hidden_dim", 128),
            output_dim=self.state_dim,
            num_layers=chse_cfg.get("num_layers", 3),
            num_attention_heads=chse_cfg.get("num_attention_heads", 4),
            causal_gate_hidden=chse_cfg.get("causal_gate_hidden", 32),
            dropout=chse_cfg.get("dropout", 0.1),
        ).to(device)

        # Module 2: SPN
        self.spn = SpikingPolicyNetwork(
            state_dim=self.state_dim,
            num_edge_nodes=self.num_edge_nodes,
            num_channels=self.num_channels,
            num_lif_layers=spn_cfg.get("num_lif_layers", 3),
            hidden_neurons=spn_cfg.get("hidden_neurons", 256),
            time_steps=spn_cfg.get("time_steps", 16),
            membrane_decay=spn_cfg.get("membrane_decay", 0.9),
            threshold=spn_cfg.get("threshold", 1.0),
            beta_init=spn_cfg.get("beta_init", 0.5),
            recency_decay=spn_cfg.get("recency_decay", 0.1),
        ).to(device)

        # Module 3: CCPG
        self.ccpg = CausalCounterfactualPolicyGradient(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_agents=num_agents,
            num_objectives=ccpg_cfg.get("num_objectives", 4),
            gamma=ccpg_cfg.get("gamma", 0.99),
            gae_lambda=ccpg_cfg.get("gae_lambda", 0.95),
            clip_epsilon=ccpg_cfg.get("clip_epsilon", 0.2),
            entropy_coeff=ccpg_cfg.get("entropy_coeff", 0.01),
            lr_critic=ccpg_cfg.get("lr_critic", 1e-3),
            lr_lambda=ccpg_cfg.get("lr_lambda", 1e-3),
            lr_dual=ccpg_cfg.get("lr_dual", 1e-2),
            max_grad_norm=ccpg_cfg.get("max_grad_norm", 0.5),
            counterfactual_samples=ccpg_cfg.get("counterfactual_samples", 8),
            device=device,
        )

        # Module 4: HFA
        self.hfa = HypergraphFederatedAggregation(
            num_clients=num_agents,
            aggregation_interval=hfa_cfg.get("aggregation_interval", 5),
            ot_regularization=hfa_cfg.get("ot_regularization", 0.1),
            ot_max_iter=hfa_cfg.get("ot_max_iter", 100),
            compression_ratio=hfa_cfg.get("compression_ratio", 0.1),
            min_participation=hfa_cfg.get("min_participation", 0.3),
        )

        # Module 5: DTCS
        self.dtcs = DigitalTwinCausalSimulator(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hypergraph_dim=self.state_dim,
            num_objectives=ccpg_cfg.get("num_objectives", 4),
            hidden_dim=dtcs_cfg.get("twin_hidden_dim", 256),
            num_layers=dtcs_cfg.get("twin_layers", 3),
            lr=dtcs_cfg.get("twin_lr", 1e-3),
            sync_interval=dtcs_cfg.get("sync_interval", 5),
            sim_to_real_threshold=dtcs_cfg.get("sim_to_real_threshold", 0.1),
            counterfactual_trajectories=dtcs_cfg.get("counterfactual_trajectories", 16),
            device=device,
        )

        # Causal infrastructure
        hg_cfg = config.get("hypergraph", {})
        self.hypergraph_builder = HypergraphBuilder(
            max_hyperedge_size=hg_cfg.get("max_hyperedge_size", 6)
        )
        self.causal_discovery = OnlineCausalDiscovery(
            window_size=hg_cfg.get("causal_discovery_window", 50),
            significance=hg_cfg.get("causal_significance", 0.05),
            update_interval=hg_cfg.get("update_interval", 10),
        )

        # Policy optimizer (for SPN + CHSE parameters)
        self.policy_optimizer = torch.optim.Adam(
            list(self.chse.parameters()) + list(self.spn.parameters()),
            lr=ccpg_cfg.get("lr_policy", 3e-4),
        )

        # Trajectory buffer
        self.trajectory_buffer = []
        self.current_hypergraph: Optional[CausalHypergraph] = None

    def select_action(self, observation: dict,
                      deterministic: bool = False) -> dict:
        """Full CHRONOS decision pipeline.

        1. Build hypergraph from observation
        2. Encode via CHSE
        3. Generate action via SPN
        4. Safety check via DTCS
        """
        # Phase 1: Build causal hypergraph
        self.current_hypergraph = self.hypergraph_builder.build_from_observation(
            observation, self.config
        )
        hg_data = self.current_hypergraph.to_pyg_data()

        # Move data to device
        for k, v in hg_data.items():
            if isinstance(v, torch.Tensor):
                hg_data[k] = v.to(self.device)

        # Phase 2: Encode state via CHSE
        z_global, vertex_embeddings = self.chse(hg_data)
        self._last_state_embedding = z_global.detach()

        # Phase 3: Generate action via SPN
        actions, log_probs = self.spn.get_action(z_global, deterministic=deterministic)
        self._last_log_probs = log_probs

        # Phase 4: Safety check via DTCS (if twin is calibrated)
        action_tensor = self._flatten_actions(actions)
        if self.dtcs.is_twin_accurate():
            is_safe, safety_info = self.dtcs.safety_check(
                z_global.detach(), action_tensor.detach(), z_global.detach()
            )
            if not is_safe:
                # Fall back to conservative action
                actions, log_probs = self.spn.get_action(z_global, deterministic=True)
                self._last_log_probs = log_probs

        # Convert to environment action format
        env_action = self._to_env_action(actions, observation)

        # Feed causal discovery
        self._feed_causal_discovery(observation)

        return env_action

    def _flatten_actions(self, actions: dict) -> torch.Tensor:
        """Flatten action dict to a single tensor."""
        parts = []
        for k in sorted(actions.keys()):
            v = actions[k]
            if v.dim() == 0:
                v = v.unsqueeze(0)
            if v.dtype == torch.long:
                v = v.float()
            parts.append(v.flatten())
        return torch.cat(parts) if parts else torch.zeros(1)

    def _to_env_action(self, actions: dict, observation: dict) -> dict:
        """Convert SPN action output to environment action format."""
        num_tasks = observation.get("num_active_tasks", 1)

        # Offloading: assign all active tasks to the selected node
        offload_idx = actions.get("offload", torch.tensor(0)).item()
        offloading = np.full(max(num_tasks, 1), offload_idx, dtype=int)

        # Resource allocation per node
        resource_frac = actions.get("resource", torch.tensor([0.5])).detach().cpu().numpy().flatten()
        resource_alloc = np.full(self.num_edge_nodes, float(resource_frac.mean()))

        # Power control per device
        power_level = actions.get("resource", torch.tensor([0.5])).detach().cpu().numpy().flatten()
        power_control = np.full(self.config["system"]["num_iot_devices"],
                                float(power_level.mean()))

        # Channel assignment
        channel_raw = actions.get("channel", torch.tensor(0))
        channel_idx = int(channel_raw.flatten()[0].item())
        channel_assign = np.full(self.config["system"]["num_iot_devices"],
                                 channel_idx, dtype=int)

        # FL parameters
        fl_params = actions.get("fl_params", torch.tensor([0.5, 0.5])).detach().cpu().numpy().flatten()
        fl_participate = np.ones(self.num_edge_nodes, dtype=int)
        fl_local_steps = np.full(self.num_edge_nodes, max(1, int(float(fl_params[0]) * 10)))

        return {
            "offloading": offloading,
            "resource_alloc": resource_alloc,
            "power_control": power_control,
            "channel_assign": channel_assign,
            "fl_participate": fl_participate,
            "fl_local_steps": fl_local_steps,
        }

    def _feed_causal_discovery(self, observation: dict):
        """Feed observation to causal discovery module."""
        variables = {}
        node_feats = observation.get("node_features")
        if node_feats is not None:
            for i in range(min(node_feats.shape[0], 10)):
                variables[f"node_{i}_load"] = node_feats[i, 4:5].numpy()  # CPU utilization
                variables[f"node_{i}_queue"] = node_feats[i, 3:4].numpy()

        self.causal_discovery.observe(variables)

    def store_transition(self, state_embedding: torch.Tensor, action: dict,
                         rewards: dict, next_state_embedding: torch.Tensor,
                         done: bool, constraint_violations: dict):
        """Store transition in trajectory buffer."""
        self.trajectory_buffer.append({
            "state": state_embedding.detach(),
            "action": action,
            "rewards": rewards,
            "next_state": next_state_embedding.detach(),
            "done": done,
            "constraint_violations": constraint_violations,
            "log_probs": {k: v.detach() for k, v in self._last_log_probs.items()},
        })

        # Feed digital twin
        reward_tensor = torch.tensor([
            rewards.get("accuracy", 0), rewards.get("latency", 0),
            rewards.get("energy", 0), rewards.get("communication", 0),
        ], dtype=torch.float32)

        action_tensor = self._flatten_actions_from_env(action)

        self.dtcs.record_transition(
            state_embedding.detach(), action_tensor,
            next_state_embedding.detach(), reward_tensor,
            state_embedding.detach(),  # hg_embedding ≈ state embedding
        )

    def _flatten_actions_from_env(self, action: dict) -> torch.Tensor:
        """Flatten environment action dict."""
        parts = []
        for k in sorted(action.keys()):
            v = action[k]
            if isinstance(v, np.ndarray):
                v = torch.tensor(v, dtype=torch.float32)
            elif not isinstance(v, torch.Tensor):
                v = torch.tensor([v], dtype=torch.float32)
            parts.append(v.flatten()[:5])  # Cap each component
        return torch.cat(parts) if parts else torch.zeros(1)

    def update(self, batch: dict | None = None) -> dict[str, float]:
        """Update all modules from collected experience.

        This is the main training step called at the end of each episode/round.
        """
        metrics = {}

        if len(self.trajectory_buffer) < 10:
            return metrics

        # --- Sync Digital Twin ---
        if self.dtcs.should_sync():
            twin_metrics = self.dtcs.sync()
            metrics.update({f"twin_{k}": v for k, v in twin_metrics.items()})

        # --- Update Causal Discovery ---
        if self.causal_discovery.should_update():
            causal_edges = self.causal_discovery.discover()
            metrics["causal_edges_found"] = len(causal_edges)

        # --- Prepare training batch ---
        def _pad_to(t, dim):
            t = t.flatten()
            if t.shape[0] > dim:
                return t[:dim]
            elif t.shape[0] < dim:
                return torch.nn.functional.pad(t, (0, dim - t.shape[0]))
            return t

        states = torch.stack([_pad_to(t["state"], self.state_dim) for t in self.trajectory_buffer]).to(self.device)
        rewards_list = [t["rewards"] for t in self.trajectory_buffer]
        dones = torch.tensor([t["done"] for t in self.trajectory_buffer], dtype=torch.float32).to(self.device)

        # Stack multi-objective rewards
        reward_keys = ["accuracy", "latency", "energy", "communication"]
        rewards_tensor = torch.tensor([
            [r.get(k, 0.0) for k in reward_keys] for r in rewards_list
        ], dtype=torch.float32).to(self.device)

        # Flatten actions for critic
        target_action_dim = self.action_dim * self.num_agents
        all_actions = torch.stack([
            _pad_to(self._flatten_actions_from_env(t["action"]), target_action_dim)
            for t in self.trajectory_buffer
        ]).to(self.device)

        # --- Update Critic ---
        with torch.no_grad():
            next_states = torch.stack([_pad_to(t["next_state"], self.state_dim) for t in self.trajectory_buffer]).to(self.device)
            next_values = self.ccpg.critic(next_states, all_actions)
            targets = rewards_tensor + self.ccpg.gamma * next_values * (1 - dones.unsqueeze(1))

        critic_loss = self.ccpg.update_critic(states, all_actions, targets)
        metrics["critic_loss"] = critic_loss

        # --- Compute Advantages ---
        with torch.no_grad():
            values = self.ccpg.critic(states, all_actions)
            advantages = self.ccpg.compute_gae(rewards_tensor, values, next_values, dones)

        # --- Update Policy (SPN + CHSE) ---
        # Collect old log probs
        old_log_probs_list = []
        for t in self.trajectory_buffer:
            lp = sum(v.item() for v in t["log_probs"].values() if v.numel() > 0)
            old_log_probs_list.append(lp)
        old_log_probs = torch.tensor(old_log_probs_list, dtype=torch.float32).to(self.device)

        # Forward pass through policy to get new log probs
        self.policy_optimizer.zero_grad()

        new_log_probs_list = []
        entropy_list = []
        for t in self.trajectory_buffer:
            state = t["state"].to(self.device)
            outputs = self.spn(state)

            # Compute log prob for taken actions
            lp = torch.tensor(0.0, device=self.device)
            ent = torch.tensor(0.0, device=self.device)

            if outputs["offload_probs"] is not None:
                dist = torch.distributions.Categorical(probs=outputs["offload_probs"])
                ent += dist.entropy().mean()

            if outputs["channel_probs"] is not None:
                dist = torch.distributions.Categorical(probs=outputs["channel_probs"])
                ent += dist.entropy().mean()

            new_log_probs_list.append(lp)
            entropy_list.append(ent)

        new_log_probs = torch.stack(new_log_probs_list)
        entropy = torch.stack(entropy_list)

        # Aggregate constraint violations
        avg_violations = {}
        for t in self.trajectory_buffer:
            for k, v in t["constraint_violations"].items():
                avg_violations[k] = avg_violations.get(k, 0) + v
        for k in avg_violations:
            avg_violations[k] /= len(self.trajectory_buffer)

        # Compute policy loss
        policy_loss = self.ccpg.compute_policy_loss(
            advantages, old_log_probs, new_log_probs, entropy, avg_violations
        )

        policy_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.chse.parameters()) + list(self.spn.parameters()),
            self.ccpg.max_grad_norm,
        )
        self.policy_optimizer.step()

        metrics["policy_loss"] = policy_loss.item()
        metrics["avg_entropy"] = entropy.mean().item()
        metrics["avg_reward"] = rewards_tensor.mean().item()

        # Update scalarizer weights
        combined_utility = torch.tensor([
            r.get("combined", 0.0) for r in rewards_list
        ], dtype=torch.float32)
        self.ccpg.update_scalarizer_weights(rewards_tensor.cpu(), combined_utility)

        metrics["lambda_weights"] = self.ccpg.scalarizer.lambda_weights.tolist()

        # Clear buffer
        self.trajectory_buffer.clear()

        return metrics

    def get_state_dict(self) -> dict:
        return {
            "chse": self.chse.state_dict(),
            "spn": self.spn.state_dict(),
            "critic": self.ccpg.critic.state_dict(),
            "dtcs_dynamics": self.dtcs.dynamics.state_dict(),
            "dtcs_reward": self.dtcs.reward_model.state_dict(),
            "scalarizer": self.ccpg.scalarizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.ccpg.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        self.chse.load_state_dict(state_dict["chse"])
        self.spn.load_state_dict(state_dict["spn"])
        self.ccpg.critic.load_state_dict(state_dict["critic"])
        self.dtcs.dynamics.load_state_dict(state_dict["dtcs_dynamics"])
        self.dtcs.reward_model.load_state_dict(state_dict["dtcs_reward"])
        self.ccpg.scalarizer.load_state_dict(state_dict["scalarizer"])
        self.policy_optimizer.load_state_dict(state_dict["policy_optimizer"])
        self.ccpg.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
