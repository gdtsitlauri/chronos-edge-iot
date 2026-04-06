"""Module 3: Causal Counterfactual Policy Gradient (CCPG).

Implements multi-agent RL training with:
- Interventional Q-function: Q_i^(m)(s, do(a_i), H)
- Counterfactual baseline for variance reduction
- Causal advantage estimation
- Causal Tchebycheff scalarization with adaptive weights
- Lagrangian constraint handling
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterventionalCritic(nn.Module):
    """Interventional Q-function: Q_i^(m)(s, do(a_i), H).

    Estimates the value of intervening on agent i's action while
    accounting for the causal hypergraph structure.
    One critic per objective.
    """

    def __init__(self, state_dim: int, action_dim: int, num_agents: int,
                 hidden_dim: int = 256, num_objectives: int = 4):
        super().__init__()
        self.num_objectives = num_objectives
        self.num_agents = num_agents

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim + action_dim * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Per-objective value heads
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_objectives)
        ])

    def forward(self, state: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all objectives.

        Args:
            state: (batch, state_dim) global state
            all_actions: (batch, num_agents * action_dim) joint action

        Returns:
            q_values: (batch, num_objectives) Q-value per objective
        """
        x = torch.cat([state, all_actions], dim=-1)
        features = self.shared(x)

        q_values = []
        for head in self.value_heads:
            q_values.append(head(features))

        return torch.cat(q_values, dim=-1)  # (batch, num_objectives)


class CounterfactualBaseline(nn.Module):
    """Counterfactual baseline for variance reduction.

    b_i^(m)(s, H) = E_{a_i ~ pi_i^cf} [Q_i^(m)(s, do(a_i), H)]

    Marginalizes over agent i's action while keeping others fixed.
    """

    def __init__(self, critic: InterventionalCritic, agent_action_dim: int):
        super().__init__()
        self.critic = critic
        self.agent_action_dim = agent_action_dim

    def compute(self, state: torch.Tensor, all_actions: torch.Tensor,
                agent_idx: int, policy_network, num_samples: int = 8) -> torch.Tensor:
        """Compute counterfactual baseline for agent_idx.

        Samples alternative actions from agent's policy and averages Q-values.

        Args:
            state: (batch, state_dim)
            all_actions: (batch, total_action_dim)
            agent_idx: which agent to marginalize
            policy_network: the agent's spiking policy
            num_samples: number of counterfactual action samples

        Returns:
            baseline: (batch, num_objectives) counterfactual baseline values
        """
        batch_size = state.shape[0]
        total_action_dim = all_actions.shape[1]
        action_start = agent_idx * self.agent_action_dim
        action_end = action_start + self.agent_action_dim

        baselines = []
        for _ in range(num_samples):
            # Sample counterfactual action
            with torch.no_grad():
                cf_actions_dict, _ = policy_network.get_action(state)
                # Flatten counterfactual action
                cf_action = self._flatten_actions(cf_actions_dict)

            # Replace agent i's action with counterfactual
            modified_actions = all_actions.clone()
            cf_dim = min(cf_action.shape[-1], action_end - action_start)
            modified_actions[:, action_start:action_start + cf_dim] = cf_action[:, :cf_dim]

            q_values = self.critic(state, modified_actions)
            baselines.append(q_values)

        return torch.stack(baselines).mean(dim=0)

    def _flatten_actions(self, actions_dict: dict) -> torch.Tensor:
        """Flatten action dict into a single tensor."""
        parts = []
        for key in sorted(actions_dict.keys()):
            val = actions_dict[key]
            if val.dim() == 1:
                val = val.unsqueeze(-1)
            if val.dtype == torch.long:
                val = val.float()
            parts.append(val)
        if parts:
            return torch.cat(parts, dim=-1)
        return torch.zeros(1, 1)


class CausalTchebycheffScalarizer(nn.Module):
    """Causal Tchebycheff scalarization with adaptive objective weights.

    A_i^scal = max_m { lambda_m(t) * A_i^(m,causal) / sigma_m }

    Weights lambda_m adapt based on causal importance:
    lambda_m(t+1) = lambda_m(t) + eta * (ACE(obj_m -> system_utility) - lambda_bar_m)
    """

    def __init__(self, num_objectives: int = 4, lr_lambda: float = 1e-3,
                 initial_weights: list[float] | None = None):
        super().__init__()
        self.num_objectives = num_objectives
        self.lr_lambda = lr_lambda

        if initial_weights is None:
            initial_weights = [1.0 / num_objectives] * num_objectives

        # Objective weights (not nn.Parameter — updated manually)
        self.register_buffer(
            "lambda_weights",
            torch.tensor(initial_weights, dtype=torch.float32)
        )

        # Running statistics for normalization
        self.register_buffer("running_mean", torch.zeros(num_objectives))
        self.register_buffer("running_var", torch.ones(num_objectives))
        self.register_buffer("update_count", torch.tensor(0, dtype=torch.long))

    def update_statistics(self, advantages: torch.Tensor):
        """Update running mean and variance of advantages per objective.

        Args:
            advantages: (batch, num_objectives) advantage values
        """
        batch_mean = advantages.mean(dim=0)
        batch_var = advantages.var(dim=0)

        momentum = 0.99
        self.running_mean = momentum * self.running_mean + (1 - momentum) * batch_mean
        self.running_var = momentum * self.running_var + (1 - momentum) * batch_var
        self.update_count += 1

    def scalarize(self, advantages: torch.Tensor) -> torch.Tensor:
        """Apply Causal Tchebycheff scalarization.

        Args:
            advantages: (batch, num_objectives) causal advantages

        Returns:
            scalar_advantage: (batch,) scalarized advantage
        """
        # Normalize
        sigma = (self.running_var + 1e-8).sqrt()
        normalized = advantages / sigma.unsqueeze(0)

        # Weighted Tchebycheff: max_m { lambda_m * A_m / sigma_m }
        weighted = self.lambda_weights.unsqueeze(0) * normalized
        # Use smooth max (log-sum-exp) for differentiability
        scalar = torch.logsumexp(weighted * 10, dim=-1) / 10

        return scalar

    def update_weights(self, causal_importances: torch.Tensor):
        """Update objective weights based on causal importance.

        Args:
            causal_importances: (num_objectives,) ACE of each objective on system utility
        """
        target = F.softmax(causal_importances, dim=0)
        self.lambda_weights = (
            self.lambda_weights + self.lr_lambda * (target - self.lambda_weights)
        )
        self.lambda_weights = F.softmax(self.lambda_weights, dim=0)


class LagrangianDualUpdater:
    """Lagrangian dual variable updates for constraint handling.

    mu_j <- [mu_j + eta_mu * g_j(omega)]^+
    """

    def __init__(self, num_constraints: int, lr_dual: float = 0.01,
                 constraint_names: list[str] | None = None):
        self.num_constraints = num_constraints
        self.lr_dual = lr_dual
        self.dual_variables = torch.zeros(num_constraints)
        self.constraint_names = constraint_names or [f"c{i}" for i in range(num_constraints)]

    def update(self, violations: dict[str, float]) -> torch.Tensor:
        """Update dual variables from constraint violations.

        Args:
            violations: {constraint_name: violation_amount} (0 = satisfied)

        Returns:
            penalty: scalar Lagrangian penalty term
        """
        for i, name in enumerate(self.constraint_names):
            g = violations.get(name, 0.0)
            self.dual_variables[i] = max(0.0, self.dual_variables[i] + self.lr_dual * g)

        return self.dual_variables.clone()

    def compute_penalty(self, violations: dict[str, float]) -> torch.Tensor:
        """Compute Lagrangian penalty: sum_j mu_j * g_j."""
        penalty = torch.tensor(0.0)
        for i, name in enumerate(self.constraint_names):
            g = violations.get(name, 0.0)
            penalty += self.dual_variables[i] * g
        return penalty


class CausalCounterfactualPolicyGradient:
    """Complete CCPG training algorithm.

    Combines:
    - Interventional critics (one per objective)
    - Counterfactual baselines
    - Causal Tchebycheff scalarization
    - Lagrangian constraint handling
    - Surrogate gradient BPTT for SNN policies
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        num_objectives: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        lr_critic: float = 1e-3,
        lr_lambda: float = 1e-3,
        lr_dual: float = 0.01,
        max_grad_norm: float = 0.5,
        counterfactual_samples: int = 8,
        num_constraints: int = 6,
        constraint_names: list[str] | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_agents = num_agents
        self.num_objectives = num_objectives
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.counterfactual_samples = counterfactual_samples
        self.device = device

        # Interventional critic (centralized)
        self.critic = InterventionalCritic(
            state_dim, action_dim, num_agents,
            num_objectives=num_objectives,
        ).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Scalarizer
        self.scalarizer = CausalTchebycheffScalarizer(
            num_objectives, lr_lambda=lr_lambda,
        ).to(device)

        # Lagrangian dual
        if constraint_names is None:
            constraint_names = ["energy_budget", "compute_capacity", "power",
                                "deadline", "task_assignment", "causality"]
        self.dual_updater = LagrangianDualUpdater(
            num_constraints, lr_dual=lr_dual,
            constraint_names=constraint_names,
        )

        # Counterfactual baselines (created per agent)
        self.baselines: dict[int, CounterfactualBaseline] = {}
        for i in range(num_agents):
            self.baselines[i] = CounterfactualBaseline(self.critic, action_dim)

    def compute_causal_advantages(
        self,
        states: torch.Tensor,
        all_actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        agent_idx: int,
        policy_network,
    ) -> torch.Tensor:
        """Compute causal advantage for agent i across all objectives.

        A_i^(m,causal) = Q_i^(m)(s, do(a_i), H) - b_i^(m)(s, H)

        Args:
            states: (T, state_dim)
            all_actions: (T, total_action_dim)
            rewards: (T, num_objectives)
            next_states: (T, state_dim)
            dones: (T,)
            agent_idx: which agent
            policy_network: agent's SPN

        Returns:
            advantages: (T, num_objectives) causal advantages
        """
        T = states.shape[0]

        with torch.no_grad():
            # Q-values for actual actions (interventional)
            q_actual = self.critic(states, all_actions)  # (T, num_obj)

            # Counterfactual baseline
            baseline = self.baselines[agent_idx].compute(
                states, all_actions, agent_idx, policy_network,
                num_samples=self.counterfactual_samples,
            )  # (T, num_obj)

            # Causal advantage
            advantages = q_actual - baseline

        return advantages

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    next_values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Generalized Advantage Estimation per objective.

        Args:
            rewards: (T, num_obj)
            values: (T, num_obj)
            next_values: (T, num_obj)
            dones: (T,)

        Returns:
            advantages: (T, num_obj)
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(self.num_objectives, device=rewards.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_values[t]
            else:
                next_val = values[t + 1]

            mask = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae

        return advantages

    def update_critic(self, states: torch.Tensor, all_actions: torch.Tensor,
                      target_values: torch.Tensor) -> float:
        """Update interventional critic towards target values.

        Args:
            states: (batch, state_dim)
            all_actions: (batch, total_action_dim)
            target_values: (batch, num_objectives)

        Returns:
            critic_loss: float
        """
        predicted = self.critic(states, all_actions)
        loss = F.mse_loss(predicted, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        return loss.item()

    def compute_policy_loss(
        self,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        entropy: torch.Tensor,
        constraint_violations: dict[str, float],
    ) -> torch.Tensor:
        """Compute policy gradient loss with PPO clipping + Lagrangian penalty.

        L = -E[min(r*A, clip(r)*A)] - c_ent * H(pi) + sum_j mu_j * g_j

        Args:
            advantages: (batch, num_objectives) causal advantages
            old_log_probs: (batch,) log probs under old policy
            new_log_probs: (batch,) log probs under new policy
            entropy: (batch,) policy entropy
            constraint_violations: dict of constraint violations

        Returns:
            total_loss: scalar
        """
        # Scalarize multi-objective advantages
        self.scalarizer.update_statistics(advantages.detach())
        scalar_advantages = self.scalarizer.scalarize(advantages)  # (batch,)

        # Normalize
        scalar_advantages = (scalar_advantages - scalar_advantages.mean()) / (scalar_advantages.std() + 1e-8)

        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * scalar_advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * scalar_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus
        entropy_loss = -self.entropy_coeff * entropy.mean()

        # Lagrangian penalty
        lagrangian_penalty = self.dual_updater.compute_penalty(constraint_violations)

        total_loss = policy_loss + entropy_loss + lagrangian_penalty

        # Update dual variables
        self.dual_updater.update(constraint_violations)

        return total_loss

    def update_scalarizer_weights(self, objective_rewards: torch.Tensor,
                                   system_utility: torch.Tensor):
        """Update Tchebycheff weights based on causal importance of each objective.

        Args:
            objective_rewards: (T, num_objectives) per-step rewards
            system_utility: (T,) combined system utility
        """
        # Estimate ACE of each objective on system utility via correlation
        causal_importances = torch.zeros(self.num_objectives)
        for m in range(self.num_objectives):
            if objective_rewards.shape[0] > 2:
                corr = torch.corrcoef(torch.stack([
                    objective_rewards[:, m], system_utility
                ]))[0, 1]
                causal_importances[m] = corr.abs()

        self.scalarizer.update_weights(causal_importances)
