"""Module 5: Digital Twin Causal Simulator (DTCS).

Implements a differentiable digital twin of the edge-IoT system for:
- Counterfactual trajectory generation via interventions
- Causal discovery validation
- Safe exploration (sim-to-real causal transfer)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableSystemModel(nn.Module):
    """Differentiable approximation of the edge-IoT environment dynamics.

    s_hat(t+1) = D(s(t), A(t), H(t); xi)

    Parameterized as an MLP with residual connections.
    """

    def __init__(self, state_dim: int, action_dim: int, hypergraph_dim: int,
                 hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.state_dim = state_dim
        input_dim = state_dim + action_dim + hypergraph_dim

        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers + [state_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(dims[i + 1]))

        self.net = nn.Sequential(*layers)

        # Residual connection for state
        self.residual_proj = nn.Linear(state_dim, state_dim) if state_dim != input_dim else None

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                hypergraph_embedding: torch.Tensor) -> torch.Tensor:
        """Predict next state.

        Args:
            state: (batch, state_dim)
            action: (batch, action_dim)
            hypergraph_embedding: (batch, hg_dim)

        Returns:
            next_state: (batch, state_dim)
        """
        x = torch.cat([state, action, hypergraph_embedding], dim=-1)
        delta = self.net(x)
        # Residual: next_state = state + delta
        return state + delta


class RewardPredictor(nn.Module):
    """Predicts multi-objective rewards from state-action pairs."""

    def __init__(self, state_dim: int, action_dim: int,
                 num_objectives: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objectives),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class DigitalTwinCausalSimulator:
    """Complete Digital Twin with causal interventional capabilities.

    Serves three purposes:
    1. Counterfactual training data generation
    2. Causal discovery validation
    3. Safe exploration (sim-to-real transfer)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hypergraph_dim: int,
        num_objectives: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
        lr: float = 1e-3,
        sync_interval: int = 5,
        sim_to_real_threshold: float = 0.1,
        counterfactual_trajectories: int = 16,
        device: torch.device = torch.device("cpu"),
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hypergraph_dim = hypergraph_dim
        self.num_objectives = num_objectives
        self.sync_interval = sync_interval
        self.sim_to_real_threshold = sim_to_real_threshold
        self.num_cf_trajectories = counterfactual_trajectories
        self.device = device

        # Dynamics model
        self.dynamics = DifferentiableSystemModel(
            state_dim, action_dim, hypergraph_dim, hidden_dim, num_layers
        ).to(device)

        # Reward predictor
        self.reward_model = RewardPredictor(
            state_dim, action_dim, num_objectives
        ).to(device)

        # Optimizers
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=lr)
        self.reward_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=lr)

        # Experience buffer for twin calibration
        self.buffer_states: list[torch.Tensor] = []
        self.buffer_actions: list[torch.Tensor] = []
        self.buffer_next_states: list[torch.Tensor] = []
        self.buffer_rewards: list[torch.Tensor] = []
        self.buffer_hg_embeddings: list[torch.Tensor] = []
        self.max_buffer_size = 10000

        # Twin accuracy tracking
        self.prediction_errors: list[float] = []
        self.steps_since_sync = 0

    def _fix_tensor(self, t: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Truncate or pad a 1D tensor to target_dim."""
        t = t.detach().cpu().flatten()
        if t.shape[0] > target_dim:
            return t[:target_dim]
        elif t.shape[0] < target_dim:
            return torch.nn.functional.pad(t, (0, target_dim - t.shape[0]))
        return t

    def record_transition(self, state: torch.Tensor, action: torch.Tensor,
                          next_state: torch.Tensor, rewards: torch.Tensor,
                          hg_embedding: torch.Tensor):
        """Record a real-world transition for twin calibration."""
        self.buffer_states.append(self._fix_tensor(state, self.state_dim))
        self.buffer_actions.append(self._fix_tensor(action, self.action_dim))
        self.buffer_next_states.append(self._fix_tensor(next_state, self.state_dim))
        self.buffer_rewards.append(rewards.detach().cpu())
        self.buffer_hg_embeddings.append(self._fix_tensor(hg_embedding, self.hypergraph_dim))

        # Evict oldest if buffer full
        if len(self.buffer_states) > self.max_buffer_size:
            self.buffer_states.pop(0)
            self.buffer_actions.pop(0)
            self.buffer_next_states.pop(0)
            self.buffer_rewards.pop(0)
            self.buffer_hg_embeddings.pop(0)

        self.steps_since_sync += 1

    def should_sync(self) -> bool:
        return self.steps_since_sync >= self.sync_interval and len(self.buffer_states) >= 32

    def sync(self, num_epochs: int = 5, batch_size: int = 64) -> dict[str, float]:
        """Synchronize twin with real system using recorded transitions.

        xi^{t+1} = xi^t - eta * grad ||s_real - D(s, A, H; xi)||^2
        """
        if len(self.buffer_states) < batch_size:
            return {"dynamics_loss": 0.0, "reward_loss": 0.0}

        states = torch.stack(self.buffer_states).to(self.device)
        actions = torch.stack(self.buffer_actions).to(self.device)
        next_states = torch.stack(self.buffer_next_states).to(self.device)
        rewards = torch.stack(self.buffer_rewards).to(self.device)
        hg_embs = torch.stack(self.buffer_hg_embeddings).to(self.device)

        n = states.shape[0]
        total_dyn_loss = 0.0
        total_rew_loss = 0.0

        for epoch in range(num_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]

                # Update dynamics model
                self.dynamics_optimizer.zero_grad()
                pred_next = self.dynamics(states[idx], actions[idx], hg_embs[idx])
                dyn_loss = F.mse_loss(pred_next, next_states[idx])
                dyn_loss.backward()
                self.dynamics_optimizer.step()
                total_dyn_loss += dyn_loss.item()

                # Update reward model
                self.reward_optimizer.zero_grad()
                pred_rewards = self.reward_model(states[idx], actions[idx])
                rew_loss = F.mse_loss(pred_rewards, rewards[idx])
                rew_loss.backward()
                self.reward_optimizer.step()
                total_rew_loss += rew_loss.item()

        num_batches = max(1, (n // batch_size) * num_epochs)
        avg_dyn_loss = total_dyn_loss / num_batches
        avg_rew_loss = total_rew_loss / num_batches

        self.prediction_errors.append(avg_dyn_loss)
        self.steps_since_sync = 0

        return {"dynamics_loss": avg_dyn_loss, "reward_loss": avg_rew_loss}

    def simulate_trajectory(
        self,
        initial_state: torch.Tensor,
        policy_fn,
        hg_embedding: torch.Tensor,
        horizon: int = 20,
    ) -> dict[str, torch.Tensor]:
        """Simulate a trajectory in the digital twin.

        Args:
            initial_state: starting state
            policy_fn: callable(state) -> action
            hg_embedding: hypergraph embedding (assumed constant over trajectory)
            horizon: number of steps

        Returns:
            dict with 'states', 'actions', 'rewards' tensors
        """
        states = [initial_state.unsqueeze(0)]
        actions = []
        rewards = []

        state = initial_state.unsqueeze(0).to(self.device)
        hg = hg_embedding.unsqueeze(0).to(self.device) if hg_embedding.dim() == 1 else hg_embedding.to(self.device)

        for t in range(horizon):
            with torch.no_grad():
                action = policy_fn(state.squeeze(0))
                if isinstance(action, dict):
                    # Flatten action dict
                    action_parts = []
                    for k in sorted(action.keys()):
                        v = action[k]
                        if v.dim() == 0:
                            v = v.unsqueeze(0)
                        if v.dtype == torch.long:
                            v = v.float()
                        action_parts.append(v.flatten())
                    action_tensor = torch.cat(action_parts).unsqueeze(0).to(self.device)
                else:
                    action_tensor = action.unsqueeze(0).to(self.device) if action.dim() == 1 else action.to(self.device)

            # Pad or truncate action to expected dim
            if action_tensor.shape[-1] < self.action_dim:
                pad = torch.zeros(1, self.action_dim - action_tensor.shape[-1], device=self.device)
                action_tensor = torch.cat([action_tensor, pad], dim=-1)
            elif action_tensor.shape[-1] > self.action_dim:
                action_tensor = action_tensor[:, :self.action_dim]

            next_state = self.dynamics(state, action_tensor, hg)
            reward = self.reward_model(state, action_tensor)

            states.append(next_state.detach())
            actions.append(action_tensor.detach())
            rewards.append(reward.detach())

            state = next_state.detach()

        return {
            "states": torch.cat(states, dim=0),        # (H+1, state_dim)
            "actions": torch.cat(actions, dim=0),       # (H, action_dim)
            "rewards": torch.cat(rewards, dim=0),       # (H, num_objectives)
        }

    def generate_counterfactual_trajectories(
        self,
        initial_state: torch.Tensor,
        actual_action: torch.Tensor,
        all_actions: torch.Tensor,
        agent_idx: int,
        agent_action_dim: int,
        policy_fn,
        hg_embedding: torch.Tensor,
        horizon: int = 10,
    ) -> list[dict[str, torch.Tensor]]:
        """Generate counterfactual trajectories by intervening on agent_idx's action.

        do(A_i = a') for different counterfactual actions a'.

        Args:
            initial_state: current state
            actual_action: actual joint action taken
            all_actions: full joint action vector
            agent_idx: which agent to intervene on
            agent_action_dim: dimension of agent's action space
            policy_fn: agent's policy for generating counterfactual actions
            hg_embedding: hypergraph embedding
            horizon: trajectory length

        Returns:
            list of trajectory dicts, one per counterfactual
        """
        trajectories = []
        action_start = agent_idx * agent_action_dim
        action_end = action_start + agent_action_dim

        for _ in range(self.num_cf_trajectories):
            # Sample counterfactual action
            with torch.no_grad():
                cf_action_dict, _ = policy_fn(initial_state)
                cf_parts = []
                for k in sorted(cf_action_dict.keys()):
                    v = cf_action_dict[k]
                    if v.dim() == 0:
                        v = v.unsqueeze(0)
                    if v.dtype == torch.long:
                        v = v.float()
                    cf_parts.append(v.flatten())
                cf_action = torch.cat(cf_parts)

            # Construct intervened joint action
            modified_action = all_actions.clone()
            cf_dim = min(cf_action.shape[0], action_end - action_start)
            modified_action[action_start:action_start + cf_dim] = cf_action[:cf_dim]

            # Simulate from this intervention
            state = initial_state.unsqueeze(0).to(self.device)
            action_tensor = modified_action.unsqueeze(0).to(self.device)
            hg = hg_embedding.unsqueeze(0).to(self.device) if hg_embedding.dim() == 1 else hg_embedding.to(self.device)

            if action_tensor.shape[-1] < self.action_dim:
                pad = torch.zeros(1, self.action_dim - action_tensor.shape[-1], device=self.device)
                action_tensor = torch.cat([action_tensor, pad], dim=-1)
            elif action_tensor.shape[-1] > self.action_dim:
                action_tensor = action_tensor[:, :self.action_dim]

            traj_states = [state]
            traj_actions = [action_tensor]
            traj_rewards = [self.reward_model(state, action_tensor)]

            for t in range(1, horizon):
                with torch.no_grad():
                    next_state = self.dynamics(state, action_tensor, hg)
                    # Use policy for subsequent actions
                    next_action_dict, _ = policy_fn(next_state.squeeze(0))
                    next_parts = []
                    for k in sorted(next_action_dict.keys()):
                        v = next_action_dict[k]
                        if v.dim() == 0:
                            v = v.unsqueeze(0)
                        if v.dtype == torch.long:
                            v = v.float()
                        next_parts.append(v.flatten())
                    action_tensor = torch.cat(next_parts).unsqueeze(0).to(self.device)
                    if action_tensor.shape[-1] < self.action_dim:
                        pad = torch.zeros(1, self.action_dim - action_tensor.shape[-1], device=self.device)
                        action_tensor = torch.cat([action_tensor, pad], dim=-1)
                    elif action_tensor.shape[-1] > self.action_dim:
                        action_tensor = action_tensor[:, :self.action_dim]

                    reward = self.reward_model(next_state, action_tensor)

                    traj_states.append(next_state)
                    traj_actions.append(action_tensor)
                    traj_rewards.append(reward)
                    state = next_state

            trajectories.append({
                "states": torch.cat(traj_states, dim=0),
                "actions": torch.cat(traj_actions, dim=0),
                "rewards": torch.cat(traj_rewards, dim=0),
                "intervention_action": cf_action.detach(),
            })

        return trajectories

    def validate_causal_edge(
        self,
        cause_state_idx: int,
        effect_state_idx: int,
        current_state: torch.Tensor,
        action: torch.Tensor,
        hg_embedding: torch.Tensor,
        delta: float = 1.0,
        num_samples: int = 50,
    ) -> float:
        """Validate a causal edge by performing intervention in the twin.

        Intervene on cause variable, observe effect on effect variable.
        Returns estimated causal effect strength.
        """
        state = current_state.unsqueeze(0).expand(num_samples, -1).clone().to(self.device)
        action_t = action.unsqueeze(0).expand(num_samples, -1).to(self.device)
        hg = hg_embedding.unsqueeze(0).expand(num_samples, -1).to(self.device)

        if action_t.shape[-1] < self.action_dim:
            pad = torch.zeros(num_samples, self.action_dim - action_t.shape[-1], device=self.device)
            action_t = torch.cat([action_t, pad], dim=-1)
        elif action_t.shape[-1] > self.action_dim:
            action_t = action_t[:, :self.action_dim]

        with torch.no_grad():
            # Baseline
            next_base = self.dynamics(state, action_t, hg)

            # Intervention: perturb cause variable
            state_intervened = state.clone()
            if cause_state_idx < state.shape[1]:
                state_intervened[:, cause_state_idx] += delta
            next_intervened = self.dynamics(state_intervened, action_t, hg)

            # Measure effect
            if effect_state_idx < next_base.shape[1]:
                effect_change = (
                    next_intervened[:, effect_state_idx] - next_base[:, effect_state_idx]
                ).mean().item()
            else:
                effect_change = 0.0

        return abs(effect_change)

    def is_twin_accurate(self) -> bool:
        """Check if the twin is sufficiently accurate for use."""
        if not self.prediction_errors:
            return False
        recent_error = np.mean(self.prediction_errors[-10:])
        return recent_error < self.sim_to_real_threshold

    def safety_check(
        self,
        state: torch.Tensor,
        proposed_action: torch.Tensor,
        hg_embedding: torch.Tensor,
    ) -> tuple[bool, dict]:
        """Check if a proposed action is safe by simulating in the twin.

        Returns (is_safe, info_dict).
        """
        state_t = state.unsqueeze(0).to(self.device)
        action_t = proposed_action.unsqueeze(0).to(self.device)
        hg = hg_embedding.unsqueeze(0).to(self.device) if hg_embedding.dim() == 1 else hg_embedding.to(self.device)

        if action_t.shape[-1] < self.action_dim:
            pad = torch.zeros(1, self.action_dim - action_t.shape[-1], device=self.device)
            action_t = torch.cat([action_t, pad], dim=-1)
        elif action_t.shape[-1] > self.action_dim:
            action_t = action_t[:, :self.action_dim]

        with torch.no_grad():
            next_state = self.dynamics(state_t, action_t, hg)
            rewards = self.reward_model(state_t, action_t)

        info = {
            "predicted_next_state": next_state.squeeze(0),
            "predicted_rewards": rewards.squeeze(0),
        }

        # Safety criteria: no extreme state values, positive rewards
        is_safe = True
        if next_state.abs().max() > 100.0:
            is_safe = False
            info["reason"] = "extreme_state_values"
        if rewards.min() < -10.0:
            is_safe = False
            info["reason"] = "very_negative_reward"

        return is_safe, info
