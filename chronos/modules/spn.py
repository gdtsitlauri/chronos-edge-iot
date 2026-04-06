"""Module 2: Spiking Policy Network (SPN).

Implements spiking neural network policies using Leaky Integrate-and-Fire (LIF) neurons
with surrogate gradient training, hybrid rate-temporal spike encoding, and
Temporal Difference Spike Timing (TDST) action decoding.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateGradient(torch.autograd.Function):
    """Surrogate gradient for the Heaviside step function.

    Forward: Theta(x) = 1 if x >= 0, else 0
    Backward: uses arctangent surrogate: d/dx = 1/(pi * (1 + (pi*x)^2))
    """
    scale = 25.0  # Surrogate slope

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output / (math.pi * (1 + (math.pi * SurrogateGradient.scale * input) ** 2))
        return grad


spike_fn = SurrogateGradient.apply


class HybridSpikeEncoder(nn.Module):
    """Learned rate-temporal hybrid spike encoding.

    s_j(t') = 1 if integral of (beta_j * z_j + (1-beta_j) * dz_j/dt) >= threshold
    beta_j in [0,1] balances rate coding (beta=1) and temporal coding (beta=0).
    """

    def __init__(self, input_dim: int, time_steps: int = 16,
                 threshold: float = 1.0, beta_init: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.threshold = threshold

        # Learnable rate-temporal balance per neuron
        self.beta_raw = nn.Parameter(torch.full((input_dim,), math.log(beta_init / (1 - beta_init))))
        # Learnable thresholds per neuron
        self.threshold_param = nn.Parameter(torch.full((input_dim,), threshold))

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_raw)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Convert continuous features to spike trains.

        Args:
            z: (batch, input_dim) continuous state embedding

        Returns:
            spikes: (batch, time_steps, input_dim) binary spike trains
        """
        batch_size = z.shape[0]
        beta = self.beta.unsqueeze(0)  # (1, input_dim)
        thresh = self.threshold_param.abs().unsqueeze(0)  # (1, input_dim)

        spikes = []
        membrane = torch.zeros(batch_size, self.input_dim, device=z.device)
        prev_z = z.clone()

        for t in range(self.time_steps):
            # Rate component: beta * z
            rate = beta * z

            # Temporal component: (1 - beta) * dz/dt (approximated as change)
            # At encoding time, we create temporal variation via learned modulation
            temporal_phase = torch.sin(2 * math.pi * t / self.time_steps * torch.arange(
                self.input_dim, device=z.device, dtype=torch.float
            ).unsqueeze(0))
            temporal = (1 - beta) * z * temporal_phase

            # Integrate
            input_current = rate + temporal
            membrane = membrane + input_current

            # Fire
            spike = spike_fn(membrane - thresh)
            spikes.append(spike)

            # Reset membrane where spike occurred
            membrane = membrane * (1 - spike)

        return torch.stack(spikes, dim=1)  # (batch, T_s, input_dim)


class LIFLayer(nn.Module):
    """Leaky Integrate-and-Fire layer with learnable parameters.

    u_j(t+1) = lambda_j * u_j(t) + sum_k w_{jk} * s_k(t) - threshold * s_j(t)
    s_j(t) = Theta(u_j(t) - threshold)
    """

    def __init__(self, input_dim: int, output_dim: int,
                 membrane_decay: float = 0.9, threshold: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Synaptic weights
        self.weight = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.kaiming_normal_(self.weight.weight)

        # Learnable membrane time constants (per neuron)
        self.decay_raw = nn.Parameter(torch.full((output_dim,),
                                                  math.log(membrane_decay / (1 - membrane_decay))))
        # Learnable thresholds
        self.threshold = nn.Parameter(torch.full((output_dim,), threshold))

    @property
    def decay(self) -> torch.Tensor:
        """Membrane decay lambda in (0, 1)."""
        return torch.sigmoid(self.decay_raw)

    def forward(self, spike_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process spike train through LIF neurons.

        Args:
            spike_input: (batch, time_steps, input_dim) input spike trains

        Returns:
            output_spikes: (batch, time_steps, output_dim)
            membrane_traces: (batch, time_steps, output_dim)
        """
        batch_size, time_steps, _ = spike_input.shape
        device = spike_input.device

        decay = self.decay.unsqueeze(0)  # (1, output_dim)
        thresh = self.threshold.abs().unsqueeze(0)

        membrane = torch.zeros(batch_size, self.output_dim, device=device)
        output_spikes = []
        membrane_traces = []

        for t in range(time_steps):
            # Synaptic input
            syn_input = self.weight(spike_input[:, t])  # (batch, output_dim)

            # Membrane dynamics: u(t+1) = lambda * u(t) + input - thresh * spike(t)
            membrane = decay * membrane + syn_input

            # Spike generation
            spike = spike_fn(membrane - thresh)
            output_spikes.append(spike)
            membrane_traces.append(membrane.clone())

            # Reset
            membrane = membrane * (1 - spike)

        return (
            torch.stack(output_spikes, dim=1),
            torch.stack(membrane_traces, dim=1),
        )


class TDSTDecoder(nn.Module):
    """Temporal Difference Spike Timing (TDST) decoder.

    Decodes actions from output spike trains by weighting spikes with
    exponential recency: later spikes carry more weight.

    a_m = softmax( sum_t kappa(t) * s(t) )_m
    kappa(t) = exp(-beta_kappa * (T_s - t))
    """

    def __init__(self, input_dim: int, num_discrete_actions: int,
                 num_continuous_actions: int, time_steps: int = 16,
                 recency_decay: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_discrete = num_discrete_actions
        self.num_continuous = num_continuous_actions
        self.time_steps = time_steps

        # Learnable recency parameter
        self.recency_decay = nn.Parameter(torch.tensor(recency_decay))

        # Projection to action space
        if num_discrete_actions > 0:
            self.discrete_proj = nn.Linear(input_dim, num_discrete_actions)
        if num_continuous_actions > 0:
            self.continuous_mean_proj = nn.Linear(input_dim, num_continuous_actions)
            self.continuous_logstd_proj = nn.Linear(input_dim, num_continuous_actions)

    def forward(self, output_spikes: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode spike trains into actions.

        Args:
            output_spikes: (batch, time_steps, input_dim) output layer spikes

        Returns:
            dict with 'discrete_logits', 'continuous_mean', 'continuous_std'
        """
        batch_size = output_spikes.shape[0]

        # Temporal weighting: kappa(t) = exp(-beta * (T-t))
        t_indices = torch.arange(self.time_steps, device=output_spikes.device, dtype=torch.float)
        kappa = torch.exp(-self.recency_decay.abs() * (self.time_steps - 1 - t_indices))
        kappa = kappa / kappa.sum()  # Normalize

        # Weighted temporal aggregation
        # output_spikes: (batch, T, dim), kappa: (T,)
        weighted = output_spikes * kappa.unsqueeze(0).unsqueeze(-1)  # (batch, T, dim)
        aggregated = weighted.sum(dim=1)  # (batch, dim)

        result = {}

        if self.num_discrete > 0:
            result["discrete_logits"] = self.discrete_proj(aggregated)
            result["discrete_probs"] = F.softmax(result["discrete_logits"], dim=-1)

        if self.num_continuous > 0:
            result["continuous_mean"] = self.continuous_mean_proj(aggregated)
            log_std = self.continuous_logstd_proj(aggregated)
            result["continuous_std"] = torch.exp(log_std.clamp(-5, 2))

        return result


class SpikingPolicyNetwork(nn.Module):
    """Complete Spiking Policy Network for a single agent.

    Architecture:
    1. HybridSpikeEncoder: continuous state -> spike trains
    2. Stack of LIF layers: spike processing
    3. TDSTDecoder: spike trains -> action distributions

    Outputs multi-head actions for the joint decision space:
    - Task offloading (discrete: which node)
    - Resource allocation (continuous: fraction)
    - Power control (continuous: power level)
    - Channel assignment (discrete: which channel)
    - FL parameters (continuous: local steps, compression)
    """

    def __init__(
        self,
        state_dim: int,
        num_edge_nodes: int,
        num_channels: int,
        num_lif_layers: int = 3,
        hidden_neurons: int = 256,
        time_steps: int = 16,
        membrane_decay: float = 0.9,
        threshold: float = 1.0,
        beta_init: float = 0.5,
        recency_decay: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_edge_nodes = num_edge_nodes
        self.num_channels = num_channels
        self.time_steps = time_steps

        # Spike encoder
        self.encoder = HybridSpikeEncoder(
            input_dim=state_dim, time_steps=time_steps,
            threshold=threshold, beta_init=beta_init,
        )

        # LIF hidden layers
        self.lif_layers = nn.ModuleList()
        dims = [state_dim] + [hidden_neurons] * num_lif_layers
        for i in range(num_lif_layers):
            self.lif_layers.append(LIFLayer(
                input_dim=dims[i], output_dim=dims[i + 1],
                membrane_decay=membrane_decay, threshold=threshold,
            ))

        # Output dim of LIF stack (state_dim if no LIF layers, hidden_neurons otherwise)
        decoder_input_dim = hidden_neurons if num_lif_layers > 0 else state_dim

        # Action decoders (separate heads for different action types)
        # Head 1: Task offloading (discrete: node selection + local)
        self.offload_decoder = TDSTDecoder(
            decoder_input_dim, num_discrete_actions=num_edge_nodes + 1,
            num_continuous_actions=0, time_steps=time_steps,
            recency_decay=recency_decay,
        )

        # Head 2: Resource allocation + power control (continuous)
        self.resource_decoder = TDSTDecoder(
            decoder_input_dim, num_discrete_actions=0,
            num_continuous_actions=2,  # resource_frac, power_level
            time_steps=time_steps, recency_decay=recency_decay,
        )

        # Head 3: Channel assignment (discrete)
        self.channel_decoder = TDSTDecoder(
            decoder_input_dim, num_discrete_actions=num_channels,
            num_continuous_actions=0, time_steps=time_steps,
            recency_decay=recency_decay,
        )

        # Head 4: FL parameters (continuous: local_steps_frac, compression_ratio)
        self.fl_decoder = TDSTDecoder(
            decoder_input_dim, num_discrete_actions=0,
            num_continuous_actions=2,
            time_steps=time_steps, recency_decay=recency_decay,
        )

    def forward(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass: state -> action distributions.

        Args:
            state: (batch, state_dim) continuous state embedding

        Returns:
            dict with all action heads' outputs
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Encode to spikes
        spikes = self.encoder(state)  # (batch, T, state_dim)

        # Process through LIF layers
        for lif_layer in self.lif_layers:
            spikes, _ = lif_layer(spikes)

        # Decode actions from each head
        offload = self.offload_decoder(spikes)
        resource = self.resource_decoder(spikes)
        channel = self.channel_decoder(spikes)
        fl_params = self.fl_decoder(spikes)

        return {
            "offload_logits": offload.get("discrete_logits"),
            "offload_probs": offload.get("discrete_probs"),
            "resource_mean": torch.sigmoid(resource.get("continuous_mean", torch.zeros(1))),
            "resource_std": resource.get("continuous_std", torch.ones(1)),
            "power_mean": torch.sigmoid(resource.get("continuous_mean", torch.zeros(1))),
            "power_std": resource.get("continuous_std", torch.ones(1)),
            "channel_logits": channel.get("discrete_logits"),
            "channel_probs": channel.get("discrete_probs"),
            "fl_mean": torch.sigmoid(fl_params.get("continuous_mean", torch.zeros(1))),
            "fl_std": fl_params.get("continuous_std", torch.ones(1)),
        }

    def get_action(self, state: torch.Tensor,
                   deterministic: bool = False) -> tuple[dict, dict]:
        """Sample actions from the policy.

        Returns:
            actions: dict of sampled action tensors
            log_probs: dict of log-probability for each action
        """
        outputs = self.forward(state)
        actions = {}
        log_probs = {}

        # Offloading (discrete)
        if deterministic:
            offload_action = outputs["offload_probs"].argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs=outputs["offload_probs"])
            offload_action = dist.sample()
            log_probs["offload"] = dist.log_prob(offload_action)
        actions["offload"] = offload_action

        # Resource allocation (continuous)
        if outputs["resource_mean"] is not None:
            if deterministic:
                actions["resource"] = outputs["resource_mean"]
                log_probs["resource"] = torch.zeros(1)
            else:
                dist = torch.distributions.Normal(outputs["resource_mean"], outputs["resource_std"])
                resource_action = dist.sample()
                actions["resource"] = torch.sigmoid(resource_action)
                log_probs["resource"] = dist.log_prob(resource_action).sum(dim=-1)

        # Channel (discrete)
        if deterministic:
            channel_action = outputs["channel_probs"].argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs=outputs["channel_probs"])
            channel_action = dist.sample()
            log_probs["channel"] = dist.log_prob(channel_action)
        actions["channel"] = channel_action

        # FL params (continuous)
        if outputs["fl_mean"] is not None:
            if deterministic:
                actions["fl_params"] = outputs["fl_mean"]
                log_probs["fl_params"] = torch.zeros(1)
            else:
                dist = torch.distributions.Normal(outputs["fl_mean"], outputs["fl_std"])
                fl_action = dist.sample()
                actions["fl_params"] = torch.sigmoid(fl_action)
                log_probs["fl_params"] = dist.log_prob(fl_action).sum(dim=-1)

        return actions, log_probs

    def compute_spike_energy(self) -> dict[str, float]:
        """Estimate energy consumption of the SNN policy.

        Based on neuromorphic energy model:
        E_AC ≈ 0.9 pJ per spike (accumulate)
        E_MAC ≈ 4.6 pJ per multiply-accumulate (ANN equivalent)
        """
        E_AC = 0.9e-12   # Joules per spike operation
        E_MAC = 4.6e-12  # Joules per MAC

        total_spikes = 0
        total_neurons = 0

        for lif in self.lif_layers:
            total_neurons += lif.output_dim * self.time_steps

        # Estimate spike count (from last forward pass, if available)
        # For now, use expected sparsity
        expected_sparsity = 0.1  # ~10% firing rate
        estimated_spikes = total_neurons * expected_sparsity

        snn_energy = estimated_spikes * E_AC
        ann_equivalent_energy = total_neurons * E_MAC

        return {
            "snn_energy_j": snn_energy,
            "ann_energy_j": ann_equivalent_energy,
            "energy_ratio": snn_energy / max(ann_equivalent_energy, 1e-20),
            "estimated_sparsity": expected_sparsity,
        }
