"""Type definitions for the CHRONOS framework."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


class NodeType(enum.Enum):
    """Vertex types in the causal hypergraph."""
    EDGE_NODE = 0
    IOT_DEVICE = 1
    TASK = 2
    CHANNEL = 3


class TaskType(enum.Enum):
    """Types of computational tasks."""
    CLASSIFICATION = 0
    DETECTION = 1
    ANOMALY = 2
    PREDICTION = 3
    SEGMENTATION = 4
    TRACKING = 5


class TaskStatus(enum.Enum):
    """Task lifecycle states."""
    PENDING = 0
    QUEUED = 1
    EXECUTING = 2
    COMPLETED = 3
    FAILED = 4
    DEADLINE_MISSED = 5


@dataclass
class TaskInfo:
    """Characterizes a computational task T_k = (delta_k, omega_k, d_k, tau_k_max)."""
    task_id: int
    source_device: int                   # Originating IoT device
    data_size_bytes: float               # delta_k: input data size
    computation_cycles: float            # omega_k: required CPU cycles
    deadline_ms: float                   # tau_k_max: deadline
    task_type: TaskType = TaskType.CLASSIFICATION
    arrival_time: float = 0.0
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: int = -1              # -1 = unassigned, 0 = local
    priority: float = 1.0
    data_partition_id: int = 0           # d_k: data partition for FL

    @property
    def data_size_mb(self) -> float:
        return self.data_size_bytes / (1024 * 1024)


@dataclass
class EdgeNodeState:
    """State of an edge node at time t."""
    node_id: int
    cpu_frequency_ghz: float
    memory_gb: float
    energy_budget_j: float
    energy_consumed_j: float = 0.0
    queue_length: int = 0
    queue_capacity: int = 50
    cpu_utilization: float = 0.0         # [0, 1]
    memory_utilization: float = 0.0      # [0, 1]
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    active_tasks: list = field(default_factory=list)
    # FL-related
    local_model_version: int = 0
    local_loss: float = float('inf')
    num_local_samples: int = 0

    @property
    def remaining_energy(self) -> float:
        return max(0.0, self.energy_budget_j - self.energy_consumed_j)

    @property
    def available_compute(self) -> float:
        return (1.0 - self.cpu_utilization) * self.cpu_frequency_ghz * 1e9  # Hz

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.cpu_frequency_ghz,
            self.memory_gb,
            self.remaining_energy / self.energy_budget_j,
            self.queue_length / self.queue_capacity,
            self.cpu_utilization,
            self.memory_utilization,
            self.position[0],
            self.position[1],
            self.local_loss if self.local_loss != float('inf') else 10.0,
            self.num_local_samples / 1000.0,
        ], dtype=torch.float32)


@dataclass
class IoTDeviceState:
    """State of an IoT device at time t."""
    device_id: int
    cpu_frequency_ghz: float
    max_power_dbm: float
    energy_budget_j: float
    energy_consumed_j: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    pending_tasks: list = field(default_factory=list)
    current_power_dbm: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # For mobility

    @property
    def remaining_energy(self) -> float:
        return max(0.0, self.energy_budget_j - self.energy_consumed_j)

    @property
    def max_power_w(self) -> float:
        return 10 ** ((self.max_power_dbm - 30) / 10)

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.cpu_frequency_ghz,
            self.remaining_energy / self.energy_budget_j,
            self.position[0],
            self.position[1],
            len(self.pending_tasks) / 10.0,
            self.current_power_dbm / self.max_power_dbm if self.max_power_dbm > 0 else 0,
            self.velocity[0],
            self.velocity[1],
        ], dtype=torch.float32)


@dataclass
class ChannelState:
    """State of a wireless channel at time t."""
    channel_id: int
    bandwidth_hz: float
    # channel_gains[i][j] = channel gain from device i to edge node j
    channel_gains: Optional[np.ndarray] = None
    noise_power_w: float = 1e-13         # Thermal noise
    interference_w: float = 0.0
    utilization: float = 0.0             # [0, 1]

    def achievable_rate(self, device_idx: int, node_idx: int, power_w: float) -> float:
        """Shannon capacity: R = B * log2(1 + SNR)."""
        if self.channel_gains is None:
            return 0.0
        gain = self.channel_gains[device_idx, node_idx]
        signal = power_w * gain
        noise_plus_interference = self.noise_power_w + self.interference_w
        if noise_plus_interference <= 0:
            noise_plus_interference = 1e-20
        sinr = signal / noise_plus_interference
        return self.bandwidth_hz * np.log2(1 + max(sinr, 1e-10))

    def to_tensor(self, num_devices: int, num_nodes: int) -> torch.Tensor:
        features = [self.bandwidth_hz / 1e6, self.utilization]
        if self.channel_gains is not None:
            # Flatten top-k gains as features (summary statistics)
            flat_gains = self.channel_gains.flatten()
            features.extend([
                float(np.mean(flat_gains)),
                float(np.std(flat_gains)),
                float(np.max(flat_gains)),
                float(np.min(flat_gains)),
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        return torch.tensor(features, dtype=torch.float32)


@dataclass
class SystemAction:
    """Composite action A(t) = (x, r, p, a, theta, phi).

    All arrays indexed consistently with the system's device/node/task/channel ordering.
    """
    task_offloading: np.ndarray          # x: (K, N+1) binary assignment
    resource_allocation: np.ndarray      # r: (K, N) fraction of compute
    power_control: np.ndarray            # p: (D, C) transmit power (watts)
    channel_assignment: np.ndarray       # a: (D, C) binary assignment
    fl_local_steps: np.ndarray           # kappa_i for each node
    fl_aggregation_weights: np.ndarray   # alpha_i for each node
    fl_compression_ratio: np.ndarray     # rho_i for each node
    fl_participation: np.ndarray         # z_i binary for each node
    snn_thresholds: np.ndarray           # vartheta_i for each agent
    snn_time_windows: np.ndarray         # Delta_i for each agent


@dataclass
class StepResult:
    """Result returned by the environment after one step."""
    next_state: dict                     # Full system state
    rewards: dict                        # {objective_name: value}
    done: bool
    info: dict                           # Additional metrics
    constraint_violations: dict          # {constraint_name: violation_amount}


@dataclass
class Transition:
    """Single transition for replay buffer."""
    state: dict
    action: dict
    rewards: dict                        # Multi-objective rewards
    next_state: dict
    done: bool
    hypergraph_state: Optional[dict] = None
    log_prob: float = 0.0
    value_estimates: Optional[dict] = None
