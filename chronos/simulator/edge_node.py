"""Edge node simulation with compute, energy, and queue management."""

from __future__ import annotations

from collections import deque

import numpy as np
from chronos.core.types import EdgeNodeState, TaskInfo, TaskStatus


class EdgeNode:
    """Simulates an edge computing node with task execution and resource management."""

    def __init__(
        self,
        node_id: int,
        cpu_frequency_ghz: float,
        memory_gb: float,
        energy_budget_j: float,
        effective_capacitance: float = 1e-28,
        queue_capacity: int = 50,
        position: np.ndarray | None = None,
    ):
        self.node_id = node_id
        self.cpu_frequency_ghz = cpu_frequency_ghz
        self.cpu_frequency_hz = cpu_frequency_ghz * 1e9
        self.memory_gb = memory_gb
        self.energy_budget_j = energy_budget_j
        self.effective_capacitance = effective_capacitance
        self.queue_capacity = queue_capacity
        self.position = position if position is not None else np.zeros(2)

        # Runtime state
        self.energy_consumed_j = 0.0
        self.task_queue: deque[TaskInfo] = deque(maxlen=queue_capacity)
        self.active_tasks: list[tuple[TaskInfo, float]] = []  # (task, remaining_cycles)
        self.resource_allocation: dict[int, float] = {}  # task_id -> fraction

        # FL state
        self.local_model_version = 0
        self.local_loss = float('inf')
        self.num_local_samples = 0

        # Metrics for this step
        self._step_energy = 0.0
        self._step_completed = 0
        self._step_dropped = 0

    def reset(self):
        """Reset node state for a new episode."""
        self.energy_consumed_j = 0.0
        self.task_queue.clear()
        self.active_tasks.clear()
        self.resource_allocation.clear()
        self.local_model_version = 0
        self.local_loss = float('inf')
        self._step_energy = 0.0
        self._step_completed = 0
        self._step_dropped = 0

    def can_accept_task(self) -> bool:
        """Check if the node can accept another task."""
        return len(self.task_queue) < self.queue_capacity

    def enqueue_task(self, task: TaskInfo) -> bool:
        """Add a task to the execution queue. Returns True if accepted."""
        if not self.can_accept_task():
            task.status = TaskStatus.FAILED
            self._step_dropped += 1
            return False

        task.status = TaskStatus.QUEUED
        task.assigned_node = self.node_id
        self.task_queue.append(task)
        return True

    def allocate_resources(self, allocations: dict[int, float]):
        """Set resource allocation for active tasks.

        Args:
            allocations: {task_id: fraction} where fractions sum to <= 1.0
        """
        total = sum(allocations.values())
        if total > 1.0 + 1e-6:
            # Normalize
            allocations = {k: v / total for k, v in allocations.items()}
        self.resource_allocation = allocations

    def step(self, dt_ms: float, current_time: float) -> list[TaskInfo]:
        """Advance simulation by dt_ms. Returns list of completed tasks."""
        self._step_energy = 0.0
        self._step_completed = 0
        self._step_dropped = 0
        dt_s = dt_ms / 1000.0
        completed = []

        # Move queued tasks to active if resources available
        while self.task_queue and len(self.active_tasks) < self.queue_capacity:
            task = self.task_queue.popleft()
            task.status = TaskStatus.EXECUTING
            self.active_tasks.append((task, task.computation_cycles))

        # Execute active tasks with allocated resources
        still_active = []
        for task, remaining_cycles in self.active_tasks:
            # Get resource fraction for this task
            frac = self.resource_allocation.get(task.task_id, 1.0 / max(len(self.active_tasks), 1))
            frac = min(frac, 1.0)

            # Compute cycles executed in this step
            allocated_freq_hz = self.cpu_frequency_hz * frac
            cycles_executed = allocated_freq_hz * dt_s
            remaining = remaining_cycles - cycles_executed

            # Energy: E = kappa * f^2 * cycles (dynamic voltage frequency scaling)
            energy = self.effective_capacitance * (allocated_freq_hz ** 2) * cycles_executed
            self._step_energy += energy
            self.energy_consumed_j += energy

            if remaining <= 0:
                # Task completed
                task.status = TaskStatus.COMPLETED
                completed.append(task)
                self._step_completed += 1
            elif current_time - task.arrival_time > task.deadline_ms:
                # Deadline missed
                task.status = TaskStatus.DEADLINE_MISSED
                completed.append(task)
                self._step_dropped += 1
            else:
                still_active.append((task, remaining))

        self.active_tasks = still_active
        return completed

    @property
    def cpu_utilization(self) -> float:
        """Current CPU utilization [0, 1]."""
        if not self.active_tasks:
            return 0.0
        total_alloc = sum(
            self.resource_allocation.get(t.task_id, 1.0 / max(len(self.active_tasks), 1))
            for t, _ in self.active_tasks
        )
        return min(total_alloc, 1.0)

    @property
    def memory_utilization(self) -> float:
        """Estimated memory utilization based on active tasks."""
        if not self.active_tasks:
            return 0.0
        total_data_gb = sum(t.data_size_bytes / 1e9 for t, _ in self.active_tasks)
        return min(total_data_gb / self.memory_gb, 1.0)

    @property
    def queue_length(self) -> int:
        return len(self.task_queue) + len(self.active_tasks)

    def get_state(self) -> EdgeNodeState:
        """Get current node state as a dataclass."""
        return EdgeNodeState(
            node_id=self.node_id,
            cpu_frequency_ghz=self.cpu_frequency_ghz,
            memory_gb=self.memory_gb,
            energy_budget_j=self.energy_budget_j,
            energy_consumed_j=self.energy_consumed_j,
            queue_length=self.queue_length,
            queue_capacity=self.queue_capacity,
            cpu_utilization=self.cpu_utilization,
            memory_utilization=self.memory_utilization,
            position=self.position.copy(),
            active_tasks=[t.task_id for t, _ in self.active_tasks],
            local_model_version=self.local_model_version,
            local_loss=self.local_loss,
            num_local_samples=self.num_local_samples,
        )

    def get_step_energy(self) -> float:
        return self._step_energy

    def get_computation_time_estimate(self, task: TaskInfo, resource_fraction: float = 1.0) -> float:
        """Estimate computation time in ms for a given task."""
        freq_hz = self.cpu_frequency_hz * resource_fraction
        if freq_hz <= 0:
            return float('inf')
        return (task.computation_cycles / freq_hz) * 1000.0  # ms
