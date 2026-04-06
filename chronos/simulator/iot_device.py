"""IoT device simulation with task generation and wireless transmission."""

from __future__ import annotations

import numpy as np
from chronos.core.types import IoTDeviceState, TaskInfo, TaskStatus


class IoTDevice:
    """Simulates an IoT device that generates tasks and offloads to edge nodes."""

    def __init__(
        self,
        device_id: int,
        cpu_frequency_ghz: float,
        max_power_dbm: float = 23.0,
        energy_budget_j: float = 1000.0,
        effective_capacitance: float = 1e-28,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
    ):
        self.device_id = device_id
        self.cpu_frequency_ghz = cpu_frequency_ghz
        self.cpu_frequency_hz = cpu_frequency_ghz * 1e9
        self.max_power_dbm = max_power_dbm
        self.max_power_w = 10 ** ((max_power_dbm - 30) / 10)
        self.energy_budget_j = energy_budget_j
        self.effective_capacitance = effective_capacitance
        self.position = position if position is not None else np.zeros(2)
        self.velocity = velocity if velocity is not None else np.zeros(2)

        # Runtime state
        self.energy_consumed_j = 0.0
        self.pending_tasks: list[TaskInfo] = []
        self.current_power_w = 0.0
        self.transmitting = False

        # Metrics
        self._step_tx_energy = 0.0
        self._step_compute_energy = 0.0

    def reset(self):
        """Reset device state for a new episode."""
        self.energy_consumed_j = 0.0
        self.pending_tasks.clear()
        self.current_power_w = 0.0
        self.transmitting = False
        self._step_tx_energy = 0.0
        self._step_compute_energy = 0.0

    def receive_task(self, task: TaskInfo):
        """Receive a newly generated task."""
        self.pending_tasks.append(task)

    def execute_local(self, task: TaskInfo, dt_ms: float) -> tuple[bool, float]:
        """Execute a task locally. Returns (completed, energy_consumed).

        Returns the energy consumed in Joules.
        """
        dt_s = dt_ms / 1000.0
        cycles_executed = self.cpu_frequency_hz * dt_s
        energy = self.effective_capacitance * (self.cpu_frequency_hz ** 2) * min(cycles_executed, task.computation_cycles)
        self.energy_consumed_j += energy
        self._step_compute_energy += energy

        completed = cycles_executed >= task.computation_cycles
        if completed:
            task.status = TaskStatus.COMPLETED
        return completed, energy

    def get_local_execution_time_ms(self, task: TaskInfo) -> float:
        """Estimate local execution time for a task."""
        if self.cpu_frequency_hz <= 0:
            return float('inf')
        return (task.computation_cycles / self.cpu_frequency_hz) * 1000.0

    def get_local_execution_energy(self, task: TaskInfo) -> float:
        """Estimate local execution energy for a task (Joules)."""
        return self.effective_capacitance * (self.cpu_frequency_hz ** 2) * task.computation_cycles

    def set_transmit_power(self, power_w: float):
        """Set current transmit power (clamped to max)."""
        self.current_power_w = min(power_w, self.max_power_w)

    def consume_tx_energy(self, energy_j: float):
        """Record transmission energy consumption."""
        self.energy_consumed_j += energy_j
        self._step_tx_energy += energy_j

    def update_position(self, dt_ms: float, area_size: float):
        """Update position based on velocity with boundary reflection."""
        dt_s = dt_ms / 1000.0
        self.position += self.velocity * dt_s

        # Boundary reflection
        for dim in range(2):
            if self.position[dim] < 0:
                self.position[dim] = -self.position[dim]
                self.velocity[dim] = -self.velocity[dim]
            elif self.position[dim] > area_size:
                self.position[dim] = 2 * area_size - self.position[dim]
                self.velocity[dim] = -self.velocity[dim]

    @property
    def remaining_energy(self) -> float:
        return max(0.0, self.energy_budget_j - self.energy_consumed_j)

    @property
    def is_alive(self) -> bool:
        return self.remaining_energy > 0

    def get_state(self) -> IoTDeviceState:
        return IoTDeviceState(
            device_id=self.device_id,
            cpu_frequency_ghz=self.cpu_frequency_ghz,
            max_power_dbm=self.max_power_dbm,
            energy_budget_j=self.energy_budget_j,
            energy_consumed_j=self.energy_consumed_j,
            position=self.position.copy(),
            pending_tasks=[t.task_id for t in self.pending_tasks],
            current_power_dbm=10 * np.log10(max(self.current_power_w, 1e-30) * 1000),
            velocity=self.velocity.copy(),
        )
