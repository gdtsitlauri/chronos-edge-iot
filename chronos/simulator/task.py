"""Task generation and management for the edge-IoT system."""

from __future__ import annotations

import numpy as np
from chronos.core.types import TaskInfo, TaskType, TaskStatus


class TaskGenerator:
    """Generates computational tasks with Poisson arrivals and configurable distributions."""

    def __init__(
        self,
        num_devices: int,
        arrival_rate: float = 0.5,
        data_size_range_mb: tuple[float, float] = (0.1, 10.0),
        computation_range_mcycles: tuple[float, float] = (10, 1000),
        deadline_range_ms: tuple[float, float] = (50, 500),
        task_types: list[str] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.num_devices = num_devices
        self.arrival_rate = arrival_rate
        self.data_size_range = (data_size_range_mb[0] * 1e6, data_size_range_mb[1] * 1e6)  # bytes
        self.computation_range = (computation_range_mcycles[0] * 1e6, computation_range_mcycles[1] * 1e6)  # cycles
        self.deadline_range = deadline_range_ms
        self.rng = rng or np.random.default_rng(42)

        self.task_type_map = {
            "classification": TaskType.CLASSIFICATION,
            "detection": TaskType.DETECTION,
            "object_detection_3d": TaskType.DETECTION,
            "anomaly": TaskType.ANOMALY,
            "anomaly_detection": TaskType.ANOMALY,
            "anomaly_alert": TaskType.ANOMALY,
            "prediction": TaskType.PREDICTION,
            "predictive_maintenance": TaskType.PREDICTION,
            "vitals_prediction": TaskType.PREDICTION,
            "segmentation": TaskType.SEGMENTATION,
            "tracking": TaskType.TRACKING,
            "quality_inspection": TaskType.CLASSIFICATION,
            "ecg_classification": TaskType.CLASSIFICATION,
            "patient_clustering": TaskType.CLASSIFICATION,
        }

        if task_types:
            self.task_types = [self.task_type_map.get(t, TaskType.CLASSIFICATION) for t in task_types]
        else:
            self.task_types = list(TaskType)

        self._next_task_id = 0

    def generate(self, current_time: float) -> list[TaskInfo]:
        """Generate tasks for the current time step.

        Each device independently generates tasks with Poisson arrival.
        """
        tasks = []
        for device_id in range(self.num_devices):
            num_arrivals = self.rng.poisson(self.arrival_rate)
            for _ in range(num_arrivals):
                task = self._create_task(device_id, current_time)
                tasks.append(task)
        return tasks

    def _create_task(self, device_id: int, current_time: float) -> TaskInfo:
        """Create a single task with random characteristics."""
        task_type = self.rng.choice(self.task_types)

        # Correlate data size and computation with task type
        type_complexity = {
            TaskType.CLASSIFICATION: 0.3,
            TaskType.DETECTION: 0.7,
            TaskType.ANOMALY: 0.4,
            TaskType.PREDICTION: 0.5,
            TaskType.SEGMENTATION: 0.9,
            TaskType.TRACKING: 0.8,
        }
        complexity = type_complexity.get(task_type, 0.5)

        # Log-uniform distribution biased by complexity
        data_size = np.exp(self.rng.uniform(
            np.log(self.data_size_range[0]),
            np.log(self.data_size_range[0] + complexity * (self.data_size_range[1] - self.data_size_range[0]))
        ))

        computation = np.exp(self.rng.uniform(
            np.log(self.computation_range[0]),
            np.log(self.computation_range[0] + complexity * (self.computation_range[1] - self.computation_range[0]))
        ))

        # Deadline inversely related to priority
        deadline = self.rng.uniform(*self.deadline_range)
        priority = (self.deadline_range[1] - deadline) / (self.deadline_range[1] - self.deadline_range[0])

        task = TaskInfo(
            task_id=self._next_task_id,
            source_device=device_id,
            data_size_bytes=data_size,
            computation_cycles=computation,
            deadline_ms=deadline,
            task_type=task_type,
            arrival_time=current_time,
            priority=max(0.1, priority),
            data_partition_id=device_id % 10,  # Simple partition scheme
        )

        self._next_task_id += 1
        return task

    def generate_batch(self, current_time: float, num_tasks: int) -> list[TaskInfo]:
        """Generate a fixed number of tasks (for controlled experiments)."""
        tasks = []
        for _ in range(num_tasks):
            device_id = self.rng.integers(0, self.num_devices)
            tasks.append(self._create_task(device_id, current_time))
        return tasks
