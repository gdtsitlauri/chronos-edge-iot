"""Main edge-IoT simulation environment (Gym-style interface)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from chronos.core.types import (
    TaskInfo, TaskStatus, StepResult, NodeType,
    EdgeNodeState, IoTDeviceState, ChannelState,
)
from chronos.simulator.edge_node import EdgeNode
from chronos.simulator.iot_device import IoTDevice
from chronos.simulator.wireless_channel import WirelessChannel
from chronos.simulator.task import TaskGenerator


class EdgeIoTEnvironment:
    """Full edge-IoT simulation environment.

    Manages edge nodes, IoT devices, wireless channels, task lifecycle,
    and computes multi-objective rewards (accuracy proxy, latency, energy, comm cost).
    """

    def __init__(self, config: dict):
        self.config = config
        sys_cfg = config["system"]
        self.num_edge_nodes = sys_cfg["num_edge_nodes"]
        self.num_iot_devices = sys_cfg["num_iot_devices"]
        self.num_channels = sys_cfg["num_channels"]
        self.area_size = sys_cfg.get("area_size_m", 500.0)

        self.rng = np.random.default_rng(sys_cfg.get("seed", 42))
        self.current_time = 0.0
        self.step_count = 0
        self.dt_ms = 10.0  # Time step duration

        # Build components
        self._build_edge_nodes()
        self._build_iot_devices()
        self._build_wireless()
        self._build_task_generator()

        # Task tracking
        self.all_tasks: list[TaskInfo] = []
        self.active_tasks: list[TaskInfo] = []

        # Step metrics accumulators
        self._episode_metrics = {
            "total_latency_ms": 0.0,
            "total_energy_j": 0.0,
            "total_comm_bits": 0.0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "deadline_violations": 0,
        }

    def _build_edge_nodes(self):
        cfg = self.config["edge_node"]
        freq_range = cfg["cpu_frequency_ghz"]
        mem_range = cfg["memory_gb"]

        self.edge_nodes: list[EdgeNode] = []
        for i in range(self.num_edge_nodes):
            pos = self.rng.uniform(0, self.area_size, size=2)
            node = EdgeNode(
                node_id=i,
                cpu_frequency_ghz=self.rng.uniform(*freq_range),
                memory_gb=self.rng.uniform(*mem_range),
                energy_budget_j=cfg["energy_budget_j"],
                effective_capacitance=cfg.get("effective_capacitance", 1e-28),
                queue_capacity=cfg.get("queue_capacity", 50),
                position=pos,
            )
            self.edge_nodes.append(node)

    def _build_iot_devices(self):
        cfg = self.config["iot_device"]
        freq_range = cfg["cpu_frequency_ghz"]

        self.iot_devices: list[IoTDevice] = []
        for i in range(self.num_iot_devices):
            pos = self.rng.uniform(0, self.area_size, size=2)
            # Random walk velocity for mobility
            speed = self.rng.uniform(0, 2.0)  # m/s
            angle = self.rng.uniform(0, 2 * np.pi)
            vel = np.array([speed * np.cos(angle), speed * np.sin(angle)])

            device = IoTDevice(
                device_id=i,
                cpu_frequency_ghz=self.rng.uniform(*freq_range),
                max_power_dbm=cfg.get("max_power_dbm", 23.0),
                energy_budget_j=cfg.get("energy_budget_j", 1000.0),
                position=pos,
                velocity=vel,
            )
            self.iot_devices.append(device)

    def _build_wireless(self):
        wcfg = self.config["wireless"]
        sys_cfg = self.config["system"]
        self.wireless = WirelessChannel(
            num_devices=self.num_iot_devices,
            num_nodes=self.num_edge_nodes,
            num_channels=self.num_channels,
            bandwidth_hz=sys_cfg.get("bandwidth_mhz", 20.0) * 1e6,
            path_loss_exponent=wcfg["path_loss_exponent"],
            shadow_fading_std_db=wcfg["shadow_fading_std_db"],
            noise_power_dbm=sys_cfg.get("noise_power_dbm", -174.0),
            carrier_frequency_ghz=wcfg["carrier_frequency_ghz"],
            coherence_time_ms=wcfg["coherence_time_ms"],
            rng=self.rng,
        )

    def _build_task_generator(self):
        tcfg = self.config["task"]
        self.task_gen = TaskGenerator(
            num_devices=self.num_iot_devices,
            arrival_rate=self.config["iot_device"].get("task_arrival_rate", 0.5),
            data_size_range_mb=tuple(tcfg["data_size_mb"]),
            computation_range_mcycles=tuple(tcfg["computation_mcycles"]),
            deadline_range_ms=tuple(tcfg["deadline_ms"]),
            task_types=tcfg.get("types"),
            rng=self.rng,
        )

    def reset(self) -> dict:
        """Reset environment for a new episode. Returns initial observation."""
        self.current_time = 0.0
        self.step_count = 0
        self.all_tasks.clear()
        self.active_tasks.clear()

        for node in self.edge_nodes:
            node.reset()
        for device in self.iot_devices:
            device.reset()

        # Initialize wireless channels
        device_pos = np.array([d.position for d in self.iot_devices])
        node_pos = np.array([n.position for n in self.edge_nodes])
        self.wireless.initialize(device_pos, node_pos)

        # Generate initial tasks
        new_tasks = self.task_gen.generate(self.current_time)
        for task in new_tasks:
            self.iot_devices[task.source_device].receive_task(task)
            self.all_tasks.append(task)
            self.active_tasks.append(task)

        self._episode_metrics = {
            "total_latency_ms": 0.0,
            "total_energy_j": 0.0,
            "total_comm_bits": 0.0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "deadline_violations": 0,
        }

        return self._get_observation()

    def step(self, actions: dict) -> StepResult:
        """Execute one environment step given agent actions.

        Args:
            actions: Dict with keys:
                - 'offloading': (K,) int array — target node per task (-1=local, 0..N-1=edge)
                - 'resource_alloc': (N,) float array — resource fraction per node
                - 'power_control': (D,) float array — transmit power per device [0,1] normalized
                - 'channel_assign': (D,) int array — channel index per device
                - 'fl_participate': (N,) binary — which nodes participate in FL
                - 'fl_local_steps': (N,) int — local SGD steps per node

        Returns:
            StepResult with next_state, rewards, done, info, constraint_violations.
        """
        self.step_count += 1
        self.current_time += self.dt_ms
        step_energy = 0.0
        step_latency = 0.0
        step_comm_bits = 0.0
        completed_tasks = []
        constraint_violations = {}

        # --- 1. Process task offloading decisions ---
        offloading = actions.get("offloading", np.array([]))
        power_norm = actions.get("power_control", np.zeros(self.num_iot_devices))
        channel_assign = actions.get("channel_assign", np.zeros(self.num_iot_devices, dtype=int))

        tasks_to_process = []
        for device in self.iot_devices:
            tasks_to_process.extend(device.pending_tasks)
            device.pending_tasks.clear()

        for idx, task in enumerate(tasks_to_process):
            if idx >= len(offloading):
                # Default: offload to nearest node
                target = self._find_nearest_node(task.source_device)
            else:
                target = int(offloading[idx])

            if target < 0 or target >= self.num_edge_nodes:
                # Local execution
                device = self.iot_devices[task.source_device]
                completed, energy = device.execute_local(task, self.dt_ms)
                step_energy += energy
                if completed:
                    latency = self.current_time - task.arrival_time
                    step_latency += latency
                    completed_tasks.append(task)
            else:
                # Offload to edge node
                device = self.iot_devices[task.source_device]
                d_idx = task.source_device
                c_idx = int(channel_assign[d_idx] % self.num_channels) if d_idx < len(channel_assign) else 0
                power_w = float(power_norm[d_idx]) * device.max_power_w if d_idx < len(power_norm) else device.max_power_w * 0.5

                # Compute transmission time and energy
                tx_time_s = self.wireless.get_transmission_time(
                    d_idx, target, c_idx, task.data_size_bytes, power_w
                )
                tx_energy = power_w * tx_time_s if tx_time_s < float('inf') else 0.0
                tx_bits = task.data_size_bytes * 8

                device.consume_tx_energy(tx_energy)
                step_energy += tx_energy
                step_comm_bits += tx_bits

                # Enqueue task at edge node
                node = self.edge_nodes[target]
                if node.can_accept_task():
                    node.enqueue_task(task)
                else:
                    task.status = TaskStatus.FAILED
                    self._episode_metrics["tasks_failed"] += 1

        # --- 2. Execute tasks at edge nodes ---
        resource_alloc = actions.get("resource_alloc", None)
        for node in self.edge_nodes:
            if resource_alloc is not None and node.node_id < len(resource_alloc):
                # Distribute resources evenly among active tasks (simplified)
                n_active = max(len(node.active_tasks), 1)
                alloc = {t.task_id: resource_alloc[node.node_id] / n_active
                         for t, _ in node.active_tasks}
                node.allocate_resources(alloc)

            node_completed = node.step(self.dt_ms, self.current_time)
            step_energy += node.get_step_energy()

            for task in node_completed:
                latency = self.current_time - task.arrival_time
                step_latency += latency
                completed_tasks.append(task)
                if task.status == TaskStatus.DEADLINE_MISSED:
                    self._episode_metrics["deadline_violations"] += 1

        # --- 3. Update wireless channels and device positions ---
        for device in self.iot_devices:
            device.update_position(self.dt_ms, self.area_size)

        device_pos = np.array([d.position for d in self.iot_devices])
        self.wireless.step(self.dt_ms, device_pos)

        # --- 4. Generate new tasks ---
        new_tasks = self.task_gen.generate(self.current_time)
        for task in new_tasks:
            self.iot_devices[task.source_device].receive_task(task)
            self.all_tasks.append(task)
            self.active_tasks.append(task)

        # Remove completed/failed tasks from active list
        self.active_tasks = [
            t for t in self.active_tasks
            if t.status in (TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.EXECUTING)
        ]

        # --- 5. Compute rewards ---
        num_completed = len([t for t in completed_tasks if t.status == TaskStatus.COMPLETED])
        num_deadline_missed = len([t for t in completed_tasks if t.status == TaskStatus.DEADLINE_MISSED])

        self._episode_metrics["tasks_completed"] += num_completed
        self._episode_metrics["total_latency_ms"] += step_latency
        self._episode_metrics["total_energy_j"] += step_energy
        self._episode_metrics["total_comm_bits"] += step_comm_bits

        rewards = self._compute_rewards(
            num_completed, num_deadline_missed,
            step_latency, step_energy, step_comm_bits, len(tasks_to_process)
        )

        # --- 6. Check constraints ---
        constraint_violations = self._check_constraints()

        # --- 7. Check done condition ---
        max_steps = self.config.get("training", {}).get("max_steps_per_episode", 200)
        done = self.step_count >= max_steps

        info = {
            "step": self.step_count,
            "time_ms": self.current_time,
            "tasks_completed_step": num_completed,
            "tasks_failed_step": num_deadline_missed,
            "active_tasks": len(self.active_tasks),
            "step_energy_j": step_energy,
            "step_latency_ms": step_latency,
            "step_comm_bits": step_comm_bits,
            "episode_metrics": self._episode_metrics.copy(),
        }

        return StepResult(
            next_state=self._get_observation(),
            rewards=rewards,
            done=done,
            info=info,
            constraint_violations=constraint_violations,
        )

    def _compute_rewards(
        self, completed: int, deadline_missed: int,
        latency: float, energy: float, comm_bits: float, total_tasks: int,
    ) -> dict[str, float]:
        """Compute multi-objective rewards (higher is better for all)."""
        # R1: Task completion rate (proxy for accuracy)
        r_accuracy = completed / max(total_tasks, 1)

        # R2: Latency (negative, lower latency = higher reward)
        avg_latency = latency / max(completed + deadline_missed, 1)
        r_latency = -avg_latency / 1000.0  # Normalize to seconds

        # R3: Energy efficiency (negative energy)
        r_energy = -energy / 100.0  # Normalize

        # R4: Communication efficiency (negative comm cost)
        r_comm = -comm_bits / 1e6  # Normalize to Mbits

        # Penalty for deadline violations
        deadline_penalty = -deadline_missed * 0.5

        return {
            "accuracy": r_accuracy,
            "latency": r_latency + deadline_penalty,
            "energy": r_energy,
            "communication": r_comm,
            "combined": 0.4 * r_accuracy + 0.3 * r_latency + 0.2 * r_energy + 0.1 * r_comm + deadline_penalty,
        }

    def _check_constraints(self) -> dict[str, float]:
        """Check constraint violations. Returns {name: violation_amount} (0 = satisfied)."""
        violations = {}

        # C2: Energy budget
        max_energy_violation = 0.0
        for device in self.iot_devices:
            excess = device.energy_consumed_j - device.energy_budget_j
            max_energy_violation = max(max_energy_violation, excess)
        violations["energy_budget"] = max(0.0, max_energy_violation)

        # C3: Compute capacity
        max_compute_violation = 0.0
        for node in self.edge_nodes:
            excess = node.cpu_utilization - 1.0
            max_compute_violation = max(max_compute_violation, excess)
        violations["compute_capacity"] = max(0.0, max_compute_violation)

        # C5: Power constraint
        max_power_violation = 0.0
        for device in self.iot_devices:
            excess = device.current_power_w - device.max_power_w
            max_power_violation = max(max_power_violation, excess)
        violations["power"] = max(0.0, max_power_violation)

        return violations

    def _find_nearest_node(self, device_id: int) -> int:
        """Find nearest edge node to a device."""
        device_pos = self.iot_devices[device_id].position
        min_dist = float('inf')
        nearest = 0
        for node in self.edge_nodes:
            dist = np.linalg.norm(device_pos - node.position)
            if dist < min_dist:
                min_dist = dist
                nearest = node.node_id
        return nearest

    def _get_observation(self) -> dict:
        """Build full observation dict for agents."""
        node_states = [n.get_state() for n in self.edge_nodes]
        device_states = [d.get_state() for d in self.iot_devices]
        channel_states = self.wireless.get_channel_states()

        # Stack into tensors
        node_features = torch.stack([s.to_tensor() for s in node_states])
        device_features = torch.stack([s.to_tensor() for s in device_states])
        channel_features = torch.stack([
            s.to_tensor(self.num_iot_devices, self.num_edge_nodes) for s in channel_states
        ])

        # Task features
        task_features_list = []
        for task in self.active_tasks[:100]:  # Cap at 100 tasks
            task_features_list.append(torch.tensor([
                task.data_size_bytes / 1e6,
                task.computation_cycles / 1e9,
                task.deadline_ms / 1000.0,
                task.priority,
                task.task_type.value / 5.0,
                (self.current_time - task.arrival_time) / task.deadline_ms,
                task.source_device / self.num_iot_devices,
            ], dtype=torch.float32))

        if task_features_list:
            task_features = torch.stack(task_features_list)
        else:
            task_features = torch.zeros(1, 7)

        return {
            "node_features": node_features,         # (N, node_feat_dim)
            "device_features": device_features,      # (D, device_feat_dim)
            "channel_features": channel_features,    # (C, channel_feat_dim)
            "task_features": task_features,           # (K, task_feat_dim)
            "channel_gains": torch.tensor(self.wireless.get_all_gains_tensor(), dtype=torch.float32),
            "num_active_tasks": len(self.active_tasks),
            "current_time": self.current_time,
            "node_positions": torch.tensor(np.array([n.position for n in self.edge_nodes]), dtype=torch.float32),
            "device_positions": torch.tensor(np.array([d.position for d in self.iot_devices]), dtype=torch.float32),
        }

    @property
    def observation_dims(self) -> dict[str, int]:
        """Return dimensionality of each observation component."""
        return {
            "node_feat_dim": 10,
            "device_feat_dim": 8,
            "channel_feat_dim": 6,
            "task_feat_dim": 7,
            "num_edge_nodes": self.num_edge_nodes,
            "num_iot_devices": self.num_iot_devices,
            "num_channels": self.num_channels,
        }

    def get_system_summary(self) -> dict:
        """Get summary statistics of the current system state."""
        return {
            "avg_node_utilization": np.mean([n.cpu_utilization for n in self.edge_nodes]),
            "avg_node_queue": np.mean([n.queue_length for n in self.edge_nodes]),
            "avg_device_energy_remaining": np.mean([d.remaining_energy / d.energy_budget_j for d in self.iot_devices]),
            "total_active_tasks": len(self.active_tasks),
            "episode_metrics": self._episode_metrics.copy(),
        }
