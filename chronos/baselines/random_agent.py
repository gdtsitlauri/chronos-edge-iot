"""Random baseline agent — selects actions uniformly at random."""

from __future__ import annotations

import numpy as np
import torch

from chronos.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Baseline: uniformly random actions."""

    def __init__(self, agent_id: int, config: dict):
        super().__init__(agent_id, config)
        self.num_edge_nodes = config["system"]["num_edge_nodes"]
        self.num_iot_devices = config["system"]["num_iot_devices"]
        self.num_channels = config["system"]["num_channels"]
        self.rng = np.random.default_rng(config["system"].get("seed", 42))

    def select_action(self, observation: dict, deterministic: bool = False) -> dict:
        num_tasks = observation.get("num_active_tasks", 1)
        return {
            "offloading": self.rng.integers(0, self.num_edge_nodes, size=max(num_tasks, 1)),
            "resource_alloc": self.rng.uniform(0.1, 1.0, size=self.num_edge_nodes),
            "power_control": self.rng.uniform(0.1, 1.0, size=self.num_iot_devices),
            "channel_assign": self.rng.integers(0, self.num_channels, size=self.num_iot_devices),
            "fl_participate": np.ones(self.num_edge_nodes, dtype=int),
            "fl_local_steps": np.full(self.num_edge_nodes, 5),
        }

    def update(self, batch: dict = None) -> dict[str, float]:
        return {}

    def get_state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict):
        pass
