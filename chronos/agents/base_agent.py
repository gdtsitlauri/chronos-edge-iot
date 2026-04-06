"""Base agent interface for all algorithms (CHRONOS and baselines)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseAgent(ABC):
    """Abstract base class for edge computing agents."""

    def __init__(self, agent_id: int, config: dict):
        self.agent_id = agent_id
        self.config = config

    @abstractmethod
    def select_action(self, observation: dict, deterministic: bool = False) -> dict:
        """Select an action given an observation.

        Returns:
            action dict with keys: offloading, resource_alloc, power_control,
                                   channel_assign, fl_participate, fl_local_steps
        """
        ...

    @abstractmethod
    def update(self, batch: dict) -> dict[str, float]:
        """Update agent from a batch of experience.

        Returns:
            dict of loss/metric values
        """
        ...

    @abstractmethod
    def get_state_dict(self) -> dict:
        """Return serializable state for checkpointing."""
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        """Load agent state from checkpoint."""
        ...
