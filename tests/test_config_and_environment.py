from __future__ import annotations

import numpy as np

from chronos.simulator.environment import EdgeIoTEnvironment
from chronos.utils.config import load_config


def _sample_action(env: EdgeIoTEnvironment, num_tasks: int) -> dict:
    task_count = max(int(num_tasks), 1)
    return {
        "offloading": np.zeros(task_count, dtype=int),
        "resource_alloc": np.full(env.num_edge_nodes, 0.5, dtype=np.float32),
        "power_control": np.full(env.num_iot_devices, 0.5, dtype=np.float32),
        "channel_assign": np.zeros(env.num_iot_devices, dtype=int),
        "fl_participate": np.ones(env.num_edge_nodes, dtype=int),
        "fl_local_steps": np.full(env.num_edge_nodes, 1, dtype=int),
    }


def test_config_inheritance_includes_base_sections():
    cfg = load_config("configs/fast_experiment.yaml")

    # Inherited from default config via _base_.
    assert "edge_node" in cfg
    assert "iot_device" in cfg
    assert "wireless" in cfg
    assert cfg["system"]["num_edge_nodes"] == 3


def test_environment_reset_contains_expected_keys():
    cfg = load_config("configs/fast_experiment.yaml")
    env = EdgeIoTEnvironment(cfg)
    obs = env.reset()

    expected_keys = {
        "node_features",
        "device_features",
        "channel_features",
        "task_features",
        "channel_gains",
        "num_active_tasks",
        "current_time",
        "node_positions",
        "device_positions",
    }
    assert expected_keys.issubset(set(obs.keys()))
    assert obs["node_features"].shape[0] == cfg["system"]["num_edge_nodes"]


def test_environment_step_returns_valid_result():
    cfg = load_config("configs/fast_experiment.yaml")
    env = EdgeIoTEnvironment(cfg)
    obs = env.reset()
    action = _sample_action(env, obs["num_active_tasks"])

    result = env.step(action)

    assert isinstance(result.rewards, dict)
    assert "combined" in result.rewards
    assert isinstance(result.constraint_violations, dict)
    assert "energy_budget" in result.constraint_violations
    assert isinstance(result.next_state, dict)


def test_environment_done_when_max_steps_reached():
    cfg = load_config("configs/fast_experiment.yaml")
    cfg["training"]["max_steps_per_episode"] = 1

    env = EdgeIoTEnvironment(cfg)
    obs = env.reset()
    action = _sample_action(env, obs["num_active_tasks"])

    result = env.step(action)
    assert result.done is True
