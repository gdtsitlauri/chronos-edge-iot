"""Wireless channel model with Rayleigh fading, path loss, and shadowing."""

from __future__ import annotations

import numpy as np
from chronos.core.types import ChannelState


class WirelessChannel:
    """Models time-varying wireless channels between IoT devices and edge nodes.

    Implements: large-scale path loss + log-normal shadowing + Rayleigh small-scale fading.
    Channel gain: |h_{ij}^c(t)|^2 = PL(d_{ij}) * Xi_{ij} * |g_{ij}^c(t)|^2
    where PL is path loss, Xi is shadowing, g is Rayleigh fading.
    """

    def __init__(
        self,
        num_devices: int,
        num_nodes: int,
        num_channels: int,
        bandwidth_hz: float = 20e6,
        path_loss_exponent: float = 3.5,
        shadow_fading_std_db: float = 8.0,
        noise_power_dbm: float = -174.0,
        carrier_frequency_ghz: float = 3.5,
        coherence_time_ms: float = 10.0,
        rng: np.random.Generator | None = None,
    ):
        self.num_devices = num_devices
        self.num_nodes = num_nodes
        self.num_channels = num_channels
        self.bandwidth_per_channel = bandwidth_hz / num_channels
        self.path_loss_exponent = path_loss_exponent
        self.shadow_fading_std_db = shadow_fading_std_db
        self.carrier_frequency_ghz = carrier_frequency_ghz
        self.coherence_time_ms = coherence_time_ms
        self.rng = rng or np.random.default_rng(42)

        # Noise power per channel (W)
        noise_power_dbm_per_channel = noise_power_dbm + 10 * np.log10(self.bandwidth_per_channel)
        self.noise_power_w = 10 ** ((noise_power_dbm_per_channel - 30) / 10)

        # Reference distance and wavelength
        self.d0 = 1.0  # Reference distance (m)
        self.wavelength = 3e8 / (carrier_frequency_ghz * 1e9)

        # Channel state: gains[c][i][j] = |h_{ij}^c|^2
        self.channel_gains = np.zeros((num_channels, num_devices, num_nodes))
        self.shadow_fading = np.zeros((num_devices, num_nodes))
        self._time_since_update = 0.0

    def initialize(self, device_positions: np.ndarray, node_positions: np.ndarray):
        """Initialize channel gains based on positions.

        Args:
            device_positions: (D, 2) array of device coordinates
            node_positions: (N, 2) array of edge node coordinates
        """
        self.device_positions = device_positions.copy()
        self.node_positions = node_positions.copy()

        # Compute distances (D, N)
        self.distances = np.linalg.norm(
            device_positions[:, None, :] - node_positions[None, :, :], axis=2
        )
        self.distances = np.maximum(self.distances, self.d0)

        # Log-normal shadowing (changes slowly)
        self.shadow_fading = self.rng.normal(
            0, self.shadow_fading_std_db, size=(self.num_devices, self.num_nodes)
        )

        self._update_fading()

    def _compute_path_loss(self) -> np.ndarray:
        """Free-space path loss + distance-dependent loss. Returns linear scale (D, N)."""
        # Friis reference at d0
        pl_d0_db = 20 * np.log10(4 * np.pi * self.d0 / self.wavelength)
        # Distance-dependent
        pl_db = pl_d0_db + 10 * self.path_loss_exponent * np.log10(self.distances / self.d0)
        # Add shadowing
        pl_db += self.shadow_fading
        return 10 ** (-pl_db / 10)

    def _update_fading(self):
        """Generate Rayleigh fading coefficients for all channels."""
        path_loss_linear = self._compute_path_loss()  # (D, N)

        for c in range(self.num_channels):
            # Rayleigh: |h|^2 ~ Exponential(1)
            rayleigh_sq = self.rng.exponential(1.0, size=(self.num_devices, self.num_nodes))
            self.channel_gains[c] = path_loss_linear * rayleigh_sq

    def step(self, dt_ms: float, device_positions: np.ndarray | None = None):
        """Advance channel state by dt_ms milliseconds.

        If dt >= coherence_time, regenerate small-scale fading.
        Optionally update device positions (mobility).
        """
        if device_positions is not None:
            self.device_positions = device_positions.copy()
            self.distances = np.linalg.norm(
                self.device_positions[:, None, :] - self.node_positions[None, :, :], axis=2
            )
            self.distances = np.maximum(self.distances, self.d0)

        self._time_since_update += dt_ms
        if self._time_since_update >= self.coherence_time_ms:
            self._update_fading()
            # Slowly evolve shadowing (correlated)
            innovation = self.rng.normal(0, 1, size=self.shadow_fading.shape)
            rho = np.exp(-self._time_since_update / (10 * self.coherence_time_ms))
            self.shadow_fading = rho * self.shadow_fading + np.sqrt(1 - rho**2) * self.shadow_fading_std_db * innovation
            self._time_since_update = 0.0

    def get_rate(
        self,
        device_idx: int,
        node_idx: int,
        channel_idx: int,
        power_w: float,
        interference_w: float = 0.0,
    ) -> float:
        """Compute achievable rate R_{ij}^c in bits/second."""
        gain = self.channel_gains[channel_idx, device_idx, node_idx]
        signal = power_w * gain
        sinr = signal / (self.noise_power_w + interference_w)
        return self.bandwidth_per_channel * np.log2(1 + max(sinr, 1e-10))

    def compute_interference(
        self,
        channel_idx: int,
        target_device: int,
        target_node: int,
        power_allocation: np.ndarray,
        channel_assignment: np.ndarray,
    ) -> float:
        """Compute interference at (target_node) on channel c from other devices.

        Args:
            power_allocation: (D,) power in watts per device on this channel
            channel_assignment: (D,) binary, which devices use this channel
        """
        interference = 0.0
        for d in range(self.num_devices):
            if d == target_device or channel_assignment[d] == 0:
                continue
            gain = self.channel_gains[channel_idx, d, target_node]
            interference += power_allocation[d] * gain
        return interference

    def get_transmission_time(
        self, device_idx: int, node_idx: int, channel_idx: int,
        data_size_bytes: float, power_w: float, interference_w: float = 0.0,
    ) -> float:
        """Compute transmission time in seconds."""
        rate = self.get_rate(device_idx, node_idx, channel_idx, power_w, interference_w)
        if rate <= 0:
            return float('inf')
        return (data_size_bytes * 8) / rate

    def get_transmission_energy(
        self, device_idx: int, node_idx: int, channel_idx: int,
        data_size_bytes: float, power_w: float, interference_w: float = 0.0,
    ) -> float:
        """Compute transmission energy in Joules: E = P * t."""
        tx_time = self.get_transmission_time(
            device_idx, node_idx, channel_idx, data_size_bytes, power_w, interference_w
        )
        return power_w * tx_time if tx_time < float('inf') else float('inf')

    def get_channel_states(self) -> list[ChannelState]:
        """Return ChannelState for each channel."""
        states = []
        for c in range(self.num_channels):
            states.append(ChannelState(
                channel_id=c,
                bandwidth_hz=self.bandwidth_per_channel,
                channel_gains=self.channel_gains[c].copy(),
                noise_power_w=self.noise_power_w,
            ))
        return states

    def get_all_gains_tensor(self) -> np.ndarray:
        """Return (C, D, N) channel gains array."""
        return self.channel_gains.copy()
