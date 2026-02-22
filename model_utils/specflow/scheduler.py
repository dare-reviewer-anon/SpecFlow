# model_utils/specflow/scheduler.py
# SpecFlow timestep/schedule utilities.
#
# Purpose:
#   - Produce the ODE integration time grid {t_k} and dt for Alg.1 Euler solve.
#   - Provide optional "strength" (anchoring / i2i) schedule per hop or per step.
#
# In the SpecFlow paper baseline, integration is performed over t in [0,1]
# using a fixed uniform grid. We keep that as default.
#
# The "strength schedule" is an optional extension (DiffThinker-style i2i anchoring)
# that blends the initial coefficients X(0) towards previous-hop coefficients.
# It is NOT required by the SpecFlow paper, but is useful if you want i2i editing behavior.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union, List

import math
import torch

GridType = Literal["uniform"]
StrengthSchedule = Literal["constant", "linear", "cosine", "sqrt"]


@dataclass
class TimestepSchedule:
    """
    Defines the integration time grid t_k, k=0..T-1.
    """
    steps: int = 8
    grid_type: GridType = "uniform"
    dtype: torch.dtype = torch.float32

    def make(self, device: torch.device) -> torch.Tensor:
        if self.steps <= 0:
            raise ValueError(f"steps must be > 0, got {self.steps}")
        if self.grid_type != "uniform":
            raise ValueError(f"Unsupported grid_type: {self.grid_type}")

        # t_k = k/T for k=0..T-1 in [0,1)
        return torch.arange(self.steps, device=device, dtype=self.dtype) / float(self.steps)

    def dt(self) -> float:
        if self.steps <= 0:
            raise ValueError(f"steps must be > 0, got {self.steps}")
        return 1.0 / float(self.steps)


@dataclass
class StrengthConfig:
    """
    Optional anchoring (i2i) schedule.

    strength in [0,1]:
      - 0   => start from pure noise coeffs
      - 1   => start from previous coeffs (full anchor)
    """
    enabled: bool = False
    base_strength: float = 0.6
    schedule: StrengthSchedule = "constant"

    # You can vary strength by hop index (often stronger in early hops)
    hop_schedule: StrengthSchedule = "constant"
    hop_min: float = 0.4
    hop_max: float = 0.8

    def _shape(self, x: float, schedule: StrengthSchedule) -> float:
        x = float(max(0.0, min(1.0, x)))
        if schedule == "constant":
            return 1.0
        if schedule == "linear":
            return x
        if schedule == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * x)
        if schedule == "sqrt":
            return math.sqrt(x)
        raise ValueError(f"Unknown schedule: {schedule}")

    def strength_for_step(self, step_idx: int, steps: int) -> float:
        """
        Optional per-step shaping within a hop.
        Many users keep this constant; we support shaping anyway.
        """
        if not self.enabled:
            return 0.0
        if steps <= 0:
            raise ValueError("steps must be > 0")
        x = step_idx / float(max(1, steps - 1))  # normalize to [0,1]
        a = self._shape(x, self.schedule)
        s = float(self.base_strength) * a if self.schedule != "constant" else float(self.base_strength)
        return float(max(0.0, min(1.0, s)))

    def strength_for_hop(self, hop_idx: int, hops: int) -> float:
        """
        Per-hop strength (common): allow stronger anchoring early, weaker later.
        """
        if not self.enabled:
            return 0.0
        if hops <= 0:
            raise ValueError("hops must be > 0")
        x = hop_idx / float(max(1, hops - 1))  # normalize to [0,1]
        a = self._shape(x, self.hop_schedule)

        # Map a in [0,1] to [hop_min, hop_max]
        s = float(self.hop_min + a * (self.hop_max - self.hop_min))
        return float(max(0.0, min(1.0, s)))


def make_time_grid(
    steps: int,
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, float]:
    """
    Convenience function: returns (t_grid, dt) for uniform Euler integration.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    sched = TimestepSchedule(steps=steps, dtype=dtype)
    t_grid = sched.make(dev)
    return t_grid, sched.dt()


def strength_schedule_for_hop(
    strength_cfg: StrengthConfig,
    hop_idx: int,
    hops: int,
) -> float:
    """
    Convenience: get anchoring strength for a hop.
    """
    return strength_cfg.strength_for_hop(hop_idx, hops)


def strength_schedule_per_step(
    strength_cfg: StrengthConfig,
    steps: int,
) -> List[float]:
    """
    Convenience: get a list of per-step strengths within a hop.
    """
    if not strength_cfg.enabled:
        return [0.0 for _ in range(steps)]
    return [strength_cfg.strength_for_step(k, steps) for k in range(steps)]

