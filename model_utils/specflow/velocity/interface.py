# model_utils/specflow/velocity/interface.py
# VelocityModel protocol / base class for SpecFlow.
#
# SpecFlow core (Alg.1/2) only assumes a callable:
#   u_theta(X_masked, t, cond) -> velocity (same shape as X_masked)
#
# This file defines:
#   - A lightweight Protocol for typing (works with any callable module)
#   - An nn.Module base class you can inherit from
#   - Small utilities for handling t formatting and cond packing
#
# Design goals:
#   - Keep SpecFlow controller/trainer completely backbone-agnostic
#   - Let UNet/DiT/VLM-adapter implementations live in velocity/*.py
#
# Conditioning:
#   - cond is Optional[Any]. For CFG, uncond is typically None.
#   - You can pass dicts, tuples, or custom objects. The velocity model decides how to use it.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union, runtime_checkable

import torch
import torch.nn as nn


TLike = Union[float, torch.Tensor]


@runtime_checkable
class VelocityModel(Protocol):
    """
    Protocol for a SpecFlow velocity model u_theta.
    Implementations can be nn.Module or any callable.
    """

    def __call__(self, X_masked: torch.Tensor, t: TLike, cond: Optional[Any]) -> torch.Tensor:
        """
        Args:
            X_masked: (B,C,H,W) coefficient tensor (already masked by M(t))
            t: scalar float in [0,1] or tensor shape (B,) or broadcastable
            cond: conditioning object (None for unconditional branch)
        Returns:
            velocity: (B,C,H,W) tensor
        """
        ...


@dataclass
class VelocityIO:
    """
    Optional structured inputs/outputs if you want to standardize internals later.
    You can ignore this and just implement __call__ directly.
    """
    X: torch.Tensor
    t: torch.Tensor  # (B,) float tensor
    cond: Optional[Any] = None


def normalize_t(t: TLike, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert t into a tensor of shape (B,) on the right device/dtype.

    Accepts:
      - float scalar
      - tensor scalar ()
      - tensor (B,)
      - tensor broadcastable to (B,) (we try to squeeze)
    """
    if isinstance(t, float):
        return torch.full((batch,), float(t), device=device, dtype=dtype)

    if not torch.is_tensor(t):
        raise TypeError(f"t must be float or torch.Tensor, got {type(t)}")

    tt = t.to(device=device, dtype=dtype)

    if tt.dim() == 0:
        return tt.expand(batch)
    if tt.dim() == 1:
        if tt.numel() == 1:
            return tt.expand(batch)
        if tt.numel() == batch:
            return tt
        raise ValueError(f"t has shape {tuple(tt.shape)} but batch={batch}")
    # If higher-dim, try squeeze to 1D
    tt2 = tt.reshape(-1)
    if tt2.numel() == 1:
        return tt2.expand(batch)
    if tt2.numel() == batch:
        return tt2
    raise ValueError(f"Could not normalize t with shape {tuple(t.shape)} to (B,) with batch={batch}")


class BaseVelocityModel(nn.Module):
    """
    Minimal nn.Module base class for velocity models.

    Subclasses should implement forward(X_masked, t, cond)->velocity.

    This base class:
      - normalizes t to (B,) float tensor (handy for embeddings)
      - provides a common place to validate shapes
    """

    def __init__(self):
        super().__init__()

    def forward(self, X_masked: torch.Tensor, t: TLike, cond: Optional[Any]) -> torch.Tensor:  # noqa: D401
        """
        Override in subclasses.
        """
        raise NotImplementedError

    def __call__(self, X_masked: torch.Tensor, t: TLike, cond: Optional[Any]) -> torch.Tensor:
        return super().__call__(X_masked, t, cond)  # nn.Module.__call__ -> forward

    @staticmethod
    def validate_io(X_masked: torch.Tensor, out: torch.Tensor) -> None:
        if X_masked.dim() != 4:
            raise ValueError(f"X_masked must be (B,C,H,W). Got {tuple(X_masked.shape)}")
        if out.shape != X_masked.shape:
            raise ValueError(f"Velocity output shape must match input. Got {tuple(out.shape)} vs {tuple(X_masked.shape)}")

    @staticmethod
    def t_to_batch(t: TLike, X_masked: torch.Tensor) -> torch.Tensor:
        B = X_masked.shape[0]
        return normalize_t(t, B, device=X_masked.device, dtype=X_masked.dtype)


class NullConditioning:
    """
    Convenience object for "unconditional" if you dislike using None.
    By default SpecFlow uses None for uncond; this is optional.
    """
    pass