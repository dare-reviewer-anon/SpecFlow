# model_utils/specflow/visualizer.py
# Debug / visualization utilities for SpecFlow.
#
# Goals:
#   - Provide simple, dependency-light visualization helpers for:
#       - workspace images (pixel or latent decoded)
#       - block-DCT coefficient magnitudes
#       - masks M(t)
#       - hop / step trajectories
#
# This file intentionally avoids heavy dependencies (matplotlib/torchvision).
# It returns tensors that you can save with your own pipeline (e.g., torchvision save_image).
# If you want PNGs without extra code, you can optionally add torchvision usage in your scripts.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .cosine_proj import block_dct
from .masking import MaskConfig, block_mask, expand_block_mask_to_grid


def _to_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize tensor to [0,1] per-sample for visualization.
    """
    B = x.shape[0]
    x_flat = x.view(B, -1)
    mn = x_flat.min(dim=1)[0].view(B, 1, 1, 1)
    mx = x_flat.max(dim=1)[0].view(B, 1, 1, 1)
    return (x - mn) / (mx - mn).clamp_min(eps)


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x
    raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")


def to_rgb(x: torch.Tensor) -> torch.Tensor:
    """
    Convert (B,C,H,W) to 3-channel RGB-like tensor for saving.
    - If C==1: replicate
    - If C>=3: take first 3
    """
    x = _ensure_bchw(x)
    B, C, H, W = x.shape
    if C == 1:
        return x.repeat(1, 3, 1, 1)
    if C >= 3:
        return x[:, :3]
    # C==2: pad third channel with zeros
    pad = torch.zeros((B, 1, H, W), device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


@dataclass
class VisualizerConfig:
    block_size: int = 8
    eps: float = 1e-8
    coeff_log: bool = True  # log(1+|X|) for coeff visualization


class SpecFlowVisualizer:
    """
    Utility class to create debug tensors for:
      - workspace snapshots
      - coeff magnitude maps
      - masks (expanded to full grid)
    """

    def __init__(self, cfg: VisualizerConfig):
        self.cfg = cfg

    @torch.no_grad()
    def workspace_preview(self, v: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Prepare workspace image tensor for saving/viewing.
        Returns (B,3,H,W) in [0,1] if normalize=True.
        """
        v = _ensure_bchw(v).detach()
        if normalize:
            v = _to_01(v, eps=self.cfg.eps)
        return to_rgb(v)

    @torch.no_grad()
    def coeff_magnitude_map(self, v: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Compute block-DCT coefficients and return |X| magnitude map (B,1,H,W) for visualization.
        """
        v = _ensure_bchw(v).detach()
        X = block_dct(v, self.cfg.block_size)
        mag = X.abs().mean(dim=1, keepdim=True)  # average over channels -> (B,1,H,W)
        if self.cfg.coeff_log:
            mag = torch.log1p(mag)
        if normalize:
            mag = _to_01(mag, eps=self.cfg.eps)
        return to_rgb(mag)

    @torch.no_grad()
    def mask_map(self, H: int, W: int, mask_cfg: MaskConfig, t: float) -> torch.Tensor:
        """
        Expand block mask M(t) to full (1,1,H,W) and return RGB-like (1,3,H,W) in [0,1].
        """
        Mb = block_mask(t, mask_cfg, device=torch.device("cpu"), dtype=torch.float32)
        grid = expand_block_mask_to_grid(Mb, H=H, W=W)  # (1,1,H,W)
        return to_rgb(grid)

    @torch.no_grad()
    def trajectory_stack(self, imgs: Sequence[torch.Tensor], normalize: bool = True) -> torch.Tensor:
        """
        Given a list of (B,C,H,W), stack them along batch dimension for easy saving.
        Returns (B*T, 3, H, W).
        """
        if len(imgs) == 0:
            raise ValueError("imgs must be non-empty")
        prepped = [self.workspace_preview(x, normalize=normalize) for x in imgs]
        return torch.cat(prepped, dim=0)


@torch.no_grad()
def make_debug_bundle(
    *,
    visual_states: List[torch.Tensor],
    mask_cfg: Optional[MaskConfig] = None,
    t_samples: Optional[Sequence[float]] = None,
    block_size: int = 8,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to generate a debug bundle of tensors from a SpecFlow run.

    Args:
      visual_states: [v0, v1, ...] each (B,C,H,W)
      mask_cfg: if provided, include mask maps at t_samples
      t_samples: list of t values to visualize masks for
      block_size: for coeff magnitude maps

    Returns:
      dict of tensors:
        - "workspace": stacked workspace previews
        - "coeff_mag": stacked coeff magnitude previews
        - "masks": stacked masks (if requested)
    """
    if len(visual_states) == 0:
        raise ValueError("visual_states must be non-empty")

    v0 = visual_states[0]
    B, C, H, W = v0.shape

    vis = SpecFlowVisualizer(VisualizerConfig(block_size=block_size))
    out: Dict[str, torch.Tensor] = {}

    out["workspace"] = vis.trajectory_stack(visual_states, normalize=True)       # (B*T,3,H,W)
    out["coeff_mag"] = torch.cat([vis.coeff_magnitude_map(v, normalize=True) for v in visual_states], dim=0)

    if mask_cfg is not None and t_samples is not None and len(t_samples) > 0:
        masks = [vis.mask_map(H, W, mask_cfg, float(t)) for t in t_samples]
        out["masks"] = torch.cat(masks, dim=0)  # (len(t),3,H,W)

    return out