# model_utils/specflow/velocity/unet_velocity.py
# UNet-style velocity model for SpecFlow (coefficient-space).
#
# Implements a compact UNet that maps X_masked (B,C,H,W) -> velocity (B,C,H,W),
# conditioned on time t and optional cond embedding via FiLM (scale/shift).
#
# Conditioning convention (recommended):
#   cond can be None or a dict containing one of:
#     - cond["emb"]: Tensor (B, D_cond)
#     - cond["text_emb"]: Tensor (B, D_cond)
#     - cond["cond_emb"]: Tensor (B, D_cond)
#   If no embedding is provided, conditioning is treated as zero (unconditional).
#
# Time embedding:
#   - sinusoidal embedding of t in [0,1], projected through an MLP
#
# FiLM:
#   - Each residual block uses (scale, shift) produced from (time_emb + cond_emb)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interface import BaseVelocityModel, TLike


# -------------------------
# Embeddings
# -------------------------

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Sinusoidal timestep embedding. t: (B,) float tensor.
    Returns: (B, dim)
    """
    # scale t for richer frequencies
    t_scaled = t * 1000.0
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=t.dtype) / half)
    args = t_scaled[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeMLP(nn.Module):
    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


# -------------------------
# Blocks
# -------------------------

class GroupNorm32(nn.GroupNorm):
    """
    GroupNorm that picks a safe group count (<= channels).
    """
    def __init__(self, channels: int, num_groups: int = 32):
        g = min(num_groups, channels)
        super().__init__(g, channels)


class FiLMResBlock(nn.Module):
    """
    Residual block with FiLM conditioning.
      h = GN(x) -> SiLU -> Conv
      (scale,shift) from cond -> apply to h: h*(1+scale)+shift
      h = GN(h) -> SiLU -> Conv
      out = h + skip(x)
    """
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.gn1 = GroupNorm32(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.cond_to_scale_shift = nn.Linear(cond_dim, 2 * out_ch)
        nn.init.zeros_(self.cond_to_scale_shift.weight)
        nn.init.zeros_(self.cond_to_scale_shift.bias)

        self.gn2 = GroupNorm32(out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

        # stabilize: start close to identity
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.gn1(x)
        h = F.silu(h)
        h = self.conv1(h)

        ss = self.cond_to_scale_shift(cond)  # (B, 2*out_ch)
        scale, shift = ss.chunk(2, dim=-1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.gn2(h)
        h = F.silu(h)
        h = self.drop(h)
        h = self.conv2(h)

        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# -------------------------
# Model
# -------------------------

@dataclass
class UNetVelocityConfig:
    in_channels: int
    base_channels: int = 128
    channel_mults: tuple = (1, 2, 4, 4)  # per resolution level
    num_res_blocks: int = 2
    dropout: float = 0.0

    # conditioning
    time_embed_dim: int = 256
    cond_embed_in_dim: int = 512
    cond_dim: int = 512  # internal cond vector dim after projections

    # output init
    zero_out: bool = True


class UNetVelocityModel(BaseVelocityModel):
    """
    UNet velocity model u_theta for SpecFlow.
    """

    def __init__(self, cfg: UNetVelocityConfig):
        super().__init__()
        self.cfg = cfg

        # input conv
        self.in_conv = nn.Conv2d(cfg.in_channels, cfg.base_channels, kernel_size=3, padding=1)

        # time + cond projections
        self.time_proj = TimeMLP(cfg.time_embed_dim, cfg.cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_embed_in_dim, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )

        # down path
        ch = cfg.base_channels
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels: List[int] = []

        for level, mult in enumerate(cfg.channel_mults):
            out_ch = cfg.base_channels * mult
            for _ in range(cfg.num_res_blocks):
                self.down_blocks.append(FiLMResBlock(ch, out_ch, cond_dim=cfg.cond_dim, dropout=cfg.dropout))
                ch = out_ch
                self.skip_channels.append(ch)
            if level != len(cfg.channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())

        # middle
        self.mid1 = FiLMResBlock(ch, ch, cond_dim=cfg.cond_dim, dropout=cfg.dropout)
        self.mid2 = FiLMResBlock(ch, ch, cond_dim=cfg.cond_dim, dropout=cfg.dropout)

        # up path
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # We'll iterate levels reversed; for each, do resblocks with skip concat
        for level, mult in list(enumerate(cfg.channel_mults))[::-1]:
            out_ch = cfg.base_channels * mult
            for _ in range(cfg.num_res_blocks):
                skip_ch = self.skip_channels.pop()
                self.up_blocks.append(
                    FiLMResBlock(ch + skip_ch, out_ch, cond_dim=cfg.cond_dim, dropout=cfg.dropout)
                )
                ch = out_ch
            if level != 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        # out
        self.out_gn = GroupNorm32(ch)
        self.out_conv = nn.Conv2d(ch, cfg.in_channels, kernel_size=3, padding=1)

        if cfg.zero_out:
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)

    def _extract_cond_emb(self, cond: Optional[Any], device, dtype, batch: int) -> torch.Tensor:
        D_in = self.cfg.cond_embed_in_dim
        if cond is None:
            return torch.zeros((batch, D_in), device=device, dtype=dtype)

        if isinstance(cond, dict):
            for k in ("emb", "text_emb", "cond_emb"):
                v = cond.get(k, None)
                if torch.is_tensor(v):
                    v = v.to(device=device, dtype=dtype)
                    if v.dim() == 1:
                        v = v.unsqueeze(0).expand(batch, -1)
                    if v.shape[0] != batch:
                        raise ValueError(f"cond[{k}] batch mismatch: got {v.shape[0]} vs {batch}")
                    if v.shape[1] != D_in:
                        raise ValueError(f"cond[{k}] dim mismatch: got {v.shape[1]} vs {D_in}")
                    return v

        if torch.is_tensor(cond):
            v = cond.to(device=device, dtype=dtype)
            if v.dim() == 1:
                v = v.unsqueeze(0).expand(batch, -1)
            if v.shape[0] != batch:
                raise ValueError(f"cond tensor batch mismatch: got {v.shape[0]} vs {batch}")
            if v.shape[1] != D_in:
                raise ValueError(f"cond tensor dim mismatch: got {v.shape[1]} vs {D_in}")
            return v

        return torch.zeros((batch, D_in), device=device, dtype=dtype)

    def forward(self, X_masked: torch.Tensor, t: TLike, cond: Optional[Any]) -> torch.Tensor:
        if X_masked.dim() != 4:
            raise ValueError(f"X_masked must be (B,C,H,W), got {tuple(X_masked.shape)}")
        B, C, H, W = X_masked.shape
        if C != self.cfg.in_channels:
            raise ValueError(f"in_channels mismatch: model expects {self.cfg.in_channels}, got {C}")

        # time embedding
        tB = self.t_to_batch(t, X_masked)  # (B,)
        t_emb = timestep_embedding(tB, self.cfg.time_embed_dim)  # (B, time_dim)
        t_cond = self.time_proj(t_emb)  # (B, cond_dim)

        # cond embedding
        c_in = self._extract_cond_emb(cond, device=X_masked.device, dtype=X_masked.dtype, batch=B)  # (B, D_in)
        c_cond = self.cond_proj(c_in)  # (B, cond_dim)

        cond_vec = t_cond + c_cond

        # down
        h = self.in_conv(X_masked)
        skips: List[torch.Tensor] = []

        down_idx = 0
        for level in range(len(self.cfg.channel_mults)):
            for _ in range(self.cfg.num_res_blocks):
                h = self.down_blocks[down_idx](h, cond_vec)
                skips.append(h)
                down_idx += 1
            h = self.downsamples[level](h)

        # mid
        h = self.mid1(h, cond_vec)
        h = self.mid2(h, cond_vec)

        # up
        up_idx = 0
        for level in range(len(self.cfg.channel_mults)):
            # level runs 0..L-1 but up path corresponds to reversed channel_mults
            # we stored up_blocks in that reversed order already.
            for _ in range(self.cfg.num_res_blocks):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[up_idx](h, cond_vec)
                up_idx += 1
            h = self.upsamples[level](h)

        # out
        h = self.out_gn(h)
        h = F.silu(h)
        vel = self.out_conv(h)

        self.validate_io(X_masked, vel)
        return vel

