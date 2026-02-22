from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Dict

import math
import torch
import torch.nn as nn


def _dct_matrix(N: int, device=None, dtype=None) -> torch.Tensor:
    n = torch.arange(N, device=device, dtype=dtype).view(1, N)
    k = torch.arange(N, device=device, dtype=dtype).view(N, 1)
    mat = torch.cos(math.pi / N * (n + 0.5) * k)
    mat[0, :] *= 1.0 / math.sqrt(N)
    if N > 1:
        mat[1:, :] *= math.sqrt(2.0 / N)
    return mat


def dct_1d(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    N = x.size(dim)
    C = _dct_matrix(N, device=x.device, dtype=x.dtype)
    x_perm = x.movedim(dim, -1)
    y = torch.matmul(x_perm, C.T)
    return y.movedim(-1, dim)


def idct_1d(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    N = y.size(dim)
    C = _dct_matrix(N, device=y.device, dtype=y.dtype)
    y_perm = y.movedim(dim, -1)
    x = torch.matmul(y_perm, C)
    return x.movedim(-1, dim)


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    x = dct_1d(x, dim=-1)
    x = dct_1d(x, dim=-2)
    return x


def idct_2d(y: torch.Tensor) -> torch.Tensor:
    y = idct_1d(y, dim=-1)
    y = idct_1d(y, dim=-2)
    return y


def _check_divisible(H: int, W: int, b: int):
    if (H % b) != 0 or (W % b) != 0:
        raise ValueError(f"H,W must be divisible by block_size b={b}. Got H={H}, W={W}.")


def blockify(x: torch.Tensor, block_size: int) -> torch.Tensor:
    B, C, H, W = x.shape
    b = block_size
    _check_divisible(H, W, b)
    HB, WB = H // b, W // b
    x = x.view(B, C, HB, b, WB, b).permute(0, 1, 2, 4, 3, 5).contiguous()
    return x


def unblockify(blocks: torch.Tensor, block_size: int) -> torch.Tensor:
    B, C, HB, WB, b1, b2 = blocks.shape
    b = block_size
    if b1 != b or b2 != b:
        raise ValueError(f"Invalid block dims: expected ({b},{b}), got ({b1},{b2}).")
    x = blocks.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, HB * b, WB * b)
    return x


def block_dct(x: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    blocks = blockify(x, block_size)
    coeff = dct_2d(blocks)
    return unblockify(coeff, block_size)


def block_idct(coeff: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    blocks = blockify(coeff, block_size)
    x = idct_2d(blocks)
    return unblockify(x, block_size)


def _freq_grid(block_size: int, device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b = block_size
    u = torch.arange(b, device=device, dtype=dtype).view(b, 1).repeat(1, b)
    v = torch.arange(b, device=device, dtype=dtype).view(1, b).repeat(b, 1)
    r = torch.sqrt(u**2 + v**2)
    r = r / (r.max().clamp_min(1e-8))
    return u, v, r


def spectral_mask(
    H: int,
    W: int,
    block_size: int,
    t: float,
    schedule: Literal["fixed", "linear", "cosine"] = "cosine",
    fixed_ratio: float = 0.25,
) -> torch.Tensor:
    b = block_size
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"t must be in [0,1], got {t}.")

    if schedule == "fixed":
        thr = float(fixed_ratio)
    elif schedule == "linear":
        min_thr, max_thr = 0.10, 1.00
        thr = min_thr + (max_thr - min_thr) * t
    elif schedule == "cosine":
        min_thr, max_thr = 0.10, 1.00
        thr = min_thr + (max_thr - min_thr) * (1 - math.cos(math.pi * t)) / 2.0
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    _, _, r = _freq_grid(b, device="cpu", dtype=torch.float32)
    bm = (r <= thr).float()

    if (H % b) != 0 or (W % b) != 0:
        raise ValueError(f"H,W must be divisible by block_size. Got H={H},W={W},b={b}.")
    HB, WB = H // b, W // b
    M = bm.repeat(HB, WB)
    return M.view(1, 1, H, W)


class VelocityModel(nn.Module):
    def forward(self, z: torch.Tensor, t: torch.Tensor, cond: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class SpecFlowConfig:
    block_size: int = 8
    ode_steps: int = 5
    cfg_scale: float = 4.0
    schedule: Literal["fixed", "linear", "cosine"] = "cosine"
    fixed_ratio: float = 0.25
    t0: float = 0.0
    t1: float = 1.0


@torch.no_grad()
def specflow_workspace_update(
    model: VelocityModel,
    x_cond: Dict[str, torch.Tensor],
    x_uncond: Dict[str, torch.Tensor],
    image_shape: Tuple[int, int, int, int],
    cfg: SpecFlowConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    B, C, H, W = image_shape
    b = cfg.block_size

    x = torch.randn((B, C, H, W), device=device, dtype=dtype)
    z = block_dct(x, block_size=b)

    T = int(cfg.ode_steps)
    t0, t1 = float(cfg.t0), float(cfg.t1)
    dt = (t1 - t0) / max(T, 1)

    for k in range(T):
        t = t0 + (k + 0.5) * dt
        t_tensor = torch.full((B,), float(t), device=device, dtype=dtype)

        M = spectral_mask(H, W, block_size=b, t=(k + 1) / T, schedule=cfg.schedule, fixed_ratio=cfg.fixed_ratio)
        M = M.to(device=device, dtype=dtype)

        z_masked = z * M

        v_u = model(z_masked, t_tensor, cond=x_uncond)
        v_c = model(z_masked, t_tensor, cond=x_cond)

        w = float(cfg.cfg_scale)
        v = v_u + w * (v_c - v_u)

        z = z + dt * (v * M)

    x_out = block_idct(z, block_size=b)
    return x_out


@torch.no_grad()
def specflow_workspace_update_from_prev(
    model: VelocityModel,
    z_prev: torch.Tensor,
    x_cond: Dict[str, torch.Tensor],
    x_uncond: Dict[str, torch.Tensor],
    cfg: SpecFlowConfig,
):
    B, C, H, W = z_prev.shape
    b = cfg.block_size
    z = z_prev.clone()

    T = int(cfg.ode_steps)
    t0, t1 = float(cfg.t0), float(cfg.t1)
    dt = (t1 - t0) / max(T, 1)

    for k in range(T):
        t = t0 + (k + 0.5) * dt
        t_tensor = torch.full((B,), float(t), device=z.device, dtype=z.dtype)
        M = spectral_mask(H, W, block_size=b, t=(k + 1) / T, schedule=cfg.schedule, fixed_ratio=cfg.fixed_ratio)
        M = M.to(device=z.device, dtype=z.dtype)

        z_masked = z * M
        v_u = model(z_masked, t_tensor, cond=x_uncond)
        v_c = model(z_masked, t_tensor, cond=x_cond)
        v = v_u + float(cfg.cfg_scale) * (v_c - v_u)
        z = z + dt * (v * M)

    return z