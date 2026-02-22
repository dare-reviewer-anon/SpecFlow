import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch

MaskSchedule = Literal["fixed", "linear", "cosine"]
MaskType = Literal["square", "radial", "zigzag"]


@dataclass
class MaskConfig:
    block_size: int = 8

    # Schedule of how many frequencies are kept
    schedule: MaskSchedule = "cosine"

    # For schedule="fixed"
    fixed_keep: int = 16

    # For progressive schedules
    min_keep: int = 8
    max_keep: int = 64  # <= block_size^2

    # Mask layout within a block
    mask_type: MaskType = "square"

    # Soft mask option (weights in [0,1]) instead of binary {0,1}
    soft: bool = False
    soft_sharpness: float = 12.0  # larger => closer to hard cutoff


def _clamp_keep(keep: int, b: int) -> int:
    return max(1, min(b * b, int(keep)))


def _alpha_from_t(t: float, schedule: MaskSchedule) -> float:
    t = float(max(0.0, min(1.0, t)))
    if schedule == "fixed":
        return 1.0
    if schedule == "linear":
        return t
    if schedule == "cosine":
        # smooth start/end; common for progressive unmasking
        return 0.5 - 0.5 * math.cos(math.pi * t)
    raise ValueError(f"Unknown schedule: {schedule}")


def keep_count(t: float, cfg: MaskConfig) -> int:
    """
    Number of kept coefficients within one block at normalized time t in [0,1].
    """
    b = cfg.block_size
    if cfg.schedule == "fixed":
        return _clamp_keep(cfg.fixed_keep, b)

    a = _alpha_from_t(t, cfg.schedule)
    keep = int(round(cfg.min_keep + a * (cfg.max_keep - cfg.min_keep)))
    return _clamp_keep(keep, b)


def _square_mask(b: int, keep: int, device, dtype) -> torch.Tensor:
    """
    Keep top-left sxs square (low frequencies) where s = floor(sqrt(keep)).
    """
    s = int(math.floor(math.sqrt(max(1, keep))))
    s = max(1, min(b, s))
    M = torch.zeros((b, b), device=device, dtype=dtype)
    M[:s, :s] = 1.0
    return M


def _radial_ranks(b: int, device, dtype) -> torch.Tensor:
    """
    Radial distance ranks (low near (0,0)).
    """
    yy, xx = torch.meshgrid(
        torch.arange(b, device=device, dtype=dtype),
        torch.arange(b, device=device, dtype=dtype),
        indexing="ij",
    )
    r = torch.sqrt(xx * xx + yy * yy)
    return r


def _radial_mask(b: int, keep: int, device, dtype, soft: bool, sharpness: float) -> torch.Tensor:
    """
    Keep lowest 'keep' entries in radial distance from (0,0).
    If soft=True, use a smooth logistic around the threshold.
    """
    r = _radial_ranks(b, device, dtype)  # (b,b)
    flat = r.flatten()
    # kth threshold (approx) - sort once per call (b is small, so fine)
    vals, _ = torch.sort(flat)
    idx = min(keep - 1, vals.numel() - 1)
    thr = vals[idx]

    if not soft:
        return (r <= thr).to(dtype=dtype)

    # smooth: sigmoid(-sharpness*(r-thr)) ~ 1 when r < thr
    return torch.sigmoid(-sharpness * (r - thr))


def _zigzag_indices(b: int) -> torch.Tensor:
    """
    Return zigzag order indices for a (b,b) block as a (b*b, 2) tensor [row,col].
    Standard JPEG-style zigzag traversal.
    """
    coords = []
    for s in range(2 * b - 1):
        if s % 2 == 0:
            # even diagonal: go down-left (i decreases)
            i_start = min(s, b - 1)
            i_end = max(0, s - (b - 1))
            for i in range(i_start, i_end - 1, -1):
                j = s - i
                coords.append((i, j))
        else:
            # odd diagonal: go up-right (i increases)
            i_start = max(0, s - (b - 1))
            i_end = min(s, b - 1)
            for i in range(i_start, i_end + 1):
                j = s - i
                coords.append((i, j))
    return torch.tensor(coords, dtype=torch.long)  # (b*b, 2)


# Cache zigzag coords per block size
_ZIGZAG_CACHE = {}


def _zigzag_mask(b: int, keep: int, device, dtype, soft: bool, sharpness: float) -> torch.Tensor:
    """
    Keep the first 'keep' coefficients in zigzag order.
    If soft=True, make a soft step using sigmoid over the rank.
    """
    global _ZIGZAG_CACHE
    if b not in _ZIGZAG_CACHE:
        _ZIGZAG_CACHE[b] = _zigzag_indices(b)
    coords = _ZIGZAG_CACHE[b].to(device=device)

    M = torch.zeros((b, b), device=device, dtype=dtype)
    if not soft:
        sel = coords[:keep]
        M[sel[:, 0], sel[:, 1]] = 1.0
        return M

    # soft: rank-based weights
    ranks = torch.arange(b * b, device=device, dtype=dtype)
    # threshold at keep-0.5
    thr = torch.tensor(float(keep) - 0.5, device=device, dtype=dtype)
    w = torch.sigmoid(-sharpness * (ranks - thr))  # high for rank < keep
    # scatter
    M = M.flatten()
    M[: b * b] = 0.0
    # put w according to zigzag rank into coords positions
    # coords gives position for each rank
    flat_idx = coords[:, 0] * b + coords[:, 1]
    M[flat_idx] = w
    return M.view(b, b)


def block_mask(t: float, cfg: MaskConfig, device=None, dtype=None) -> torch.Tensor:
    """
    Compute block-level mask M(t) of shape (b,b).
    """
    b = cfg.block_size
    device = device if device is not None else torch.device("cpu")
    dtype = dtype if dtype is not None else torch.float32

    keep = keep_count(t, cfg)

    if cfg.mask_type == "square":
        M = _square_mask(b, keep, device, dtype)
    elif cfg.mask_type == "radial":
        M = _radial_mask(b, keep, device, dtype, soft=cfg.soft, sharpness=cfg.soft_sharpness)
    elif cfg.mask_type == "zigzag":
        M = _zigzag_mask(b, keep, device, dtype, soft=cfg.soft, sharpness=cfg.soft_sharpness)
    else:
        raise ValueError(f"Unknown mask_type: {cfg.mask_type}")

    return M


def expand_block_mask_to_grid(Mb: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Expand a (b,b) block mask Mb to a full (H,W) mask by tiling over blocks.
    Returned shape: (1,1,H,W) to broadcast over (B,C,H,W).

    Assumes H and W divisible by b.
    """
    if Mb.dim() != 2 or Mb.shape[0] != Mb.shape[1]:
        raise ValueError(f"Mb must be (b,b). Got {tuple(Mb.shape)}")
    b = Mb.shape[0]
    if H % b != 0 or W % b != 0:
        raise ValueError(f"H,W must be divisible by b. Got H={H}, W={W}, b={b}")

    Hb, Wb = H // b, W // b
    grid = Mb.view(1, 1, 1, 1, b, b).expand(1, 1, Hb, Wb, b, b)
    # reorder back to (H,W)
    grid = grid.permute(0, 1, 2, 4, 3, 5).contiguous().view(1, 1, H, W)
    return grid


def apply_block_mask(X: torch.Tensor, Mb: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Apply a (b,b) block mask Mb to full coefficient grid X (B,C,H,W) by
    reshaping into blocks and multiplying per block.

    This avoids allocating a big (H,W) mask.
    """
    if X.dim() != 4:
        raise ValueError(f"X must be (B,C,H,W). Got {tuple(X.shape)}")
    B, C, H, W = X.shape
    b = int(block_size)
    if b <= 0 or H % b != 0 or W % b != 0:
        raise ValueError(f"Invalid block_size={b} for H={H}, W={W}")
    if Mb.shape != (b, b):
        raise ValueError(f"Mb must be (b,b) with b={b}. Got {tuple(Mb.shape)}")

    Xb = X.view(B, C, H // b, b, W // b, b).permute(0, 1, 2, 4, 3, 5).contiguous()  # (B,C,Hb,Wb,b,b)
    Xb = Xb * Mb.view(1, 1, 1, 1, b, b)
    X_masked = Xb.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
    return X_masked

