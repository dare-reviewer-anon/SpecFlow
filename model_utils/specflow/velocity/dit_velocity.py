import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interface import BaseVelocityModel, TLike



def _timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:

    t_scaled = t * 1000.0

    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=t.dtype) / half)
    args = t_scaled[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int, device, dtype) -> torch.Tensor:

    if embed_dim % 4 != 0:
        pass

    y = torch.arange(grid_h, device=device, dtype=dtype)
    x = torch.arange(grid_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # (H,W)

    # Flatten
    yy = yy.reshape(-1)  # (N,)
    xx = xx.reshape(-1)  # (N,)

    # Split dims
    dim_half = embed_dim // 2
    dim_quarter = dim_half // 2

    def pe_1d(pos: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device, dtype=dtype) / max(1, half))
        args = pos[:, None] * freqs[None]
        out = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            out = torch.cat([out, torch.zeros_like(out[:, :1])], dim=-1)
        return out

    pe_x = pe_1d(xx, dim_half)
    pe_y = pe_1d(yy, dim_half)
    pe = torch.cat([pe_y, pe_x], dim=-1)  # (N, embed_dim') where embed_dim' ~ 2*dim_half

    if pe.shape[1] < embed_dim:
        pe = torch.cat([pe, torch.zeros((pe.shape[0], embed_dim - pe.shape[1]), device=device, dtype=dtype)], dim=-1)
    elif pe.shape[1] > embed_dim:
        pe = pe[:, :embed_dim]

    return pe.unsqueeze(0)  # (1, N, D)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * hidden_mult
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdaLN(nn.Module):

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * dim)

        # Initialize to near-zero modulation (stability)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        ss = self.to_scale_shift(cond)  # (B, 2D)
        scale, shift = ss.chunk(2, dim=-1)
        # Broadcast to tokens: (B,1,D)
        return h * (1.0 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, cond_dim: int, mlp_mult: int = 4, attn_dropout: float = 0.0, resid_dropout: float = 0.0):
        super().__init__()
        self.adaln1 = AdaLN(dim, cond_dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=attn_dropout, batch_first=True)
        self.drop1 = nn.Dropout(resid_dropout)

        self.adaln2 = AdaLN(dim, cond_dim)
        self.mlp = MLP(dim, hidden_mult=mlp_mult, dropout=resid_dropout)
        self.drop2 = nn.Dropout(resid_dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Attention
        h = self.adaln1(x, cond)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(a)

        # MLP
        h = self.adaln2(x, cond)
        m = self.mlp(h)
        x = x + self.drop2(m)
        return x



@dataclass
class DiTVelocityConfig:
    in_channels: int
    model_dim: int = 512
    depth: int = 8
    num_heads: int = 8
    mlp_mult: int = 4

    patch_size: int = 2  # patchify on coeff grid
    cond_dim: int = 512  # dimension after cond projection (time + optional cond)

    time_embed_dim: int = 256
    cond_embed_in_dim: int = 512  # expected cond embedding input dim (you can change)

    attn_dropout: float = 0.0
    resid_dropout: float = 0.0


class DiTVelocityModel(BaseVelocityModel):

    def __init__(self, cfg: DiTVelocityConfig):
        super().__init__()
        self.cfg = cfg

        p = cfg.patch_size
        if p <= 0:
            raise ValueError(f"patch_size must be > 0, got {p}")

        # Patchify coefficients to tokens
        self.patch_embed = nn.Conv2d(cfg.in_channels, cfg.model_dim, kernel_size=p, stride=p, padding=0)

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_embed_in_dim, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=cfg.model_dim,
                n_heads=cfg.num_heads,
                cond_dim=cfg.cond_dim,
                mlp_mult=cfg.mlp_mult,
                attn_dropout=cfg.attn_dropout,
                resid_dropout=cfg.resid_dropout,
            )
            for _ in range(cfg.depth)
        ])

        # Final AdaLN + projection back to patch pixels (velocity in coeff space)
        self.final_adaln = AdaLN(cfg.model_dim, cfg.cond_dim)
        self.to_patch = nn.Linear(cfg.model_dim, (p * p) * cfg.in_channels)

        nn.init.zeros_(self.to_patch.weight)
        nn.init.zeros_(self.to_patch.bias)

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

        # Unknown cond type: ignore (zeros)
        return torch.zeros((batch, D_in), device=device, dtype=dtype)

    def forward(self, X_masked: torch.Tensor, t: TLike, cond: Optional[Any]) -> torch.Tensor:
        if X_masked.dim() != 4:
            raise ValueError(f"X_masked must be (B,C,H,W), got {tuple(X_masked.shape)}")
        B, C, H, W = X_masked.shape
        if C != self.cfg.in_channels:
            raise ValueError(f"in_channels mismatch: model expects {self.cfg.in_channels}, got {C}")

        p = self.cfg.patch_size
        if H % p != 0 or W % p != 0:
            raise ValueError(f"H,W must be divisible by patch_size={p}. Got H={H}, W={W}")

        # Normalize t to (B,)
        tB = self.t_to_batch(t, X_masked)  # (B,)
        t_emb = _timestep_embedding(tB, self.cfg.time_embed_dim)  # (B, time_dim)
        t_cond = self.time_mlp(t_emb)  # (B, cond_dim)

        # Optional conditioning embedding
        c_in = self._extract_cond_emb(cond, device=X_masked.device, dtype=X_masked.dtype, batch=B)  # (B, D_in)
        c_cond = self.cond_proj(c_in)  # (B, cond_dim)

        # Combined conditioning vector (B, cond_dim)
        cond_vec = t_cond + c_cond

        # Patchify: (B, D, H/p, W/p)
        x = self.patch_embed(X_masked)
        Gh, Gw = x.shape[2], x.shape[3]
        # Tokens: (B, N, D)
        x = x.flatten(2).transpose(1, 2)

        # Positional embedding (1, N, D)
        pos = _get_2d_sincos_pos_embed(self.cfg.model_dim, Gh, Gw, device=x.device, dtype=x.dtype)
        x = x + pos

        # Transformer
        for blk in self.blocks:
            x = blk(x, cond_vec)

        # Final modulation + project to patches
        x = self.final_adaln(x, cond_vec)                 # (B,N,D)
        patch = self.to_patch(x)                          # (B,N,p*p*C)

        # Unpatchify
        patch = patch.transpose(1, 2).contiguous()        # (B,p*p*C,N)
        patch = patch.view(B, (p * p) * C, Gh, Gw)        # (B,p*p*C,Gh,Gw)

        # Fold to (B,C,H,W)
        # Reshape channels into (C, p, p) and interleave
        patch = patch.view(B, C, p, p, Gh, Gw)
        patch = patch.permute(0, 1, 4, 2, 5, 3).contiguous()  # (B,C,Gh,p,Gw,p)
        vel = patch.view(B, C, Gh * p, Gw * p)

        self.validate_io(X_masked, vel)
        return vel


