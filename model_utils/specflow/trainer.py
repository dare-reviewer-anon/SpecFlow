from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union, List

import torch

from .cosine_proj import block_dct
from .masking import MaskConfig, block_mask


VelocityModel = Callable[[torch.Tensor, Union[float, torch.Tensor], Optional[Any]], torch.Tensor]


@dataclass
class SpecFlowTrainConfig:
    block_size: int = 8
    mask: MaskConfig = MaskConfig()
    cond_dropout_prob: float = 0.1
    use_spectral_mask: bool = True
    eps: float = 1e-6


def _maybe_drop_cond(cond: Any, p: float, device: torch.device, batch: int) -> List[Optional[Any]]:
    if p <= 0.0:
        return [cond for _ in range(batch)]
    if p >= 1.0:
        return [None for _ in range(batch)]
    drop = (torch.rand((batch,), device=device) < float(p))
    return [None if bool(drop[i].item()) else cond for i in range(batch)]


def _sample_t(B: int, device, dtype) -> torch.Tensor:
    return torch.rand((B,), device=device, dtype=dtype)


def flow_matching_loss(
    cfg: SpecFlowTrainConfig,
    velocity_model: VelocityModel,
    x0: torch.Tensor,
    cond: Any,
) -> torch.Tensor:
    if x0.dim() != 4:
        raise ValueError(f"x0 must be (B,C,H,W), got {tuple(x0.shape)}")
    B, C, H, W = x0.shape
    device, dtype = x0.device, x0.dtype

    cfg.mask.block_size = cfg.block_size

    x1 = torch.randn_like(x0)

    X0 = block_dct(x0, cfg.block_size)
    X1 = block_dct(x1, cfg.block_size)

    t = _sample_t(B, device, dtype)
    t_view = t.view(B, 1, 1, 1)

    Xt = (1.0 - t_view) * X1 + t_view * X0
    target = (X0 - X1)

    cond_list = _maybe_drop_cond(cond, cfg.cond_dropout_prob, device=device, batch=B)

    if cfg.use_spectral_mask:
        b = cfg.block_size
        Xt_b = Xt.view(B, C, H // b, b, W // b, b).permute(0, 1, 2, 4, 3, 5).contiguous()
        for i in range(B):
            Mb = block_mask(float(t[i].item()), cfg.mask, device=device, dtype=dtype)
            Xt_b[i] = Xt_b[i] * Mb.view(1, 1, 1, b, b)
        Xt_masked = Xt_b.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
    else:
        Xt_masked = Xt

    preds = []
    for i in range(B):
        preds.append(velocity_model(Xt_masked[i : i + 1], float(t[i].item()), cond_list[i]))
    pred = torch.cat(preds, dim=0)

    loss = (pred - target).pow(2).mean()
    return loss


@torch.no_grad()
def flow_matching_batch_sanity(
    cfg: SpecFlowTrainConfig,
    velocity_model: VelocityModel,
    x0: torch.Tensor,
    cond: Any,
) -> Tuple[torch.Tensor, dict]:
    loss = flow_matching_loss(cfg, velocity_model, x0, cond)
    info = {
        "loss": float(loss.item()),
        "block_size": cfg.block_size,
        "cond_dropout_prob": cfg.cond_dropout_prob,
        "use_spectral_mask": cfg.use_spectral_mask,
        "mask_schedule": cfg.mask.schedule,
        "mask_type": cfg.mask.mask_type,
    }
    return loss, info