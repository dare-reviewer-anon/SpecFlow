from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

import torch


Cond = Optional[Dict[str, Any]]  # None means unconditional
PosNegCond = Dict[str, Cond]     # {"pos": Cond, "neg": Cond}


def make_cond(
    *,
    emb: Optional[torch.Tensor] = None,
    text_emb: Optional[torch.Tensor] = None,
    trace_emb: Optional[torch.Tensor] = None,
    vis_emb: Optional[torch.Tensor] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Cond:
    """
    Build a simple conditioning dict.
    Only includes keys that are not None.
    """
    d: Dict[str, Any] = {}
    if emb is not None:
        d["emb"] = emb
    if text_emb is not None:
        d["text_emb"] = text_emb
    if trace_emb is not None:
        d["trace_emb"] = trace_emb
    if vis_emb is not None:
        d["vis_emb"] = vis_emb
    if meta is not None:
        d["meta"] = meta
    return d if len(d) > 0 else None


def make_posneg_cond(pos: Cond, neg: Cond) -> PosNegCond:
    """
    Build a pos/neg conditioning container.
    """
    return {"pos": pos, "neg": neg}

def is_posneg(cond: Any) -> bool:
    return isinstance(cond, dict) and ("pos" in cond or "neg" in cond)


def get_cond_branch(cond: Any, branch: str) -> Cond:

    if branch == "uncond":
        return None

    if not is_posneg(cond):
        # simple condition dict or None
        return cond if branch in ("cond", "pos") else None

    # pos/neg container
    if branch == "cond" or branch == "pos":
        return cond.get("pos", None)
    if branch == "neg":
        return cond.get("neg", None)

    raise ValueError(f"Unknown branch: {branch}")

@dataclass
class PackedCond:
   
    cond: Any  # Cond or PosNegCond

    def pos(self) -> Cond:
        return get_cond_branch(self.cond, "pos")

    def neg(self) -> Cond:
        return get_cond_branch(self.cond, "neg")

    def uncond(self) -> Cond:
        return None


def pack(cond: Any) -> PackedCond:
    return PackedCond(cond=cond)


def unpack(p: PackedCond) -> Any:
    return p.cond


def cfg_dropout_mask(batch: int, p: float, device: torch.device) -> torch.Tensor:
    """
    Returns a boolean mask (B,) where True means "drop condition" (use uncond).
    """
    if p <= 0.0:
        return torch.zeros((batch,), device=device, dtype=torch.bool)
    if p >= 1.0:
        return torch.ones((batch,), device=device, dtype=torch.bool)
    return torch.rand((batch,), device=device) < float(p)


def apply_cfg_dropout(cond: Any, drop_mask: torch.Tensor) -> List[Cond]:

    B = int(drop_mask.numel())
    out: List[Cond] = []

    base = get_cond_branch(cond, "cond")  # pos if pos/neg, else simple dict

    for i in range(B):
        out.append(None if bool(drop_mask[i].item()) else base)
    return out


def cfg_combine(u_uncond: torch.Tensor, u_cond: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Standard CFG in velocity space:
      u = u_uncond + s * (u_cond - u_uncond)
    """
    return u_uncond + float(scale) * (u_cond - u_uncond)


def cfg_combine_posneg(
    u_pos: torch.Tensor,
    u_neg: torch.Tensor,
    scale: float,
) -> torch.Tensor:

    return u_neg + float(scale) * (u_pos - u_neg)


def resolve_cfg_branches(cond: Any) -> Tuple[Cond, Cond]:

    if not is_posneg(cond):
        return None, cond

    pos = cond.get("pos", None)
    neg = cond.get("neg", None)
    if neg is None:
        return None, pos
    return neg, pos



def ensure_batch_emb(emb: torch.Tensor, batch: int) -> torch.Tensor:
    """
    Ensure emb is (B,D). If (D,), expand to (B,D).
    """
    if emb.dim() == 1:
        return emb.unsqueeze(0).expand(batch, -1)
    if emb.dim() == 2:
        if emb.shape[0] != batch:
            raise ValueError(f"Batch mismatch: emb has B={emb.shape[0]} but expected {batch}")
        return emb
    raise ValueError(f"emb must be 1D or 2D, got {tuple(emb.shape)}")