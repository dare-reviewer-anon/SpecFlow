from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .cosine_proj import block_dct, block_idct
from .masking import MaskConfig, block_mask, apply_block_mask
from .ode import euler_integrate


VelocityModel = Callable[[torch.Tensor, Union[float, torch.Tensor], Optional[Any]], torch.Tensor]
TextGenerator = Callable[[Any, List[Any], torch.Tensor], Any]


@dataclass
class SpecFlowControllerConfig:
    # DCT block config
    block_size: int = 8

    # Mask config (M(t))
    mask: MaskConfig = MaskConfig()

    # ODE / Euler
    euler_steps: int = 8

    # CFG
    cfg_scale: float = 3.0

    # Optional: decode intermediate visuals during integration (debug)
    return_intermediate_visuals: bool = False

    # Optional: anchoring (i2i editing style)
    use_anchoring: bool = False
    denoising_strength: float = 0.6  # 0 => pure noise init, 1 => start from previous coeffs

    # Numerical
    eps: float = 1e-6


class SpecFlowController:

    def __init__(
        self,
        cfg: SpecFlowControllerConfig,
        velocity_model: VelocityModel,
        text_generator: TextGenerator,
    ):
        self.cfg = cfg
        self.velocity_model = velocity_model
        self.text_generator = text_generator

        # Ensure mask cfg uses the same block size
        self.cfg.mask.block_size = self.cfg.block_size

    def _make_cond(self, query: Any, trace: List[Any], hop_idx: int) -> Dict[str, Any]:

        return {"query": query, "trace": list(trace), "hop": hop_idx}

    @torch.no_grad()
    def _one_hop_visual_update(
        self,
        v_prev: torch.Tensor,
        cond: Any,
        *,
        coeff_anchor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        One hop update:
          - initialize X(0) in coeff space
          - Euler integrate dX/dt = u_theta(M(t)⊙X, t, cond) with CFG
          - decode v_next = Db^{-1}(X(1))

        Returns:
          v_next: (B,C,H,W)
          X_next: (B,C,H,W) coeffs at final time
          intermediates: optional list of decoded visuals per Euler step (incl. start)
        """
        cfg = self.cfg
        B, C, H, W = v_prev.shape
        device, dtype = v_prev.device, v_prev.dtype

        # Alg.1 init: sample noise and project to cosine coeffs
        x_noise = torch.randn((B, C, H, W), device=device, dtype=dtype)
        X0 = block_dct(x_noise, cfg.block_size)

        # Optional anchoring (not required by SpecFlow paper Alg.1)
        if cfg.use_anchoring and (coeff_anchor is not None):
            lam = float(max(0.0, min(1.0, cfg.denoising_strength)))
            X0 = (1.0 - lam) * X0 + lam * coeff_anchor

        # Mask function (block-level)
        def mask_fn(t: float) -> torch.Tensor:
            return block_mask(t, cfg.mask, device=device, dtype=dtype)

        # Apply mask function without allocating big tensors
        def apply_mask_fn(X: torch.Tensor, Mb: torch.Tensor) -> torch.Tensor:
            return apply_block_mask(X, Mb, cfg.block_size)

        # Integrate with Euler, with CFG in velocity space
        # cond/uncond branches: uncond is simply None by convention.
        if cfg.return_intermediate_visuals:
            XT, traj = euler_integrate(
                X0,
                self.velocity_model,
                steps=cfg.euler_steps,
                cfg_scale=cfg.cfg_scale,
                cond=cond,
                uncond=None,
                mask_fn=mask_fn,
                apply_mask_fn=apply_mask_fn,
                return_all=True,
            )
            inter_vis = [block_idct(Xk, cfg.block_size) for Xk in traj]
        else:
            XT = euler_integrate(
                X0,
                self.velocity_model,
                steps=cfg.euler_steps,
                cfg_scale=cfg.cfg_scale,
                cond=cond,
                uncond=None,
                mask_fn=mask_fn,
                apply_mask_fn=apply_mask_fn,
                return_all=False,
            )
            inter_vis = None

        v_next = block_idct(XT, cfg.block_size)
        return v_next, XT, inter_vis

    @torch.no_grad()
    def run(
        self,
        x_vis: torch.Tensor,
        query: Any,
        hops: int,
    ) -> Tuple[List[Any], List[torch.Tensor], Optional[List[List[torch.Tensor]]]]:

        if hops <= 0:
            raise ValueError(f"hops must be > 0, got {hops}")
        if x_vis.dim() != 4:
            raise ValueError(f"x_vis must be (B,C,H,W), got {tuple(x_vis.shape)}")

        text_trace: List[Any] = []
        visual_states: List[torch.Tensor] = [x_vis]
        per_hop_inter: Optional[List[List[torch.Tensor]]] = [] if self.cfg.return_intermediate_visuals else None

        v_prev = x_vis
        coeff_anchor: Optional[torch.Tensor] = None

        for i in range(hops):
            cond_i = self._make_cond(query, text_trace, i)

            # Visual update (bounded workspace: overwrite)
            v_next, X_next, inter_vis = self._one_hop_visual_update(
                v_prev,
                cond=cond_i,
                coeff_anchor=coeff_anchor,
            )
            visual_states.append(v_next)

            if per_hop_inter is not None:
                # inter_vis includes start + each step end; keep as-is for debugging
                per_hop_inter.append(inter_vis if inter_vis is not None else [])

            # Text thought update (autoregressive over trace, conditioned on new visual)
            t_next = self.text_generator(query, list(text_trace), v_next)
            text_trace.append(t_next)

            # prepare for next hop
            v_prev = v_next
            coeff_anchor = X_next

        return text_trace, visual_states, per_hop_inter