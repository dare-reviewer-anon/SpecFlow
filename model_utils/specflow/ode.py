from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Any, Tuple, List, Union

import torch


VelocityFn = Callable[[torch.Tensor, Union[float, torch.Tensor], Optional[Any]], torch.Tensor]


@dataclass
class ODESolverConfig:
    steps: int = 8
    use_heun: bool = False  # default is Euler


def make_t_grid(steps: int, device=None, dtype=None) -> torch.Tensor:
    """
    Create a uniform grid t_k for k=0..steps-1 in [0,1).
    Shape: (steps,)
    """
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    device = device if device is not None else torch.device("cpu")
    dtype = dtype if dtype is not None else torch.float32
    # t_k = k/steps, k=0..steps-1
    return torch.arange(steps, device=device, dtype=dtype) / float(steps)


def euler_step(X: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """
    One Euler step: X_next = X + dt * u
    """
    return X + dt * u


def heun_step(
    X: torch.Tensor,
    t: float,
    dt: float,
    velocity_fn: VelocityFn,
    cond: Any,
    mask_fn: Optional[Callable[[float], torch.Tensor]] = None,
    apply_mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:

    def f(X_in: torch.Tensor, tt: float, c: Any) -> torch.Tensor:
        if mask_fn is not None and apply_mask_fn is not None:
            Mb = mask_fn(tt)
            X_in = apply_mask_fn(X_in, Mb)
        return velocity_fn(X_in, tt, c)

    u1 = f(X, t, cond)
    X_tilde = X + dt * u1
    u2 = f(X_tilde, t + dt, cond)
    return X + dt * 0.5 * (u1 + u2)


@torch.no_grad()
def euler_integrate(
    X0: torch.Tensor,
    velocity_fn: VelocityFn,
    *,
    steps: int = 8,
    t_grid: Optional[Sequence[float]] = None,
    cfg_scale: float = 0.0,
    cond: Any = None,
    uncond: Any = None,
    mask_fn: Optional[Callable[[float], torch.Tensor]] = None,
    apply_mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    return_all: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:

    if t_grid is None:
        t = make_t_grid(steps, device=X0.device, dtype=X0.dtype)
        t_list = [float(v.item()) for v in t]
        T = steps
    else:
        t_list = [float(v) for v in t_grid]
        T = len(t_list)
        if T <= 0:
            raise ValueError("t_grid must be non-empty")

    dt = 1.0 / float(T)

    X = X0
    traj: List[torch.Tensor] = []
    if return_all:
        traj.append(X)

    for tk in t_list:
        X_in = X
        if mask_fn is not None and apply_mask_fn is not None:
            Mb = mask_fn(tk)
            X_in = apply_mask_fn(X_in, Mb)

        if cfg_scale and cfg_scale != 0.0:
            u_uncond = velocity_fn(X_in, tk, uncond)
            u_cond = velocity_fn(X_in, tk, cond)
            u = u_uncond + float(cfg_scale) * (u_cond - u_uncond)
        else:
            u = velocity_fn(X_in, tk, cond)

        X = euler_step(X, u, dt)

        if return_all:
            traj.append(X)

    return (X, traj) if return_all else X


@torch.no_grad()
def heun_integrate(
    X0: torch.Tensor,
    velocity_fn: VelocityFn,
    *,
    steps: int = 8,
    t_grid: Optional[Sequence[float]] = None,
    cond: Any = None,
    mask_fn: Optional[Callable[[float], torch.Tensor]] = None,
    apply_mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    return_all: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Optional RK2 integrator (Heun). You can ignore/delete if you only want Euler.
    No CFG here (keep minimal); if needed, you can wrap velocity_fn outside.
    """
    if t_grid is None:
        t = make_t_grid(steps, device=X0.device, dtype=X0.dtype)
        t_list = [float(v.item()) for v in t]
        T = steps
    else:
        t_list = [float(v) for v in t_grid]
        T = len(t_list)
        if T <= 0:
            raise ValueError("t_grid must be non-empty")

    dt = 1.0 / float(T)

    X = X0
    traj: List[torch.Tensor] = []
    if return_all:
        traj.append(X)

    for tk in t_list:
        X = heun_step(
            X, tk, dt,
            velocity_fn=velocity_fn,
            cond=cond,
            mask_fn=mask_fn,
            apply_mask_fn=apply_mask_fn,
        )
        if return_all:
            traj.append(X)

    return (X, traj) if return_all else X

