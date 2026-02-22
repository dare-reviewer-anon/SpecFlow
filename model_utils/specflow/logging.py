import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Union

import torch


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    """
    Quick numeric summary for a tensor.
    """
    x = x.detach()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "absmax": float(x.abs().max().item()),
    }


@dataclass
class MetricMeter:
    """
    Tracks running sums/means for scalar metrics.
    """
    sums: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def update(self, metrics: Dict[str, Union[float, int]], n: int = 1) -> None:
        for k, v in metrics.items():
            val = float(v)
            self.sums[k] = self.sums.get(k, 0.0) + val * n
            self.counts[k] = self.counts.get(k, 0) + n

    def mean(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, s in self.sums.items():
            c = max(1, self.counts.get(k, 0))
            out[k] = s / c
        return out

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()


@dataclass
class SpecFlowLogger:
    """
    Simple logger:
      - accumulates metrics in a window
      - prints periodically
      - can write to a text file
    """
    name: str = "specflow"
    log_every: int = 50
    to_file: Optional[str] = None  # path to .txt log file
    with_time: bool = True

    meter: MetricMeter = field(default_factory=MetricMeter)
    step: int = 0
    _t0: float = field(default_factory=time.time)

    def _format_line(self, metrics: Dict[str, float]) -> str:
        parts = []
        if self.with_time:
            parts.append(f"[{_now()}]")
        parts.append(f"[{self.name}] step={self.step}")
        elapsed = time.time() - self._t0
        parts.append(f"elapsed={elapsed:.1f}s")
        for k, v in sorted(metrics.items()):
            parts.append(f"{k}={v:.6g}")
        return " ".join(parts)

    def log(self, metrics: Dict[str, Union[float, int]], n: int = 1, force: bool = False) -> None:
        """
        Update meter and print if needed.
        """
        self.step += 1
        self.meter.update(metrics, n=n)

        if force or (self.log_every > 0 and self.step % self.log_every == 0):
            means = self.meter.mean()
            line = self._format_line(means)
            print(line)
            if self.to_file is not None:
                os.makedirs(os.path.dirname(self.to_file) or ".", exist_ok=True)
                with open(self.to_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            # reset window
            self.meter.reset()

    def log_tensor(self, name: str, x: torch.Tensor, extra: Optional[Dict[str, Any]] = None, force: bool = False) -> None:
        """
        Log tensor statistics as scalars.
        """
        stats = {f"{name}/{k}": v for k, v in tensor_stats(x).items()}
        if extra:
            for k, v in extra.items():
                stats[f"{name}/{k}"] = float(v) if isinstance(v, (int, float)) else 0.0
        self.log(stats, n=1, force=force)


def save_image_grid(
    x: torch.Tensor,
    path: str,
    *,
    clamp: bool = True,
    value_range: Optional[tuple] = None,
    max_items: int = 8,
) -> None:

    if x.dim() != 4:
        raise ValueError(f"x must be (B,C,H,W), got {tuple(x.shape)}")

    x = x.detach().cpu()
    if x.shape[0] > max_items:
        x = x[:max_items]

    if value_range is not None:
        lo, hi = value_range
        x = (x - lo) / max(1e-8, (hi - lo))
        x = x.clamp(0.0, 1.0)
    elif clamp:
        x = x.clamp(0.0, 1.0)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"images": x}, path)


def debug_dict(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Namespaces a dict's keys for logging.
    """
    return {f"{prefix}/{k}": v for k, v in d.items()}