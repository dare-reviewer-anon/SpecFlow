# model_utils/specflow/__init__.py
"""
SpecFlow — Spectral-Progressive Thought Flow

Core components for bounded multimodal reasoning with:
  - Blockwise DCT projection
  - Frequency masking schedules M(t)
  - Flow-matching velocity models
  - ODE integration (Euler)
  - CFG conditioning (simple / pos-neg)
  - Multi-hop controller (Alg.1)
  - Flow-matching trainer (Alg.2)
  - Debug / visualization tools

Recommended usage pattern:

    from model_utils.specflow import (
        SpecFlowController,
        SpecFlowTrainer,
        UNetVelocityModel,
        VLMAdapter,
        MaskConfig,
        SchedulerConfig,
    )

The velocity models live in:
    model_utils.specflow.velocity.*

Conditioning utilities:
    model_utils.specflow.conditioning
"""

# ---- Projection (DCT) ----
from .cosine_proj import block_dct, block_idct

# ---- Masking ----
from .masking import (
    MaskConfig,
    mask_ratio,
    block_mask,
    expand_block_mask_to_grid,
)

# ---- ODE ----
from .ode import euler_step

# ---- Scheduler ----
from .scheduler import SchedulerConfig, build_time_grid

# ---- Conditioning ----
from .conditioning import (
    make_cond,
    make_posneg_cond,
    get_cond_branch,
    resolve_cfg_branches,
    cfg_dropout_mask,
    apply_cfg_dropout,
    cfg_combine,
    cfg_combine_posneg,
)

# ---- Trainer / Controller ----
from .trainer import SpecFlowTrainer
from .controller import SpecFlowController

# ---- Debug / Logging ----
from .logging import SpecFlowLogger, tensor_stats
from .visualizer import SpecFlowVisualizer, VisualizerConfig, make_debug_bundle

# ---- Velocity subpackage ----
from .velocity import (
    VelocityModel,
    BaseVelocityModel,
    DiTVelocityModel,
    DiTVelocityConfig,
    UNetVelocityModel,
    UNetVelocityConfig,
    VLMAdapter,
    VLMAdapterConfig,
)

__all__ = [
    # projection
    "block_dct",
    "block_idct",

    # masking
    "MaskConfig",
    "mask_ratio",
    "block_mask",
    "expand_block_mask_to_grid",

    # ODE
    "euler_step",

    # scheduler
    "SchedulerConfig",
    "build_time_grid",

    # conditioning
    "make_cond",
    "make_posneg_cond",
    "get_cond_branch",
    "resolve_cfg_branches",
    "cfg_dropout_mask",
    "apply_cfg_dropout",
    "cfg_combine",
    "cfg_combine_posneg",

    # trainer / controller
    "SpecFlowTrainer",
    "SpecFlowController",

    # logging / visualization
    "SpecFlowLogger",
    "tensor_stats",
    "SpecFlowVisualizer",
    "VisualizerConfig",
    "make_debug_bundle",

    # velocity models
    "VelocityModel",
    "BaseVelocityModel",
    "DiTVelocityModel",
    "DiTVelocityConfig",
    "UNetVelocityModel",
    "UNetVelocityConfig",
    "VLMAdapter",
    "VLMAdapterConfig",
]