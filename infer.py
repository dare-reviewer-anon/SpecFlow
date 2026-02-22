import os
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from utils.load_model import load_model
from utils.load_data import load_data
from utils.evaluator import VisualizationEvaluator

from model_utils.specflow import (
    SpecFlowController,
    SpecFlowControllerConfig,
    MaskConfig,
)
from model_utils.specflow.velocity import (
    UNetVelocityModel,
    UNetVelocityConfig,
    DiTVelocityModel,
    DiTVelocityConfig,
    VLMAdapter,
    VLMAdapterConfig,
)

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_velocity(ckpt_path: str, kind: str, in_channels: int, cond_dim: int, device: torch.device):
    if kind == "unet":
        cfg = UNetVelocityConfig(in_channels=in_channels, cond_embed_in_dim=cond_dim)
        model = UNetVelocityModel(cfg)
    elif kind == "dit":
        cfg = DiTVelocityConfig(in_channels=in_channels, cond_embed_in_dim=cond_dim)
        model = DiTVelocityModel(cfg)
    else:
        raise ValueError(f"Unknown velocity kind: {kind}")

    sd = torch.load(ckpt_path, map_location="cpu")
    state_dict = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if len(missing) > 0:
        logger.warning(f"[velocity] missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if len(unexpected) > 0:
        logger.warning(f"[velocity] unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    model.to(device).eval()
    return model


def _dummy_text_generator(query: Any, trace: List[Any], visual: torch.Tensor) -> str:
    return f"hop{len(trace)}"


def run_standalone_specflow(args):
    device = _device()
    torch.set_grad_enabled(False)

    adapter_cfg = VLMAdapterConfig(
        out_dim=args.cond_dim,
        vis_in_channels=args.vis_channels,
        return_parts=False,
    )
    adapter = VLMAdapter(adapter_cfg).to(device).eval()

    velocity = _load_velocity(
        ckpt_path=args.velocity_ckpt,
        kind=args.velocity_kind,
        in_channels=args.vis_channels,
        cond_dim=args.cond_dim,
        device=device,
    )

    mask_cfg = MaskConfig(
        block_size=args.block_size,
        schedule=args.mask_schedule,
        mask_type=args.mask_type,
        ratio_start=args.mask_ratio_start,
        ratio_end=args.mask_ratio_end,
    )

    ctrl_cfg = SpecFlowControllerConfig(
        block_size=args.block_size,
        mask=mask_cfg,
        euler_steps=args.euler_steps,
        cfg_scale=args.cfg_scale,
        return_intermediate_visuals=args.return_intermediate,
        use_anchoring=args.use_anchoring,
        denoising_strength=args.denoising_strength,
    )

    def velocity_fn(X_masked, t, cond):
        return velocity(X_masked, t, cond)

    def text_gen(q, tr, v):
        return _dummy_text_generator(q, tr, v)

    controller = SpecFlowController(ctrl_cfg, velocity_fn, text_gen)

    B = args.batch_size
    H, W = args.image_size, args.image_size
    x0 = torch.randn(B, args.vis_channels, H, W, device=device)

    text_emb = torch.randn(B, args.cond_dim, device=device)
    query = {"emb": text_emb}

    def _make_cond_override(q, trace, hop_idx):
        return adapter(text={"emb": q["emb"]}, trace=None, visual=None)

    controller._make_cond = _make_cond_override  # type: ignore

    trace, visuals, inter = controller.run(x_vis=x0, query=query, hops=args.hops)

    print("Text trace:", trace)
    print("Final visual:", visuals[-1].shape)
    if args.return_intermediate:
        print("Intermediate per hop:", len(inter), "steps in hop0:", len(inter[0]) if len(inter) else 0)

    if args.save_pt is not None:
        os.makedirs(os.path.dirname(args.save_pt) or ".", exist_ok=True)
        torch.save(
            {
                "trace": trace,
                "visuals": [v.detach().cpu() for v in visuals],
                "intermediate": [[vv.detach().cpu() for vv in hop] for hop in inter] if inter is not None else None,
            },
            args.save_pt,
        )
        print(f"Saved outputs to: {args.save_pt}")


def run_attach_to_llm(args):
    device = _device()
    torch.set_grad_enabled(False)

    data = load_data(dataset=args.data, data_dir=args.data_dir)
    test_split = data.get("test", None)
    if test_split is None:
        raise ValueError(f"Expected test split in dataset. Got keys: {list(data.keys())}")

    model_processor = load_model(args)
    model, processor = model_processor["model"], model_processor["processor"]
    model.to(device).eval()

    if args.enable_specflow:
        try:
            from model_utils.specflow import DAREConfig, DAREController, attach_specflow_to_anole
            cfg = DAREConfig(
                hidden_size=getattr(model.config, "hidden_size", None),
                num_layers=getattr(model.config, "num_hidden_layers", None),
                num_heads=getattr(model.config, "num_attention_heads", None),
                rho_text_target=args.rho_text_target,
                rho_vis_target=args.rho_vis_target,
                tau=args.specflow_tau,
                lambda_ratio=args.specflow_lambda_ratio,
                lambda_soft=args.specflow_lambda_soft,
                lambda_hard=args.specflow_lambda_hard,
                prefix_kappa=args.specflow_prefix_kappa,
            )
            controller = DAREController(cfg)
            model = attach_specflow_to_anole(model, controller)
            logger.info("SpecFlow attached to LLM backbone for inference.")
        except ImportError:
            logger.warning("enable_specflow set but attach API is missing; running vanilla model.")

    evaluator = VisualizationEvaluator(args=args)

    num = min(args.max_examples, len(test_split))
    test_split = test_split.select(list(range(num)))

    outputs = []
    for i in range(num):
        ex = test_split[i]
        model_inputs = processor(ex, return_tensors="pt")
        model_inputs = {k: v.to(device) for k, v in model_inputs.items() if torch.is_tensor(v)}

        with torch.no_grad():
            gen = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens)
        outputs.append(gen.detach().cpu())

    if args.save_pt is not None:
        os.makedirs(os.path.dirname(args.save_pt) or ".", exist_ok=True)
        torch.save({"outputs": outputs}, args.save_pt)
        print(f"Saved outputs to: {args.save_pt}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="standalone_specflow", choices=["standalone_specflow", "attach_to_llm"])

    parser.add_argument("--velocity_ckpt", type=str, default=None)
    parser.add_argument("--velocity_kind", type=str, default="unet", choices=["unet", "dit"])
    parser.add_argument("--vis_channels", type=int, default=4)
    parser.add_argument("--cond_dim", type=int, default=512)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--hops", type=int, default=4)
    parser.add_argument("--euler_steps", type=int, default=8)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--block_size", type=int, default=8)

    parser.add_argument("--mask_schedule", type=str, default="cosine", choices=["fixed", "linear", "cosine"])
    parser.add_argument("--mask_type", type=str, default="lowpass", choices=["lowpass", "bandpass", "highpass"])
    parser.add_argument("--mask_ratio_start", type=float, default=0.25)
    parser.add_argument("--mask_ratio_end", type=float, default=1.0)

    parser.add_argument("--use_anchoring", action="store_true")
    parser.add_argument("--denoising_strength", type=float, default=0.6)

    parser.add_argument("--return_intermediate", action="store_true")
    parser.add_argument("--save_pt", type=str, default=None)

    parser.add_argument("--data", type=str, nargs="+", default=None)
    parser.add_argument("--data_dir", type=str, default="data_samples")
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--enable_specflow", action="store_true")
    parser.add_argument("--rho_text_target", type=float, default=0.7)
    parser.add_argument("--rho_vis_target", type=float, default=0.4)
    parser.add_argument("--specflow_tau", type=float, default=0.5)
    parser.add_argument("--specflow_lambda_ratio", type=float, default=1.0)
    parser.add_argument("--specflow_lambda_soft", type=float, default=1.0)
    parser.add_argument("--specflow_lambda_hard", type=float, default=1.0)
    parser.add_argument("--specflow_prefix_kappa", type=int, default=16)

    args = parser.parse_args()

    if args.mode == "standalone_specflow":
        if args.velocity_ckpt is None:
            raise ValueError("--velocity_ckpt is required for standalone_specflow mode")
        run_standalone_specflow(args)
    else:
        run_attach_to_llm(args)