"""serve_policy_ricl_bimanual.py — single-process dual-arm RICL server.

Why this exists
---------------
A standard RICL server (serve_policy_ricl.py) pre-loads the Pi0-FAST model
(~6 GB bfloat16) and occupies ~77 % of a 16 GB GPU.  Running two independent
servers for bimanual (one per arm) would require ~154 % VRAM — impossible on a
single card.

This script loads the model **once**, builds **two** RiclPolicy instances that
share the same JAX model object (same device arrays, same VRAM), and serves
them on **two ports** concurrently inside a single asyncio event loop.

  Port layout (defaults):
    --right_port  8000  ← right-arm policy (right-arm demos)
    --left_port   8001  ← left-arm policy  (left-arm demos)

The RICLAgent in agents/ricl_agent.py connects to these two ports as before; no
changes are needed on the evaluation side.

Usage (openpi venv, after `uv pip install "nvidia-cuda-nvcc-cu12>=12.8"` once)
-------------------------------------------------------------------------------
    cd ricl_openpi
    uv run --no-sync scripts/serve_policy_ricl_bimanual.py \\
        --right_port 8000 \\
        --left_port  8001 \\
        --config=pi0_fast_droid_ricl \\
        --dir=pi0_fast_droid_ricl_checkpoint \\
        --right_demos_dir=preprocessing/collected_demos/{date}_{task}_right_arm \\
        --left_demos_dir=preprocessing/collected_demos/{date}_{task}_left_arm
"""

import os

# Must be set BEFORE jax is imported.  The default pool allocator pre-grabs
# 75 % of GPU VRAM (~12 GB on a 16 GB card), leaving too little room for the
# first JIT compilation and activation buffers (~6 GB weights + 6 GB workspace).
# 'platform' uses CUDA's native cudaMalloc/cudaFree instead: memory is
# allocated on demand and freed after use, so the JIT workspace never
# competes with a fixed pool ceiling.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import asyncio
import dataclasses
import logging
import socket

import jax.numpy as jnp
import tyro

import openpi.transforms as transforms
from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """Arguments for the bimanual serve_policy script."""

    # Training config name (e.g., "pi0_fast_droid_ricl").
    config: str = "pi0_fast_droid_ricl"
    # Checkpoint directory (e.g., "pi0_fast_droid_ricl_checkpoint").
    dir: str = "pi0_fast_droid_ricl_checkpoint"
    # Demos directory for the right arm.
    right_demos_dir: str = ""
    # Demos directory for the left arm.
    left_demos_dir: str = ""
    # Port to serve the right-arm policy on.
    right_port: int = 8000
    # Port to serve the left-arm policy on.
    left_port: int = 8001
    # Override the lambda parameter that controls ICL vs model interpolation.
    # Formula: weight = exp(-lamda * normalized_distance)
    # Lower lamda → ICL examples dominate even when distance > 0.
    # Config default is 10.0 (aggressive fallback to model); 1.0-3.0 gives much
    # more weight to ICL examples at typical retrieval distances (~0.05-0.15).
    # Set to None to use the value from the training config.
    lamda: float | None = None


def create_shared_model(args: Args):
    """Load model weights and DINOv2 once; return (model, train_config, norm_stats, dinov2)."""
    from openpi.policies.utils import load_dinov2

    train_config = _config.get_config(args.config)
    checkpoint_dir = args.dir

    logging.info("Loading shared model (loaded once for both arms)...")
    model = train_config.model.load(
        _model.restore_params(f"{checkpoint_dir}/params", dtype=jnp.bfloat16)
    )

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(
        f"{checkpoint_dir}/assets", data_config.asset_id
    )

    logging.info("Loading shared DINOv2 (loaded once for both arms)...")
    dinov2 = load_dinov2()
    return model, train_config, norm_stats, dinov2


def create_arm_policy(
    arm: str,
    model,
    train_config,
    norm_stats,
    demos_dir: str,
    dinov2=None,
    lamda_override: float | None = None,
) -> _policy.RiclPolicy:
    """Build a RiclPolicy for one arm, sharing the already-loaded model."""
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    logging.info("Building %s-arm policy (demos: %s)...", arm, demos_dir)
    return _policy.RiclPolicy(
        model,  # ← shared: same JAX arrays, same VRAM allocation
        transforms=[
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.UnnormalizeRicl(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=train_config.policy_metadata,
        demos_dir=demos_dir,
        use_action_interpolation=train_config.model.use_action_interpolation,
        lamda=lamda_override if lamda_override is not None else train_config.model.lamda,
        action_horizon=train_config.model.action_horizon,
        dinov2=dinov2,  # ← shared: one DINOv2 GPU allocation for both arms
    )


def warmup_policy(policy: _policy.RiclPolicy, label: str) -> None:
    """Run one dummy inference to trigger JAX JIT compilation before serving.

    Both arm policies share the same model, so JIT is cached after one warmup.
    The dummy input is taken from the first timestep of the first loaded demo —
    a real observation that exercises the full pipeline (DINOv2 → FAISS →
    tokenize → JAX forward pass).
    """
    import numpy as np

    logging.info("Warming up %s-arm policy (triggers JIT compilation)...", label)
    demo = policy._demos[0]
    obs = {
        "query_top_image":   demo["top_image"][0],
        "query_right_image": demo["right_image"][0],
        "query_wrist_image": demo["wrist_image"][0],
        "query_state":       demo["state"][0],
        "query_prompt":      demo["prompt"].item() if hasattr(demo["prompt"], "item") else str(demo["prompt"]),
        "prefix":            "warmup",
    }
    try:
        policy.infer(obs, debug=False)
        logging.info("%s-arm warmup complete.", label)
    except Exception as exc:
        logging.warning("%s-arm warmup failed (non-fatal): %s", label, exc)


async def run_both_servers(
    right_policy: _policy.RiclPolicy,
    left_policy: _policy.RiclPolicy,
    right_port: int,
    left_port: int,
) -> None:
    """Run two websocket servers concurrently in one event loop."""
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating servers (host: %s, ip: %s)", hostname, local_ip)

    right_server = websocket_policy_server.WebsocketPolicyServer(
        policy=right_policy,
        host="0.0.0.0",
        port=right_port,
        metadata=right_policy.metadata,
    )
    left_server = websocket_policy_server.WebsocketPolicyServer(
        policy=left_policy,
        host="0.0.0.0",
        port=left_port,
        metadata=left_policy.metadata,
    )

    # Both servers run concurrently; requests are handled sequentially per server
    # (the JAX model is not thread-safe, but each server's policy is independent).
    await asyncio.gather(right_server.run(), left_server.run())


def main(args: Args) -> None:
    model, train_config, norm_stats, dinov2 = create_shared_model(args)

    if args.lamda is not None:
        logging.info(
            "Overriding lamda: %.4f → %.4f (ICL weight at d=0.1: %.2f → %.2f)",
            train_config.model.lamda, args.lamda,
            __import__('math').exp(-train_config.model.lamda * 0.1),
            __import__('math').exp(-args.lamda * 0.1),
        )

    right_policy = create_arm_policy(
        "right", model, train_config, norm_stats, args.right_demos_dir,
        dinov2=dinov2, lamda_override=args.lamda,
    )
    left_policy = create_arm_policy(
        "left", model, train_config, norm_stats, args.left_demos_dir,
        dinov2=dinov2, lamda_override=args.lamda,
    )

    logging.info(
        "Serving right-arm on port %d and left-arm on port %d",
        args.right_port,
        args.left_port,
    )

    # Warmup: trigger JAX JIT compilation before accepting real requests.
    # Both policies share the same model so one warmup suffices for JIT caching;
    # we warm up the right arm only (left arm reuses the cached executable).
    warmup_policy(right_policy, "right")

    asyncio.run(run_both_servers(right_policy, left_policy, args.right_port, args.left_port))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
