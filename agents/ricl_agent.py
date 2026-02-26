"""
RICL (Re-training a VLA for In-Context Learning) agent for RLBench bimanual.

This agent connects to **two** RICL policy servers (one per arm) and converts
the predicted joint-velocity action chunks into end-effector pose targets that
RLBench's ``BimanualMoveArmThenGripper`` action mode can execute.

Architecture
~~~~~~~~~~~~
* RICL is a *single-arm* VLA (Pi0-FAST-DROID) that predicts 15-step action
  chunks of 8 dims (7 joint velocities + 1 gripper position).
* For bimanual control we run **two** server instances — one for
  right-arm demos (port ``ricl_right_port``, default 8000) and one for
  left-arm demos (port ``ricl_left_port``, default 8001).
* At each ``act()`` call the agent:
    1. Extracts camera images + joint/gripper state from the live observation.
    2. Queries both servers via websocket.
    3. Integrates the predicted joint velocities over the chunk to compute a
       target joint configuration.
    4. Uses PyRep FK (temporarily setting joint positions in the simulation)
       to obtain the target end-effector pose.
    5. Formats the result as an 18-dim bimanual action
       ``[pos(3), quat(4), grip(1), ign_coll(1)] x 2 arms``.

Environment
~~~~~~~~~~~
This agent is imported and executed by ``main.py``, which runs inside the
**icl_bimanual conda environment**.  Before evaluating, install the
``openpi_client`` websocket library into that env once::

    conda activate icl_bimanual
    pip install -e ricl_openpi/packages/openpi-client

The two RICL policy servers (one per arm) must be started separately using
the **openpi venv** *before* launching the evaluation.

**One-time setup for RTX 5080 (CC 12.0):** the default lockfile ships
``nvidia-cuda-nvcc-cu12==12.6.x``, which is too old for sm_120.  Run once
after ``uv sync``::

    cd ricl_openpi
    uv pip install "nvidia-cuda-nvcc-cu12>=12.8"

Then start **one bimanual server** (one process, shared model, two ports) with
``--no-sync`` so ``uv run`` skips the venv sync and preserves the upgraded
nvcc::

    # Single terminal — bimanual server (both arms, one model in VRAM):
    cd ricl_openpi && source .venv/bin/activate
    uv run --no-sync scripts/serve_policy_ricl_bimanual.py \\
        --right_port 8000 \\
        --left_port  8001 \\
        --config=pi0_fast_droid_ricl \\
        --dir=pi0_fast_droid_ricl_checkpoint \\
        --right_demos_dir=preprocessing/collected_demos/{date}_{task}_right_arm \\
        --left_demos_dir=preprocessing/collected_demos/{date}_{task}_left_arm

    # Terminal C — evaluation (icl_bimanual conda env):
    conda activate icl_bimanual
    xvfb-run -a python main.py method.name=RICLAgent ...

See ``test_ricl.sh`` for the fully automated version.
"""

from typing import List
import json
import os
import sys

import numpy as np
from PIL import Image

from yarr.agents.agent import Agent, Summary, ActResult
from icl_utils import SCENE_BOUNDS, CAMERAS

# ── Lazily-imported heavy deps (avoid import errors when server not needed) ───
_ws_client = None
_image_tools = None


def _ensure_openpi_client():
    global _ws_client, _image_tools
    if _ws_client is not None:
        return
    # Try direct import first, then fall back to repo path
    ricl_pkg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ricl_openpi", "packages", "openpi-client", "src",
    )
    if ricl_pkg not in sys.path:
        sys.path.insert(0, ricl_pkg)
    from openpi_client import websocket_client_policy, image_tools  # noqa: E402
    _ws_client = websocket_client_policy
    _image_tools = image_tools


# ──────────────────── Helpers ────────────────────

def _resize_with_pad(img, h, w):
    """Resize a (H, W, 3) uint8 image to (h, w, 3) with aspect-preserving padding."""
    _ensure_openpi_client()
    return _image_tools.resize_with_pad(img, h, w)


def _extract_rgb(obs, camera):
    """Extract an (H, W, 3) uint8 numpy image from a YARR observation tensor.

    YARR delivers RGB as float32 in [0, 255] (cast directly from uint8 without
    any normalisation), channel-first: (1, 3, H, W).  Convert to channel-last
    (H, W, 3) uint8.  Previously this applied a [-1, 1]→[0, 255] de-norm,
    which saturated nearly every pixel to 255 (all-white images).
    """
    rgb = obs[f"{camera}_rgb"]
    rgb = rgb.squeeze().permute(1, 2, 0).cpu().numpy()  # (H, W, 3) float32 [0,255]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


# ──────────────────── Agent ────────────────────

class RICLAgent(Agent):
    """RICL VLA agent for RLBench bimanual evaluation."""

    def __init__(self, task_name, model_config):
        self.episode_id = -1
        self.device = "cuda"
        self.task_name = task_name
        self.model_config = model_config
        # RICL integration parameters (configurable via model_config)
        self.ricl_right_port = getattr(model_config, "ricl_right_port", 8000)
        self.ricl_left_port = getattr(model_config, "ricl_left_port", 8001)
        self.ricl_host = getattr(model_config, "ricl_host", "127.0.0.1")
        # How many steps of the 15-step velocity chunk to integrate
        self.integration_steps = getattr(model_config, "ricl_integration_steps", 10)
        # dt per integration step (seconds) — tune to match the velocity scale
        self.integration_dt = getattr(model_config, "ricl_integration_dt", 0.1)
        # Task prompt for RICL
        self.task_prompt = task_name.replace("_", " ")
        # PyRep arm handles — no longer needed (using joint positions directly)
        self._right_arm = None
        self._left_arm = None

    # ── Observation → RICL request ────────────────────────────────────

    def _build_request(self, obs, arm):
        """
        Map a YARR observation dict to a RICL websocket request.

        Parameters
        ----------
        obs  : dict — YARR observation
        arm  : str  — "right" or "left"
        """
        # Camera mapping
        if arm == "right":
            top_cam, side_cam, wrist_cam = "front", "over_shoulder_right", "wrist_right"
            joint_pos = obs["right_joint_positions"]
            grip_joints = obs["right_gripper_joint_positions"]
        else:
            top_cam, side_cam, wrist_cam = "front", "over_shoulder_left", "wrist_left"
            joint_pos = obs["left_joint_positions"]
            grip_joints = obs["left_gripper_joint_positions"]

        # Convert tensors to numpy if needed
        if hasattr(joint_pos, "cpu"):
            joint_pos = joint_pos.cpu().numpy().flatten()
        if hasattr(grip_joints, "cpu"):
            grip_joints = grip_joints.cpu().numpy().flatten()

        # Normalise gripper to [0, 1] (RLBench finger range is [0, 0.04])
        gripper_norm = float(np.clip(np.mean(grip_joints) / 0.04, 0.0, 1.0))

        # State: (8,) = joint_pos (7) + gripper (1)
        state = np.concatenate([joint_pos, [gripper_norm]]).astype(np.float64)

        # Images: extract, resize to 224×224
        top_img = _resize_with_pad(_extract_rgb(obs, top_cam), 224, 224)
        side_img = _resize_with_pad(_extract_rgb(obs, side_cam), 224, 224)
        wrist_img = _resize_with_pad(_extract_rgb(obs, wrist_cam), 224, 224)

        return {
            "query_top_image": top_img,
            "query_right_image": side_img,
            "query_wrist_image": wrist_img,
            "query_state": state,
            "query_prompt": self.task_prompt,
            "prefix": f"{self.task_name}_{arm}",
        }

    # ── Velocity chunk → joint positions ─────────────────────────────

    def _velocity_chunk_to_joint_pos(self, action_chunk, obs, arm):
        """
        Integrate predicted joint velocities to obtain target joint positions.

        The RICL model predicts 15-step joint-velocity chunks (8 dims each:
        7 joint velocities + 1 gripper position).  We integrate a configurable
        number of those steps at a fixed dt to get an absolute target joint
        configuration.  The caller then commands that configuration via
        BimanualJointPosition, which uses a PD controller — crucially, the PD
        controller never penetrates objects kinematically, so objects cannot be
        "bolted away" by path-planning sweeps.

        Parameters
        ----------
        action_chunk : (N, 8) — joint_vel (7) + gripper (1)
        obs          : dict   — YARR observation
        arm          : str    — "right" or "left"

        Returns
        -------
        target_joint_pos : (7,) — absolute joint positions to command
        gripper          : float — binarised gripper state (0 or 1)
        """
        # Current joint positions from observation
        key = f"{arm}_joint_positions"
        joint_pos = obs[key]
        if hasattr(joint_pos, "cpu"):
            joint_pos = joint_pos.cpu().numpy().flatten()
        joint_pos = joint_pos.astype(np.float64).copy()

        # Integrate the first n_steps velocity predictions
        n_steps = min(self.integration_steps, action_chunk.shape[0])
        dt = self.integration_dt
        for t in range(n_steps):
            joint_pos += action_chunk[t, :7] * dt

        # Gripper: average over integrated steps, binarise
        avg_gripper = float(np.mean(action_chunk[:n_steps, 7]))
        gripper_open = 1.0 if avg_gripper > 0.5 else 0.0

        return joint_pos, gripper_open

    # ── Main inference ────────────────────────────────────────────────

    def _preprocess(self, obs, step, **kwargs):
        # Save RGBs for logging
        for camera in CAMERAS:
            rgb_img = _extract_rgb(obs, camera)
            img = Image.fromarray(rgb_img)
            rgb_dir = os.path.join(
                self.savedir, "rgb_dir", camera, str(self.episode_id)
            )
            os.makedirs(rgb_dir, exist_ok=True)
            img.save(os.path.join(rgb_dir, f"{self.step}.png"))

        if len(self.actions) != 0:
            return  # still executing cached actions

        # ── Query RICL for both arms ──────────────────────────────
        right_req = self._build_request(obs, "right")
        left_req = self._build_request(obs, "left")

        right_result = self.right_client.infer(right_req)
        right_chunk = right_result["query_actions"]  # (15, 8)

        left_result = self.left_client.infer(left_req)
        left_chunk = left_result["query_actions"]  # (15, 8)

        print(f"[RICL] Right chunk shape: {right_chunk.shape}, "
              f"Left chunk shape: {left_chunk.shape}")

        # ── Integrate velocities → target joint positions ─────────
        right_jpos, right_grip = self._velocity_chunk_to_joint_pos(
            right_chunk, obs, "right"
        )
        left_jpos, left_grip = self._velocity_chunk_to_joint_pos(
            left_chunk, obs, "left"
        )

        # ── Format as bimanual joint-position action ──────────────
        # BimanualMoveArmThenGripper with BimanualJointPosition expects:
        #   [right_q0..q6 (7), right_gripper (1), right_ignore_coll (1),
        #    left_q0..q6  (7), left_gripper  (1), left_ignore_coll  (1)]  = 18
        continuous_action = np.concatenate([
            right_jpos.astype(np.float64),  # right joint positions (7)
            [right_grip],                    # right gripper
            [0.0],                           # right ignore_collisions = False
            left_jpos.astype(np.float64),   # left joint positions (7)
            [left_grip],                     # left gripper
            [0.0],                           # left ignore_collisions = False
        ])

        self.actions.append(continuous_action)

    # ── YARR Agent interface ──────────────────────────────────────────

    def act(self, step: int, observation: dict,
            deterministic=False, **kwargs) -> ActResult:
        self._preprocess(observation, step, **kwargs)

        continuous_action = self.actions.pop(0)
        self.step += 1

        copy_obs = {k: v.cpu() for k, v in observation.items()}
        return ActResult(
            continuous_action, observation_elements=copy_obs, info=None
        )

    def act_summaries(self) -> List[Summary]:
        return []

    def reset(self):
        super().reset()
        self.step = 0
        self.episode_id += 1
        self._prev_action = None
        self.actions = []

    def load_weights(self, savedir: str):
        _ensure_openpi_client()
        self.savedir = savedir

        print(f"[RICL] Connecting to right-arm server at "
              f"{self.ricl_host}:{self.ricl_right_port}")
        self.right_client = _ws_client.WebsocketClientPolicy(
            host=self.ricl_host, port=self.ricl_right_port,
        )

        print(f"[RICL] Connecting to left-arm server at "
              f"{self.ricl_host}:{self.ricl_left_port}")
        self.left_client = _ws_client.WebsocketClientPolicy(
            host=self.ricl_host, port=self.ricl_left_port,
        )

        print("[RICL] Connected to both servers.")

    def build(self, training: bool, device=None):
        return

    def update(self, step: int, replay_sample: dict) -> dict:
        return {}

    def update_summaries(self) -> List[Summary]:
        return []

    def save_weights(self, savedir: str):
        return
