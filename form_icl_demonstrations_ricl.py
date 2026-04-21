"""
Prepare RLBench bimanual episodes in RICL's ``processed_demo.npz`` format.

For each task this script creates **two** demo pools — one per arm — so that
the RICL server can be launched separately for each arm:

    preprocessing/collected_demos/{date}_{task}_right_arm/demo_N/processed_demo.npz
    preprocessing/collected_demos/{date}_{task}_left_arm/demo_N/processed_demo.npz

Each ``processed_demo.npz`` contains:

    state               (T, 8)  float32  — joint_positions (7) + gripper_open (1)
    actions             (T, 8)  float32  — joint_velocities (7) + gripper_open (1)
    top_image           (T, 224, 224, 3) uint8
    right_image         (T, 224, 224, 3) uint8
    wrist_image         (T, 224, 224, 3) uint8
    top_image_embeddings   (T, EMBED_DIM) float32
    right_image_embeddings (T, EMBED_DIM) float32
    wrist_image_embeddings (T, EMBED_DIM) float32
    prompt              str

Environment
-----------
This script must be run with the **openpi venv** (``ricl_openpi/.venv``), NOT
the ``icl_bimanual`` conda env.  The openpi venv provides ``openpi``,
``openpi_client``, and the DINOv2 torch model.  RLBench/PyRep are loaded via
direct file-path imports (bypassing their compiled C extensions), so they do
not need to be installed in this venv.

Usage
-----
    # from the project root:
    source ricl_openpi/.venv/bin/activate
    python form_icl_demonstrations_ricl.py [--tasks bimanual_lift_tray ...]

    # or for a quick single-task test:
    source ricl_openpi/.venv/bin/activate
    python form_icl_demonstrations_ricl.py --tasks bimanual_lift_tray --num_episodes 1
"""

import argparse
import os
import pickle
import sys
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

# --------------- path setup ------------------------------------------------
import importlib.util as _iutil
import types as _types

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RICL_REPO = os.path.join(PROJECT_ROOT, "ricl_openpi")

# RICL imports (openpi + openpi_client)
sys.path.insert(0, os.path.join(RICL_REPO, "src"))
sys.path.insert(0, os.path.join(RICL_REPO, "packages", "openpi-client", "src"))

# RLBench is needed to unpickle low_dim_obs.pkl.  The pickle references only:
#   rlbench.backend.observation.{BimanualObservation,UnimanualObservationData}
#   rlbench.demo.Demo
# Both source files only depend on numpy/stdlib — they never touch pyrep.
# We load them directly by file path, bypassing rlbench/__init__.py (which would
# otherwise fail because it tries to import the compiled PyRep C extension).
_RLBENCH_SRC = os.path.join(PROJECT_ROOT, "RLBench")

# Create lightweight stub packages so Python resolves the dotted names correctly
# without executing any __init__.py.
for _pkg in ["rlbench", "rlbench.backend"]:
    if _pkg not in sys.modules:
        _stub = _types.ModuleType(_pkg)
        _stub.__path__ = [os.path.join(_RLBENCH_SRC, *_pkg.split("."))]
        _stub.__package__ = _pkg
        sys.modules[_pkg] = _stub

# Load the two leaf modules directly from their .py files.
for _dotted in ["rlbench.backend.observation", "rlbench.demo"]:
    if _dotted not in sys.modules:
        _path = os.path.join(_RLBENCH_SRC, *_dotted.split(".")) + ".py"
        _spec = _iutil.spec_from_file_location(_dotted, _path)
        _mod = _iutil.module_from_spec(_spec)
        sys.modules[_dotted] = _mod
        _spec.loader.exec_module(_mod)

from openpi_client.image_tools import resize_with_pad
from openpi.policies.utils import embed_with_batches, load_dinov2, EMBED_DIM

# --------------- paths --------------------------------------------------------
ROOT = "/media/nvme/palma/icl_bimanual/generated_data/train"
RICL_DEMOS_ROOT = os.path.join(RICL_REPO, "preprocessing", "collected_demos")
DATE_PREFIX = datetime.now().strftime("%Y-%m-%d")
NUM_EPISODES = 100  # number of demos per task (used as RICL retrieval pool)

# Camera mapping: RLBench camera name → RICL camera slot
# Right arm uses right-side cameras, left arm uses left-side cameras.
CAMERA_MAP = {
    "right": {
        "top": "front",
        "right": "over_shoulder_right",
        "wrist": "wrist_right",
    },
    "left": {
        "top": "front",
        "right": "over_shoulder_left",
        "wrist": "wrist_left",
    },
}


def _load_images(epis_path, camera_name, num_timesteps):
    """Load all RGB images for a camera from disk and resize to 224×224."""
    imgs = []
    for t in range(num_timesteps):
        path = os.path.join(epis_path, f"{camera_name}_rgb", f"rgb_{t:04d}.png")
        img = np.array(Image.open(path).convert("RGB"))
        imgs.append(img)
    imgs = np.stack(imgs)  # (T, H, W, 3)
    imgs = resize_with_pad(imgs, 224, 224)  # (T, 224, 224, 3)
    return imgs


def prepare_single_episode(epis_path, arm, dinov2, prompt):
    """
    Convert one RLBench episode into a ``processed_demo`` dict for one arm.

    Parameters
    ----------
    epis_path : str   — path to the episode directory
    arm       : str   — "right" or "left"
    dinov2    : model  — loaded DINOv2 model for embedding
    prompt    : str   — language instruction

    Returns
    -------
    dict with keys matching RICL's processed_demo.npz format
    """
    with open(os.path.join(epis_path, "low_dim_obs.pkl"), "rb") as f:
        demo = pickle.load(f)

    T = len(demo)
    cam_map = CAMERA_MAP[arm]

    # ── State & actions ──────────────────────────────────────────────
    states = np.zeros((T, 8), dtype=np.float32)
    actions = np.zeros((T, 8), dtype=np.float32)

    for t in range(T):
        obs_arm = demo[t].right if arm == "right" else demo[t].left
        joint_pos = obs_arm.joint_positions  # (7,)
        joint_vel = obs_arm.joint_velocities  # (7,)
        gripper_open = float(obs_arm.gripper_open)  # 0.0 or 1.0

        states[t] = np.concatenate([joint_pos, [gripper_open]])
        actions[t] = np.concatenate([joint_vel, [gripper_open]])

    # ── Images ───────────────────────────────────────────────────────
    top_images = _load_images(epis_path, cam_map["top"], T)
    right_images = _load_images(epis_path, cam_map["right"], T)
    wrist_images = _load_images(epis_path, cam_map["wrist"], T)

    # ── DINOv2 embeddings ────────────────────────────────────────────
    top_emb = embed_with_batches(top_images, dinov2)
    right_emb = embed_with_batches(right_images, dinov2)
    wrist_emb = embed_with_batches(wrist_images, dinov2)

    assert top_emb.shape == (T, EMBED_DIM), f"{top_emb.shape=}"
    assert right_emb.shape == (T, EMBED_DIM), f"{right_emb.shape=}"
    assert wrist_emb.shape == (T, EMBED_DIM), f"{wrist_emb.shape=}"

    return {
        "state": states,
        "actions": actions,
        "top_image": top_images,
        "right_image": right_images,
        "wrist_image": wrist_images,
        "top_image_embeddings": top_emb,
        "right_image_embeddings": right_emb,
        "wrist_image_embeddings": wrist_emb,
        "prompt": prompt,
    }


def prepare_task(task_name, dinov2, num_episodes=NUM_EPISODES):
    """Prepare RICL demos for both arms of a bimanual task."""
    task_root = os.path.join(ROOT, task_name, "all_variations", "episodes")

    # Get task prompt from the first episode
    with open(os.path.join(task_root, "episode0", "variation_descriptions.pkl"), "rb") as f:
        descriptions = pickle.load(f)
    prompt = descriptions[0] if descriptions else task_name.replace("_", " ")

    for arm in ["right", "left"]:
        out_dir = os.path.join(
            RICL_DEMOS_ROOT, f"{DATE_PREFIX}_{task_name}_{arm}_arm"
        )
        os.makedirs(out_dir, exist_ok=True)

        for epi_id in tqdm(
            range(num_episodes),
            desc=f"{task_name} ({arm} arm)",
        ):
            epis_path = os.path.join(task_root, f"episode{epi_id}")
            if not os.path.exists(epis_path):
                print(f"  Skipping episode{epi_id} — not found")
                continue

            demo_dir = os.path.join(out_dir, f"demo_{epi_id}")
            npz_path = os.path.join(demo_dir, "processed_demo.npz")
            if os.path.exists(npz_path):
                print(f"  episode{epi_id} ({arm}) already processed — skipping")
                continue

            data = prepare_single_episode(epis_path, arm, dinov2, prompt)

            os.makedirs(demo_dir, exist_ok=True)
            np.savez(npz_path, **data)
            print(f"  Saved {npz_path}  (T={data['state'].shape[0]})")


# ──────────────────── Bimanual task list ──────────────────────────────────────
BIMANUAL_TASKS = [
    "bimanual_handover_item",
    "bimanual_dual_push_buttons",
    "bimanual_handover_item_easy",
    "bimanual_lift_ball",
    "bimanual_lift_tray",
    "bimanual_pick_laptop",
    "bimanual_pick_plate",
    "bimanual_push_box",
    "bimanual_put_bottle_in_fridge",
    "bimanual_straighten_rope",
    "bimanual_sweep_to_dustpan",
    "bimanual_take_tray_out_of_oven",
    "bimanual_put_item_in_drawer",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare RLBench bimanual episodes for RICL"
    )
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="Tasks to process (default: all bimanual tasks)",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=NUM_EPISODES,
        help=f"Number of episodes per task (default: {NUM_EPISODES})",
    )
    args = parser.parse_args()

    tasks = args.tasks if args.tasks else BIMANUAL_TASKS

    print("Loading DINOv2 for image embedding...")
    dinov2 = load_dinov2()

    for task_name in tasks:
        print(f"\n{'=' * 60}")
        print(f"Processing task: {task_name}")
        print(f"{'=' * 60}")
        prepare_task(task_name, dinov2, args.num_episodes)

    print("\nDone!  Demos saved under:", RICL_DEMOS_ROOT)
    print(
        "\nTo serve RICL for a task, start TWO servers:\n"
        f"  # Right arm (port 8000):\n"
        f"  uv run scripts/serve_policy_ricl.py \\\n"
        f"    --port 8000 \\\n"
        f"    policy:checkpoint \\\n"
        f"    --policy.config=pi0_fast_droid_ricl \\\n"
        f"    --policy.dir=pi0_fast_droid_ricl_checkpoint \\\n"
        f"    --policy.demos_dir=preprocessing/collected_demos/"
        f"{DATE_PREFIX}_<TASK>_right_arm\n"
        f"\n  # Left arm (port 8001):\n"
        f"  uv run scripts/serve_policy_ricl.py \\\n"
        f"    --port 8001 \\\n"
        f"    policy:checkpoint \\\n"
        f"    --policy.config=pi0_fast_droid_ricl \\\n"
        f"    --policy.dir=pi0_fast_droid_ricl_checkpoint \\\n"
        f"    --policy.demos_dir=preprocessing/collected_demos/"
        f"{DATE_PREFIX}_<TASK>_left_arm\n"
    )
