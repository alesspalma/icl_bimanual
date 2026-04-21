"""
VLM baseline for bimanual ICL demonstrations.

Instead of text-based object positions (LLM baseline) or DINO keypoints (KAT
baseline), this module uses *images* as observations.  Each ICL example pairs
a concatenated image (front RGB | front depth, side-by-side) of the initial
scene with the discretised bimanual action trajectory.

Demonstrations are stored as **JSON** files (one per batch of 10 episodes),
where each entry contains:
    - ``image_path``: absolute path to the pre-built combined PNG
    - ``actions``:    list of 14-dim discrete action vectors (7 right + 7 left)

At test time, the task handler builds multimodal OpenAI-style content blocks
(interleaved images and text) that can be sent to a VLM backend (e.g.
Qwen-2.5 VL via vLLM, or GPT-5 via the OpenAI API).
"""

import argparse
import base64
import glob
import io
import json
import os
import pickle
import random
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

from icl_utils import (
    _image_to_float_array,
    normalize_quaternion,
    point_to_voxel_index,
    quaternion_to_discrete_euler,
    CAMERAS,
    IMAGE_SIZE,
)
from helpers.demo_loading_utils import keypoint_discovery

ROOT = "/media/nvme/palma/icl_bimanual/generated_data/train"  # TODO: change this

DEPTH_SCALE = 2 ** 24 - 1  # 24-bit depth encoding used by RLBench

# ──────────────────── VLM system prompts ────────────────────
SYSTEM_PROMPT_RIGHT = (
    "You are the right arm of a bimanual Franka Panda robot with parallel "
    "grippers. We provide you with some demos: each demo shows an image of "
    "the initial scene (RGB on the left, depth on the right) followed by "
    "a list of discretised actions. Then you will receive a new image and "
    "you need to output a list of actions that matches the trend in the "
    "demos. Do not output anything else."
)

SYSTEM_PROMPT_LEFT = (
    "You are the left arm of a bimanual Franka Panda robot with parallel "
    "grippers. We provide you with some demos: each demo shows an image of "
    "the initial scene (RGB on the left, depth on the right) followed by "
    "a list of discretised actions. Then you will receive a new image and "
    "you need to output a list of actions that matches the trend in the "
    "demos. Do not output anything else."
)

SYSTEM_PROMPT_FOLLOWER = (
    "You are the follower arm of a bimanual Franka Panda robot with parallel grippers. "
    "We provide you with some demos: each demo shows an image of the initial "
    "scene (RGB on the left, depth on the right) together with the leader "
    "arm's actions, followed by the follower arm's actions. Given a new "
    "image and the leader arm's actions, output the follower arm's actions "
    "matching the trend in the demos. Do not output anything else."
)


# ──────────────────── Action helper (same discretisation) ────────────────────
def _get_action(obs_tp1, obs_tm1):
    """Discretise translation, rotation, gripper open."""
    quat = normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = quaternion_to_discrete_euler(quat)
    index = point_to_voxel_index(obs_tp1.gripper_pose[:3])
    rot_and_grip = disc_rot.tolist()
    rot_and_grip.extend([int(obs_tp1.gripper_open)])
    return index.tolist() + rot_and_grip


# ──────────────────── Image helpers ────────────────────
def _depth_to_visualization(depth_raw):
    """Normalise a raw depth array to 0-255 uint8."""
    depth_min, depth_max = depth_raw.min(), depth_raw.max()
    if depth_max - depth_min > 1e-6:
        depth_norm = ((depth_raw - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth_raw, dtype=np.uint8)
    return depth_norm


def _create_combined_image_offline(rgb_path, depth_path, epis_path, frame_idx):
    """
    Build a side-by-side (front RGB | front depth) image from stored episode data.

    The depth PNG is decoded from its 24-bit encoding, converted to metric
    depth via the near/far planes, and then normalised to a grayscale image.
    """
    rgb_img = Image.open(rgb_path).convert("RGB")

    # Decode depth
    with open(os.path.join(epis_path, "low_dim_obs.pkl"), "rb") as f:
        demo = pickle.load(f)
    depth_raw = _image_to_float_array(Image.open(depth_path), DEPTH_SCALE)
    near = demo[frame_idx].misc["front_camera_near"]
    far = demo[frame_idx].misc["front_camera_far"]
    depth_metric = (far - near) * depth_raw + near

    depth_vis = _depth_to_visualization(depth_metric)
    depth_img = Image.fromarray(depth_vis).convert("RGB")

    # Concatenate side by side
    w, h = rgb_img.size
    combined = Image.new("RGB", (w * 2, h))
    combined.paste(rgb_img, (0, 0))
    combined.paste(depth_img, (w, 0))
    return combined


def depth_from_point_cloud(point_cloud_hw3, extrinsics):
    """
    Reconstruct a normalised depth visualisation from a world-frame point cloud.

    Parameters
    ----------
    point_cloud_hw3 : (H, W, 3) point cloud in world frame
    extrinsics      : (4, 4) camera-to-world transform (as used by RLBench/PyRep)

    Returns
    -------
    depth_vis : (H, W) uint8 normalised depth image
    """
    H, W, _ = point_cloud_hw3.shape
    pts = point_cloud_hw3.reshape(-1, 3)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])        # (N, 4)
    world_to_cam = np.linalg.inv(extrinsics)
    pts_cam = (world_to_cam @ pts_h.T).T[:, :3]                 # (N, 3) cam frame
    depth = np.abs(pts_cam[:, 2]).reshape(H, W)                  # Z = depth
    return _depth_to_visualization(depth)


# ──────────────────── Base64 encoding for OpenAI API ────────────────────
def _image_path_to_data_url(image_path):
    """Load an image file and return a ``data:image/png;base64,…`` URL."""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _pil_to_data_url(pil_image):
    """Convert a PIL image to a ``data:image/png;base64,…`` URL."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ──────────────────── Task handler (VLM version) ────────────────────
class base_task_handler_vlm:
    """
    VLM variant of the task handler.

    Observations are images (front RGB + front depth concatenated side-by-side)
    rather than text-based object positions or keypoints.
    """

    def __init__(self):
        self.save_root = os.path.join(ROOT, type(self).__name__)
        self.num_demos = 10
        print(f"[VLM] Task handler {type(self).__name__} "
              f"using demonstrations from {self.save_root}")

    # ── Test-time prompt construction ──────────────────────────────────
    def get_user_prompt(self, combined_image_pil, agent):
        """
        Build multimodal content blocks for the VLM LeaderFollower agent.

        Parameters
        ----------
        combined_image_pil : PIL.Image
            The live query image (front RGB | depth, side-by-side).
        agent : VLMLeaderFollower
            The calling agent (used to read ``agent.leader``).

        Returns
        -------
        leader_content     : list[dict]
            Interleaved ``image_url`` / ``text`` blocks for the leader prompt.
        follower_examples  : list[dict]
            Per-demo dicts (``image_path``, ``leader_actions``,
            ``follower_actions``) for building the follower prompt later.
        """
        assert os.path.exists(self.save_root), \
            f"Cannot find save root {self.save_root}"

        # Pick a random batch of pre-saved demonstrations
        demo_files = glob.glob(
            os.path.join(self.save_root, "vlm_demonstrations", "*.json"))
        path = random.choice(demo_files)
        with open(path, "r") as f:
            demos = json.load(f)

        leader_is_right = agent.leader == "right"

        leader_content: List[dict] = []
        follower_examples: List[dict] = []

        for demo_entry in demos:
            img_path = demo_entry["image_path"]
            actions_bi = demo_entry["actions"]
            right_actions = [a[:7] for a in actions_bi]
            left_actions = [a[7:] for a in actions_bi]
            leader_actions = right_actions if leader_is_right else left_actions
            follower_actions = left_actions if leader_is_right else right_actions

            # Image block
            leader_content.append({
                "type": "image_url",
                "image_url": {"url": _image_path_to_data_url(img_path)},
            })
            # Trajectory block
            leader_content.append({
                "type": "text",
                "text": f">{json.dumps(leader_actions)}",
            })

            follower_examples.append({
                "image_path": img_path,
                "leader_actions": leader_actions,
                "follower_actions": follower_actions,
            })

        # Query image (live)
        leader_content.append({
            "type": "image_url",
            "image_url": {"url": _pil_to_data_url(combined_image_pil)},
        })
        leader_content.append({"type": "text", "text": ">"})

        return leader_content, follower_examples

    def build_follower_prompt(self, combined_image_pil, follower_examples,
                              leader_prediction):
        """
        Build the follower multimodal content after the leader has predicted.

        Each ICL example shows: image + leader trajectory → follower trajectory.
        The query shows: live image + leader prediction → (to be predicted).
        """
        follower_content: List[dict] = []

        for ex in follower_examples:
            follower_content.append({
                "type": "image_url",
                "image_url": {"url": _image_path_to_data_url(ex["image_path"])},
            })
            follower_content.append({
                "type": "text",
                "text": (f"Leader: {json.dumps(ex['leader_actions'])}\n"
                         f">{json.dumps(ex['follower_actions'])}"),
            })

        # Query with leader prediction
        follower_content.append({
            "type": "image_url",
            "image_url": {"url": _pil_to_data_url(combined_image_pil)},
        })
        follower_content.append({
            "type": "text",
            "text": f"Leader: {json.dumps(leader_prediction)}\n>",
        })

        return follower_content

    # ── Offline demonstration generation ──────────────────────────────
    def save_in_context_demonstrations(self):
        """
        Pre-compute and save VLM-style in-context demonstration batches.

        For each of the first 100 episodes:
            1. Concatenate front RGB + front depth (frame 0) into one image.
            2. Extract keyframe actions (same discretisation as the LLM baseline).
            3. Save the combined image to ``vlm_images/``.

        Batches of ``self.num_demos`` episodes are written as JSON files to
        ``vlm_demonstrations/``.
        """
        task_root = os.path.join(
            ROOT, type(self).__name__, "all_variations", "episodes")

        all_demos = []
        num_episodes = 100

        for epi_id in tqdm(range(num_episodes), desc=type(self).__name__):
            epis_path = os.path.join(task_root, f"episode{epi_id}")
            if not os.path.exists(epis_path):
                continue

            with open(os.path.join(epis_path, "low_dim_obs.pkl"), "rb") as f:
                demo = pickle.load(f)
            with open(os.path.join(epis_path, "variation_number.pkl"), "rb") as f:
                demo.variation_number = pickle.load(f)

            episode_keypoints = keypoint_discovery(demo)

            # Build combined image from first frame
            rgb_path = os.path.join(epis_path, "front_rgb", "rgb_0000.png")
            depth_path = os.path.join(epis_path, "front_depth", "depth_0000.png")

            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                print(f"Skipping episode {epi_id}: missing front images")
                continue

            combined = _create_combined_image_offline(
                rgb_path, depth_path, epis_path, 0)

            # Save combined image
            img_dir = os.path.join(self.save_root, "vlm_images")
            os.makedirs(img_dir, exist_ok=True)
            combined_path = os.path.join(img_dir, f"episode{epi_id}.png")
            combined.save(combined_path)

            # Extract keyframe actions
            actions = []
            for keypoint in episode_keypoints:
                obs_tp1 = demo[keypoint]
                action_right = _get_action(obs_tp1.right, obs_tp1.right)
                action_left = _get_action(obs_tp1.left, obs_tp1.left)
                actions.append(action_right + action_left)

            all_demos.append({
                "episode_id": epi_id,
                "image_path": combined_path,
                "actions": actions,
            })

        # Batch into groups of num_demos and save as JSON
        demo_dir = os.path.join(self.save_root, "vlm_demonstrations")
        os.makedirs(demo_dir, exist_ok=True)
        for i, start_idx in enumerate(range(0, len(all_demos), self.num_demos)):
            if start_idx + self.num_demos <= len(all_demos):
                batch = all_demos[start_idx : start_idx + self.num_demos]
                with open(os.path.join(demo_dir, f"{i}.json"), "w") as f:
                    json.dump(batch, f)

        print(f"[VLM] Saved {len(all_demos)} demos for {type(self).__name__}")


# ──────────────────── Concrete task classes ────────────────────
# Each class name must match its task directory under generated_data/train/

class bimanual_push_box(base_task_handler_vlm):
    pass

class bimanual_dual_push_buttons(base_task_handler_vlm):
    pass

class bimanual_put_bottle_in_fridge(base_task_handler_vlm):
    pass

class bimanual_handover_item(base_task_handler_vlm):
    pass

class bimanual_handover_item_easy(base_task_handler_vlm):
    pass

class bimanual_lift_ball(base_task_handler_vlm):
    pass

class bimanual_lift_tray(base_task_handler_vlm):
    pass

class bimanual_pick_laptop(base_task_handler_vlm):
    pass

class bimanual_pick_plate(base_task_handler_vlm):
    pass

class bimanual_straighten_rope(base_task_handler_vlm):
    pass

class bimanual_sweep_to_dustpan(base_task_handler_vlm):
    pass

class bimanual_take_tray_out_of_oven(base_task_handler_vlm):
    pass

class bimanual_put_item_in_drawer(base_task_handler_vlm):
    pass


task_name_to_handler = {
    "bimanual_handover_item": bimanual_handover_item,
    "bimanual_dual_push_buttons": bimanual_dual_push_buttons,
    "bimanual_handover_item_easy": bimanual_handover_item_easy,
    "bimanual_lift_ball": bimanual_lift_ball,
    "bimanual_lift_tray": bimanual_lift_tray,
    "bimanual_pick_laptop": bimanual_pick_laptop,
    "bimanual_pick_plate": bimanual_pick_plate,
    "bimanual_push_box": bimanual_push_box,
    "bimanual_put_bottle_in_fridge": bimanual_put_bottle_in_fridge,
    "bimanual_straighten_rope": bimanual_straighten_rope,
    "bimanual_sweep_to_dustpan": bimanual_sweep_to_dustpan,
    "bimanual_take_tray_out_of_oven": bimanual_take_tray_out_of_oven,
    "bimanual_put_item_in_drawer": bimanual_put_item_in_drawer,
}


def create_task_handler(task_name):
    return task_name_to_handler[task_name]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate VLM-style in-context learning demonstrations"
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task names to process (default: all bimanual tasks)",
    )
    args = parser.parse_args()

    tasks = args.tasks if args.tasks else list(task_name_to_handler.keys())
    for task_name in tasks:
        print(f"\n{'=' * 60}")
        print(f"Processing task: {task_name}")
        print(f"{'=' * 60}")
        handler = task_name_to_handler[task_name]()
        handler.save_in_context_demonstrations()
