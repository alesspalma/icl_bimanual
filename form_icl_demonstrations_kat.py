"""
KAT (Keypoint Action Tokens) baseline for bimanual ICL demonstrations.

Instead of using per-object mask-based 3D center positions as observations,
this module uses DINO-ViT feature correspondences to extract keypoints from
RGB images and projects them to 3D using depth maps and camera parameters.

The keyframe extraction mechanism is the same as in form_icl_demonstrations.py.
"""
import argparse
import copy
import glob
import itertools
import os
import pickle
import random
import string
import sys
from typing import List

import numpy as np
from PIL import Image
from pyrep.objects import VisionSensor
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import json
import torch

from icl_utils import (
    _image_to_float_array,
    normalize_quaternion,
    point_to_voxel_index,
    quaternion_to_discrete_euler,
    CAMERAS,
    IMAGE_SIZE,
)
from helpers.demo_loading_utils import keypoint_discovery

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dino-vit-features'))
import keypoint_utils

ROOT = "/media/nvme/palma/icl_bimanual/generated_data/train"  # TODO: change this

# KAT-style system prompts — observations are keypoint coordinates, not named objects
SYSTEM_PROMPT = (
    "You are a bimanual Franka Panda robot with parallel grippers. "
    "We provide you with some demos in the format of observation>[action_1, action_2, ...]. "
    "Each observation is a list of 3D keypoint coordinates extracted from the scene. "
    "Then you will receive a new observation and you need to output a list of actions "
    "that matches the trend in the demos. Do not output anything else."
)
SYSTEM_PROMPT_RIGHT = (
    "You are the right arm of a bimanual Franka Panda robot with parallel grippers. "
    "We provide you with some demos in the format of observation>[action_1, action_2, ...]. "
    "Each observation is a list of 3D keypoint coordinates extracted from the scene. "
    "Then you will receive a new observation and you need to output a list of actions "
    "that matches the trend in the demos. Do not output anything else."
)
SYSTEM_PROMPT_LEFT = (
    "You are the left arm of a bimanual Franka Panda robot with parallel grippers. "
    "We provide you with some demos in the format of observation>[action_1, action_2, ...]. "
    "Each observation is a list of 3D keypoint coordinates extracted from the scene. "
    "Then you will receive a new observation and you need to output a list of actions "
    "that matches the trend in the demos. Do not output anything else."
)


# ──────────────────── KAT configuration ────────────────────
NUM_KEYPOINTS = 15          # number of DINO keypoints to extract
KAT_LOAD_SIZE = 480         # DINO ViT expects square images at this resolution
KAT_CAMERA = "front"        # camera used for keypoint extraction
DEPTH_SCALE = 2 ** 24 - 1   # depth encoding scale used by RLBench


# ──────────────────── Action helpers (same as original) ────────────────────
def _get_action(obs_tp1, obs_tm1):
    """Discretize translation, rotation, gripper open."""
    quat = normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = quaternion_to_discrete_euler(quat)
    trans_indicies = []
    index = point_to_voxel_index(obs_tp1.gripper_pose[:3])
    trans_indicies.extend(index.tolist())
    rot_and_grip_indicies = disc_rot.tolist()
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return trans_indicies + rot_and_grip_indicies


# ──────────────────── Depth / point-cloud helpers ────────────────────
def _get_depth(epis_path, idx, camera):
    """Load the depth map for a given camera and frame index, converted to metric depth."""
    with open(os.path.join(epis_path, "low_dim_obs.pkl"), "rb") as f:
        demo = pickle.load(f)
    cam_depth = _image_to_float_array(
        Image.open(os.path.join(epis_path, f"{camera}_depth", f"depth_{idx:04d}.png")),
        DEPTH_SCALE,
    )
    near = demo[idx].misc[f"{camera}_camera_near"]
    far = demo[idx].misc[f"{camera}_camera_far"]
    cam_depth = (far - near) * cam_depth + near
    return cam_depth


def _get_camera_params(epis_path, idx, camera):
    """Return (extrinsics, intrinsics) for the given camera and frame."""
    with open(os.path.join(epis_path, "low_dim_obs.pkl"), "rb") as f:
        demo = pickle.load(f)
    extrinsics = demo[idx].misc[f"{camera}_camera_extrinsics"]
    intrinsics = demo[idx].misc[f"{camera}_camera_intrinsics"]
    return extrinsics, intrinsics


def _get_rgb_path(epis_path, idx, camera):
    """Return the path to the RGB image for a camera at a given frame."""
    return os.path.join(epis_path, f"{camera}_rgb", f"rgb_{idx:04d}.png")


# ──────────────────── 2-D to 3-D projection ────────────────────
def _project_keypoints_to_3d(pixel_coords, depth, extrinsics, intrinsics, img_h, img_w):
    """
    Project a set of 2-D pixel keypoints into world-frame 3-D coordinates.

    Parameters
    ----------
    pixel_coords : list of (y_pixel, x_pixel) in the KAT_LOAD_SIZE space
    depth        : (H, W) metric depth map at native resolution
    extrinsics   : 4x4 camera extrinsic matrix (world ← camera)
    intrinsics   : 3x3 camera intrinsic matrix
    img_h, img_w : native image resolution (before DINO resizing)

    Returns
    -------
    world_points : list of [x, y, z] in world frame
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    depth_h, depth_w = depth.shape[:2]

    world_points = []
    for (y_kat, x_kat) in pixel_coords:
        # Map from KAT_LOAD_SIZE coordinates back to native image coordinates
        y_native = int(round(y_kat * (depth_h / KAT_LOAD_SIZE)))
        x_native = int(round(x_kat * (depth_w / KAT_LOAD_SIZE)))
        y_native = np.clip(y_native, 0, depth_h - 1)
        x_native = np.clip(x_native, 0, depth_w - 1)

        d = depth[y_native, x_native]
        # Back-project to camera frame
        z_cam = d
        x_cam = (x_native - cx) * z_cam / fx
        y_cam = (y_native - cy) * z_cam / fy

        # Camera frame → world frame
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        point_world = extrinsics @ point_cam
        world_points.append(point_world[:3].tolist())

    return world_points


# ──────────────────── DINO descriptor extraction ────────────────────
def _extract_kat_descriptors(epis_path_0, epis_path_1):
    """
    Extract DINO-ViT descriptors from the kat_init.jpg images in two episodes.

    These dedicated initialization images are only used for descriptor
    computation. All subsequent keypoint extraction uses the front camera RGB.

    Returns the descriptor_vectors and num_patches needed to find
    correspondences in any subsequent image.
    """
    img_path_0 = os.path.join(epis_path_0, "kat_init.jpg")
    img_path_1 = os.path.join(epis_path_1, "kat_init.jpg")

    assert os.path.exists(img_path_0), f"kat_init.jpg not found in {epis_path_0}"
    assert os.path.exists(img_path_1), f"kat_init.jpg not found in {epis_path_1}"

    patches_xy, desc1, descriptor_vectors, num_patches = keypoint_utils.extract_descriptors(
        img_path_0, img_path_1,
        num_pairs=NUM_KEYPOINTS,
        load_size=KAT_LOAD_SIZE,
    )
    return descriptor_vectors, num_patches


def _extract_keypoints_2d(image_path, descriptor_vectors, num_patches):
    """
    Given pre-computed descriptor vectors, find the corresponding 2-D keypoints
    in a new image.

    Returns lists (ys, xs) in the KAT_LOAD_SIZE pixel space.
    """
    desc_maps, _ = keypoint_utils.extract_desc_maps(image_path, load_size=KAT_LOAD_SIZE)
    ys, xs = keypoint_utils.extract_descriptor_nn(
        descriptor_vectors, desc_maps[0], num_patches, return_heatmaps=False,
    )
    return ys, xs


# ──────────────────── Observation formation ────────────────────
def _form_obs_kat(epis_path, frame_idx, descriptor_vectors, num_patches, camera=KAT_CAMERA):
    """
    Build a KAT-style observation: a list of discretized 3-D keypoint positions.

    Steps:
        1. Extract 2-D keypoints via DINO NN matching on the camera RGB
        2. Load depth + camera params for the same camera
        3. Project keypoints to 3-D world coordinates
        4. Discretize with point_to_voxel_index (same grid as original project)

    Returns a string representation of the keypoints list, e.g.
        "[[vx, vy, vz], [vx, vy, vz], ...]"
    """
    img_path = _get_rgb_path(epis_path, frame_idx, camera)
    ys, xs = _extract_keypoints_2d(img_path, descriptor_vectors, num_patches)

    depth = _get_depth(epis_path, frame_idx, camera)
    extrinsics, intrinsics = _get_camera_params(epis_path, frame_idx, camera)
    img_h, img_w = IMAGE_SIZE  # native RLBench image size

    pixel_coords = list(zip(ys, xs))  # (y, x) in KAT_LOAD_SIZE space
    world_points = _project_keypoints_to_3d(
        pixel_coords, depth, extrinsics, intrinsics, img_h, img_w,
    )

    # Discretize to voxel indices (same grid as the original project)
    voxel_keypoints = []
    for wp in world_points:
        voxel = point_to_voxel_index(np.array(wp))
        voxel_keypoints.append(voxel.tolist())

    return str(voxel_keypoints)


def _form_obs_kat_test_pointcloud(rgb_image_np, point_cloud_hw3,
                                  descriptor_vectors, num_patches):
    """
    KAT observation at test-time using a world-frame point cloud.

    At runtime the simulator provides a (H, W, 3) point cloud in world
    coordinates, so we can look up the 3-D position of each 2-D DINO
    keypoint directly — no depth back-projection needed.

    Parameters
    ----------
    rgb_image_np     : (H, W, 3) uint8 numpy array
    point_cloud_hw3  : (H, W, 3) world-frame point cloud
    descriptor_vectors : pre-computed DINO descriptor vectors
    num_patches      : DINO patch grid shape

    Returns
    -------
    Observation string with voxelized 3-D keypoints.
    """
    desc_maps, _ = keypoint_utils.extract_desc_maps(rgb_image_np, load_size=KAT_LOAD_SIZE)
    ys, xs = keypoint_utils.extract_descriptor_nn(
        descriptor_vectors, desc_maps[0], num_patches, return_heatmaps=False,
    )

    img_h, img_w = rgb_image_np.shape[:2]

    voxel_keypoints = []
    for (y_kat, x_kat) in zip(ys, xs):
        # Map from KAT_LOAD_SIZE coordinates back to native image coordinates
        y_native = int(round(y_kat * (img_h / KAT_LOAD_SIZE)))
        x_native = int(round(x_kat * (img_w / KAT_LOAD_SIZE)))
        y_native = np.clip(y_native, 0, img_h - 1)
        x_native = np.clip(x_native, 0, img_w - 1)

        world_point = point_cloud_hw3[y_native, x_native]
        voxel = point_to_voxel_index(world_point)
        voxel_keypoints.append(voxel.tolist())

    return str(voxel_keypoints)


# ──────────────────── Replay buffer building ────────────────────
def _add_keypoints_to_replay(buffer, frame_idx, demo, episode_keypoints,
                             epis_path, descriptor_vectors, num_patches):
    """
    Append one (observation, actions) pair to the buffer.

    observation : KAT keypoint-based voxelized 3-D positions
    actions     : list of discretized bimanual actions at each keyframe
    """
    obs = _form_obs_kat(epis_path, frame_idx, descriptor_vectors, num_patches)

    buffer.append(obs)

    actions = []
    for keypoint in episode_keypoints:
        obs_tp1 = demo[keypoint]
        action_right = _get_action(obs_tp1.right, obs_tp1.right)
        action_left = _get_action(obs_tp1.left, obs_tp1.left)
        actions.append(action_right + action_left)

    buffer.append(actions)


def get_stored_demos(dataset_root, task_name, amount):
    """
    Load `amount` episodes for `task_name`, extract KAT observations + actions.

    Descriptors are extracted once from the first two episodes and reused for all
    subsequent episodes (as in the original KAT paper).
    """
    task_root = os.path.join(dataset_root, task_name, "all_variations", "episodes")

    # ── 1. Extract DINO descriptors from first two episodes ──
    epis_path_0 = os.path.join(task_root, "episode0")
    epis_path_1 = os.path.join(task_root, "episode1")
    descriptor_vectors, num_patches = _extract_kat_descriptors(epis_path_0, epis_path_1)

    # ── 2. Iterate over episodes ──
    buffer = []
    for epi_id in tqdm(range(amount)):
        epis_path = os.path.join(task_root, f"episode{epi_id}")

        with open(os.path.join(epis_path, "low_dim_obs.pkl"), "rb") as f:
            demo = pickle.load(f)
        with open(os.path.join(epis_path, "variation_number.pkl"), "rb") as f:
            demo.variation_number = pickle.load(f)
        with open(os.path.join(epis_path, "variation_descriptions.pkl"), "rb") as f:
            demo.language_descriptions = pickle.load(f)

        episode_keypoints = keypoint_discovery(demo)

        tmp = []
        _add_keypoints_to_replay(
            tmp, 0, demo, episode_keypoints, epis_path,
            descriptor_vectors, num_patches,
        )
        buffer.append(tmp)

    print("Average number of steps: ",
          sum(len(each[1]) for each in buffer) / len(buffer))
    return buffer, descriptor_vectors, num_patches


# ──────────────────── Task handlers (KAT version) ────────────────────
class base_task_handler_kat:
    """
    KAT variant of the task handler. Observations are DINO keypoints
    instead of named object positions.
    """
    def __init__(self):
        self.save_root = os.path.join(ROOT, type(self).__name__)
        self.num_demos = 10
        self.descriptor_vectors = None
        self.num_patches = None
        print(f"[KAT] Task handler {type(self).__name__} using demonstrations from {self.save_root}")

    def _ensure_descriptors(self):
        """Load pre-computed descriptors from disk, or compute on-the-fly."""
        desc_path = os.path.join(self.save_root, "kat_descriptors.pkl")
        if self.descriptor_vectors is not None:
            return
        if os.path.exists(desc_path):
            with open(desc_path, "rb") as f:
                data = pickle.load(f)
            self.descriptor_vectors = data["descriptor_vectors"]
            self.num_patches = data["num_patches"]
            print(f"[KAT] Loaded descriptors from {desc_path}")
        else:
            # Compute from first two episodes
            task_root = os.path.join(ROOT, type(self).__name__,
                                     "all_variations", "episodes")
            epis0 = os.path.join(task_root, "episode0")
            epis1 = os.path.join(task_root, "episode1")
            self.descriptor_vectors, self.num_patches = _extract_kat_descriptors(epis0, epis1)
            # Save for next time
            os.makedirs(self.save_root, exist_ok=True)
            with open(desc_path, "wb") as f:
                pickle.dump({
                    "descriptor_vectors": self.descriptor_vectors,
                    "num_patches": self.num_patches,
                }, f)
            print(f"[KAT] Computed and saved descriptors to {desc_path}")

    # ── Test-time prompt construction ──
    def get_user_prompt(self, rgb_dict, point_cloud_dict, agent):
        """
        Build the prompt at test time.

        For the KAT baseline the observation comes from DINO keypoints on a single
        camera view (KAT_CAMERA). The 2-D keypoints are mapped to 3-D using
        the world-frame point cloud provided by the simulator.
        """
        self._ensure_descriptors()
        assert os.path.exists(self.save_root), f"Cannot find save root {self.save_root}"

        camera = KAT_CAMERA
        rgb_img = rgb_dict[camera]
        point_cloud = point_cloud_dict[camera]

        obs = _form_obs_kat_test_pointcloud(
            rgb_img, point_cloud,
            self.descriptor_vectors, self.num_patches,
        )

        # Load a random pre-saved demonstration batch
        path = random.choice(
            glob.glob(os.path.join(self.save_root, "kat_demonstrations", "*.txt"))
        )
        demonstration = open(path, "r").read()

        if type(agent).__name__ in ["KATAgentOnePerArm"]:
            # For KAT the separator is ", [[" since observations are lists
            examples = demonstration.split(", [[")
            right_demonstration = ""
            left_demonstration = ""
            for i, example in enumerate(examples):
                if i > 0:
                    example = "[[" + example
                if example.endswith(", "):
                    example = example[:-2]
                parts = example.split(">")
                if len(parts) != 2:
                    continue
                objects_str, actions_str = parts
                actions_list = json.loads(actions_str)
                right_actions = []
                left_actions = []
                for j, action in enumerate(actions_list):
                    if type(agent).__name__ == "OnePerArmDummyContext":
                        right_actions.append(action[:7] + [j] * 7)
                        left_actions.append(action[7:] + [j] * 7)
                    else:
                        right_actions.append(action[:7])
                        left_actions.append(action[7:])
                right_demonstration += objects_str + ">" + str(right_actions) + ", "
                left_demonstration += objects_str + ">" + str(left_actions) + ", "

            return right_demonstration + obs + ">", left_demonstration + obs + ">"

        return demonstration + obs + ">"

    # ── Offline demonstration generation ──
    def save_in_context_demonstrations(self):
        """
        Pre-compute and save KAT-style in-context demonstration batches
        (observations = DINO keypoints, actions = discretized bimanual actions).
        """
        train_demos, descriptor_vectors, num_patches = get_stored_demos(
            ROOT, type(self).__name__, 100,
        )
        self.descriptor_vectors = descriptor_vectors
        self.num_patches = num_patches

        # Persist descriptors
        desc_path = os.path.join(self.save_root, "kat_descriptors.pkl")
        os.makedirs(self.save_root, exist_ok=True)
        with open(desc_path, "wb") as f:
            pickle.dump({
                "descriptor_vectors": descriptor_vectors,
                "num_patches": num_patches,
            }, f)
        print(f"[KAT] Saved descriptors to {desc_path}")

        # Save demonstration batches (groups of num_demos episodes)
        for i, start_idx in enumerate(range(0, len(train_demos), self.num_demos)):
            if start_idx + self.num_demos <= len(train_demos):
                output = ""
                for epi in train_demos[start_idx : start_idx + self.num_demos]:
                    output += f"{epi[0]}>{epi[1]}, "

                d = os.path.join(self.save_root, "kat_demonstrations")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f"{i}.txt"), "w") as f:
                    f.write(output)


# ──────────────────── Concrete task classes ────────────────────
# Each class name matches its task directory under generated_data/train/

class bimanual_push_box(base_task_handler_kat):
    pass

class bimanual_dual_push_buttons(base_task_handler_kat):
    pass

class bimanual_put_bottle_in_fridge(base_task_handler_kat):
    pass

class bimanual_handover_item(base_task_handler_kat):
    pass

class bimanual_handover_item_easy(base_task_handler_kat):
    pass

class bimanual_lift_ball(base_task_handler_kat):
    pass

class bimanual_lift_tray(base_task_handler_kat):
    pass

class bimanual_pick_laptop(base_task_handler_kat):
    pass

class bimanual_pick_plate(base_task_handler_kat):
    pass

class bimanual_straighten_rope(base_task_handler_kat):
    pass

class bimanual_sweep_to_dustpan(base_task_handler_kat):
    pass

class bimanual_take_tray_out_of_oven(base_task_handler_kat):
    pass

class bimanual_put_item_in_drawer(base_task_handler_kat):
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
        description="Generate KAT-style in-context learning demonstrations"
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task names to process (default: all bimanual tasks)",
    )
    parser.add_argument(
        "--num_keypoints",
        type=int,
        default=NUM_KEYPOINTS,
        help=f"Number of DINO keypoints to extract (default: {NUM_KEYPOINTS})",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=KAT_CAMERA,
        help=f"Camera to use for keypoint extraction (default: {KAT_CAMERA})",
    )
    args = parser.parse_args()

    # Allow overriding globals from CLI
    NUM_KEYPOINTS = args.num_keypoints
    KAT_CAMERA = args.camera

    tasks = args.tasks if args.tasks else list(task_name_to_handler.keys())
    for task_name in tasks:
        print(f"\n{'='*60}")
        print(f"Processing task: {task_name}")
        print(f"{'='*60}")
        handler = task_name_to_handler[task_name]()
        handler.save_in_context_demonstrations()
