import argparse
import copy
import glob
import itertools
import os
import pickle
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import numpy as np
from PIL import Image
from pyrep.objects import VisionSensor
from pyrep.objects.object import Object
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import json
import open3d as o3d
from icl_utils import _image_to_float_array, normalize_quaternion, point_to_voxel_index, quaternion_to_discrete_euler, CAMERAS
from helpers.demo_loading_utils import keypoint_discovery
from helpers.utils import visualize_point_cloud_to_file, visualize_point_cloud_live

ROOT = "/media/nvme/palma/icl_bimanual/generated_data/train" # TODO: change this
SYSTEM_PROMPT = "You are a bimanual Franka Panda robot with parallel grippers. We provide you with some demos in the format of observation>[action_1, action_2, ...]. Then you will receive a new observation and you need to output a list of actions that matches the trend in the demos. Do not output anything else."
SYSTEM_PROMPT_RIGHT = "You are the right arm of a bimanual Franka Panda robot with parallel grippers. We provide you with some demos in the format of observation>[action_1, action_2, ...]. Then you will receive a new observation and you need to output a list of actions that matches the trend in the demos. Do not output anything else."
SYSTEM_PROMPT_LEFT = "You are the left arm of a bimanual Franka Panda robot with parallel grippers. We provide you with some demos in the format of observation>[action_1, action_2, ...]. Then you will receive a new observation and you need to output a list of actions that matches the trend in the demos. Do not output anything else."


# discretize translation, rotation, gripper open
def _get_action(
        obs_tp1,
        obs_tm1):
    quat = normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = quaternion_to_discrete_euler(quat)
    trans_indicies = []
    ignore_collisions = int(obs_tm1.ignore_collisions)

    index = point_to_voxel_index(
        obs_tp1.gripper_pose[:3])
    trans_indicies.extend(index.tolist())

    rot_and_grip_indicies = disc_rot.tolist()
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return trans_indicies + rot_and_grip_indicies

def _get_point_cloud_dict(epis_path, idx):
    # This function gets the point cloud using the same operations as PerAct Colab Tutorial
    DEPTH_SCALE = 2**24 - 1
    point_cloud_dict = {}
    for camera_type in CAMERAS:
        with open(os.path.join(epis_path, 'low_dim_obs.pkl'), 'rb') as f:
            demo = pickle.load(f)
        cam_extrinsics = demo[idx].misc[f"{camera_type}_camera_extrinsics"]
        cam_intrinsics = demo[idx].misc[f"{camera_type}_camera_intrinsics"]
        cam_depth = _image_to_float_array(Image.open(os.path.join(epis_path, f"{camera_type}_depth", f"depth_{idx:04d}.png")), DEPTH_SCALE)
        near = demo[idx].misc[f"{camera_type}_camera_near"]
        far = demo[idx].misc[f"{camera_type}_camera_far"]
        cam_depth = (far - near) * cam_depth + near
        point_cloud_dict[camera_type] = VisionSensor.pointcloud_from_depth_and_camera_params(cam_depth, cam_extrinsics, cam_intrinsics) # reconstructed 3D point cloud in world coordinate frame

    return point_cloud_dict

def _get_mask_dict(epis_path, idx):
    mask_dict = {}
    for camera in CAMERAS:
        img = Image.open(os.path.join(epis_path, f"{camera}_mask", f"mask_{idx:04d}.png"))
        rgb_mask = np.array(img, dtype=int)
        mask_dict[camera] = rgb_mask[:, :, 0] + rgb_mask[:, :, 1]*256 + rgb_mask[:, :, 2]*256*256
    return mask_dict

def _get_mask_id_to_name_dict(epis_path, idx):
    with open(os.path.join(epis_path, "low_dim_obs.pkl"), "rb") as f:
        low_dim_obs = pickle.load(f)
    mask_id_to_name_dict = {}
    for camera in CAMERAS:
        mask_id_to_name_dict[camera] = low_dim_obs[idx].misc[f"{camera}_mask_id_to_name"]
    return mask_id_to_name_dict

# add individual data points to replay
def _add_keypoints_to_replay(
        buffer,
        i,
        demo,
        episode_keypoints,
        epis_path_depth,
        epis_path_char,
        sim_name_to_real_name,
        use_gt_pos
    ):
    prev_action = None
    cur_index = i

    mask_dict = _get_mask_dict(epis_path_char, cur_index)

    mask_id_to_sim_name_dict = _get_mask_id_to_name_dict(epis_path_char, cur_index)
    point_cloud_dict = _get_point_cloud_dict(epis_path_depth, cur_index)
    
    mask_id_to_sim_name = {}
    for camera in CAMERAS:
        mask_id_to_sim_name.update(mask_id_to_sim_name_dict[camera]) # unifies the various dicts in mask_id_to_sim_name_dict

    mask_id_to_real_name = {mask_id: sim_name_to_real_name[name] for mask_id, name in mask_id_to_sim_name.items()
                        if name in sim_name_to_real_name}
    
    if len(sim_name_to_real_name) != len(mask_id_to_real_name):
        print(f"WARNING objects not found: {set(sim_name_to_real_name.values()) - set(mask_id_to_real_name.values())}")

    if use_gt_pos:
        avg_coord = form_obs_gt(demo, cur_index, sim_name_to_real_name)
    else:
        # avg_coord = form_obs(mask_dict, mask_id_to_real_name, point_cloud_dict)
        # avg_coord = form_obs_concatenate_clouds(mask_dict, mask_id_to_real_name, point_cloud_dict)
        avg_coord = form_obs_prune_points(mask_dict, mask_id_to_real_name, point_cloud_dict)
    # sort avg_coord alphabetically by key
    avg_coord = str({k: eval(avg_coord)[k] for k in sorted(eval(avg_coord))})
    
    ####### for statistics
    avg_coord_gt = eval(form_obs_gt(demo, cur_index, sim_name_to_real_name))
    avg_coord_std = eval(form_obs(mask_dict, mask_id_to_real_name, point_cloud_dict))
    avg_coord_concat = eval(form_obs_concatenate_clouds(mask_dict, mask_id_to_real_name, point_cloud_dict))
    avg_coord_prune = eval(form_obs_prune_points(mask_dict, mask_id_to_real_name, point_cloud_dict))

    diff_gt_std = {}
    diff_gt_conc = {}
    diff_gt_prune = {}
    for key in avg_coord_gt:
        diff_gt_std[key] = np.linalg.norm(np.array(avg_coord_gt[key]) - np.array(avg_coord_std[key]))
        diff_gt_conc[key] = np.linalg.norm(np.array(avg_coord_gt[key]) - np.array(avg_coord_concat[key]))
        diff_gt_prune[key] = np.linalg.norm(np.array(avg_coord_gt[key]) - np.array(avg_coord_prune[key]))
    #######

    buffer.append(avg_coord) # appends center coordinates of the objects in the scene
    actions = []
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        action_right = _get_action(
            obs_tp1.right,
            obs_tp1.right)
        action_left = _get_action(
            obs_tp1.left,
            obs_tp1.left)

        actions.append(action_right + action_left)
    
    buffer.append(actions) # appends actions taken at each keyframe
    return diff_gt_std, diff_gt_conc, diff_gt_prune

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def _keypoint_discovery(demo, delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or
                        last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    #print('Found %d keypoints.' % len(episode_keypoints), episode_keypoints)
    return episode_keypoints

def get_stored_demos(dataset_root, task_name, amount, sim_name_to_real_name, use_gt_pos):
    total_num_keypoints = 0
    buffer = []
    task_root = os.path.join(dataset_root, task_name, 'all_variations', 'episodes')
    tot_diff_gt_std, tot_diff_gt_conc, tot_diff_gt_prune = {}, {}, {}

    for epi_id in tqdm(range(amount)):
        epis_path_depth = os.path.join(task_root, f'episode{epi_id}')
        epis_path_char = os.path.join(task_root, f'episode{epi_id}')

        with open(os.path.join(epis_path_depth, 'low_dim_obs.pkl'), 'rb') as f:
            demo = pickle.load(f)
        with open(os.path.join(epis_path_depth, 'variation_number.pkl'), 'rb') as f:
            demo.variation_number = pickle.load(f)

        # language description
        with open(os.path.join(epis_path_depth, 'variation_descriptions.pkl'), 'rb') as f:
            demo.language_descriptions = pickle.load(f)

        episode_keypoints = keypoint_discovery(demo) # _keypoint_discovery(demo) # extract indices of keyframes in the demo

        tmp = []
        diff_gt_std, diff_gt_conc, diff_gt_prune = _add_keypoints_to_replay(
            tmp, 0, demo, episode_keypoints, epis_path_depth, epis_path_char, sim_name_to_real_name, use_gt_pos)
        buffer.append(tmp)

        # accumulate statistics
        for key in diff_gt_std:
            tot_diff_gt_std[key] = tot_diff_gt_std.get(key, 0) + diff_gt_std[key]
            tot_diff_gt_conc[key] = tot_diff_gt_conc.get(key, 0) + diff_gt_conc[key]
            tot_diff_gt_prune[key] = tot_diff_gt_prune.get(key, 0) + diff_gt_prune[key]

    # average distances
    for key in tot_diff_gt_std:
        tot_diff_gt_std[key] = round(tot_diff_gt_std[key] / amount, 3)
        tot_diff_gt_conc[key] = round(tot_diff_gt_conc[key] / amount, 3)
        tot_diff_gt_prune[key] = round(tot_diff_gt_prune[key] / amount, 3)

    print("Average number of steps: ", sum([len(each[1]) for each in buffer])/len(buffer))
    print("Task", task_name, "\navg diff gt std:", tot_diff_gt_std, "\navg diff gt conc:", tot_diff_gt_conc, "\navg diff gt prune:", tot_diff_gt_prune, "\n")
    return buffer

def form_obs_gt(demo, cur_index, sim_name_to_real_name):
    real_name_to_avg_coord = {}
    for sim_name, real_name in sim_name_to_real_name.items():
        obj_pos = demo[cur_index].misc['object_positions'][sim_name]
        real_name_to_avg_coord[real_name] = obj_pos
    return str(real_name_to_avg_coord)

def form_obs_gt_test(sim_name_to_real_name):
    # Extract gt object positions from live simulation
    object_positions = {}
    for sim_name in sim_name_to_real_name:
        obj = Object.get_object(sim_name)
        position = obj.get_position()
        voxel = point_to_voxel_index(position)
        object_positions[sim_name_to_real_name[sim_name]] = list(voxel)
    return str(object_positions)

def form_obs(
    mask_dict,
    mask_id_to_real_name,
    point_cloud_dict):
    
    # convert object id to char and average and discretize point cloud per object
    uniques = np.unique(np.concatenate(list(mask_dict.values()), axis=0))
    real_name_to_avg_coord = {}
    for _, mask_id in enumerate(uniques):
        if mask_id not in mask_id_to_real_name: # for each object id I am interested in
            continue
        avg_point_list = []
        for camera in CAMERAS: # for each camera
            mask = mask_dict[camera] # take the mask from that camera
            point_cloud = point_cloud_dict[camera] # take the point cloud from that camera
            if not np.any(mask == mask_id): # if the object is visible in that camera
                continue
            object_points = point_cloud[mask == mask_id].reshape(-1, 3)
            avg_point_list.append(np.mean(object_points, axis = 0)) # take the average point (center) in the point cloud for that object

        avg_point = sum(avg_point_list) / len(avg_point_list) # take average of the centers
        real_name = mask_id_to_real_name[mask_id]
        real_name_to_avg_coord[real_name] = list(point_to_voxel_index(avg_point))
    return str(real_name_to_avg_coord)

def form_obs_concatenate_clouds(
    mask_dict,
    mask_id_to_real_name,
    point_cloud_dict):
    # Merge point clouds from all cameras first, then compute center
    
    uniques = np.unique(np.concatenate(list(mask_dict.values()), axis=0))
    real_name_to_avg_coord = {}
    for _, mask_id in enumerate(uniques):
        if mask_id not in mask_id_to_real_name: # for each object id I am interested in
            continue
        
        # Collect all points for this object from all cameras
        all_object_points = []
        for camera in CAMERAS: # for each camera
            mask = mask_dict[camera] # take the mask from that camera
            point_cloud = point_cloud_dict[camera] # take the point cloud from that camera
            if not np.any(mask == mask_id): # if the object is visible in that camera
                continue
            object_points = point_cloud[mask == mask_id].reshape(-1, 3)
            all_object_points.append(object_points)
        
        # Merge all points from different cameras and compute center once
        merged_points = np.concatenate(all_object_points, axis=0)
        avg_point = np.mean(merged_points, axis=0)
        real_name = mask_id_to_real_name[mask_id]
        real_name_to_avg_coord[real_name] = list(point_to_voxel_index(avg_point))
    
    return str(real_name_to_avg_coord)

def form_obs_prune_points(
    mask_dict,
    mask_id_to_real_name,
    point_cloud_dict):
    # Merge point clouds from all cameras first, then remove crowded areas and compute center
    
    uniques = np.unique(np.concatenate(list(mask_dict.values()), axis=0))
    real_name_to_avg_coord = {}
    for _, mask_id in enumerate(uniques):
        if mask_id not in mask_id_to_real_name: # for each object id I am interested in
            continue
        
        # Collect all points for this object from all cameras
        all_object_points = []
        for camera in CAMERAS: # for each camera
            mask = mask_dict[camera] # take the mask from that camera
            point_cloud = point_cloud_dict[camera] # take the point cloud from that camera
            if not np.any(mask == mask_id): # if the object is visible in that camera
                continue
            object_points = point_cloud[mask == mask_id].reshape(-1, 3)
            all_object_points.append(object_points)
        
        # Merge all points from different cameras and compute center once
        merged_points = np.concatenate(all_object_points, axis=0)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(merged_points)
        pc = pc.voxel_down_sample(voxel_size=0.02)
        avg_point = np.mean(np.asarray(pc.points), axis=0)
        real_name = mask_id_to_real_name[mask_id]
        real_name_to_avg_coord[real_name] = list(point_to_voxel_index(avg_point))
    
    return str(real_name_to_avg_coord)

class base_task_handler:
    def __init__(self, sim_name_to_real_name):
        self.sim_name_to_real_name = sim_name_to_real_name
        self.save_root = os.path.join(ROOT, type(self).__name__)
        self.num_demos = 10
        print(f"Task handler {type(self).__name__} using demonstrations from {self.save_root}")
        # random.seed(42) # TODO

    def get_user_prompt(self, mask_dict, mask_id_to_sim_name, point_cloud_dict, agent):
        assert os.path.exists(self.save_root), f"Cannot find save root {self.save_root}"
        mask_id_to_real_name = {mask_id: self.sim_name_to_real_name[name] for mask_id, name in mask_id_to_sim_name.items()
                            if name in self.sim_name_to_real_name}
        if agent.model_config.use_gt_obj_pos:
            obs = form_obs_gt_test(self.sim_name_to_real_name)
        else:
            # obs = form_obs(mask_dict, mask_id_to_real_name, point_cloud_dict)
            # obs = form_obs_concatenate_clouds(mask_dict, mask_id_to_real_name, point_cloud_dict)
            obs = form_obs_prune_points(mask_dict, mask_id_to_real_name, point_cloud_dict)
        # sort obs alphabetically by key
        obs = str({k: eval(obs)[k] for k in sorted(eval(obs))})

        # during evaluation, randomly choose one batch of 10 demonstrations from the saved demonstrations
        path = random.choice(glob.glob(os.path.join(self.save_root, "demonstrations", "*.txt")))
        demonstration = open(path, "r").read()

        if type(agent).__name__ in ["RoboPromptAgentOnePerArm", "BestOfN", "LeaderFollower", "LeaderFollowerConversational", "ArmsDebate", "ArmsDebateBestOfN"]:
            examples = demonstration.split(", {") # split over episodes
            right_demonstration = ""
            left_demonstration = ""
            for i, example in enumerate(examples):
                if i == len(examples) - 1:
                    example = "{"+example[:-2]
                elif i > 0:
                    example = "{"+example
                objects_dict, actions_list = example.split(">")
                actions_list = json.loads(actions_list)
                right_actions = []
                left_actions = []
                for action in actions_list:
                    right_actions.append(action[:7])
                    left_actions.append(action[7:])
                right_demonstration += objects_dict + ">" + str(right_actions) + ", "
                left_demonstration += objects_dict + ">" + str(left_actions) + ", "
            
            if type(agent).__name__ in ["BestOfN", "LeaderFollower", "LeaderFollowerConversational", "ArmsDebate", "ArmsDebateBestOfN"]:
                return right_demonstration + obs + ">", left_demonstration + obs + ">", demonstration + obs + ">"
            return right_demonstration + obs + ">", left_demonstration + obs + ">"

        return demonstration + obs + ">"
    
    def save_in_context_demonstrations(self, use_gt_pos):
        train_demos = get_stored_demos(ROOT, type(self).__name__, 100, self.sim_name_to_real_name, use_gt_pos)
        # iterate over 100 demonstrations, each time take 10 demonstrations
        for i, start_idx in enumerate(range(0, len(train_demos), self.num_demos)):
            if start_idx + self.num_demos <= len(train_demos):
                output = ""
                for epi in train_demos[start_idx:start_idx+self.num_demos]:
                    output += f"{epi[0]}>{epi[1]}, "

                d = os.path.join(ROOT, type(self).__name__, f"demonstrations")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f'{i}.txt'), "w") as f:
                    f.write(output)

#####################################################

class bimanual_push_box(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "cube": "box",
            # "target": "target area" # target is not captured by mask cameras for some reason, but task is easy so not really needed
        }
        super().__init__(sim_name_to_real_name)

class bimanual_dual_push_buttons(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "target_button_wrap0": "button 1", # pressed by right arm
            "target_button_wrap1": "button 2", # pressed by left arm
            "target_button_wrap2": "button 3" # distractor
        }
        super().__init__(sim_name_to_real_name)

class bimanual_put_bottle_in_fridge(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "fridge_base_visual": "fridge",
            "bottle_visual": "bottle"
        }
        super().__init__(sim_name_to_real_name)

class bimanual_handover_item(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "item0": "item 1", # item to be handed over
            "item1": "item 2",
            "item2": "item 3",
            "item3": "item 4",
            "item4": "item 5"
        }
        super().__init__(sim_name_to_real_name)

class bimanual_handover_item_easy(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "item": "item",
        }
        super().__init__(sim_name_to_real_name)

class bimanual_lift_ball(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "ball": "ball",
        }
        super().__init__(sim_name_to_real_name)

class bimanual_lift_tray(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "tray_visual": "tray",
        }
        super().__init__(sim_name_to_real_name)

class bimanual_pick_laptop(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "lid_visual": "laptop",
        }
        super().__init__(sim_name_to_real_name)

class bimanual_pick_plate(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "plate_visual": "plate",
        }
        super().__init__(sim_name_to_real_name)

class bimanual_straighten_rope(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "tail": "rope",
            "head_tail": "target 1",
            "head_target": "target 2"
        }
        super().__init__(sim_name_to_real_name)

class bimanual_sweep_to_dustpan(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "sweep_to_dustpan_broom_visual": "broom",
            "Dustpan_5": "dustpan"
        }
        super().__init__(sim_name_to_real_name)

class bimanual_take_tray_out_of_oven(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "oven_frame0": "oven",
            # "tray_visual": "tray", # tray is not visible in the first frame because it's inside the oven
        }
        super().__init__(sim_name_to_real_name)

class bimanual_put_item_in_drawer(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "drawer_frame": "drawer",
            "item": "item"
        }
        super().__init__(sim_name_to_real_name)

class bimanual_close_jar(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "jar_lid0": "lid",
            "jar0": "jar 1",
            "jar1": "jar 2",
            "handover": "handover"
        }
        super().__init__(sim_name_to_real_name)

class bimanual_take_item_out_of_box(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "box_lid": "box"
        }
        super().__init__(sim_name_to_real_name)

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
                        "bimanual_close_jar": bimanual_close_jar,
                        "bimanual_take_item_out_of_box": bimanual_take_item_out_of_box,
                        }

def create_task_handler(task_name):
    return task_name_to_handler[task_name]()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate in-context learning examples')
    parser.add_argument('--gt_pos', action='store_true', help='Use ground truth object positions (default: False)')
    args = parser.parse_args()
    
    use_gt_positions = args.gt_pos
    if use_gt_positions:
        print("Using GT object positions")
    
    for class_name in task_name_to_handler.values():
        handler = class_name()
        handler.save_in_context_demonstrations(use_gt_positions)

