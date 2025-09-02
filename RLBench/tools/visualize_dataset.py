#!/usr/bin/env python3

from typing import List

import os

import numpy as np
import open3d as o3d

from rlbench import CameraConfig
from rlbench import ObservationConfig

from rlbench.utils import get_stored_demos
from pyrep.const import RenderMode

import rich_click as click
from click_prompt import filepath_option


box_length = 0.01
box_dim = dict(zip(["width", "height", "depth"], [box_length] * 3))
box_offset = np.identity(4)
box_offset[:3, 3] = - box_length / 2

camera_names = ["front", "overhead", "over_shoulder_left", "over_shoulder_right", "wrist_left", "wrist_right"]


@click.command()
@filepath_option("--task-folder", default="/home/alessiopalma/Desktop/icl_bimanual/generated_data/test/bimanual_push_box")
@click.option("--show-visualization/--hide-visualization", is_flag=True, default=True)
@click.option("--add-trajectory/--hide-trajectory", is_flag=True, default=True)
@click.option("--episode-number", "-e", help="Which episode to select from the dataset", default=93, type=int)
def cli(task_folder, show_visualization, add_trajectory, episode_number):

    task_folder = os.path.expanduser(task_folder)
    dataset_root = os.path.dirname(task_folder)
    task_name = os.path.basename(task_folder)

    obs_config = create_obs_config(camera_names, [128, 128])
    episodes = get_stored_demos(1, False, dataset_root, -1, task_name, obs_config, random_selection=False, from_episode_number=episode_number)

    visualization_data = []


    # visualize trajectories
    if add_trajectory:
        for obs in episodes[0]:
            right_action_box = o3d.geometry.TriangleMesh.create_box(**box_dim)
            left_action_box = o3d.geometry.TriangleMesh.create_box(**box_dim)

            right_action_box.paint_uniform_color([0, 1, 0]) 
            left_action_box.paint_uniform_color([1, 0, 0]) 

            right_action_box.transform(obs.right.gripper_matrix.dot(box_offset))
            left_action_box.transform(obs.left.gripper_matrix.dot(box_offset))

            visualization_data.append(right_action_box)
            visualization_data.append(left_action_box)



    # point cloud from first observation of the first episode
    obs = episodes[0][0]
    for camera_name in camera_names:

        xyz = obs.perception_data[f"{camera_name}_point_cloud"].reshape(-1, 3)
        rgb = obs.perception_data[f"{camera_name}_rgb"].reshape(-1, 3) / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        visualization_data.append(pcd)

    if show_visualization:

        o3d.visualization.draw_geometries(visualization_data)
    else:

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for g in visualization_data:
            vis.add_geometry(g)
            vis.update_geometry(g)


        vis.poll_events()
        vis.update_renderer()

        view_control = vis.get_view_control()

        # copied from the viewer
        view_pose = {"front" : [ 0.012339452449784705, -0.90282996236235535, 0.42982065675584702 ],
                    "lookat" : [ -0.12694871669500873, 0.11746709358396157, 0.88075116287669197 ],
                    "up" : [ 0.025976937544841906, 0.42999774206857738, 0.90245617097547559 ],
                    "zoom" : 0.33999999999999964 }

        view_control.set_front(view_pose["front"])
        view_control.set_up(view_pose["up"])
        view_control.set_lookat(view_pose["lookat"])
        view_control.set_zoom(view_pose["zoom"])


        vis.capture_screen_image(f"/tmp/rlbench_{task_name}_episode{episode_number}.png", do_render=True)
        vis.destroy_window()


def create_obs_config(
    camera_names: List[str],
    camera_resolution: List[int],
):
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=False,
        depth=False,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL3,
    )

    camera_configs = {camera_name: used_cams for camera_name in camera_names}

    obs_config = ObservationConfig(
        camera_configs=camera_configs,
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
        robot_name="bimanual"
    )
    return obs_config


if __name__ == "__main__":
    cli()

