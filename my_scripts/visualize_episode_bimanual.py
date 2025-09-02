from rlbench.environment import Environment
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning, BimanualJointPosition
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.bimanual_tasks.bimanual_push_box import BimanualPushBox  # Or your specific task
import time
import numpy as np

# Path to your saved episode dataset
DATASET_PATH = '/home/alessiopalma/Desktop/icl_bimanual/generated_data/test'

# Define action mode and observation config
action_mode = BimanualMoveArmThenGripper(
    arm_action_mode=BimanualJointPosition(),
    gripper_action_mode=BimanualDiscrete())

camera_names = ["over_shoulder_left", "over_shoulder_right", "overhead", "wrist_right", "wrist_left", "front"]
obs_config = ObservationConfig()
obs_config.set_all(True)
default_config_params = {"image_size": [128,128], "depth_in_meters": False, "masks_as_one_channel": False}
camera_configs = {camera_name: CameraConfig(**default_config_params) for camera_name in camera_names}
obs_config.camera_configs = camera_configs

# Launch environment with path to demos
env = Environment(action_mode, DATASET_PATH, obs_config=obs_config, headless=False, robot_setup='dual_panda')
env.launch()

# Choose the task to replay
task = env.get_task(BimanualPushBox)

# Load saved demos (set live_demos=False to use disk demos)
demo = task.get_demos(amount=1, live_demos=False, random_selection=False, from_episode_number=25)[0]
task.reset_to_demo(demo)
# env._scene.robot.right_arm.set_joint_positions(env._scene._start_arm_joint_pos[0], disable_dynamics=True)
# env._scene.robot.left_arm.set_joint_positions(env._scene._start_arm_joint_pos[1], disable_dynamics=True)

none_actions_right = 0
none_actions_left = 0

# Replay each observation as action towards the next state
for i in range(len(demo) - 1):
    next_obs = demo[i + 1]

    # Right arm action
    arm_pose_right = (next_obs.misc['right_executed_demo_joint_position_action']
                 if ('right_executed_demo_joint_position_action' in next_obs.misc) and
                 (next_obs.misc['right_executed_demo_joint_position_action'] is not None)
                 else next_obs.right.joint_positions) # 7D: pos (3) + quat (4)
    gripper_right = [1.0 if next_obs.right.gripper_open else 0.0]
    ignore_collisions_right = np.array([next_obs.right.ignore_collisions])
    action_right = np.concatenate([arm_pose_right, gripper_right, ignore_collisions_right])  # 7 + 1 + 1 = 9

    # Left arm action
    arm_pose_left = (next_obs.misc['left_executed_demo_joint_position_action'] 
                 if ('left_executed_demo_joint_position_action' in next_obs.misc) and
                 (next_obs.misc['left_executed_demo_joint_position_action'] is not None)
                 else next_obs.left.joint_positions)  # 7D: pos (3) + quat (4)
    gripper_left = [1.0 if next_obs.left.gripper_open else 0.0]
    ignore_collisions_left = np.array([next_obs.left.ignore_collisions])
    action_left = np.concatenate([arm_pose_left, gripper_left, ignore_collisions_left]) # 7 + 1 + 1 = 9

    action = np.concatenate([action_right, action_left])  # 9 + 9 = 18
    obs, reward, terminate = task.step(action)

    # time.sleep(0.05)

    if ('right_executed_demo_joint_position_action' not in next_obs.misc) or (next_obs.misc['right_executed_demo_joint_position_action'] is None):
        none_actions_right += 1
    if ('left_executed_demo_joint_position_action' not in next_obs.misc) or (next_obs.misc['left_executed_demo_joint_position_action'] is None):
        none_actions_left += 1

print(f"None actions right: {none_actions_right}, left: {none_actions_left}")
print(f"Terminated: {terminate}")
env.shutdown()
