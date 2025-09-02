from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import CloseJar  # Or your specific task
import time
import numpy as np

# Path to your saved episode dataset
DATASET_PATH = '/home/alessiopalma/Desktop/roboprompt/generated_data/test'

# Define action mode and observation config
action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaPlanning(),
    gripper_action_mode=Discrete())

obs_config = ObservationConfig()
obs_config.set_all(True)

# Launch environment with path to demos
env = Environment(action_mode, DATASET_PATH, obs_config=obs_config, headless=False)
env.launch()

# Choose the task to replay
task = env.get_task(CloseJar)

# Load saved demos (set live_demos=False to use disk demos)
demo = task.get_demos(amount=1, live_demos=False, random_selection=False, from_episode_number=0)[0]  # Take First episode
task.reset_to_demo(demo)

# Replay each observation as action towards the next state
for i in range(len(demo) - 1):
    next_obs = demo[i + 1]
    
    ee_pose = next_obs.gripper_pose  # 7D: pos (3) + quat (4)
    gripper = [1.0 if next_obs.gripper_open else 0.0]

    action = np.concatenate([ee_pose, gripper])  # 7 + 1 = 8
    obs, reward, terminate = task.step(action)

    # time.sleep(0.05)

env.shutdown()
