from scipy.spatial.transform import Rotation
import numpy as np

SCENE_BOUNDS = np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6])
DEFAULT_ROTATION_RESOLUTION = 5
DEFAULT_VOXEL_SIZE = 100
CAMERAS = ["over_shoulder_left", "over_shoulder_right", "overhead", "wrist_right", "wrist_left", "front"]
IMAGE_SIZE = [128, 128]
DEFAULT_SINGLE_ARM_ACTION_100 = [57, 49, 87, 0, 39, 0, 1]


def _get_config_value(config, key, default):
    if config is None:
        return default
    value = getattr(config, key, default)
    return default if value is None else value


def get_voxel_size(config=None):
    return int(_get_config_value(config, "voxel_size", DEFAULT_VOXEL_SIZE))


def get_rotation_resolution(config=None):
    return float(_get_config_value(config, "rotation_resolution", DEFAULT_ROTATION_RESOLUTION))


def _format_resolution_value(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def get_demonstrations_dir(config=None, voxel_size=None, rotation_resolution=None):
    explicit = _get_config_value(config, "demonstrations_dir", None)
    if explicit:
        return str(explicit)
    voxel_size = get_voxel_size(config) if voxel_size is None else int(voxel_size)
    rotation_resolution = (
        get_rotation_resolution(config)
        if rotation_resolution is None
        else float(rotation_resolution)
    )
    if (
        voxel_size == DEFAULT_VOXEL_SIZE
        and rotation_resolution == DEFAULT_ROTATION_RESOLUTION
    ):
        return "demonstrations"
    return f"demonstrations_vox{voxel_size}_rot{_format_resolution_value(rotation_resolution)}"

# From https://github.com/stepjam/RLBench/blob/master/rlbench/backend/utils.py
def _image_to_float_array(image, scale_factor):
    image_array = np.array(image)
    image_dtype = image_array.dtype
    image_shape = image_array.shape

    channels = image_shape[2] if len(image_shape) > 2 else 1
    assert 2 <= len(image_shape) <= 3
    if channels == 3:
        # RGB image needs to be converted to 24 bit integer.
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
    else:
        float_array = image_array.astype(np.float32)
    scaled_array = float_array / scale_factor
    return scaled_array

def quaternion_to_discrete_euler(quaternion, rotation_resolution=DEFAULT_ROTATION_RESOLUTION):
    euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / rotation_resolution)).astype(int)
    disc[disc == int(360 / rotation_resolution)] = 0
    return disc

def rotation_matrix_to_discrete_euler(R, rotation_resolution=DEFAULT_ROTATION_RESOLUTION):
    R = np.array(R).copy()  # ensure writable (Open3D OBB R can be read-only)
    euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True) + 180
    euler = np.clip(euler, 0, 360)
    disc = np.around((euler / rotation_resolution)).astype(int)
    disc[disc == int(360 / rotation_resolution)] = 0
    return disc

def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)

def point_to_voxel_index(
        point: np.ndarray,
        voxel_size=DEFAULT_VOXEL_SIZE):
    point = np.asarray(point)
    bb_mins = np.array(SCENE_BOUNDS[0:3])[None]
    bb_maxs = np.array(SCENE_BOUNDS[3:])[None]
    dims_m_one = np.array([voxel_size] * 3)[None] - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(
            np.int32), dims_m_one)
    return voxel_indicy.reshape(point.shape)


def discrete_euler_to_quaternion(discrete_euler, rotation_resolution=DEFAULT_ROTATION_RESOLUTION):
    euluer = (discrete_euler * rotation_resolution) - 180
    return Rotation.from_euler('xyz', euluer, degrees=True).as_quat()


def rotation_bin_count(rotation_resolution):
    return int(round(360 / rotation_resolution))


def scale_default_single_arm_action(voxel_size, rotation_resolution):
    trans = [
        int(np.clip(np.rint((idx + 0.5) * voxel_size / 100 - 0.5), 0, voxel_size - 1))
        for idx in DEFAULT_SINGLE_ARM_ACTION_100[:3]
    ]
    bins = rotation_bin_count(rotation_resolution)
    rot = [
        int(round(idx * DEFAULT_ROTATION_RESOLUTION / rotation_resolution)) % bins
        for idx in DEFAULT_SINGLE_ARM_ACTION_100[3:6]
    ]
    return trans + rot + [DEFAULT_SINGLE_ARM_ACTION_100[6]]


def scale_voxel_index(voxel_index, target_voxel_size, source_voxel_size=DEFAULT_VOXEL_SIZE):
    values = np.asarray(voxel_index, dtype=float)
    scaled = np.rint((values + 0.5) * target_voxel_size / source_voxel_size - 0.5)
    return np.clip(scaled.astype(int), 0, target_voxel_size - 1)


def fallback_single_arm_sequence(voxel_size, rotation_resolution, length=26):
    return [scale_default_single_arm_action(voxel_size, rotation_resolution) for _ in range(length)]


def fallback_dual_arm_sequence(voxel_size, rotation_resolution, length=26):
    fallback = scale_default_single_arm_action(voxel_size, rotation_resolution)
    return [fallback + fallback for _ in range(length)]


def sanitize_single_arm_action(action, voxel_size, rotation_resolution):
    if len(action) != 7:
        return scale_default_single_arm_action(voxel_size, rotation_resolution)
    values = np.rint(np.array(action, dtype=float)).astype(int)
    values[:3] = np.clip(values[:3], 0, voxel_size - 1)
    values[3:6] = np.mod(values[3:6], rotation_bin_count(rotation_resolution))
    values[6] = 1 if values[6] >= 1 else 0
    return values.tolist()


def single_arm_discrete_action_to_continuous(action, voxel_size, rotation_resolution):
    action = sanitize_single_arm_action(action, voxel_size, rotation_resolution)
    trans_indices = np.array(action[:3])
    rot_indices = np.array(action[3:6])
    is_gripper_open = 1 if action[6] >= 0.5 else 0

    bounds = SCENE_BOUNDS
    res = (bounds[3:] - bounds[:3]) / voxel_size
    attention_coordinate = bounds[:3] + res * trans_indices + res / 2
    quat = discrete_euler_to_quaternion(rot_indices, rotation_resolution)

    return np.concatenate([
        attention_coordinate,
        quat,
        [is_gripper_open],
        [1],
    ])


def single_arm_discrete_actions_to_continuous(actions, voxel_size, rotation_resolution, limit=26):
    if len(np.array(actions).shape) == 1:
        actions = [actions]
    return [
        single_arm_discrete_action_to_continuous(action, voxel_size, rotation_resolution)
        for action in actions
    ][:limit]


def dual_arm_discrete_action_to_continuous(action, voxel_size, rotation_resolution):
    if len(action) != 14:
        action = scale_default_single_arm_action(voxel_size, rotation_resolution) * 2
    right = single_arm_discrete_action_to_continuous(action[:7], voxel_size, rotation_resolution)
    left = single_arm_discrete_action_to_continuous(action[7:], voxel_size, rotation_resolution)
    return np.concatenate([right, left], axis=0)


def dual_arm_discrete_actions_to_continuous(actions, voxel_size, rotation_resolution, limit=26):
    if len(np.array(actions).shape) == 1:
        actions = [actions]
    return [
        dual_arm_discrete_action_to_continuous(action, voxel_size, rotation_resolution)
        for action in actions
    ][:limit]
