import ast
import json
import re

import numpy as np

from agents.leader_follower import LeaderFollower
from form_icl_demonstrations import SYSTEM_PROMPT_LEFT, SYSTEM_PROMPT_RIGHT
from icl_utils import CAMERAS, fallback_dual_arm_sequence


LEADER_SELECTOR_SYSTEM_PROMPT = """You have to choose the leader arm for a bimanual Franka Panda leader-follower controller.

You will receive completed in-context demonstrations formatted as:
observation>trajectory, observation>trajectory, ...

Each observation is a dictionary of object names mapped to discretized 3D positions.
Each trajectory is a list of bimanual keyframe actions. Every bimanual action has
14 integers:
[right_x, right_y, right_z, right_roll, right_pitch, right_yaw, right_gripper,
 left_x,  left_y,  left_z,  left_roll,  left_pitch,  left_yaw,  left_gripper]

Select the arm that should be predicted first. Base the decision on the
demonstrated action patterns and object geometry. Output exactly one JSON object:
{"leader": "right"} or {"leader": "left"}."""


LEADER_SELECTOR_CRITERIA = (
    (
        "temporal",
        "Which arm tends to move toward or contact task objects first in the examples?",
    ),
    (
        "causal",
        "Which arm creates state changes or constraints that the other arm must follow?",
    ),
    (
        "role",
        "Which arm appears to perform the primary manipulation rather than only support or react?",
    ),
)


class AdaptiveLeaderFollower(LeaderFollower):
    """LeaderFollower with one per-episode LLM call to choose the leader arm."""

    def __init__(self, task_name, model_config):
        super().__init__(task_name, model_config)
        self.default_leader = model_config.leader
        self.current_leader = self.default_leader
        self._episode_selected_leader = None
        self.adaptive_leader_history = []

    def _build_leader_selector_user_prompt(self, user_prompt_bi):
        criteria = "\n".join(
            f"- {name}: {description}"
            for name, description in LEADER_SELECTOR_CRITERIA
        )
        selector_examples = user_prompt_bi.rsplit(", {", 1)[0]
        return (
            "Choose the leader arm for this demonstration batch.\n\n"
            "Use these criteria, in order, when reading the examples:\n"
            f"{criteria}\n\n"
            "Demonstrations:\n"
            f"{selector_examples}\n\n"
            "Return only one of these JSON objects: "
            "{\"leader\": \"right\"} or {\"leader\": \"left\"}"
        )

    def _parse_leader_choice(self, output_text):
        text = output_text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        if fenced:
            text = fenced.group(1)

        try:
            parsed = json.loads(text)
            leader = str(parsed.get("leader", "")).strip().lower()
            if leader in {"right", "left"}:
                return leader
        except Exception:
            pass

        lowered = output_text.lower()
        right_count = lowered.count("right")
        left_count = lowered.count("left")
        if right_count > left_count:
            return "right"
        if left_count > right_count:
            return "left"
        return self.default_leader

    def _select_leader_arm(self, user_prompt_bi):
        user_prompt = self._build_leader_selector_user_prompt(user_prompt_bi)
        messages = [
            {"role": "system", "content": LEADER_SELECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        print(LEADER_SELECTOR_SYSTEM_PROMPT)
        print()
        print(user_prompt)
        try:
            output_text = self.llm_call(messages)
        except Exception as e:
            print(f"AdaptiveLeaderFollower leader-selection call failed: {e!r}")
            return self.default_leader

        selected = self._parse_leader_choice(output_text)
        print(f"Adaptive leader selection llm prediction: {output_text}")
        print(f"Selected leader: {selected}")
        self.adaptive_leader_history.append(selected)
        return selected

    def _preprocess(self, obs, step, **kwargs):
        rgb_dict = {}
        mask_id_to_sim_name = {}
        mask_dict = {}
        point_cloud_dict = {}
        for camera in CAMERAS:
            rgb_img = obs[f'{camera}_rgb']
            rgb_img = rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
            rgb_img = np.clip(((rgb_img + 1.0) / 2 * 255).astype(np.uint8), 0, 255)
            rgb_dict[camera] = rgb_img

            mask_id_to_sim_name.update(kwargs["mapping_dict"][f"{camera}_mask_id_to_name"])
            mask_dict[camera] = obs[f'{camera}_mask'].squeeze().cpu().numpy()
            point_cloud_dict[camera] = (
                obs[f'{camera}_point_cloud'].cpu().squeeze().permute(1, 2, 0).numpy()
            )

        if len(self.actions) != 0:
            return None

        user_prompt_right, user_prompt_left, user_prompt_bi = self.handler.get_user_prompt(
            mask_dict, mask_id_to_sim_name, point_cloud_dict, self
        )
        if self._episode_selected_leader is None:
            leader = self._select_leader_arm(user_prompt_bi)
            self._episode_selected_leader = leader
        else:
            leader = self._episode_selected_leader
            print(f"Adaptive leader reused for replanning: {leader}")
        self.current_leader = leader
        follower = "left" if leader == "right" else "right"

        system_prompt_leader = SYSTEM_PROMPT_RIGHT if leader == "right" else SYSTEM_PROMPT_LEFT
        user_prompt_leader = user_prompt_right if leader == "right" else user_prompt_left

        print(system_prompt_leader)
        print()
        print(user_prompt_leader)

        messages = [
            {"role": "system", "content": system_prompt_leader},
            {"role": "user", "content": user_prompt_leader},
        ]
        try:
            output_text_leader = self.llm_call(messages)
        except Exception as e:
            print(f"AdaptiveLeaderFollower leader call failed: {e!r}")
            return json.dumps(fallback_dual_arm_sequence(self.voxel_size, self.rotation_resolution, length=1))
        print("Prediction:", output_text_leader)
        output_list_leader = self._postprocess_single_arm(output_text_leader)

        examples = user_prompt_bi.split(", {")
        user_prompt_follower = ""
        for i, example in enumerate(examples[:-1]):
            if i > 0:
                example = "{" + example
            objects_dict, actions_list = example.split(">")
            objects_dict = ast.literal_eval(objects_dict)
            actions_list = json.loads(actions_list)
            right_actions = []
            left_actions = []
            for action in actions_list:
                right_actions.append(action[:7])
                left_actions.append(action[7:])
            leader_actions = right_actions if leader == "right" else left_actions
            follower_actions = left_actions if leader == "right" else right_actions
            objects_dict[f'{leader}_arm'] = leader_actions
            user_prompt_follower += str(objects_dict) + ">" + str(follower_actions) + ", "

        example = "{" + examples[-1]
        objects_dict, _ = example.split(">")
        objects_dict = ast.literal_eval(objects_dict)
        objects_dict[f'{leader}_arm'] = output_list_leader
        user_prompt_follower += str(objects_dict) + ">"

        system_prompt_follower = SYSTEM_PROMPT_LEFT if leader == "right" else SYSTEM_PROMPT_RIGHT
        print(system_prompt_follower)
        print()
        print(user_prompt_follower)

        messages = [
            {"role": "system", "content": system_prompt_follower},
            {"role": "user", "content": user_prompt_follower},
        ]
        try:
            output_text_follower = self.llm_call(messages)
        except Exception as e:
            print(f"AdaptiveLeaderFollower follower call failed: {e!r}")
            return json.dumps(fallback_dual_arm_sequence(self.voxel_size, self.rotation_resolution, length=1))
        print("Prediction:", output_text_follower)
        output_list_follower = self._postprocess_single_arm(output_text_follower)

        combined_actions = []
        len_leader = len(output_list_leader)
        len_follower = len(output_list_follower)
        if len_leader > len_follower:
            output_list_follower.extend([output_list_follower[-1]] * (len_leader - len_follower))
        elif len_follower > len_leader:
            output_list_leader.extend([output_list_leader[-1]] * (len_follower - len_leader))

        for leader_action, follower_action in zip(output_list_leader, output_list_follower):
            if leader == "right":
                combined_action = leader_action + follower_action
            else:
                combined_action = follower_action + leader_action
            combined_actions.append(combined_action)

        return json.dumps(combined_actions)

    def reset(self):
        super().reset()
        self.current_leader = self.default_leader
        self._episode_selected_leader = None
