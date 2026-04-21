from typing import List
import re
import random
import time
from yarr.agents.agent import Agent, Summary, ActResult
import json
import ast
import numpy as np
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError
from form_icl_demonstrations import create_task_handler, SYSTEM_PROMPT_RIGHT, SYSTEM_PROMPT_LEFT
from icl_utils import SCENE_BOUNDS, ROTATION_RESOLUTION, discrete_euler_to_quaternion, CAMERAS
import openai
from openai import OpenAI

N_CANDIDATES = 5

_VALIDATOR_SYSTEM = (
    "You are a strict judge evaluating bimanual robot action plans.\n\n"
    "CONTEXT: Two Franka Panda arms (right=indices 0-6, left=indices 7-13) in a 100x100x100 voxel workspace. "
    "Each 14-dim action is [right_x, right_y, right_z, right_rot1, right_rot2, right_rot3, right_gripper, "
    "left_x, left_y, left_z, left_rot1, left_rot2, left_rot3, left_gripper].\n\n"
    "TASK: Score the CANDIDATE plan from 1 to 5. START AT 3 and adjust:\n\n"
    "CHECK 1 \u2014 Arm collision risk (+1 or -1):\n"
    "  At each timestep, compute the Euclidean distance between right [x,y,z] and left [x,y,z]. "
    "If ANY step has distance < 10 voxels AND both arms are actively moving (not stationary), that is a collision risk: -1. "
    "If all steps have safe separation: +1.\n\n"
    "CHECK 2 \u2014 Target + trajectory match vs demos (+1 or -1):\n"
    "  Does the candidate approach the SAME objects as in demos (first action within 5 voxels of demo first action)? "
    "Does the z-trajectory follow the same shape (e.g. approach high, descend to grasp, lift)? "
    "Both must be true for +1. Either failing: -1.\n\n"
    "CHECK 3 \u2014 Gripper logic (0 or -1):\n"
    "  For EACH arm: does the gripper open/close at the correct step relative to when the arm reaches the object? "
    "Closing too early (before reaching), or gripper sequence inverted vs demos: -1.\n\n"
    "CHECK 4 \u2014 Workspace reachability (0 or -1):\n"
    "  Right arm should mostly operate in x > 30 (its reachable zone). Left arm should mostly operate in x < 70. "
    "If an arm consistently reaches into the opposite side of the workspace (>3 steps): -1.\n\n"
    "Final score = 3 + check1 + check2 + check3 + check4, clamped to [1, 5].\n\n"
    "You MUST show your work for each check, then give the final score.\n"
    "Output ONLY valid JSON:\n"
    '{"check1": "+1 or -1: <reason>", "check2": "+1 or -1: <reason>", '
    '"check3": "0 or -1: <reason>", "check4": "0 or -1: <reason>", "score": <int 1-5>}'
)


_RETRYABLE = (
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.RateLimitError,
)


def openai_call(client, model_name, messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            return completion.choices[0].message.content or ""
        except _RETRYABLE as e:
            if attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                print(f"openai_call retry {attempt+1}/{max_retries}: {e!r}  "
                      f"(sleeping {wait}s)")
                time.sleep(wait)
            else:
                raise


def huggingface_call(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


class ArmsDebateBestOfN(Agent):
    """Hybrid agent: N parallel ArmsDebate (4-call iterative refinement) candidates
    scored by an LLM-as-judge validator.  Combines the diversity of BestOfN with
    the coordination quality of ArmsDebate's iterative leader-follower debate."""

    def __init__(self, task_name, model_config):
        self.episode_id = -1
        self.device = 'cuda'
        self.task_name = task_name
        self.model_config = model_config
        self.leader = model_config.leader
        self.follower = "left" if self.leader == "right" else "right"

        # Use the same system prompts as the base ArmsDebate agent
        self._sys_leader = SYSTEM_PROMPT_RIGHT if self.leader == "right" else SYSTEM_PROMPT_LEFT
        self._sys_follower = SYSTEM_PROMPT_LEFT if self.leader == "right" else SYSTEM_PROMPT_RIGHT

    # ── Observation preprocessing ────────────────────────────────────────────

    def _preprocess_observation(self, obs, step, **kwargs):
        rgb_dict, mask_id_to_sim_name, mask_dict, point_cloud_dict = {}, {}, {}, {}
        for camera in CAMERAS:
            rgb_img = obs[f'{camera}_rgb']
            rgb_img = rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
            rgb_img = np.clip(((rgb_img + 1.0) / 2 * 255).astype(np.uint8), 0, 255)
            rgb_dict[camera] = rgb_img

            mask_id_to_sim_name.update(kwargs["mapping_dict"][f"{camera}_mask_id_to_name"])

            mask = obs[f'{camera}_mask'].squeeze().cpu().numpy()
            mask_dict[camera] = mask

            point_cloud = obs[f'{camera}_point_cloud'].cpu().squeeze().permute(1, 2, 0).numpy()
            point_cloud_dict[camera] = point_cloud

        return mask_dict, mask_id_to_sim_name, point_cloud_dict

    # ── Prompt building ──────────────────────────────────────────────────────

    def _build_prompts(self, mask_dict, mask_id_to_sim_name, point_cloud_dict):
        """Parse ICL prompts into reusable components (called once per episode)."""
        user_prompt_right, user_prompt_left, user_prompt_bi = self.handler.get_user_prompt(
            mask_dict, mask_id_to_sim_name, point_cloud_dict, self
        )

        # --- Single-arm leader examples (for call 1) ---
        leader_prompt = user_prompt_right if self.leader == "right" else user_prompt_left
        parts_leader = leader_prompt.split(", {")
        parsed_leader_examples = []
        for i, ex in enumerate(parts_leader[:-1]):
            if i > 0:
                ex = "{" + ex
            parsed_leader_examples.append(ex)
        leader_obs_suffix = "{" + parts_leader[-1]  # live observation, ends with ">"

        # --- Bimanual examples (for calls 2-4) ---
        parts_bi = user_prompt_bi.split(", {")
        parsed_bi_examples = []
        for i, ex in enumerate(parts_bi[:-1]):
            if i > 0:
                ex = "{" + ex
            objects_dict, actions_str = ex.split(">")
            objects_dict = ast.literal_eval(objects_dict)
            actions_list = json.loads(actions_str)
            right_actions = [a[:7] for a in actions_list]
            left_actions = [a[7:] for a in actions_list]
            parsed_bi_examples.append((objects_dict, right_actions, left_actions))

        # Live observation dict (shared across candidates, augmented at generation time)
        last_obs_raw = "{" + parts_bi[-1]
        last_obs_str, _ = last_obs_raw.split(">")
        last_obs_dict = ast.literal_eval(last_obs_str)

        return (parsed_leader_examples, leader_obs_suffix,
                parsed_bi_examples, last_obs_dict, user_prompt_bi)

    def _build_shuffled_prompts(self, parsed_leader_examples, leader_obs_suffix,
                                parsed_bi_examples, candidate_idx):
        """Build shuffled prompts for all 4 ArmsDebate calls of one candidate."""
        rng = random.Random(self.episode_id * N_CANDIDATES + candidate_idx)
        order = list(range(len(parsed_leader_examples)))
        rng.shuffle(order)

        # Call 1: shuffled single-arm leader prompt
        user_prompt_leader = ", ".join(
            parsed_leader_examples[i] for i in order
        ) + ", " + leader_obs_suffix

        # Calls 2 & 4: follower prefix (obs has leader_arm → predict follower arm)
        follower_prefix = ""
        for i in order:
            obs_dict, r_acts, l_acts = parsed_bi_examples[i]
            obs_copy = dict(obs_dict)
            leader_demo_actions = r_acts if self.leader == "right" else l_acts
            follower_target = l_acts if self.leader == "right" else r_acts
            obs_copy[f'{self.leader}_arm'] = leader_demo_actions
            follower_prefix += str(obs_copy) + ">" + str(follower_target) + ", "

        # Call 3: augmented leader prefix (obs has follower_arm → predict leader arm)
        leader_aug_prefix = ""
        for i in order:
            obs_dict, r_acts, l_acts = parsed_bi_examples[i]
            obs_copy = dict(obs_dict)
            follower_demo_actions = l_acts if self.leader == "right" else r_acts
            leader_target = r_acts if self.leader == "right" else l_acts
            obs_copy[f'{self.follower}_arm'] = follower_demo_actions
            leader_aug_prefix += str(obs_copy) + ">" + str(leader_target) + ", "

        return user_prompt_leader, follower_prefix, leader_aug_prefix

    # ── Candidate generation (4-call ArmsDebate pipeline) ──────────────────────

    def _generate_candidate(self, user_prompt_leader, follower_prefix,
                            leader_aug_prefix, last_obs_dict, candidate_idx):
        """Run the full 4-call iterative refinement pipeline for one candidate.
        Thread-safe for OpenAI calls."""

        # Call 1: Initial leader prediction (single-arm prompt)
        leader_out = self.llm_call([
            {"role": "system", "content": self._sys_leader},
            {"role": "user", "content": user_prompt_leader}
        ])
        leader_actions = self._postprocess_single_arm(leader_out)

        # Call 2: Initial follower prediction (sees leader's plan)
        obs_f = dict(last_obs_dict)
        obs_f[f'{self.leader}_arm'] = leader_actions
        follower_out = self.llm_call([
            {"role": "system", "content": self._sys_follower},
            {"role": "user", "content": follower_prefix + str(obs_f) + ">"}
        ])
        follower_actions = self._postprocess_single_arm(follower_out)

        # Call 3: Refined leader (sees follower's plan)
        obs_l = dict(last_obs_dict)
        obs_l[f'{self.follower}_arm'] = follower_actions
        refined_leader_out = self.llm_call([
            {"role": "system", "content": self._sys_leader},
            {"role": "user", "content": leader_aug_prefix + str(obs_l) + ">"}
        ])
        refined_leader_actions = self._postprocess_single_arm(refined_leader_out)

        # Call 4: Refined follower (sees refined leader's plan)
        obs_f2 = dict(last_obs_dict)
        obs_f2[f'{self.leader}_arm'] = refined_leader_actions
        refined_follower_out = self.llm_call([
            {"role": "system", "content": self._sys_follower},
            {"role": "user", "content": follower_prefix + str(obs_f2) + ">"}
        ])
        refined_follower_actions = self._postprocess_single_arm(refined_follower_out)

        # Pad shorter arm to match longer
        len_l = len(refined_leader_actions)
        len_f = len(refined_follower_actions)
        if len_l > len_f:
            refined_follower_actions.extend(
                [refined_follower_actions[-1]] * (len_l - len_f)
            )
        elif len_f > len_l:
            refined_leader_actions.extend(
                [refined_leader_actions[-1]] * (len_f - len_l)
            )

        # Combine into 14-dim bimanual actions
        combined = []
        for la, fa in zip(refined_leader_actions, refined_follower_actions):
            if self.leader == "right":
                combined.append(la + fa)
            else:
                combined.append(fa + la)

        return json.dumps(combined)

    # ── Validator ────────────────────────────────────────────────────────────

    def _validate_prediction(self, prediction, user_prompt_bi):
        examples = user_prompt_bi.split(", {")
        last_observation = "{" + examples[-1]
        icl_examples = ""
        for i, ex in enumerate(examples[:-1]):
            if i > 0:
                ex = "{" + ex
            icl_examples += ex + "\n"

        prompt_content = (
            f"REFERENCE DEMOS (observation>actions):\n{icl_examples}\n"
            f"NEW OBSERVATION:\n{last_observation}\n\n"
            f"CANDIDATE PLAN:\n{prediction}"
        )

        response = self.llm_call([
            {"role": "system", "content": _VALIDATOR_SYSTEM},
            {"role": "user", "content": prompt_content}
        ])

        content = response.strip()
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                data = json.loads(content[start:end])
                score = int(data.get("score", 1))
                score = max(1, min(5, score))
                checks = {k: data.get(k, "") for k in
                          ["check1", "check2", "check3", "check4"]}
                return score, str(checks)
            match = re.search(r'\b([1-5])\b', content)
            if match:
                return int(match.group(1)), "fallback parsing"
            return 1, "no score found"
        except Exception as e:
            print(f"Error parsing validator response: {e}")
            return 1, f"error {e}"

    # ── Postprocessing ───────────────────────────────────────────────────────

    def _postprocess_single_arm(self, output_text):
        try:
            regex = r'^```json(\s*\[\s*(?:\[(?:\d+\s*,\s*){6}\d+\]\s*,\s*)*\[(?:\d+\s*,\s*){6}\d+\]\s*\])\s*```$'
            match = re.search(regex, output_text)
            if match:
                actions = json.loads(match.group(1))
            else:
                regex = r'^```(\s*\[\s*(?:\[(?:\d+\s*,\s*){6}\d+\]\s*,\s*)*\[(?:\d+\s*,\s*){6}\d+\]\s*\])\s*```$'
                match = re.search(regex, output_text)
                if match:
                    actions = json.loads(match.group(1))
                else:
                    try:
                        actions = json.loads(output_text)
                    except JSONDecodeError:
                        if "\n" in output_text:
                            output_text = output_text.replace("\n", ",")
                        if (not output_text.startswith('[[')) and output_text.endswith(']]'):
                            output_text = output_text[:-1]
                        elif output_text.startswith('[[') and (not output_text.endswith(']]')):
                            output_text = output_text[1:]
                        actions = json.loads('[' + output_text + ']')
            if len(np.array(actions).shape) == 1:
                actions = [actions]
        except Exception as e:
            actions = [[57, 49, 87, 0, 39, 0, 1] for _ in range(26)]
            print(e)
            print('Error when parsing actions')
        output = []
        for action in actions:
            if len(action) != 7:
                action = [57, 49, 87, 0, 39, 0, 1]
            output.append(action)
        return output[:26]

    def _postprocess_dual_arm(self, output_text):
        try:
            regex = r'^```json(\s*\[\s*(?:\[(?:\d+\s*,\s*){13}\d+\]\s*,\s*)*\[(?:\d+\s*,\s*){13}\d+\]\s*\])\s*```$'
            match = re.search(regex, output_text)
            if match:
                actions = np.array(json.loads(match.group(1)))
            else:
                regex = r'^```(\s*\[\s*(?:\[(?:\d+\s*,\s*){13}\d+\]\s*,\s*)*\[(?:\d+\s*,\s*){13}\d+\]\s*\])\s*```$'
                match = re.search(regex, output_text)
                if match:
                    actions = np.array(json.loads(match.group(1)))
                else:
                    try:
                        actions = np.array(json.loads(output_text))
                    except JSONDecodeError:
                        if "\n" in output_text:
                            output_text = output_text.replace("\n", ",")
                        if (not output_text.startswith('[[')) and output_text.endswith(']]'):
                            output_text = output_text[:-1]
                        elif output_text.startswith('[[') and (not output_text.endswith(']]')):
                            output_text = output_text[1:]
                        actions = np.array(json.loads('[' + output_text + ']'))
        except Exception as e:
            actions = [[57, 49, 87, 0, 39, 0, 1, 57, 49, 87, 0, 39, 0, 1]
                       for _ in range(26)]
            print(e)
            print('Error when parsing actions')
        if len(np.array(actions).shape) == 1:
            actions = [actions]
        output = []
        for action in actions:
            if len(action) != 14:
                action = [57, 49, 87, 0, 39, 0, 1, 57, 49, 87, 0, 39, 0, 1]
            temp_actions = []
            for i in range(2):
                arm_action = action[i * 7:(i + 1) * 7]
                trans_indicies = np.array(arm_action[:3])
                rot_and_grip_indicies = np.array(arm_action[3:6])
                is_gripper_open = 1 if arm_action[6] >= 0.5 else 0
                bounds = SCENE_BOUNDS
                res = (bounds[3:] - bounds[:3]) / 100
                attention_coordinate = bounds[:3] + res * trans_indicies + res / 2
                quat = discrete_euler_to_quaternion(rot_and_grip_indicies)
                continuous_action = np.concatenate([
                    attention_coordinate, quat, [is_gripper_open], [1],
                ])
                temp_actions.append(continuous_action)
            output.append(np.concatenate(temp_actions, axis=0))
        return output[:26]

    # ── Main act loop ────────────────────────────────────────────────────────

    def act(self, step: int, observation: dict,
            deterministic=False, **kwargs) -> ActResult:
        if len(self.actions) == 0:
            # 1. Preprocess observation once
            mask_dict, mask_id_to_sim_name, pc_dict = self._preprocess_observation(
                observation, step, **kwargs
            )

            # 2. Build all prompt components once
            (parsed_leader_examples, leader_obs_suffix,
             parsed_bi_examples, last_obs_dict,
             user_prompt_bi) = self._build_prompts(
                mask_dict, mask_id_to_sim_name, pc_dict
            )

            # 3. Generate N candidates in parallel (each runs 4 sequential LLM calls)
            _DEFAULT_CANDIDATE = json.dumps(
                [[57, 49, 87, 0, 39, 0, 1, 57, 49, 87, 0, 39, 0, 1]] * 4
            )
            candidates = [None] * N_CANDIDATES
            with ThreadPoolExecutor(max_workers=N_CANDIDATES) as pool:
                futures = {}
                for i in range(N_CANDIDATES):
                    lp, fp, lap = self._build_shuffled_prompts(
                        parsed_leader_examples, leader_obs_suffix,
                        parsed_bi_examples, i
                    )
                    futures[pool.submit(
                        self._generate_candidate,
                        lp, fp, lap, last_obs_dict, i
                    )] = i
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        candidates[idx] = future.result()
                    except Exception as e:
                        print(f"Candidate {idx} generation failed: {e!r}")
                        candidates[idx] = _DEFAULT_CANDIDATE

            # 4. Score all candidates in parallel
            scores = [0] * N_CANDIDATES
            reasonings = [""] * N_CANDIDATES
            with ThreadPoolExecutor(max_workers=N_CANDIDATES) as pool:
                futures = {
                    pool.submit(self._validate_prediction, cand, user_prompt_bi): i
                    for i, cand in enumerate(candidates)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        scores[idx], reasonings[idx] = future.result()
                    except Exception as e:
                        print(f"Validation of candidate {idx} failed: {e!r}")
                        scores[idx] = 0
                        reasonings[idx] = f"validation error: {e}"

            # 5. Select best
            best_idx = int(np.argmax(scores))
            print(f"ArmsDebateBestOfN scores: {scores}, selected candidate {best_idx} "
                  f"(score={scores[best_idx]})")
            for i, (s, r) in enumerate(zip(scores, reasonings)):
                print(f"  Candidate {i}: score={s}, reasoning={r}")

            output = self._postprocess_dual_arm(candidates[best_idx])
            self.actions = output

        continuous_action = self.actions.pop(0)
        self.step += 1
        copy_obs = {k: v.cpu() for k, v in observation.items()}
        return ActResult(continuous_action, observation_elements=copy_obs, info=None)

    def act_summaries(self) -> List[Summary]:
        return []

    def reset(self):
        super().reset()
        self.step = 0
        self.episode_id += 1
        self._prev_action = None
        self.actions = []

    def load_weights(self, savedir: str):
        self.savedir = savedir
        self.handler = create_task_handler(self.task_name)

        if self.model_config.llm_call_style == "openai":
            print("using openai model")
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.llm_call = lambda messages: openai_call(
                client, self.model_config.name, messages
            )
        elif self.model_config.llm_call_style == "huggingface":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("loading model from huggingface")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.name, torch_dtype="auto", device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.name)
            for param in model.parameters():
                param.requires_grad = False
            self.llm_call = lambda messages: huggingface_call(
                model, tokenizer, messages
            )
        elif self.model_config.llm_call_style == "vllm":
            print("using remote vllm-served model")
            client = OpenAI(
                base_url="http://127.0.0.1:8000/v1",
                api_key="password",
            )
            model_name = ("/leonardo_scratch/large/userexternal/apalma01/"
                          "llm_models/" + self.model_config.name.split("/")[-1])
            self.llm_call = lambda messages: openai_call(
                client, model_name, messages
            )

    def build(self, training: bool, device=None):
        return

    def update(self, step: int, replay_sample: dict) -> dict:
        return {}

    def update_summaries(self) -> List[Summary]:
        return []

    def save_weights(self, savedir: str):
        return
