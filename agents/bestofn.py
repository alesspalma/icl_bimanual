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
from icl_utils import CAMERAS, dual_arm_discrete_actions_to_continuous, fallback_dual_arm_sequence, fallback_single_arm_sequence, get_rotation_resolution, get_voxel_size, sanitize_single_arm_action
import openai
from openai import OpenAI
from agents.llm_tracking import LLMTrackingMixin

N_CANDIDATES = 5

_RETRYABLE = (
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.RateLimitError,
)


def openai_call(client, model_name, messages, service_tier=None, max_retries=5):
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **({"service_tier": service_tier} if service_tier else {}),
            )
            content = completion.choices[0].message.content or ""
            usage = {
                'prompt_tokens': completion.usage.prompt_tokens,
                'completion_tokens': completion.usage.completion_tokens,
                'total_tokens': completion.usage.total_tokens,
            }
            return content, usage
        except _RETRYABLE as e:
            if attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                print(f"openai_call retry {attempt + 1}/{max_retries}: {e!r}  "
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

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response, None


class BestOfN(LLMTrackingMixin, Agent):
    def __init__(self, task_name, model_config):
        self.episode_id = -1
        self.device = 'cuda'
        self.task_name = task_name
        self.model_config = model_config
        self.leader = model_config.leader
        self._parallel_factor = N_CANDIDATES
        self.voxel_size = get_voxel_size(model_config)
        self.rotation_resolution = get_rotation_resolution(model_config)
        voxel_scale = self.voxel_size / 100
        self.validator_system_prompt = (
            "You are a strict judge evaluating bimanual robot action plans.\n\n"
            f"CONTEXT: Two Franka Panda arms (right=indices 0-6, left=indices 7-13) in a {self.voxel_size}x{self.voxel_size}x{self.voxel_size} voxel workspace. "
            "Each 14-dim action is [right_x, right_y, right_z, right_rot1, right_rot2, right_rot3, right_gripper, "
            "left_x, left_y, left_z, left_rot1, left_rot2, left_rot3, left_gripper].\n\n"
            "TASK: Score the CANDIDATE plan from 1 to 5. START AT 3 and adjust:\n\n"
            "CHECK 1 — Arm collision risk (+1 or -1):\n"
            "  At each timestep, compute the Euclidean distance between right [x,y,z] and left [x,y,z]. "
            f"If ANY step has distance < {round(10 * voxel_scale, 1)} voxels AND both arms are actively moving (not stationary), that is a collision risk: -1. "
            "If all steps have safe separation: +1.\n\n"
            "CHECK 2 — Target + trajectory match vs demos (+1 or -1):\n"
            f"  Does the candidate approach the SAME objects as in demos (first action within {round(5 * voxel_scale, 1)} voxels of demo first action)? "
            "Does the z-trajectory follow the same shape (e.g. approach high, descend to grasp, lift)? "
            "Both must be true for +1. Either failing: -1.\n\n"
            "CHECK 3 — Gripper logic (0 or -1):\n"
            "  For EACH arm: does the gripper open/close at the correct step relative to when the arm reaches the object? "
            "Closing too early (before reaching), or gripper sequence inverted vs demos: -1.\n\n"
            "CHECK 4 — Workspace reachability (0 or -1):\n"
            f"  Right arm should mostly operate in x > {round(30 * voxel_scale, 1)} (its reachable zone). Left arm should mostly operate in x < {round(70 * voxel_scale, 1)}. "
            "If an arm consistently reaches into the opposite side of the workspace (>3 steps): -1.\n\n"
            "Final score = 3 + check1 + check2 + check3 + check4, clamped to [1, 5].\n\n"
            "You MUST show your work for each check, then give the final score.\n"
            "Output ONLY valid JSON:\n"
            '{"check1": "+1 or -1: <reason>", "check2": "+1 or -1: <reason>", '
            '"check3": "0 or -1: <reason>", "check4": "0 or -1: <reason>", "score": <int 1-5>}'
        )

    def _preprocess_observation(self, obs, step, **kwargs):
        """Extract observation data once. Returns dicts needed for prompt building."""
        rgb_dict = {}
        mask_id_to_sim_name = {}
        mask_dict = {}
        point_cloud_dict = {}
        for camera in CAMERAS:
            rgb_img = obs[f'{camera}_rgb']
            rgb_img = rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
            rgb_img = np.clip(((rgb_img + 1.0) / 2 * 255).astype(np.uint8), 0, 255)
            rgb_dict[camera] = rgb_img

            # # Save RGB for debugging
            # img = Image.fromarray(rgb_img)
            # rgb_dir = os.path.join(self.savedir, 'rgb_dir', camera, str(self.episode_id))
            # os.makedirs(rgb_dir, exist_ok=True)
            # img.save(os.path.join(rgb_dir, f'{self.step}.png'))

            mask_id_to_sim_name.update(kwargs["mapping_dict"][f"{camera}_mask_id_to_name"])

            mask = obs[f'{camera}_mask']
            mask = mask.squeeze().cpu().numpy()
            mask_dict[camera] = mask

            # # Save mask for debugging
            # mask_dir = os.path.join(self.savedir, 'input_masks', camera, str(self.episode_id))
            # os.makedirs(mask_dir, exist_ok=True)
            # mask_pil = Image.fromarray(mask.astype(np.uint8))
            # mask_pil.save(os.path.join(mask_dir, f'{self.step}.png'))

            point_cloud = obs[f'{camera}_point_cloud'].cpu().squeeze().permute(1, 2, 0).numpy()
            point_cloud_dict[camera] = point_cloud

        return mask_dict, mask_id_to_sim_name, point_cloud_dict

    def _build_prompts(self, mask_dict, mask_id_to_sim_name, point_cloud_dict):
        """Build all prompt components once from the observation."""
        user_prompt_right, user_prompt_left, user_prompt_bi = self.handler.get_user_prompt(
            mask_dict, mask_id_to_sim_name, point_cloud_dict, self
        )
        system_prompt_leader = SYSTEM_PROMPT_RIGHT if self.leader == "right" else SYSTEM_PROMPT_LEFT
        system_prompt_follower = SYSTEM_PROMPT_LEFT if self.leader == "right" else SYSTEM_PROMPT_RIGHT

        # Parse ALL ICL examples + live observation (shared across candidates)
        examples_bi = user_prompt_bi.split(", {")
        examples_leader_raw = (user_prompt_right if self.leader == "right" else user_prompt_left).split(", {")

        # Parse individual ICL examples (without the live observation suffix)
        parsed_leader_examples = []
        for i, ex in enumerate(examples_leader_raw[:-1]):
            if i > 0:
                ex = "{" + ex
            parsed_leader_examples.append(ex)

        # Live observation suffix for leader prompt
        leader_obs_suffix = "{" + examples_leader_raw[-1]  # ends with ">"

        parsed_bi_examples = []
        for i, ex in enumerate(examples_bi[:-1]):
            if i > 0:
                ex = "{" + ex
            parsed_bi_examples.append(ex)

        # Parse the last observation dict (the live one) for follower
        last_example = "{" + examples_bi[-1]
        last_obs_str, _ = last_example.split(">")
        last_obs_dict = ast.literal_eval(last_obs_str)

        return (system_prompt_leader, system_prompt_follower,
                parsed_leader_examples, leader_obs_suffix,
                parsed_bi_examples, last_obs_dict, user_prompt_bi)

    def _build_shuffled_prompts(self, parsed_leader_examples, leader_obs_suffix,
                                parsed_bi_examples, candidate_idx):
        """Build shuffled leader + follower prompts for one candidate."""
        # Deterministic but different shuffle per candidate
        rng = random.Random(self.episode_id * N_CANDIDATES + candidate_idx)
        order = list(range(len(parsed_leader_examples)))
        rng.shuffle(order)

        # Shuffled leader prompt
        user_prompt_leader = ", ".join(parsed_leader_examples[i] for i in order) + ", " + leader_obs_suffix

        # Shuffled follower prefix (parse bimanual examples in shuffled order)
        follower_prefix = ""
        for i in order:
            ex = parsed_bi_examples[i]
            objects_dict, actions_list = ex.split(">")
            objects_dict = ast.literal_eval(objects_dict)
            actions_list = json.loads(actions_list)
            right_actions = [a[:7] for a in actions_list]
            left_actions = [a[7:] for a in actions_list]
            objects_dict[f'{self.leader}_arm'] = right_actions if self.leader == "right" else left_actions
            follower_target = left_actions if self.leader == "right" else right_actions
            follower_prefix += str(objects_dict) + ">" + str(follower_target) + ", "

        return user_prompt_leader, follower_prefix

    def _generate_candidate(self, system_prompt_leader, user_prompt_leader,
                            system_prompt_follower, follower_prefix, last_obs_dict,
                            candidate_idx):
        """Generate one leader+follower candidate. Thread-safe for OpenAI calls."""
        leader_messages = [
            {"role": "system", "content": system_prompt_leader},
            {"role": "user", "content": user_prompt_leader}
        ]
        output_text_leader = self.llm_call(leader_messages)
        output_list_leader = self._postprocess_single_arm(output_text_leader)
        print(f"Candidate {candidate_idx} leader: {output_list_leader}")

        # Build follower prompt with this leader's prediction
        follower_obs = dict(last_obs_dict)
        follower_obs[f'{self.leader}_arm'] = output_list_leader
        user_prompt_follower = follower_prefix + str(follower_obs) + ">"

        follower_messages = [
            {"role": "system", "content": system_prompt_follower},
            {"role": "user", "content": user_prompt_follower}
        ]
        output_text_follower = self.llm_call(follower_messages)
        output_list_follower = self._postprocess_single_arm(output_text_follower)
        print(f"Candidate {candidate_idx} follower: {output_list_follower}")

        # Combine leader + follower
        len_leader = len(output_list_leader)
        len_follower = len(output_list_follower)
        if len_leader > len_follower:
            output_list_follower.extend([output_list_follower[-1]] * (len_leader - len_follower))
        elif len_follower > len_leader:
            output_list_leader.extend([output_list_leader[-1]] * (len_follower - len_leader))

        combined_actions = []
        for leader_action, follower_action in zip(output_list_leader, output_list_follower):
            if self.leader == "right":
                combined_actions.append(leader_action + follower_action)
            else:
                combined_actions.append(follower_action + leader_action)

        return json.dumps(combined_actions)

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
                else: # Try direct JSON parsing
                    try:
                        actions = json.loads(output_text)
                    except JSONDecodeError:
                        if "\n" in output_text:
                            output_text = output_text.replace("\n", ",")
                        if (not output_text.startswith('[[')) and output_text.endswith(']]'):
                            output_text = output_text[:-1]
                        elif output_text.startswith('[[') and (not output_text.endswith(']]')):
                            output_text = output_text[1:]
                        actions = json.loads('['+output_text+']')
            if len(np.array(actions).shape) == 1:
                actions = [actions]
        except Exception as e:
            actions = fallback_single_arm_sequence(self.voxel_size, self.rotation_resolution, length=1)
            print(e)
            print('Error when parsing actions')
        output = []
        for action in actions:
            output.append(sanitize_single_arm_action(action, self.voxel_size, self.rotation_resolution))
        if not output:
            return fallback_single_arm_sequence(self.voxel_size, self.rotation_resolution, length=1)
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
                        actions = np.array(json.loads('['+output_text+']'))
        except Exception as e:
            actions = fallback_dual_arm_sequence(self.voxel_size, self.rotation_resolution, length=1)
            print(e)
            print('Error when parsing actions')
        return dual_arm_discrete_actions_to_continuous(
            actions,
            self.voxel_size,
            self.rotation_resolution,
        )

    def _validate_prediction(self, prediction, user_prompt_bi):
        """Score a candidate plan using LLM-as-judge with deterministic (temp=0) call."""
        examples = user_prompt_bi.split(", {")
        last_observation = "{" + examples[-1]
        icl_examples = ""
        for i, example in enumerate(examples[:-1]):
            if i > 0:
                example = "{" + example
            icl_examples += example + "\n"
        
        prompt_content = (
            f"REFERENCE DEMOS (observation>actions):\n{icl_examples}\n"
            f"NEW OBSERVATION:\n{last_observation}\n\n"
            f"CANDIDATE PLAN:\n{prediction}"
        )

        response = self.llm_call(
            [
                {"role": "system", "content": self.validator_system_prompt},
                {"role": "user", "content": prompt_content}
            ],
        )

        content = response.strip()
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                data = json.loads(content[start:end])
                score = int(data.get("score", 1))
                score = max(1, min(5, score))
                checks = {k: data.get(k, "") for k in ["check1", "check2", "check3", "check4"]}
                return score, str(checks)
            match = re.search(r'\b([1-5])\b', content)
            if match:
                return int(match.group(1)), "fallback parsing"
            return 1, "no score found"
        except Exception as e:
            print(f"Error parsing validator response: {e}")
            return 1, f"error {e}"

    def act(self, step: int, observation: dict,
            deterministic=False, **kwargs) -> ActResult:
        if len(self.actions) == 0:
            # 1. Preprocess observation ONCE
            mask_dict, mask_id_to_sim_name, point_cloud_dict = self._preprocess_observation(
                observation, step, **kwargs
            )

            # 2. Build all prompt components ONCE
            (system_prompt_leader, system_prompt_follower,
             parsed_leader_examples, leader_obs_suffix,
             parsed_bi_examples, last_obs_dict,
             user_prompt_bi) = self._build_prompts(mask_dict, mask_id_to_sim_name, point_cloud_dict)

            # 3. Generate N candidates in parallel (each with shuffled ICL order)
            _DEFAULT_CANDIDATE = json.dumps(
                fallback_dual_arm_sequence(self.voxel_size, self.rotation_resolution, length=1)
            )
            candidates = []
            with ThreadPoolExecutor(max_workers=N_CANDIDATES) as pool:
                futures = {}
                for i in range(N_CANDIDATES):
                    user_prompt_leader_i, follower_prefix_i = self._build_shuffled_prompts(
                        parsed_leader_examples, leader_obs_suffix,
                        parsed_bi_examples, i
                    )
                    futures[pool.submit(
                        self._generate_candidate,
                        system_prompt_leader, user_prompt_leader_i,
                        system_prompt_follower, follower_prefix_i, last_obs_dict, i
                    )] = i
                results = [None] * N_CANDIDATES
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        print(f"Candidate {idx} generation failed: {e!r}")
                        results[idx] = _DEFAULT_CANDIDATE
                candidates = results

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
                        score, reasoning = future.result()
                        scores[idx] = score
                        reasonings[idx] = reasoning
                    except Exception as e:
                        print(f"Validation of candidate {idx} failed: {e!r}")
                        scores[idx] = 0
                        reasonings[idx] = f"validation error: {e}"

            # 5. Select the best candidate
            best_idx = int(np.argmax(scores))
            print(f"BestOfN scores: {scores}, selected candidate {best_idx} (score={scores[best_idx]})")
            for i, (s, r) in enumerate(zip(scores, reasonings)):
                print(f"  Candidate {i}: score={s}, reasoning={r}")

            output = self._postprocess_dual_arm(candidates[best_idx])
            self.actions = output
            
        continuous_action = self.actions.pop(0)
        self.step += 1
        copy_obs = {k: v.cpu() for k, v in observation.items()}

        return ActResult(continuous_action,
                         observation_elements=copy_obs,
                         info=None)
    
    def act_summaries(self) -> List[Summary]:
        return []

    def reset(self):
        self._finalize_episode_stats()
        super().reset()
        self.step = 0
        self.episode_id += 1
        self._prev_action = None
        self.actions = []

    def load_weights(self, savedir: str):
        self.savedir = savedir
        self.handler = create_task_handler(self.task_name, self.model_config)
        
        if self.model_config.llm_call_style == "openai":
            print("using openai model")
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            service_tier = self.model_config.openai_service_tier
            print(f"using OpenAI {service_tier} service tier")
            raw_call = lambda messages: openai_call(
                client, self.model_config.name, messages, service_tier
            )
        elif self.model_config.llm_call_style == "huggingface":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("loading model from huggingface")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.name,
                torch_dtype="auto",
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.name)
            for param in model.parameters():
                param.requires_grad = False
            raw_call = lambda messages: huggingface_call(
                model, tokenizer, messages
            )
        elif self.model_config.llm_call_style == "vllm":
            print("using remote vllm-served model")
            client = OpenAI(
                base_url="http://127.0.0.1:8000/v1",
                api_key="password",
            )
            model_name = "/leonardo_scratch/large/userexternal/apalma01/llm_models/" + self.model_config.name.split("/")[-1]
            raw_call = lambda messages: openai_call(
                client, model_name, messages
            )

        self._setup_tracking_if_enabled(raw_call)
        return

    def build(self, training: bool, device=None):
        return

    def update(self, step: int, replay_sample: dict) -> dict:
        return {}
    
    def update_summaries(self) -> List[Summary]:
        return []

    def save_weights(self, savedir: str):
        return
