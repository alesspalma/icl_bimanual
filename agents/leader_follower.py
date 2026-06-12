from typing import List
import re
import time
from yarr.agents.agent import Agent, Summary, ActResult
import json
import ast
import numpy as np
from PIL import Image
import os
from json import JSONDecodeError
from form_icl_demonstrations import create_task_handler, SYSTEM_PROMPT_RIGHT, SYSTEM_PROMPT_LEFT
from icl_utils import (
    CAMERAS,
    dual_arm_discrete_actions_to_continuous,
    fallback_dual_arm_sequence,
    fallback_single_arm_sequence,
    get_rotation_resolution,
    get_voxel_size,
    sanitize_single_arm_action,
)
import openai
from openai import OpenAI
from agents.llm_tracking import LLMTrackingMixin

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
                messages=messages
            )
            content = completion.choices[0].message.content
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

class LeaderFollower(LLMTrackingMixin, Agent):
    def __init__(self, task_name, model_config):
        self.episode_id = -1
        self.device = 'cuda'
        self.task_name = task_name
        self.model_config = model_config
        self.leader = model_config.leader
        self.voxel_size = get_voxel_size(model_config)
        self.rotation_resolution = get_rotation_resolution(model_config)

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

            # img = Image.fromarray(rgb_img)
            # rgb_dir = os.path.join(self.savedir, 'rgb_dir', camera, str(self.episode_id))
            # os.makedirs(rgb_dir, exist_ok=True)
            # # Save the image as PNG
            # img.save(os.path.join(rgb_dir, f'{self.step}.png'))

            mask_id_to_sim_name.update(kwargs["mapping_dict"][f"{camera}_mask_id_to_name"])

            mask = obs[f'{camera}_mask']
            mask = mask.squeeze().cpu().numpy() 

            mask_dict[camera] = mask

            # mask_dir = os.path.join(self.savedir, 'input_masks', camera, str(self.episode_id))
            # os.makedirs(mask_dir, exist_ok=True)
            # mask_pil = Image.fromarray(mask.astype(np.uint8))
            # mask_pil.save(os.path.join(mask_dir, f'{self.step}.png'))

            point_cloud = obs[f'{camera}_point_cloud'].cpu().squeeze().permute(1, 2, 0).numpy()
            point_cloud_dict[camera] = point_cloud
        if len(self.actions) == 0:
            user_prompt_right, user_prompt_left, user_prompt_bi = self.handler.get_user_prompt(mask_dict, mask_id_to_sim_name, point_cloud_dict, self)
            system_prompt_leader = SYSTEM_PROMPT_RIGHT if self.leader == "right" else SYSTEM_PROMPT_LEFT
            user_prompt_leader = user_prompt_right if self.leader == "right" else user_prompt_left

            print(system_prompt_leader)
            print()
            print(user_prompt_leader)

            messages = [
                    {"role": "system", "content": system_prompt_leader},
                    {"role": "user", "content": user_prompt_leader}
                ]
            try:
                output_text_leader = self.llm_call(messages)
            except Exception as e:
                print(f"LeaderFollower leader call failed: {e!r}")
                return json.dumps(fallback_dual_arm_sequence(self.voxel_size, self.rotation_resolution, length=1))
            print(f"Prediction:", output_text_leader)
            output_list_leader = self._postprocess_single_arm(output_text_leader)

            # now prepare the follower prompt
            examples = user_prompt_bi.split(", {") # split over ICL episodes
            user_prompt_follower = ""
            for i, example in enumerate(examples[:-1]):
                if i > 0:
                    example = "{"+example
                objects_dict, actions_list = example.split(">")
                objects_dict = ast.literal_eval(objects_dict)
                actions_list = json.loads(actions_list)
                right_actions = []
                left_actions = []
                for action in actions_list:
                    right_actions.append(action[:7])
                    left_actions.append(action[7:])
                objects_dict[f'{self.leader}_arm'] = right_actions if self.leader == "right" else left_actions
                user_prompt_follower += str(objects_dict) + ">" + str(left_actions if self.leader == "right" else right_actions) + ", "
            # add last live obs with the leader prediction
            example = "{"+examples[-1]
            objects_dict, _ = example.split(">")
            objects_dict = ast.literal_eval(objects_dict)
            objects_dict[f'{self.leader}_arm'] = output_list_leader
            user_prompt_follower += str(objects_dict) + ">"

            system_prompt_follower = SYSTEM_PROMPT_LEFT if self.leader == "right" else SYSTEM_PROMPT_RIGHT
            print(system_prompt_follower)
            print()
            print(user_prompt_follower)
            
            messages = [
                    {"role": "system", "content": system_prompt_follower},
                    {"role": "user", "content": user_prompt_follower}
                ]
            try:
                output_text_follower = self.llm_call(messages)
            except Exception as e:
                print(f"LeaderFollower follower call failed: {e!r}")
                return json.dumps(fallback_dual_arm_sequence(self.voxel_size, self.rotation_resolution, length=1))
            print(f"Prediction:", output_text_follower)
            output_list_follower = self._postprocess_single_arm(output_text_follower)

            # now combine leader and follower discrete actions
            combined_actions = []
            # first match length of the leader and follower outputs -> make the shorter equal to the longer by repeating last action
            len_leader = len(output_list_leader)
            len_follower = len(output_list_follower)
            if len_leader > len_follower:
                for _ in range(len_leader - len_follower):
                    output_list_follower.append(output_list_follower[-1])
            elif len_follower > len_leader:
                for _ in range(len_follower - len_leader):
                    output_list_leader.append(output_list_leader[-1])
            for leader_action, follower_action in zip(output_list_leader, output_list_follower):
                if self.leader == "right":
                    combined_action = leader_action + follower_action
                else:
                    combined_action = follower_action + leader_action
                combined_actions.append(combined_action)
            
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
                            output_text = output_text[:-1]  # Remove misaligned trailing ]
                        elif output_text.startswith('[[') and (not output_text.endswith(']]')):
                            output_text = output_text[1:]  # Remove misaligned leading [
                        actions = json.loads('['+output_text+']') # handle cases in which just external brackets are missing
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
        
        # get subsequent predicted actions
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
                else: # Try direct JSON parsing
                    try:
                        actions = np.array(json.loads(output_text))
                    except JSONDecodeError:
                        if "\n" in output_text:
                            output_text = output_text.replace("\n", ",")
                        if (not output_text.startswith('[[')) and output_text.endswith(']]'):
                            output_text = output_text[:-1]  # Remove misaligned trailing ]
                        elif output_text.startswith('[[') and (not output_text.endswith(']]')):
                            output_text = output_text[1:]  # Remove misaligned leading [
                        actions = np.array(json.loads('['+output_text+']')) # handle cases in which just external brackets are missing
        except Exception as e:
            actions = fallback_dual_arm_sequence(self.voxel_size, self.rotation_resolution, length=1)
            print(e)
            print('Error when parsing actions')
        return dual_arm_discrete_actions_to_continuous(
            actions,
            self.voxel_size,
            self.rotation_resolution,
        )
        
    def act(self, step: int, observation: dict,
            deterministic=False, **kwargs) -> ActResult:
        # inference
        output_text = self._preprocess(observation, step, **kwargs)
        if len(self.actions) == 0:
            output = self._postprocess_dual_arm(output_text) # extract continuous actions from the LLM prediction
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
        # no weight to load
        # only build task handler
        self.savedir = savedir

        self.handler = create_task_handler(self.task_name, self.model_config)
        
        if self.model_config.llm_call_style == "openai":
            print("using openai model")
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            raw_call = lambda messages: openai_call(client, self.model_config.name, messages)
        elif self.model_config.llm_call_style == "huggingface":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("loading model from huggingface")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.name,
                torch_dtype="auto",
                device_map="auto",
                # max_memory={0: "12GB"}
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.name)
            for param in model.parameters():
                param.requires_grad = False # no fine-tuning
            raw_call = lambda messages: huggingface_call(model, tokenizer, messages)
        elif self.model_config.llm_call_style == "vllm":
            print("using remote vllm-served model")
            client = OpenAI(
                base_url=f"http://127.0.0.1:8000/v1",
                api_key="password",
            )
            model_name = "/leonardo_scratch/large/userexternal/apalma01/llm_models/" + self.model_config.name.split("/")[-1]
            raw_call = lambda messages: openai_call(client, model_name, messages)

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
