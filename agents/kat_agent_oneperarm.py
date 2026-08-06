from typing import List
import re
import time
from yarr.agents.agent import Agent, Summary, ActResult
import json
import numpy as np
from PIL import Image
import os
from json import JSONDecodeError
from form_icl_demonstrations_kat import create_task_handler, SYSTEM_PROMPT_RIGHT, SYSTEM_PROMPT_LEFT
from icl_utils import CAMERAS, fallback_single_arm_sequence, get_rotation_resolution, get_voxel_size, single_arm_discrete_actions_to_continuous
import openai
from openai import OpenAI
from agents.llm_tracking import LLMTrackingMixin

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


class KATAgentOnePerArm(LLMTrackingMixin, Agent):
    """KAT baseline agent (one-per-arm): uses DINO keypoint observations, with separate LLM calls for right and left arms."""

    def __init__(self, task_name, model_config):
        self.episode_id = -1
        self.device = 'cuda'
        self.task_name = task_name
        self.model_config = model_config
        self.voxel_size = get_voxel_size(model_config)
        self.rotation_resolution = get_rotation_resolution(model_config)

    def _preprocess(self, obs, step, **kwargs):
        rgb_dict = {}
        point_cloud_dict = {}
        for camera in CAMERAS:
            # Decode RGB
            rgb_img = obs[f'{camera}_rgb']
            rgb_img = rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
            rgb_img = np.clip(((rgb_img + 1.0) / 2 * 255).astype(np.uint8), 0, 255)
            rgb_dict[camera] = rgb_img

            # # Save RGB for debugging
            # img = Image.fromarray(rgb_img)
            # rgb_dir = os.path.join(self.savedir, 'rgb_dir', camera, str(self.episode_id))
            # os.makedirs(rgb_dir, exist_ok=True)
            # img.save(os.path.join(rgb_dir, f'{self.step}.png'))

            # Point cloud (H, W, 3) in world frame
            point_cloud = obs[f'{camera}_point_cloud'].cpu().squeeze().permute(1, 2, 0).numpy()
            point_cloud_dict[camera] = point_cloud

        if (len(self.right_actions) == 0) or (len(self.left_actions) == 0):
            user_prompt_right, user_prompt_left = self.handler.get_user_prompt(rgb_dict, point_cloud_dict, self)

            output_text_right = None
            if len(self.right_actions) == 0:
                print(SYSTEM_PROMPT_RIGHT)
                print()
                print(user_prompt_right)

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_RIGHT},
                    {"role": "user", "content": user_prompt_right}
                ]
                try:
                    output_text_right = self.llm_call(messages)
                except Exception as e:
                    print(f"KATAgentOnePerArm right-arm call failed: {e!r}")
                    fallback = json.dumps(fallback_single_arm_sequence(self.voxel_size, self.rotation_resolution))
                    return fallback, fallback
                print(f"Prediction:", output_text_right)

            output_text_left = None
            if len(self.left_actions) == 0:
                print(SYSTEM_PROMPT_LEFT)
                print()
                print(user_prompt_left)

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_LEFT},
                    {"role": "user", "content": user_prompt_left}
                ]
                try:
                    output_text_left = self.llm_call(messages)
                except Exception as e:
                    print(f"KATAgentOnePerArm left-arm call failed: {e!r}")
                    fallback = json.dumps(fallback_single_arm_sequence(self.voxel_size, self.rotation_resolution))
                    return fallback, fallback
                print(f"Prediction:", output_text_left)

            return output_text_right, output_text_left

    def _postprocess(self, output_text):
        try:
            regex = r'^```json(\s*\[\s*(?:\[(?:\d+\s*,\s*){6}\d+\]\s*,\s*)*\[(?:\d+\s*,\s*){6}\d+\]\s*\])\s*```$'
            match = re.search(regex, output_text)
            if match:
                actions = np.array(json.loads(match.group(1)))
            else:
                regex = r'^```(\s*\[\s*(?:\[(?:\d+\s*,\s*){6}\d+\]\s*,\s*)*\[(?:\d+\s*,\s*){6}\d+\]\s*\])\s*```$'
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
            actions = fallback_single_arm_sequence(self.voxel_size, self.rotation_resolution)
            print(e)
            print('Error when parsing actions')
        return single_arm_discrete_actions_to_continuous(
            actions,
            self.voxel_size,
            self.rotation_resolution,
        )

    def act(self, step: int, observation: dict,
            deterministic=False, **kwargs) -> ActResult:
        output_text = self._preprocess(observation, step, **kwargs)
        if len(self.right_actions) == 0:
            output_right = self._postprocess(output_text[0])
            self.right_actions = output_right
        if len(self.left_actions) == 0:
            output_left = self._postprocess(output_text[1])
            self.left_actions = output_left

        continuous_action_right = self.right_actions.pop(0)
        continuous_action_left = self.left_actions.pop(0)
        continuous_action = np.concatenate([continuous_action_right, continuous_action_left], axis=0)

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
        self.right_actions = []
        self.left_actions = []

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
