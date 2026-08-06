from typing import List
import json
import os
import re
import time

import numpy as np
import openai
from openai import OpenAI
from json import JSONDecodeError
from yarr.agents.agent import Agent, Summary, ActResult

from agents.llm_tracking import LLMTrackingMixin
from form_icl_demonstrations import create_task_handler
from icl_utils import (
    CAMERAS,
    dual_arm_discrete_actions_to_continuous,
    fallback_dual_arm_sequence,
    fallback_single_arm_sequence,
    get_rotation_resolution,
    get_voxel_size,
    sanitize_single_arm_action,
)

_RETRYABLE = (
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.RateLimitError,
)

SYSTEM_PROMPT_SEQUENTIAL = (
    "You are controlling at each turn one arm of a bimanual Franka Panda robot with "
    "parallel grippers. We provide you with some demos in the format of "
    "observation>[action_1, action_2, ...]. Then you will receive a new "
    "observation and you need to output a list of actions that matches the "
    "trend in the demos. Do not output anything else."
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
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
            return content, usage
        except _RETRYABLE as e:
            if attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                print(
                    f"openai_call retry {attempt + 1}/{max_retries}: {e!r}  "
                    f"(sleeping {wait}s)"
                )
                time.sleep(wait)
            else:
                raise


def huggingface_call(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024,
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response, None


class SequentialArms(LLMTrackingMixin, Agent):
    """Shared-history 7D+7D ablation without leader-follower prompt restructuring."""

    def __init__(self, task_name, model_config):
        self.episode_id = -1
        self.device = "cuda"
        self.task_name = task_name
        self.model_config = model_config
        self.first_arm = model_config.leader
        self.voxel_size = get_voxel_size(model_config)
        self.rotation_resolution = get_rotation_resolution(model_config)

    def _collect_prompt_inputs(self, obs, **kwargs):
        mask_id_to_sim_name = {}
        mask_dict = {}
        point_cloud_dict = {}

        for camera in CAMERAS:
            mask_id_to_sim_name.update(
                kwargs["mapping_dict"][f"{camera}_mask_id_to_name"]
            )
            mask_dict[camera] = obs[f"{camera}_mask"].squeeze().cpu().numpy()
            point_cloud_dict[camera] = (
                obs[f"{camera}_point_cloud"].cpu().squeeze().permute(1, 2, 0).numpy()
            )

        return mask_dict, mask_id_to_sim_name, point_cloud_dict

    def _preprocess(self, obs, step, **kwargs):
        if len(self.actions) != 0:
            return None

        mask_dict, mask_id_to_sim_name, point_cloud_dict = self._collect_prompt_inputs(
            obs, **kwargs
        )
        user_prompt_right, user_prompt_left = self.handler.get_user_prompt(
            mask_dict,
            mask_id_to_sim_name,
            point_cloud_dict,
            self,
        )

        if self.first_arm == "right":
            first_arm = "right"
            second_arm = "left"
            first_user_prompt = user_prompt_right
            second_user_prompt = user_prompt_left
        else:
            first_arm = "left"
            second_arm = "right"
            first_user_prompt = user_prompt_left
            second_user_prompt = user_prompt_right

        print(f"SequentialArms first arm: {first_arm}")
        print(SYSTEM_PROMPT_SEQUENTIAL)
        print()
        print(first_user_prompt)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SEQUENTIAL},
            {"role": "user", "content": first_user_prompt},
        ]
        try:
            output_text_first = self.llm_call(messages)
        except Exception as e:
            print(f"SequentialArms {first_arm}-arm call failed: {e!r}")
            return json.dumps(
                fallback_dual_arm_sequence(
                    self.voxel_size,
                    self.rotation_resolution,
                    length=1,
                )
            )
        print("Prediction:", output_text_first)
        output_list_first = self._postprocess_single_arm(output_text_first)

        messages.append({"role": "assistant", "content": output_text_first})
        messages.append({"role": "user", "content": second_user_prompt})

        print(second_user_prompt)

        try:
            output_text_second = self.llm_call(messages)
        except Exception as e:
            print(f"SequentialArms {second_arm}-arm call failed: {e!r}")
            return json.dumps(
                fallback_dual_arm_sequence(
                    self.voxel_size,
                    self.rotation_resolution,
                    length=1,
                )
            )
        print("Prediction:", output_text_second)
        output_list_second = self._postprocess_single_arm(output_text_second)

        combined_actions = self._combine_actions(
            first_arm,
            output_list_first,
            output_list_second,
        )
        return json.dumps(combined_actions)

    def _postprocess_single_arm(self, output_text):
        try:
            regex = r"^```json(\s*\[\s*(?:\[(?:\d+\s*,\s*){6}\d+\]\s*,\s*)*\[(?:\d+\s*,\s*){6}\d+\]\s*\])\s*```$"
            match = re.search(regex, output_text)
            if match:
                actions = json.loads(match.group(1))
            else:
                regex = r"^```(\s*\[\s*(?:\[(?:\d+\s*,\s*){6}\d+\]\s*,\s*)*\[(?:\d+\s*,\s*){6}\d+\]\s*\])\s*```$"
                match = re.search(regex, output_text)
                if match:
                    actions = json.loads(match.group(1))
                else:
                    try:
                        actions = json.loads(output_text)
                    except JSONDecodeError:
                        if "\n" in output_text:
                            output_text = output_text.replace("\n", ",")
                        if (not output_text.startswith("[[")) and output_text.endswith("]]"):
                            output_text = output_text[:-1]
                        elif output_text.startswith("[[") and (
                            not output_text.endswith("]]")
                        ):
                            output_text = output_text[1:]
                        actions = json.loads("[" + output_text + "]")
            if len(np.array(actions).shape) == 1:
                actions = [actions]
        except Exception as e:
            actions = fallback_single_arm_sequence(
                self.voxel_size,
                self.rotation_resolution,
                length=1,
            )
            print(e)
            print("Error when parsing actions")

        output = [
            sanitize_single_arm_action(action, self.voxel_size, self.rotation_resolution)
            for action in actions
        ]
        if not output:
            return fallback_single_arm_sequence(
                self.voxel_size,
                self.rotation_resolution,
                length=1,
            )
        return output[:26]

    def _combine_actions(self, first_arm, first_actions, second_actions):
        if len(first_actions) > len(second_actions):
            second_actions = second_actions + [second_actions[-1]] * (
                len(first_actions) - len(second_actions)
            )
        elif len(second_actions) > len(first_actions):
            first_actions = first_actions + [first_actions[-1]] * (
                len(second_actions) - len(first_actions)
            )

        combined_actions = []
        for first_action, second_action in zip(first_actions, second_actions):
            if first_arm == "right":
                combined_actions.append(first_action + second_action)
            else:
                combined_actions.append(second_action + first_action)
        return combined_actions

    def _postprocess_dual_arm(self, output_text):
        try:
            actions = np.array(json.loads(output_text))
        except Exception as e:
            actions = fallback_dual_arm_sequence(
                self.voxel_size,
                self.rotation_resolution,
                length=1,
            )
            print(e)
            print("Error when parsing actions")
        return dual_arm_discrete_actions_to_continuous(
            actions,
            self.voxel_size,
            self.rotation_resolution,
        )

    def act(self, step: int, observation: dict, deterministic=False, **kwargs) -> ActResult:
        output_text = self._preprocess(observation, step, **kwargs)
        if len(self.actions) == 0:
            output = self._postprocess_dual_arm(output_text)
            self.actions = output

        continuous_action = self.actions.pop(0)
        self.step += 1

        copy_obs = {k: v.cpu() for k, v in observation.items()}
        return ActResult(continuous_action, observation_elements=copy_obs, info=None)

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
                client,
                self.model_config.name,
                messages,
                service_tier,
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
                base_url="http://127.0.0.1:8000/v1",
                api_key="password",
            )
            model_name = (
                "/leonardo_scratch/large/userexternal/apalma01/llm_models/"
                + self.model_config.name.split("/")[-1]
            )
            raw_call = lambda messages: openai_call(client, model_name, messages)
        else:
            raise ValueError(f"Unsupported llm_call_style: {self.model_config.llm_call_style}")

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
