"""
VLM Leader-Follower agent for bimanual manipulation.

Uses a Vision-Language Model (e.g. Qwen-2.5 VL via vLLM, or GPT-5) instead
of a text-only LLM.  Observations are images (concatenated front RGB + depth
visualisation, side-by-side) rather than text-based object positions.

The leaderfollower pattern is the same as in ``leader_follower.py``:
    1. Call the VLM with single-arm ICL demos → leader prediction.
    2. Build follower prompt that includes the leader's prediction,
       call the VLM again → follower prediction.
    3. Combine leader + follower into 14-dim bimanual actions.

Supported backends: OpenAI API (``llm_call_style: openai``) and
vLLM OpenAI-compatible server (``llm_call_style: vllm``).
"""

from typing import List
import re
import json
import os
import time

import numpy as np
from PIL import Image
from json import JSONDecodeError

from yarr.agents.agent import Agent, Summary, ActResult
from form_icl_demonstrations_vlm import (
    create_task_handler,
    SYSTEM_PROMPT_RIGHT,
    SYSTEM_PROMPT_LEFT,
    SYSTEM_PROMPT_FOLLOWER,
    depth_from_point_cloud,
    _depth_to_visualization,
    _pil_to_data_url,
)
from icl_utils import CAMERAS, dual_arm_discrete_actions_to_continuous, fallback_dual_arm_sequence, fallback_single_arm_sequence, get_rotation_resolution, get_voxel_size, sanitize_single_arm_action
import openai
from openai import OpenAI
import io

_RETRYABLE = (
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.RateLimitError,
)


# ──────────────────── VLM call ────────────────────
def vlm_call_openai(client, model_name, messages, max_retries=5):
    """Send a multimodal (vision) request via the OpenAI-compatible API."""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            return completion.choices[0].message.content
        except _RETRYABLE as e:
            if attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                print(f"vlm_call_openai retry {attempt + 1}/{max_retries}: {e!r}  "
                      f"(sleeping {wait}s)")
                time.sleep(wait)
            else:
                raise


def _data_url_to_pil(data_url):
    """Decode a ``data:image/…;base64,…`` URL back to a PIL Image."""
    import base64 as _b64
    header, b64data = data_url.split(",", 1)
    raw = _b64.b64decode(b64data)
    return Image.open(io.BytesIO(raw))


def huggingface_vlm_call(model, processor, messages):
    """
    Local VLM inference via HuggingFace ``transformers``.

    Accepts OpenAI-style multimodal messages (with ``image_url`` content
    blocks containing base64 data-URLs) and converts them to the format
    expected by HuggingFace VLM processors.
    """
    # Collect PIL images in order
    images: list = []
    hf_messages = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # System / plain-text messages
        if isinstance(content, str):
            hf_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue

        # Multimodal content list
        hf_content = []
        for block in content:
            if block["type"] == "text":
                hf_content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image_url":
                url = block["image_url"]["url"]
                pil_img = _data_url_to_pil(url)
                images.append(pil_img)
                hf_content.append({"type": "image"})
        hf_messages.append({"role": role, "content": hf_content})

    # Build prompt text via the processor's chat template
    text = processor.apply_chat_template(
        hf_messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenise text + images together
    inputs = processor(
        text=[text], images=images if images else None,
        return_tensors="pt", padding=True,
    ).to("cuda")

    generated_ids = model.generate(
        **inputs
    )
    # Strip the prompt tokens
    generated_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    return response


# ──────────────────── Agent ────────────────────
class VLMLeaderFollower(Agent):
    def __init__(self, task_name, model_config):
        self.episode_id = -1
        self.device = "cuda"
        self.task_name = task_name
        self.model_config = model_config
        self.leader = model_config.leader
        self.voxel_size = get_voxel_size(model_config)
        self.rotation_resolution = get_rotation_resolution(model_config)

    # ── live query image construction ──────────────────────────────────
    def _create_query_image(self, obs):
        """
        Build the combined (front RGB | depth) query image from the live
        simulator observation.

        Depth is reconstructed from the front camera's world-frame point
        cloud and camera extrinsics (no raw depth buffer needed).
        """
        # Front RGB ────────────────────────────────────────────────────
        rgb = obs["front_rgb"]
        rgb = rgb.squeeze().permute(1, 2, 0).cpu().numpy()
        rgb = np.clip(((rgb + 1.0) / 2 * 255).astype(np.uint8), 0, 255)
        rgb_img = Image.fromarray(rgb)

        # Front depth from point cloud ─────────────────────────────────
        point_cloud = (
            obs["front_point_cloud"].cpu().squeeze().permute(1, 2, 0).numpy()
        )
        try:
            extrinsics = obs["front_camera_extrinsics"]
            if hasattr(extrinsics, "cpu"):
                extrinsics = extrinsics.cpu().numpy()
            extrinsics = np.squeeze(extrinsics)
            depth_vis = depth_from_point_cloud(point_cloud, extrinsics)
        except (KeyError, Exception):
            # Fallback: Euclidean norm as pseudo-depth
            depth_raw = np.linalg.norm(point_cloud, axis=-1)
            depth_vis = _depth_to_visualization(depth_raw)

        depth_img = Image.fromarray(depth_vis).convert("RGB")

        # Concatenate side by side ─────────────────────────────────────
        w, h = rgb_img.size
        combined = Image.new("RGB", (w * 2, h))
        combined.paste(rgb_img, (0, 0))
        combined.paste(depth_img, (w, 0))
        return combined

    # ── main inference logic ──────────────────────────────────────────
    def _preprocess(self, obs, step, **kwargs):
        # # Save RGB for every step (logging / debugging)
        # for camera in CAMERAS:
        #     rgb_img = obs[f"{camera}_rgb"]
        #     rgb_img = rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
        #     rgb_img = np.clip(((rgb_img + 1.0) / 2 * 255).astype(np.uint8), 0, 255)
        #     img = Image.fromarray(rgb_img)
        #     rgb_dir = os.path.join(
        #         self.savedir, "rgb_dir", camera, str(self.episode_id)
        #     )
        #     os.makedirs(rgb_dir, exist_ok=True)
        #     img.save(os.path.join(rgb_dir, f"{self.step}.png"))

        if len(self.actions) == 0:
            # ── 1. Build query image from live observation ────────────
            combined_image = self._create_query_image(obs)

            # ── 2. Retrieve ICL demonstrations ───────────────────────
            leader_content, follower_examples = self.handler.get_user_prompt(
                combined_image, self
            )

            # ── 3. Leader VLM call ───────────────────────────────────
            system_prompt_leader = (
                SYSTEM_PROMPT_RIGHT if self.leader == "right" else SYSTEM_PROMPT_LEFT
            )

            leader_messages = [
                {"role": "system", "content": system_prompt_leader},
                {"role": "user", "content": leader_content},
            ]

            print(system_prompt_leader)
            print()
            print(leader_content[-1:][0])

            try:
                output_text_leader = self.vlm_call(leader_messages)
            except Exception as e:
                print(f"VLMLeaderFollower leader call failed: {e!r}")
                return json.dumps(fallback_dual_arm_sequence(self.voxel_size, self.rotation_resolution))
            print(f"Leader prediction: {output_text_leader}")
            output_list_leader = self._postprocess_single_arm(output_text_leader)

            # ── 4. Follower VLM call ─────────────────────────────────
            follower_content = self.handler.build_follower_prompt(
                combined_image, follower_examples, output_list_leader
            )

            system_prompt_follower = SYSTEM_PROMPT_LEFT if self.leader == "right" else SYSTEM_PROMPT_RIGHT
            follower_messages = [
                {"role": "system", "content": system_prompt_follower},
                {"role": "user", "content": follower_content},
            ]

            print(system_prompt_follower)
            print()
            print(follower_content[-1:][0])

            try:
                output_text_follower = self.vlm_call(follower_messages)
            except Exception as e:
                print(f"VLMLeaderFollower follower call failed: {e!r}")
                return json.dumps(fallback_dual_arm_sequence(self.voxel_size, self.rotation_resolution))
            print(f"Follower prediction: {output_text_follower}")
            output_list_follower = self._postprocess_single_arm(output_text_follower)

            # ── 5. Combine leader + follower ─────────────────────────
            # Pad shorter list by repeating last action
            len_leader = len(output_list_leader)
            len_follower = len(output_list_follower)
            if len_leader > len_follower:
                for _ in range(len_leader - len_follower):
                    output_list_follower.append(output_list_follower[-1])
            elif len_follower > len_leader:
                for _ in range(len_follower - len_leader):
                    output_list_leader.append(output_list_leader[-1])

            combined_actions = []
            for leader_action, follower_action in zip(
                output_list_leader, output_list_follower
            ):
                if self.leader == "right":
                    combined_actions.append(leader_action + follower_action)
                else:
                    combined_actions.append(follower_action + leader_action)

            return json.dumps(combined_actions)

    # ── postprocessing (same as LeaderFollower) ───────────────────────
    def _postprocess_single_arm(self, output_text):
        """Parse a VLM response into a list of 7-dim discrete actions."""
        if output_text.startswith(">"):
            output_text = output_text[1:]
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
                        if (not output_text.startswith("[[")) and output_text.endswith("]]"):
                            output_text = output_text[:-1]
                        elif output_text.startswith("[[") and (not output_text.endswith("]]")):
                            output_text = output_text[1:]
                        actions = json.loads("[" + output_text + "]")
            if len(np.array(actions).shape) == 1:
                actions = [actions]
        except Exception as e:
            actions = fallback_single_arm_sequence(self.voxel_size, self.rotation_resolution)
            print(e)
            print("Error when parsing actions")
        output = []
        for action in actions:
            output.append(sanitize_single_arm_action(action, self.voxel_size, self.rotation_resolution))
        return output[:26]

    def _postprocess_dual_arm(self, output_text):
        """Convert discrete 14-dim predictions to continuous actions."""
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
                        if (not output_text.startswith("[[")) and output_text.endswith("]]"):
                            output_text = output_text[:-1]
                        elif output_text.startswith("[[") and (not output_text.endswith("]]")):
                            output_text = output_text[1:]
                        actions = np.array(json.loads("[" + output_text + "]"))
        except Exception as e:
            actions = fallback_dual_arm_sequence(self.voxel_size, self.rotation_resolution)
            print(e)
            print("Error when parsing actions")
        return dual_arm_discrete_actions_to_continuous(
            actions,
            self.voxel_size,
            self.rotation_resolution,
        )

    # ── YARR Agent interface ──────────────────────────────────────────
    def act(self, step: int, observation: dict,
            deterministic=False, **kwargs) -> ActResult:
        output_text = self._preprocess(observation, step, **kwargs)
        if len(self.actions) == 0:
            output = self._postprocess_dual_arm(output_text)
            self.actions = output

        continuous_action = self.actions.pop(0)
        self.step += 1

        copy_obs = {k: v.cpu() for k, v in observation.items()}
        return ActResult(
            continuous_action, observation_elements=copy_obs, info=None
        )

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
        self.handler = create_task_handler(self.task_name, self.model_config)

        if self.model_config.llm_call_style == "openai":
            print("Using OpenAI VLM")
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.vlm_call = lambda messages: vlm_call_openai(
                client, self.model_config.name, messages
            )
        elif self.model_config.llm_call_style == "huggingface":
            from transformers import AutoProcessor, AutoModelForImageTextToText
            print("Loading VLM from HuggingFace")
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_config.name,
                torch_dtype="auto",
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(self.model_config.name)
            for param in model.parameters():
                param.requires_grad = False  # no fine-tuning
            self.vlm_call = lambda messages: huggingface_vlm_call(
                model, processor, messages
            )
        elif self.model_config.llm_call_style == "vllm":
            print("Using remote vLLM-served VLM")
            client = OpenAI(
                base_url="http://127.0.0.1:8000/v1",
                api_key="password",
            )
            model_name = (
                "/leonardo_scratch/large/userexternal/apalma01/llm_models/"
                + self.model_config.name.split("/")[-1]
            )
            self.vlm_call = lambda messages: vlm_call_openai(
                client, model_name, messages
            )
        else:
            raise ValueError(
                f"Unsupported llm_call_style '{self.model_config.llm_call_style}' "
                f"for VLM agent. Use 'openai', 'huggingface', or 'vllm'."
            )

    def build(self, training: bool, device=None):
        return

    def update(self, step: int, replay_sample: dict) -> dict:
        return {}

    def update_summaries(self) -> List[Summary]:
        return []

    def save_weights(self, savedir: str):
        return
