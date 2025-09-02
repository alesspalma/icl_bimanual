from typing import List
import re
from yarr.agents.agent import Agent, Summary, ActResult
import json
import numpy as np
from PIL import Image
import os
from json import JSONDecodeError
from form_icl_demonstrations import create_task_handler, SYSTEM_PROMPT_RIGHT, SYSTEM_PROMPT_LEFT
from utils import SCENE_BOUNDS, ROTATION_RESOLUTION, discrete_euler_to_quaternion, CAMERAS
from openai import OpenAI

def openai_call(model_name, messages):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return completion.choices[0].message.content

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
    return response

class RoboPromptAgentOnePerArm(Agent):
    def __init__(self, task_name, model_config):
        self.episode_id = -1
        self.device = 'cuda'
        self.task_name = task_name
        self.model_config = model_config
        
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

            img = Image.fromarray(rgb_img)
            rgb_dir = os.path.join(self.savedir, 'rgb_dir', camera, str(self.episode_id))
            os.makedirs(rgb_dir, exist_ok=True)
            # Save the image as PNG
            img.save(os.path.join(rgb_dir, f'{self.step}.png'))

            mask_id_to_sim_name.update(kwargs["mapping_dict"][f"{camera}_mask_id_to_name"])

            mask = obs[f'{camera}_mask']
            mask = mask.squeeze().cpu().numpy() 

            mask_dict[camera] = mask

            mask_dir = os.path.join(self.savedir, 'input_masks', camera, str(self.episode_id))

            os.makedirs(mask_dir, exist_ok=True)
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            mask_pil.save(os.path.join(mask_dir, f'{self.step}.png'))

            point_cloud = obs[f'{camera}_point_cloud'].cpu().squeeze().permute(1, 2, 0).numpy()
            point_cloud_dict[camera] = point_cloud
        if (len(self.right_actions) == 0) or (len(self.left_actions) == 0):
            user_prompt_right, user_prompt_left = self.handler.get_user_prompt(mask_dict, mask_id_to_sim_name, point_cloud_dict, self)

            output_text_right = None
            if len(self.right_actions) == 0:
                print(SYSTEM_PROMPT_RIGHT) 
                print()
                print(user_prompt_right)

                messages = [
                        {"role": "system", "content": SYSTEM_PROMPT_RIGHT},
                        {"role": "user", "content": user_prompt_right}
                    ]
                output_text_right = self.llm_call(messages)
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
                output_text_left = self.llm_call(messages)
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
                else: # Try direct JSON parsing
                    try:
                        actions = np.array(json.loads(output_text))
                    except JSONDecodeError:
                        if (not output_text.startswith('[[')) and output_text.endswith(']]'):
                            output_text = output_text[:-1]  # Remove misaligned trailing ]
                        elif output_text.startswith('[[') and (not output_text.endswith(']]')):
                            output_text = output_text[1:]  # Remove misaligned leading [
                        actions = np.array(json.loads('['+output_text+']')) # handle cases in which just external brackets are missing
        except Exception as e:
            actions = [[57, 49, 87, 0, 39, 0, 1] for _ in range(26)]
            print(e)
            print('Error when parsing actions')
        if len(np.array(actions).shape) == 1:
            actions = [actions]
        output = []
        for action in actions:
            if len(action) != 7:
                action = [57, 49, 87, 0, 39, 0, 1]
            trans_indicies = np.array(action[:3])
            rot_and_grip_indicies = np.array(action[3:6])
            is_gripper_open = 1 if action[6] >= 0.5 else 0

            bounds = SCENE_BOUNDS
            res = (bounds[3:] - bounds[:3]) / 100
            attention_coordinate = bounds[:3] + res * trans_indicies + res / 2
            quat = discrete_euler_to_quaternion(rot_and_grip_indicies)
            
            continuous_action = np.concatenate([
                attention_coordinate,
                quat,
                [is_gripper_open],
                [1],
            ])
            output.append(continuous_action)
        
        # get subsequent predicted actions
        return output[:26]
        

    def act(self, step: int, observation: dict,
            deterministic=False, **kwargs) -> ActResult:
        # inference
        output_text = self._preprocess(observation, step, **kwargs)            
        if len(self.right_actions) == 0:
            output_right = self._postprocess(output_text[0]) # extract continuous actions from the LLM prediction
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
        super().reset()
        self.step = 0
        self.episode_id += 1
        self._prev_action = None
        self.right_actions = []
        self.left_actions = []

    def load_weights(self, savedir: str):
        # no weight to load
        # only build task handler
        self.savedir = savedir

        self.handler = create_task_handler(self.task_name)
        
        if self.model_config.llm_call_style == "openai":
            self.llm_call = lambda messages: openai_call(self.model_config.name, messages)
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
            self.llm_call = lambda messages: huggingface_call(model, tokenizer, messages)
        return

    def build(self, training: bool, device=None):
        return

    def update(self, step: int, replay_sample: dict) -> dict:
        return {}
    
    def update_summaries(self) -> List[Summary]:
        return []

    def save_weights(self, savedir: str):
        return