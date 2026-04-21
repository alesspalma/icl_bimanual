from typing import List
import re
from yarr.agents.agent import Agent, Summary, ActResult
import json
import numpy as np
from PIL import Image
import os
from json import JSONDecodeError
from form_icl_demonstrations import create_task_handler, SYSTEM_PROMPT_RIGHT, SYSTEM_PROMPT_LEFT, SYSTEM_PROMPT_FOLLOWER_FILLIN
from icl_utils import SCENE_BOUNDS, ROTATION_RESOLUTION, discrete_euler_to_quaternion, CAMERAS
from openai import OpenAI

def openai_call(client, model_name, messages):
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

class LeaderFollowerFillIn(Agent):
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

            # img = Image.fromarray(rgb_img)
            # rgb_dir = os.path.join(self.savedir, 'rgb_dir', camera, str(self.episode_id))
            # os.makedirs(rgb_dir, exist_ok=True)
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
            user_prompt_leader = user_prompt_right

            print(SYSTEM_PROMPT_RIGHT) 
            print()
            print(user_prompt_leader)

            messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_RIGHT},
                    {"role": "user", "content": user_prompt_leader}
                ]
            output_text_leader = self.llm_call(messages)
            print(f"Prediction:", output_text_leader)
            output_list_leader = self._postprocess_single_arm(output_text_leader)

            masked_action = ["X"] * 7
            output_with_mask = []
            for action in output_list_leader:
                combined_action = action + masked_action
                output_with_mask.append(combined_action)
            user_prompt_follower = user_prompt_bi + str(output_with_mask).replace("'", "")

            print(SYSTEM_PROMPT_FOLLOWER_FILLIN) 
            print()
            print(user_prompt_follower)
            
            messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_FOLLOWER_FILLIN},
                    {"role": "user", "content": user_prompt_follower}
                ]
            output_text_follower = self.llm_call(messages)
            print(f"Prediction:", output_text_follower)

            return output_text_follower

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
            actions = [[57, 49, 87, 0, 39, 0, 1] for _ in range(26)]
            print(e)
            print('Error when parsing actions')
        output = []
        for action in actions:
            if len(action) != 7:
                action = [57, 49, 87, 0, 39, 0, 1]
            output.append(action)
        
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
            actions = [[57, 49, 87, 0, 39, 0, 1, 57, 49, 87, 0, 39, 0, 1] for _ in range(26)]
            print(e)
            print('Error when parsing actions')
        if len(np.array(actions).shape) == 1:
            actions = [actions]
        output = []
        for action in actions:
            if len(action) != 7*2: # predicting bimanual action directly for now
                action = [57, 49, 87, 0, 39, 0, 1, 57, 49, 87, 0, 39, 0, 1]
            temp_actions = []
            for i in range(2): # because two arms
                arm_action = action[i*7:(i+1)*7]
                trans_indicies = np.array(arm_action[:3])
                rot_and_grip_indicies = np.array(arm_action[3:6])
                is_gripper_open = 1 if arm_action[6] >= 0.5 else 0

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
                temp_actions.append(continuous_action)
            
            temp_actions = np.concatenate(temp_actions, axis=0)
            output.append(temp_actions)

        # get subsequent predicted actions
        return output[:26]
        
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
        super().reset()
        self.step = 0
        self.episode_id += 1
        self._prev_action = None
        self.actions = []

    def load_weights(self, savedir: str):
        # no weight to load
        # only build task handler
        self.savedir = savedir

        self.handler = create_task_handler(self.task_name)
        
        if self.model_config.llm_call_style == "openai":
            print("using openai model")
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.llm_call = lambda messages: openai_call(client, self.model_config.name, messages)
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
        elif self.model_config.llm_call_style == "vllm":
            print("using remote vllm-served model")
            client = OpenAI(
                base_url=f"http://127.0.0.1:8000/v1",
                api_key="password",
            )
            model_name = "/leonardo_scratch/large/userexternal/apalma01/llm_models/" + self.model_config.name.split("/")[-1]
            self.llm_call = lambda messages: openai_call(client, model_name, messages)

        return

    def build(self, training: bool, device=None):
        return

    def update(self, step: int, replay_sample: dict) -> dict:
        return {}
    
    def update_summaries(self) -> List[Summary]:
        return []

    def save_weights(self, savedir: str):
        return