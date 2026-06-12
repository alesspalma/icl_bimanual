import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from yarr.agents.agent import ActResult

from agents.bestofn import BestOfN, N_CANDIDATES
from icl_utils import fallback_single_arm_sequence


class BestOfNV2(BestOfN):
    """Best-of-N with separate leader and follower selection stages."""

    def __init__(self, task_name, model_config):
        super().__init__(task_name, model_config)
        voxel_scale = self.voxel_size / 100
        leader_reachability = (
            f"The right arm should mostly operate in x > {round(30 * voxel_scale, 1)}."
            if self.leader == "right"
            else f"The left arm should mostly operate in x < {round(70 * voxel_scale, 1)}."
        )
        self.leader_validator_system_prompt = (
            f"You are a strict judge evaluating {self.leader}-arm robot action plans.\n\n"
            f"CONTEXT: The candidate controls only the {self.leader} arm of a bimanual "
            f"Franka Panda robot in a {self.voxel_size}x{self.voxel_size}x{self.voxel_size} "
            "voxel workspace. Each action is [x, y, z, rot1, rot2, rot3, gripper].\n\n"
            "TASK: Score the CANDIDATE plan from 1 to 5. START AT 3 and adjust:\n\n"
            "CHECK 1 - Target + trajectory match vs demos (+1 or -1):\n"
            f"  Does the candidate approach the same target region as the demos "
            f"(first action within {round(5 * voxel_scale, 1)} voxels of the demo trend) "
            "and follow the same trajectory shape? Both must be true for +1.\n\n"
            "CHECK 2 - Gripper logic (+1 or -1):\n"
            "  Does the gripper open and close at the same manipulation phase as the demos? "
            "Premature closure, inverted logic, or a missing grasp/release transition is -1.\n\n"
            "CHECK 3 - Workspace reachability (0 or -1):\n"
            f"  {leader_reachability} Penalize a plan that repeatedly moves into the "
            "opposite arm's workspace.\n\n"
            "CHECK 4 - Action-sequence coherence (0 or -1):\n"
            "  Penalize malformed, degenerate, or abrupt trajectories that do not follow "
            "the temporal pattern in the demonstrations.\n\n"
            "Final score = 3 + check1 + check2 + check3 + check4, clamped to [1, 5].\n\n"
            "You MUST show your work for each check, then give the final score.\n"
            "Output ONLY valid JSON:\n"
            '{"check1": "+1 or -1: <reason>", "check2": "+1 or -1: <reason>", '
            '"check3": "0 or -1: <reason>", "check4": "0 or -1: <reason>", '
            '"score": <int 1-5>}'
        )

    def _generate_leader_candidate(
        self,
        system_prompt_leader,
        user_prompt_leader,
        candidate_idx,
    ):
        output_text = self.llm_call([
            {"role": "system", "content": system_prompt_leader},
            {"role": "user", "content": user_prompt_leader},
        ])
        output_list = self._postprocess_single_arm(output_text)
        print(f"BestOfNV2 leader candidate {candidate_idx}: {output_list}")
        return output_list

    def _generate_follower_candidate(
        self,
        system_prompt_follower,
        follower_prefix,
        last_obs_dict,
        leader_actions,
        candidate_idx,
    ):
        follower_obs = dict(last_obs_dict)
        follower_obs[f"{self.leader}_arm"] = leader_actions
        output_text = self.llm_call([
            {"role": "system", "content": system_prompt_follower},
            {"role": "user", "content": follower_prefix + str(follower_obs) + ">"},
        ])
        output_list = self._postprocess_single_arm(output_text)
        print(f"BestOfNV2 follower candidate {candidate_idx}: {output_list}")
        return output_list

    def _validate_leader_prediction(self, prediction, user_prompt_leader):
        examples = user_prompt_leader.split(", {")
        last_observation = "{" + examples[-1]
        icl_examples = ""
        for i, example in enumerate(examples[:-1]):
            if i > 0:
                example = "{" + example
            icl_examples += example + "\n"

        prompt_content = (
            f"REFERENCE DEMOS (observation>{self.leader}-arm actions):\n"
            f"{icl_examples}\n"
            f"NEW OBSERVATION:\n{last_observation}\n\n"
            f"CANDIDATE {self.leader.upper()}-ARM PLAN:\n{json.dumps(prediction)}"
        )
        response = self.llm_call([
            {"role": "system", "content": self.leader_validator_system_prompt},
            {"role": "user", "content": prompt_content},
        ])
        return self._parse_validator_response(response)

    def _parse_validator_response(self, response):
        content = response.strip()
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != 0:
                data = json.loads(content[start:end])
                score = max(1, min(5, int(data.get("score", 1))))
                checks = {
                    key: data.get(key, "")
                    for key in ["check1", "check2", "check3", "check4"]
                }
                return score, str(checks)
            match = re.search(r"\b([1-5])\b", content)
            if match:
                return int(match.group(1)), "fallback parsing"
            return 1, "no score found"
        except Exception as exc:
            print(f"Error parsing BestOfNV2 validator response: {exc}")
            return 1, f"error {exc}"

    def _combine_actions(self, leader_actions, follower_actions):
        leader_padded = [list(action) for action in leader_actions]
        follower_padded = [list(action) for action in follower_actions]
        if len(leader_padded) > len(follower_padded):
            follower_padded.extend(
                [list(follower_padded[-1])] * (len(leader_padded) - len(follower_padded))
            )
        elif len(follower_padded) > len(leader_padded):
            leader_padded.extend(
                [list(leader_padded[-1])] * (len(follower_padded) - len(leader_padded))
            )

        combined = []
        for leader_action, follower_action in zip(leader_padded, follower_padded):
            if self.leader == "right":
                combined.append(leader_action + follower_action)
            else:
                combined.append(follower_action + leader_action)
        return json.dumps(combined)

    def _run_parallel(self, pool_inputs, work, default, failure_label):
        results = [None] * N_CANDIDATES
        with ThreadPoolExecutor(max_workers=N_CANDIDATES) as pool:
            futures = {
                pool.submit(work, *args): idx
                for idx, args in enumerate(pool_inputs)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f"{failure_label} {idx} failed: {exc!r}")
                    results[idx] = default()
        return results

    def _score_parallel(self, candidates, validate, failure_label):
        scores = [0] * N_CANDIDATES
        reasonings = [""] * N_CANDIDATES
        with ThreadPoolExecutor(max_workers=N_CANDIDATES) as pool:
            futures = {
                pool.submit(validate, idx, candidate): idx
                for idx, candidate in enumerate(candidates)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    scores[idx], reasonings[idx] = future.result()
                except Exception as exc:
                    print(f"{failure_label} {idx} failed: {exc!r}")
                    reasonings[idx] = f"validation error: {exc}"
        return scores, reasonings

    def act(self, step: int, observation: dict, deterministic=False, **kwargs):
        if len(self.actions) == 0:
            mask_dict, mask_id_to_sim_name, point_cloud_dict = self._preprocess_observation(
                observation, step, **kwargs
            )
            (
                system_prompt_leader,
                system_prompt_follower,
                parsed_leader_examples,
                leader_obs_suffix,
                parsed_bi_examples,
                last_obs_dict,
                user_prompt_bi,
            ) = self._build_prompts(mask_dict, mask_id_to_sim_name, point_cloud_dict)

            shuffled_prompts = [
                self._build_shuffled_prompts(
                    parsed_leader_examples,
                    leader_obs_suffix,
                    parsed_bi_examples,
                    candidate_idx,
                )
                for candidate_idx in range(N_CANDIDATES)
            ]
            user_prompt_leaders = [prompts[0] for prompts in shuffled_prompts]
            follower_prefixes = [prompts[1] for prompts in shuffled_prompts]

            leader_inputs = [
                (system_prompt_leader, user_prompt, idx)
                for idx, user_prompt in enumerate(user_prompt_leaders)
            ]
            leader_candidates = self._run_parallel(
                leader_inputs,
                self._generate_leader_candidate,
                lambda: fallback_single_arm_sequence(
                    self.voxel_size,
                    self.rotation_resolution,
                    length=1,
                ),
                "Leader candidate",
            )
            leader_scores, leader_reasonings = self._score_parallel(
                leader_candidates,
                lambda idx, candidate: self._validate_leader_prediction(
                    candidate,
                    user_prompt_leaders[idx],
                ),
                "Leader validation",
            )
            best_leader_idx = int(np.argmax(leader_scores))
            best_leader = leader_candidates[best_leader_idx]
            print(
                f"BestOfNV2 leader scores: {leader_scores}, selected candidate "
                f"{best_leader_idx} (score={leader_scores[best_leader_idx]})"
            )
            for idx, (score, reasoning) in enumerate(zip(leader_scores, leader_reasonings)):
                print(f"  Leader candidate {idx}: score={score}, reasoning={reasoning}")

            follower_inputs = [
                (
                    system_prompt_follower,
                    follower_prefix,
                    last_obs_dict,
                    best_leader,
                    idx,
                )
                for idx, follower_prefix in enumerate(follower_prefixes)
            ]
            follower_candidates = self._run_parallel(
                follower_inputs,
                self._generate_follower_candidate,
                lambda: fallback_single_arm_sequence(
                    self.voxel_size,
                    self.rotation_resolution,
                    length=1,
                ),
                "Follower candidate",
            )
            combined_candidates = [
                self._combine_actions(best_leader, follower_candidate)
                for follower_candidate in follower_candidates
            ]
            follower_scores, follower_reasonings = self._score_parallel(
                combined_candidates,
                lambda _idx, candidate: self._validate_prediction(
                    candidate,
                    user_prompt_bi,
                ),
                "Follower validation",
            )
            best_follower_idx = int(np.argmax(follower_scores))
            print(
                f"BestOfNV2 follower scores: {follower_scores}, selected candidate "
                f"{best_follower_idx} (score={follower_scores[best_follower_idx]})"
            )
            for idx, (score, reasoning) in enumerate(zip(follower_scores, follower_reasonings)):
                print(f"  Follower candidate {idx}: score={score}, reasoning={reasoning}")

            self.actions = self._postprocess_dual_arm(combined_candidates[best_follower_idx])

        continuous_action = self.actions.pop(0)
        self.step += 1
        copy_obs = {key: value.cpu() for key, value in observation.items()}
        return ActResult(continuous_action, observation_elements=copy_obs, info=None)
