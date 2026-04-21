#!/bin/bash

# Record baseline episodes for paper qualitative figure
# - Lift Ball      → RoboPrompt-SA (RP-SA has natural advantage on symmetric tasks)
# - Straighten Rope → RoboPrompt-DA (RP-DA has natural advantage on asymmetric tasks)
# - Tray Out of Oven → RoboPrompt-DA (loosely coupled, RP-DA edges RP-SA)
# Uses vLLM + Qwen2.5 for speed

log_dir="/media/nvme/palma/icl_bimanual/paper_videos/baselines"
test_data_path="/media/nvme/palma/icl_bimanual/generated_data/test"
llm_call_style="vllm"
model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"

tasks=("bimanual_straighten_rope" "bimanual_sweep_to_dustpan" "bimanual_take_tray_out_of_oven")

declare -A task_init_rotation
task_init_rotation["bimanual_lift_ball"]=280
task_init_rotation["bimanual_straighten_rope"]=280
task_init_rotation["bimanual_take_tray_out_of_oven"]=-245

for task in "${tasks[@]}"; do
    agent="${task_agents[$task]:-RoboPromptAgentOnePerArm}"
    init_rot=${task_init_rotation[$task]:-280}
    echo "Recording $agent on task: $task (init_rotation=$init_rot)"
    mkdir -p "${log_dir}/RPDA/${task}/videos"

    xvfb-run -a -s "-screen 0 1280x1024x24" \
    python main.py \
        method.name="$agent" \
        model.llm_call_style="$llm_call_style" \
        model.name="$model_name" \
        rlbench.tasks="[$task]" \
        rlbench.task_name="$task" \
        rlbench.episode_length=25 \
        rlbench.demo_path="$test_data_path" \
        framework.gpu=0 \
        framework.logdir="${log_dir}/RPDA/${task}" \
        framework.eval_episodes=40 \
        framework.record_every_n=1 \
        rlbench.headless=True \
        framework.repeat_eval=1 \
        cinematic_recorder.enabled=True \
        cinematic_recorder.camera_resolution="[1280,720]" \
        cinematic_recorder.rotate_speed=0 \
        cinematic_recorder.init_rotation="$init_rot" \
        cinematic_recorder.fps=30 \
        > "${log_dir}/RPDA/${task}/run.log" 2>&1

    echo "Done: ${task}. Videos saved to ${log_dir}/RPDA/${task}/videos/"
    echo ""
done
