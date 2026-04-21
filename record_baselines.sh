#!/bin/bash

# Record baseline episodes for paper qualitative figure
# - Lift Ball      → RoboPrompt-SA (RP-SA has natural advantage on symmetric tasks)
# - Straighten Rope → RoboPrompt-DA (RP-DA has natural advantage on asymmetric tasks)
# - Tray Out of Oven → RoboPrompt-DA (loosely coupled, RP-DA edges RP-SA)
# Uses vLLM for speed

log_dir="/home/alessio/Desktop/icl_bimanual/paper_videos/baselines"
test_data_path="/home/alessio/Desktop/icl_bimanual/generated_data/test"
llm_call_style="openai"
model_name="gpt-5-mini"

declare -A task_agents
task_agents["bimanual_lift_ball"]="RoboPromptAgentBimanual"
task_agents["bimanual_straighten_rope"]="RoboPromptAgentOnePerArm"
task_agents["bimanual_take_tray_out_of_oven"]="RoboPromptAgentOnePerArm"

tasks=("bimanual_lift_ball" "bimanual_straighten_rope" "bimanual_take_tray_out_of_oven")

declare -A task_init_rotation
task_init_rotation["bimanual_lift_ball"]=280
task_init_rotation["bimanual_straighten_rope"]=280
task_init_rotation["bimanual_take_tray_out_of_oven"]=-245

for task in "${tasks[@]}"; do
    agent="${task_agents[$task]}"
    init_rot=${task_init_rotation[$task]}
    echo "Recording $agent on task: $task (init_rotation=$init_rot)"
    mkdir -p "${log_dir}/${task}/videos"

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
        framework.logdir="${log_dir}/${task}" \
        framework.eval_episodes=30 \
        framework.record_every_n=1 \
        rlbench.headless=True \
        framework.repeat_eval=1 \
        cinematic_recorder.enabled=True \
        cinematic_recorder.camera_resolution="[1280,720]" \
        cinematic_recorder.rotate_speed=0 \
        cinematic_recorder.init_rotation="$init_rot" \
        cinematic_recorder.fps=30 \
        > "${log_dir}/${task}/run.log" 2>&1

    echo "Done: ${task}. Videos saved to ${log_dir}/${task}/videos/"
    echo ""
done
