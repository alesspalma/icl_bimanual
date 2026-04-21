#!/bin/bash

# Record BiCICLe (LeaderFollower) episodes for paper qualitative figure
# Tasks: Lift Ball (tightly coupled symmetric), Straighten Rope (tightly coupled asymmetric),
#         Take Tray Out of Oven (loosely coupled)

log_dir="/home/alessio/Desktop/icl_bimanual/paper_videos/bicicle"
test_data_path="/home/alessio/Desktop/icl_bimanual/generated_data/test"
agent="LeaderFollower"
llm_call_style="openai"
model_name="gpt-5-mini"

tasks=("bimanual_lift_ball" "bimanual_straighten_rope" "bimanual_take_tray_out_of_oven")

declare -A task_init_rotation
task_init_rotation["bimanual_lift_ball"]=280
task_init_rotation["bimanual_straighten_rope"]=280
task_init_rotation["bimanual_take_tray_out_of_oven"]=-245

for task in "${tasks[@]}"; do
    init_rot=${task_init_rotation[$task]:-280}
    echo "Recording BiCICLe on task: $task (init_rotation=$init_rot)"
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
