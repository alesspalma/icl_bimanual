#!/bin/bash

set -euo pipefail

log_dir="/media/nvme/palma/icl_bimanual/logs"
train_data_path="/media/nvme/palma/icl_bimanual/generated_data/train"
test_data_path="/media/nvme/palma/icl_bimanual/generated_data/test"
llm_call_style="openai"
agent="LeaderFollower"
voxel_size=200
rotation_resolution=2.5
demonstrations_dir="demonstrations_vox200_rot2p5"

export CUDA_VISIBLE_DEVICES=0

tasks=(
    "bimanual_pick_laptop"
    "bimanual_straighten_rope"
    "bimanual_take_tray_out_of_oven"
    "bimanual_put_item_in_drawer"
)

run_name="${agent}_highres_vox${voxel_size}_rot${rotation_resolution//./p}"
redirect_dir="redirections/${llm_call_style}/${run_name}"

if [[ "$llm_call_style" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set. Export it before running this script."
    exit 1
fi

mkdir -p "$redirect_dir"

missing_demos=0
for task in "${tasks[@]}"; do
    for idx in {0..9}; do
        if [[ ! -f "${train_data_path}/${task}/${demonstrations_dir}/${idx}.txt" ]]; then
            missing_demos=1
            break 2
        fi
    done
done

if [[ "$missing_demos" -eq 1 ]]; then
    echo "Generating high-resolution ICL demonstrations in ${demonstrations_dir}"
    python form_icl_demonstrations.py \
        --voxel_size "$voxel_size" \
        --rotation_resolution "$rotation_resolution" \
        --demonstrations_dir "$demonstrations_dir" \
        --tasks "${tasks[@]}"
fi

for task in "${tasks[@]}"; do
    echo "Running high-resolution ${agent} evaluation for task: $task"

    xvfb-run -a -s "-screen 0 1280x1024x24" \
    python main.py \
        method.name="$agent" \
        model.llm_call_style="$llm_call_style" \
        model.name=gpt-5-mini \
        model.leader=right \
        model.voxel_size="$voxel_size" \
        model.rotation_resolution="$rotation_resolution" \
        model.demonstrations_dir="$demonstrations_dir" \
        rlbench.tasks="[$task]" \
        rlbench.task_name="$task" \
        rlbench.episode_length=25 \
        rlbench.demo_path="$test_data_path" \
        framework.gpu=0 \
        framework.logdir="${log_dir}/${run_name}" \
        framework.eval_episodes=100 \
        rlbench.headless=True \
        framework.repeat_eval=1 \
        > "${redirect_dir}/${task}.txt" 2>&1

    echo ""
done
