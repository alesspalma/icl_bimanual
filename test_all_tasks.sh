#!/bin/bash

# Parameters
log_dir="/home/alessio/Desktop/icl_bimanual/logs"
test_data_path="/home/alessio/Desktop/icl_bimanual/generated_data/test"
llm_call_style="openai"
agent="LeaderFollower"
mkdir -p "redirections/$llm_call_style/$agent"

methods=("bimanual_dual_push_buttons" "bimanual_handover_item_easy" "bimanual_handover_item" "bimanual_lift_ball" "bimanual_lift_tray" "bimanual_pick_laptop" "bimanual_pick_plate" "bimanual_push_box" "bimanual_put_bottle_in_fridge" "bimanual_put_item_in_drawer" "bimanual_straighten_rope" "bimanual_sweep_to_dustpan" "bimanual_take_tray_out_of_oven")

# Loop through each method
for method in "${methods[@]}"; do
    echo "Running evaluation for method: $method"

    xvfb-run -a -s "-screen 0 1280x1024x24" \
    python main.py \
        method.name="$agent" \
        model.llm_call_style="$llm_call_style" \
        model.name=gpt-5-mini \
        rlbench.tasks="[$method]" \
        rlbench.task_name="$method" \
        rlbench.episode_length=25 \
        rlbench.demo_path="$test_data_path" \
        framework.gpu=0 \
        framework.logdir="$log_dir" \
        framework.eval_episodes=100 \
        rlbench.headless=True \
        framework.repeat_eval=1 \
        > "redirections/$llm_call_style/$agent/$method.txt" 2>&1

    echo ""
done
