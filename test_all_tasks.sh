#!/bin/bash

# Parameters
log_dir="/home/alessio/Desktop/icl_bimanual/logs"
test_data_path="/home/alessio/Desktop/icl_bimanual/generated_data/test"
agent="RoboPromptAgentOnePerArm"
mkdir -p "redirections/$agent"

methods=("bimanual_dual_push_buttons" "bimanual_handover_item" "bimanual_handover_item_easy" "bimanual_lift_ball" "bimanual_lift_tray" "bimanual_pick_laptop" "bimanual_pick_plate" "bimanual_push_box" "bimanual_put_bottle_in_fridge" "bimanual_straighten_rope" "bimanual_sweep_to_dustpan" "bimanual_take_tray_out_of_oven" "bimanual_put_item_in_drawer")
# methods=("bimanual_take_tray_out_of_oven")

# Loop through each method
for method in "${methods[@]}"; do
    echo "Running evaluation for method: $method"

    xvfb-run -a -s "-screen 0 1280x1024x24" \
    python main.py \
        method.name="$agent" \
        model.llm_call_style=openai \
        model.name=gpt-4.1-mini \
        rlbench.tasks="[$method]" \
        rlbench.task_name="$method" \
        rlbench.episode_length=25 \
        rlbench.demo_path="$test_data_path" \
        framework.gpu=0 \
        framework.logdir="$log_dir" \
        framework.eval_episodes=100 \
        rlbench.headless=True \
        framework.repeat_eval=1 \
        > "redirections/$agent/$method.txt" 2>&1

    echo ""
done
