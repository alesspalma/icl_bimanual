#!/bin/bash

# Parameters
log_dir="/home/alessio/Desktop/icl_bimanual/logs"
test_data_path="/home/alessio/Desktop/icl_bimanual/generated_data/test"

methods=("bimanual_dual_push_buttons" "bimanual_handover_item" "bimanual_handover_item_easy" "bimanual_lift_ball" "bimanual_lift_tray" "bimanual_pick_laptop" "bimanual_pick_plate" "bimanual_push_box" "bimanual_put_bottle_in_fridge" "bimanual_straighten_rope" "bimanual_sweep_to_dustpan" "bimanual_take_tray_out_of_oven" "bimanual_put_item_in_drawer")
# methods=("bimanual_take_tray_out_of_oven")

# Loop through each method
for method in "${methods[@]}"; do
    echo "Running evaluation for method: $method"

    python main.py \
        method.name=RoboPromptAgentOnePerArm \
        model.llm_call_style=huggingface \
        model.name=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8 \
        rlbench.tasks="[$method]" \
        rlbench.task_name="$method" \
        rlbench.episode_length=25 \
        rlbench.demo_path="$test_data_path" \
        framework.gpu=0 \
        framework.logdir="$log_dir" \
        framework.eval_episodes=25 \
        rlbench.headless=True \
        framework.repeat_eval=1

    echo ""
done
