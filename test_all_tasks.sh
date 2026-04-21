#!/bin/bash

# Parameters
log_dir="/media/nvme/palma/icl_bimanual/logs"
test_data_path="/media/nvme/palma/icl_bimanual/generated_data/test"
llm_call_style="openai"
agent="PingPongBestOfN"
mkdir -p "redirections/$llm_call_style/$agent"

methods=("bimanual_lift_ball" "bimanual_push_box" "bimanual_put_item_in_drawer")
# methods=("bimanual_take_tray_out_of_oven")

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
