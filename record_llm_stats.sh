#!/bin/bash

# Record LLM call statistics for the base BiCICLe agent on bimanual_lift_ball.
# This produces the data for the supplementary material table:
# Variant | Calls/ep | Prompt tok/ep | Completion tok/ep | Total tok/ep | Median wall-time/ep

log_dir="/home/alessio/Desktop/icl_bimanual/logs/llm_stats"
test_data_path="/home/alessio/Desktop/icl_bimanual/generated_data/test"
llm_call_style="openai"
model_name="gpt-5-mini"
task="bimanual_lift_ball"
eval_episodes=100

mkdir -p "redirections/llm_stats"

# Base agent to test
agents=("LeaderFollower" "ArmsDebate" "BestOfN")

for agent in "${agents[@]}"; do
    echo "=============================================="
    echo "Recording LLM stats for agent: $agent"
    echo "=============================================="

    xvfb-run -a -s "-screen 0 1280x1024x24" \
    python main.py \
        method.name="$agent" \
        model.llm_call_style="$llm_call_style" \
        model.name="$model_name" \
        model.track_llm_stats=True \
        rlbench.tasks="[$task]" \
        rlbench.task_name="$task" \
        rlbench.episode_length=25 \
        rlbench.demo_path="$test_data_path" \
        framework.gpu=0 \
        framework.logdir="$log_dir" \
        framework.eval_episodes="$eval_episodes" \
        rlbench.headless=True \
        framework.repeat_eval=1 \
        > "redirections/llm_stats/${agent}.txt" 2>&1

    echo "Done with $agent. Output saved to redirections/llm_stats/${agent}.txt"
    echo ""
done

echo "=============================================="
echo "All configured agents tested. Summary of results:"
echo "=============================================="
for agent in "${agents[@]}"; do
    echo ""
    echo "--- $agent ---"
    grep -A 10 "LLM CALL STATISTICS" "redirections/llm_stats/${agent}.txt" 2>/dev/null || echo "  (no stats found)"
done
