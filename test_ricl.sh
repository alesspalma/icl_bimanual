#!/bin/bash
#
# test_ricl.sh — Full RICL evaluation pipeline for bimanual tasks.
#
# TWO ENVIRONMENTS are used, each for a specific purpose:
#
#   openpi venv  (ricl_openpi/.venv)
#     → RICL policy servers (Pi0-FAST model, DINOv2 retrieval, JAX inference)
#     → Activated via: source ricl_openpi/.venv/bin/activate
#     → Launched with:  uv run  (from inside ricl_openpi/)
#
#   icl_bimanual conda env
#     → RLBench evaluation (main.py + RICLAgent)
#     → Provides: PyRep, RLBench, YARR, torch, openpi_client websocket client
#     → Launched with: conda run -n icl_bimanual
#
# For each task this script:
#   1. Starts TWO RICL servers with the openpi venv
#   2. Waits for both to be ready
#   3. Runs the RLBench evaluation with the icl_bimanual conda env
#   4. Kills both servers before moving to the next task
#
# Prerequisites (run once):
#   1. Install the openpi venv:
#        cd ricl_openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync
#        source .venv/bin/activate
#        uv pip install tensorflow-datasets tensorflow-cpu autofaiss google-genai openai
#
#   2. Checkpoint (clone to ricl_openpi/pi0_fast_droid_ricl_checkpoint)
#        cd ricl_openpi/pi0_fast_droid_ricl_checkpoint && git lfs pull
#
#   3. RTX 5080 (CC 12.0) fix — upgrade ptxas to CUDA 12.8+ (run once after uv sync):
#        cd ricl_openpi
#        uv pip install "nvidia-cuda-nvcc-cu12>=12.8"
#        # The servers must then be launched with 'uv run --frozen' (see below) to
#        # prevent uv from reverting the nvcc upgrade on each run.
#
#   4. Install openpi_client into the icl_bimanual conda env:
#        conda activate icl_bimanual
#        pip install -e ricl_openpi/packages/openpi-client
#
#   4. Prepare demo retrieval pools (openpi venv, run once per task set):
#        source ricl_openpi/.venv/bin/activate
#        python form_icl_demonstrations_ricl.py [--tasks bimanual_lift_tray ...]
#
# Usage:
#   chmod +x test_ricl.sh
#   ./test_ricl.sh                                     # run all tasks
#   RICL_TASKS="bimanual_lift_tray" ./test_ricl.sh     # run a single task
#

set -euo pipefail

# ───────────────────────── Configuration ─────────────────────────
PROJECT_DIR="/media/nvme/palma/icl_bimanual"
RICL_DIR="${PROJECT_DIR}/ricl_openpi"

# Python environments
# openpi venv: used for the RICL servers (uv run handles activation automatically)
OPENPI_VENV="${RICL_DIR}/.venv"
# icl_bimanual conda env: used for main.py / RICLAgent evaluation
CONDA_ENV="icl_bimanual"
LOG_DIR="${PROJECT_DIR}/logs"
TEST_DATA_PATH="${PROJECT_DIR}/generated_data/test"
CHECKPOINT_DIR="pi0_fast_droid_ricl_checkpoint"
DEMOS_ROOT="${RICL_DIR}/preprocessing/collected_demos"

RIGHT_PORT=8000
LEFT_PORT=8002
RICL_HOST="127.0.0.1"

# Agent settings
AGENT="RICLAgent"

# RICL velocity integration parameters
# The model predicts joint velocities (rad/s).  These are integrated n steps
# at dt seconds to obtain a target joint position.  Smaller (steps * dt) =>
# finer / safer per-query motion.  With BimanualJointPosition the PD
# controller executes the target smoothly via physics simulation.
INTEGRATION_STEPS=3
INTEGRATION_DT=0.05

# RICL lambda: controls ICL-vs-model interpolation weight.
# Formula: ICL_weight = exp(-LAMDA * normalized_distance)
# Lower = more ICL weight at typical distances; config default is 10.0.
# Recommended: 1.0-3.0 to strongly prefer ICL examples.
LAMDA=0.5

# Evaluation settings
EVAL_EPISODES=100
EPISODE_LENGTH=250
REPEAT_EVAL=1

# All bimanual tasks
ALL_TASKS=(
    # "bimanual_dual_push_buttons"
    "bimanual_handover_item"
    "bimanual_handover_item_easy"
    "bimanual_lift_ball"
    "bimanual_lift_tray"
    "bimanual_pick_laptop"
    "bimanual_pick_plate"
    "bimanual_push_box"
    "bimanual_put_bottle_in_fridge"
    "bimanual_put_item_in_drawer"
    "bimanual_straighten_rope"
    "bimanual_sweep_to_dustpan"
    "bimanual_take_tray_out_of_oven"
)

# Allow overriding from environment variable
if [[ -n "${RICL_TASKS:-}" ]]; then
    IFS=' ' read -ra TASKS <<< "$RICL_TASKS"
else
    TASKS=("${ALL_TASKS[@]}")
fi

# ───────────────────────── Helper functions ──────────────────────

find_demos_dir() {
    # Find the demo directory for a given task and arm.
    # Searches for any date-prefixed folder matching the pattern.
    local task="$1"
    local arm="$2"
    local pattern="${task}_${arm}_arm"

    local found
    found=$(find "${DEMOS_ROOT}" -maxdepth 1 -type d -name "*_${pattern}" | sort -r | head -1)

    if [[ -z "$found" ]]; then
        echo ""
        return 1
    fi
    echo "$found"
}

wait_for_server() {
    # Wait until a RICL server is accepting websocket connections.
    local port="$1"
    local name="$2"
    local max_wait=600  # 10 minutes max
    local waited=0

    echo "  Waiting for ${name} server on port ${port}..."
    while ! python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1)
try:
    s.connect(('${RICL_HOST}', ${port}))
except Exception:
    raise SystemExit(1)
finally:
    s.close()
" 2>/dev/null; do
        sleep 2
        waited=$((waited + 2))
        if [[ $waited -ge $max_wait ]]; then
            echo "  ERROR: ${name} server did not start within ${max_wait}s"
            return 1
        fi
    done
    echo "  ${name} server is ready (waited ${waited}s)"
}

kill_ricl_servers() {
    # Kill any RICL servers on the configured ports.
    echo "  Stopping RICL servers..."
    # Kill by PID file if available
    for pidfile in /tmp/ricl_bimanual.pid /tmp/ricl_right.pid /tmp/ricl_left.pid; do
        if [[ -f "$pidfile" ]]; then
            local pid
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                wait "$pid" 2>/dev/null || true
            fi
            rm -f "$pidfile"
        fi
    done
    # Also kill by port as fallback
    for port in $RIGHT_PORT $LEFT_PORT; do
        local pid
        pid=$(lsof -ti :"$port" 2>/dev/null || true)
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
}

# ───────────────────────── Cleanup on exit ───────────────────────
cleanup() {
    echo ""
    echo "Cleaning up..."
    kill_ricl_servers
}
trap cleanup EXIT

# ───────────────────────── Main loop ─────────────────────────────

mkdir -p "redirections/ricl/${AGENT}"

echo "============================================================"
echo "  RICL Evaluation Pipeline"
echo "  Tasks: ${TASKS[*]}"
echo "  Eval episodes: ${EVAL_EPISODES}, Repeats: ${REPEAT_EVAL}"
echo "============================================================"
echo ""

for task in "${TASKS[@]}"; do
    echo "============================================================"
    echo "  Task: ${task}"
    echo "============================================================"

    # ── Find demo directories ──────────────────────────────────
    RIGHT_DEMOS=$(find_demos_dir "$task" "right")
    LEFT_DEMOS=$(find_demos_dir "$task" "left")

    if [[ -z "$RIGHT_DEMOS" ]]; then
        echo "  ERROR: No right-arm demos found for ${task}. Run form_icl_demonstrations_ricl.py first."
        echo "  Skipping ${task}."
        echo ""
        continue
    fi
    if [[ -z "$LEFT_DEMOS" ]]; then
        echo "  ERROR: No left-arm demos found for ${task}. Run form_icl_demonstrations_ricl.py first."
        echo "  Skipping ${task}."
        echo ""
        continue
    fi

    echo "  Right demos: ${RIGHT_DEMOS}"
    echo "  Left demos:  ${LEFT_DEMOS}"

    # ── Kill any leftover servers ──────────────────────────────
    kill_ricl_servers
    sleep 10

    # ── Start RICL servers ─────────────────────────────────────
    # One process, two websocket ports — the Pi0 model is loaded ONCE and shared
    # between both arm policies (same JAX device arrays = same VRAM allocation).
    # This halves VRAM vs. two independent servers (single server = ~6 GB bfloat16
    # weights; two independent servers would require ~12.5 GB each on 16 GB VRAM).
    echo "  Starting bimanual RICL server (right:${RIGHT_PORT}, left:${LEFT_PORT})..."
    cd "${RICL_DIR}"
    CUDA_VISIBLE_DEVICES=1 TMPDIR=/media/nvme/palma/icl_bimanual/tmp uv run --no-sync scripts/serve_policy_ricl_bimanual.py \
        --right_port "${RIGHT_PORT}" \
        --left_port  "${LEFT_PORT}" \
        --config=pi0_fast_droid_ricl \
        --dir="${CHECKPOINT_DIR}" \
        --right_demos_dir="${RIGHT_DEMOS}" \
        --left_demos_dir="${LEFT_DEMOS}" \
        --lamda="${LAMDA}" \
        > "/tmp/ricl_bimanual_${task}.log" 2>&1 &
    echo $! > /tmp/ricl_bimanual.pid
    # Keep the old pid files for compatibility with kill_ricl_servers
    cp /tmp/ricl_bimanual.pid /tmp/ricl_right.pid
    cp /tmp/ricl_bimanual.pid /tmp/ricl_left.pid

    cd "${PROJECT_DIR}"

    # ── Wait for both servers ──────────────────────────────────
    if ! wait_for_server "$RIGHT_PORT" "right-arm"; then
        echo "  Server log: /tmp/ricl_bimanual_${task}.log"
        tail -20 "/tmp/ricl_bimanual_${task}.log" | sed 's/^/    /'
        kill_ricl_servers
        continue
    fi
    if ! wait_for_server "$LEFT_PORT" "left-arm"; then
        echo "  Server log: /tmp/ricl_bimanual_${task}.log"
        tail -20 "/tmp/ricl_bimanual_${task}.log" | sed 's/^/    /'
        kill_ricl_servers
        continue
    fi

    echo "  Both servers ready. Starting evaluation..."

    # ── Run evaluation (icl_bimanual conda env) ───────────────
    # main.py uses PyRep/RLBench/YARR which are installed in the conda env.
    # The openpi_client websocket library must also be installed there.
    # conda run starts a clean subprocess — propagate CoppeliaSim env vars
    # that PyRep needs to find libcoppeliaSim.so at import time.
    # Temporarily disable set -e so a failed evaluation doesn't kill the whole pipeline.
    REDIR_FILE="${PROJECT_DIR}/redirections/ricl/${AGENT}/${task}.txt"
    set +e
    conda run -n "${CONDA_ENV}" --no-capture-output \
    bash -c "
    export PYTHONUNBUFFERED=1
    export COPPELIASIM_ROOT=\"\${HOME}/CoppeliaSim\"
    export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:-}:\${COPPELIASIM_ROOT}\"
    export QT_QPA_PLATFORM_PLUGIN_PATH=\"\${COPPELIASIM_ROOT}\"
    export CUDA_VISIBLE_DEVICES=1
    xvfb-run -a -s \"-screen 0 1280x1024x24\" \
    python main.py \
        method.name=\"${AGENT}\" \
        model.llm_call_style=\"openai\" \
        model.name=\"ricl\" \
        +model.ricl_host=\"${RICL_HOST}\" \
        +model.ricl_right_port=\"${RIGHT_PORT}\" \
        +model.ricl_left_port=\"${LEFT_PORT}\" \
        +model.ricl_integration_steps=\"${INTEGRATION_STEPS}\" \
        +model.ricl_integration_dt=\"${INTEGRATION_DT}\" \
        rlbench.tasks=\"[${task}]\" \
        rlbench.task_name=\"${task}\" \
        rlbench.episode_length=\"${EPISODE_LENGTH}\" \
        rlbench.demo_path=\"${TEST_DATA_PATH}\" \
        framework.gpu=0 \
        framework.logdir=\"${LOG_DIR}\" \
        framework.eval_episodes=\"${EVAL_EPISODES}\" \
        rlbench.headless=True \
        framework.repeat_eval=\"${REPEAT_EVAL}\" \
    > \"${REDIR_FILE}\" 2>&1
    "
    eval_exit=$?
    set -e
    if [[ $eval_exit -eq 0 ]]; then
        echo "  Evaluation completed successfully."
        # Print the final results from the log
        echo "  Results:"
        tail -5 "redirections/ricl/${AGENT}/${task}.txt" | sed 's/^/    /'
    else
        echo "  Evaluation FAILED (exit code ${eval_exit})."
        echo "  Last 10 lines of log:"
        tail -10 "redirections/ricl/${AGENT}/${task}.txt" | sed 's/^/    /'
    fi

    # ── Kill servers before next task ──────────────────────────
    kill_ricl_servers

    echo ""
done

echo "============================================================"
echo "  All tasks complete."
echo "  Logs saved in: redirections/ricl/${AGENT}/"
echo "  Server logs in: /tmp/ricl_bimanual_*.log"
echo "============================================================"
