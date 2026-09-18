#!/usr/bin/env bash
#
# run_all_trainings.sh
#
# Launches train_staghunt.py in parallel for multiple algorithms, all with the
# same hyperparameters, so the results stay comparable with each other.
#
# Usage:
#   ./run_all_trainings.sh                # launches the 4 DQN-based algorithms (default)
#   ./run_all_trainings.sh --all           # also launches a2c and mappo (ONLY if their
#                                           # interfaces have been made compatible with
#                                           # train_staghunt.py: they currently are not)
#   ./run_all_trainings.sh --algos "vanillaDQN doubleDQN"   # pick which ones to run
#
# Each run writes its own log to logs/<timestamp>/<algorithm>.log
# The script waits for all runs to finish before exiting (wait).

set -euo pipefail

### ================================================================================================================
### Hyperparameters shared across all runs (same values for every algorithm => fair comparison)
### ================================================================================================================
MAX_EPISODES=2000
MAX_STEPS_PER_EPISODE=200

GRID_SIZE=10
FORAGE_QTA=2
FORAGE_REWARD=1
STAG_REWARD=5
MAULING_PENALTY=-3

DISCOUNT_FACTOR=0.99
LEARNING_RATE=5e-3 #5e-4
BATCH_SIZE=64
EPSILON_START=1.0
EPSILON_DECAY=0.995
EPSILON_MIN=0.1
TARGET_UPDATE_FREQ=500
REPLAY_BUFFER_SIZE=10000

### ================================================================================================================
### Training script and algorithms to launch
### ================================================================================================================
TRAIN_SCRIPT="src/train_staghunt.py"

DQN_ALGOS=(vanillaDQN standardDQN doubleDQN duelingDQN)
EXTRA_ALGOS=(a2c mappo)   # incompatible with train_staghunt.py at the moment, see note above

ALGOS=("${DQN_ALGOS[@]}")
# ALGOS=("vanillaDQN" "standardDQN")
# ALGOS=("doubleDQN" "duelingDQN") 
RUN_ALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            RUN_ALL=true
            shift
            ;;
        --algos)
            read -r -a ALGOS <<< "$2"
            shift 2
            ;;
        *)
            echo "Unrecognized argument: $1"
            echo "Usage: $0 [--all] [--algos \"algo1 algo2 ...\"]"
            exit 1
            ;;
    esac
done

if [[ "$RUN_ALL" == true ]]; then
    ALGOS=("${DQN_ALGOS[@]}" "${EXTRA_ALGOS[@]}")
    echo "WARNING: including a2c/mappo. Make sure their interface (store_sample,"
    echo "select_action, constructor signature) is compatible with train_staghunt.py"
    echo "first, otherwise they will fail immediately."
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Error: cannot find $TRAIN_SCRIPT. Run this script from the project root."
    exit 1
fi

### ================================================================================================================
### Log directory setup
### ================================================================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Launching ${#ALGOS[@]} runs in parallel: ${ALGOS[*]}"
echo "Logs saved to: $LOG_DIR"
echo

### ================================================================================================================
### Launch runs in the background
### ================================================================================================================
PIDS=()

for ALGO in "${ALGOS[@]}"; do
    LOG_FILE="${LOG_DIR}/${ALGO}.log"

    echo "-> Starting $ALGO (log: $LOG_FILE)"

    python3 -u "$TRAIN_SCRIPT" \
        --algorithm "$ALGO" \
        --max_episodes "$MAX_EPISODES" \
        --max_steps_per_episode "$MAX_STEPS_PER_EPISODE" \
        --grid_size "$GRID_SIZE" \
        --forage_qta "$FORAGE_QTA" \
        --forage_reward "$FORAGE_REWARD" \
        --stag_reward "$STAG_REWARD" \
        --mauling_penalty "$MAULING_PENALTY" \
        --discount_factor "$DISCOUNT_FACTOR" \
        --learning_rate "$LEARNING_RATE" \
        --batch_size "$BATCH_SIZE" \
        --epsilon_start "$EPSILON_START" \
        --epsilon_decay "$EPSILON_DECAY" \
        --epsilon_min "$EPSILON_MIN" \
        --target_update_freq "$TARGET_UPDATE_FREQ" \
        --replay_buffer_size "$REPLAY_BUFFER_SIZE" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo
echo "PIDs of launched processes: ${PIDS[*]}"
echo "Waiting for all training runs to finish..."
echo "(you can follow a log live with: tail -f ${LOG_DIR}/<algorithm>.log)"
echo

### ================================================================================================================
### Wait for all runs and check outcome
### ================================================================================================================
FAILED=()

for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    ALGO="${ALGOS[$i]}"
    if wait "$PID"; then
        echo "[OK]    $ALGO (PID $PID) finished successfully."
    else
        echo "[ERROR] $ALGO (PID $PID) exited with an error. Check ${LOG_DIR}/${ALGO}.log"
        FAILED+=("$ALGO")
    fi
done

echo
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All runs completed successfully."
else
    echo "Failed runs: ${FAILED[*]}"
    exit 1
fi