# #!/usr/bin/env bash
# #
# # run_all_trainings.sh
# #
# # Launches train_staghunt.py in parallel for multiple algorithms, all with the
# # same hyperparameters, so the results stay comparable with each other.
# #
# # Usage:
# #   ./run_all_trainings.sh                # launches the 4 DQN-based algorithms (default)
# #   ./run_all_trainings.sh --all           # also launches a2c and mappo (ONLY if their
# #                                           # interfaces have been made compatible with
# #                                           # train_staghunt.py: they currently are not)
# #   ./run_all_trainings.sh --algos "vanillaDQN doubleDQN"   # pick which ones to run
# #
# # Each run writes its own log to logs/<timestamp>/<algorithm>.log
# # The script waits for all runs to finish before exiting (wait).

# set -euo pipefail

# ### ================================================================================================================
# ### Hyperparameters shared across all runs (same values for every algorithm => fair comparison)
# ### ================================================================================================================
# MAX_EPISODES=2000
# MAX_STEPS_PER_EPISODE=200

# GRID_SIZE=10
# FORAGE_QTA=2
# FORAGE_REWARD=1
# STAG_REWARD=5
# MAULING_PENALTY=-3

# DISCOUNT_FACTOR=0.99
# LEARNING_RATE=5e-3 #5e-4
# BATCH_SIZE=64
# EPSILON_START=1.0
# EPSILON_DECAY=0.995
# EPSILON_MIN=0.1
# TARGET_UPDATE_FREQ=500
# REPLAY_BUFFER_SIZE=10000

# ### ================================================================================================================
# ### Training script and algorithms to launch
# ### ================================================================================================================
# TRAIN_SCRIPT="src/train_staghunt.py"

# DQN_ALGOS=(vanillaDQN standardDQN doubleDQN duelingDQN)
# EXTRA_ALGOS=(a2c mappo)   # incompatible with train_staghunt.py at the moment, see note above

# ALGOS=("${DQN_ALGOS[@]}")
# # ALGOS=("vanillaDQN" "standardDQN")
# # ALGOS=("doubleDQN" "duelingDQN") 
# RUN_ALL=false

# while [[ $# -gt 0 ]]; do
#     case "$1" in
#         --all)
#             RUN_ALL=true
#             shift
#             ;;
#         --algos)
#             read -r -a ALGOS <<< "$2"
#             shift 2
#             ;;
#         *)
#             echo "Unrecognized argument: $1"
#             echo "Usage: $0 [--all] [--algos \"algo1 algo2 ...\"]"
#             exit 1
#             ;;
#     esac
# done

# if [[ "$RUN_ALL" == true ]]; then
#     ALGOS=("${DQN_ALGOS[@]}" "${EXTRA_ALGOS[@]}")
#     echo "WARNING: including a2c/mappo. Make sure their interface (store_sample,"
#     echo "select_action, constructor signature) is compatible with train_staghunt.py"
#     echo "first, otherwise they will fail immediately."
# fi

# if [[ ! -f "$TRAIN_SCRIPT" ]]; then
#     echo "Error: cannot find $TRAIN_SCRIPT. Run this script from the project root."
#     exit 1
# fi

# ### ================================================================================================================
# ### Log directory setup
# ### ================================================================================================================
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# LOG_DIR="logs/${TIMESTAMP}"
# mkdir -p "$LOG_DIR"

# echo "Launching ${#ALGOS[@]} runs in parallel: ${ALGOS[*]}"
# echo "Logs saved to: $LOG_DIR"
# echo

# ### ================================================================================================================
# ### Launch runs in the background
# ### ================================================================================================================
# PIDS=()

# for ALGO in "${ALGOS[@]}"; do
#     LOG_FILE="${LOG_DIR}/${ALGO}.log"

#     echo "-> Starting $ALGO (log: $LOG_FILE)"

#     python3 -u "$TRAIN_SCRIPT" \
#         --algorithm "$ALGO" \
#         --max_episodes "$MAX_EPISODES" \
#         --max_steps_per_episode "$MAX_STEPS_PER_EPISODE" \
#         --grid_size "$GRID_SIZE" \
#         --forage_qta "$FORAGE_QTA" \
#         --forage_reward "$FORAGE_REWARD" \
#         --stag_reward "$STAG_REWARD" \
#         --mauling_penalty "$MAULING_PENALTY" \
#         --discount_factor "$DISCOUNT_FACTOR" \
#         --learning_rate "$LEARNING_RATE" \
#         --batch_size "$BATCH_SIZE" \
#         --epsilon_start "$EPSILON_START" \
#         --epsilon_decay "$EPSILON_DECAY" \
#         --epsilon_min "$EPSILON_MIN" \
#         --target_update_freq "$TARGET_UPDATE_FREQ" \
#         --replay_buffer_size "$REPLAY_BUFFER_SIZE" \
#         > "$LOG_FILE" 2>&1 &

#     PIDS+=($!)
# done

# echo
# echo "PIDs of launched processes: ${PIDS[*]}"
# echo "Waiting for all training runs to finish..."
# echo "(you can follow a log live with: tail -f ${LOG_DIR}/<algorithm>.log)"
# echo

# ### ================================================================================================================
# ### Wait for all runs and check outcome
# ### ================================================================================================================
# FAILED=()

# for i in "${!PIDS[@]}"; do
#     PID="${PIDS[$i]}"
#     ALGO="${ALGOS[$i]}"
#     if wait "$PID"; then
#         echo "[OK]    $ALGO (PID $PID) finished successfully."
#     else
#         echo "[ERROR] $ALGO (PID $PID) exited with an error. Check ${LOG_DIR}/${ALGO}.log"
#         FAILED+=("$ALGO")
#     fi
# done

# echo
# if [[ ${#FAILED[@]} -eq 0 ]]; then
#     echo "All runs completed successfully."
# else
#     echo "Failed runs: ${FAILED[*]}"
#     exit 1
# fi


#!/usr/bin/env bash
#
# run_parallel.sh
#
# Runs all 6 algorithms for ONE selected game (stag hunt / harvest / escalation)
# in parallel, max 3 at a time, each one in its own macOS Terminal window with
# output printed directly in that window (nothing is saved to a log file).
#
# Usage:
#   ./run_parallel.sh --game staghunt
#   ./run_parallel.sh --game harvest
#   ./run_parallel.sh --game escalation
#
# Any extra arguments are forwarded as-is to every training run, so you can
# quickly test different hyperparameters without editing this file, e.g.:
#   ./run_parallel.sh --game staghunt --learning_rate 1e-3 --max_episodes 500
#
# To change the DEFAULT hyperparameters for every run, just edit the values
# in the "Hyperparameters" section below.

#!/usr/bin/env bash
#
# run_parallel.sh
#
# Runs all 6 algorithms for ONE selected game (stag hunt / harvest / escalation)
# in parallel, max 3 at a time, each one in its own macOS Terminal window with
# output printed directly in that window (nothing is saved to a log file).
#
# Usage:
#   ./run_parallel.sh --game staghunt
#   ./run_parallel.sh --game harvest
#   ./run_parallel.sh --game escalation
#
# Any extra arguments are forwarded as-is to every training run, so you can
# quickly test different hyperparameters without editing this file, e.g.:
#   ./run_parallel.sh --game staghunt --learning_rate 1e-3 --max_episodes 500
#
# To change the DEFAULT hyperparameters for every run, just edit the values
# in the "Hyperparameters" section below.

set -o pipefail

### ================================================================================================================
### Hyperparameters (edit these to change the defaults for every run)
### ================================================================================================================
MAX_EPISODES=2000
MAX_STEPS_PER_EPISODE=200
GRID_SIZE=5

DISCOUNT_FACTOR=0.99
LEARNING_RATE=1e-3
BATCH_SIZE=32
EPSILON_START=1.0
EPSILON_DECAY=0.995
EPSILON_MIN=0.1
TARGET_UPDATE_FREQ=500
REPLAY_BUFFER_SIZE=10000
GAE_LAMBDA=0.95
PPO_EPOCHS=5
CRITIC_COEFF=0.5
ENTROPY_COEFF=0.01
EPSILON_CLIP=0.2

# Stag Hunt specific
FORAGE_QTA=2
FORAGE_REWARD=1
STAG_REWARD=5
MAULING_PENALTY=-3

# Harvest specific
MAX_PLANTS=5
CHANCE_MATURE=0.1
CHANCE_DIE=0.1
YOUNG_REWARD=1
MATURE_REWARD=3

# Escalation specific
PUNISHMENT_FACTOR=0.5

MAX_PARALLEL=3
ALGOS=(vanillaDQN standardDQN doubleDQN duelingDQN a2c mappo)
# ALGOS=(vanillaDQN standardDQN doubleDQN duelingDQN)
# ALGOS=(a2c mappo)
# ALGOS=(duelingDQN a2c mappo)
# ALGOS=(duelingDQN a2c mappo)

# Path to the virtualenv activate script, relative to the project root.
# New Terminal windows do NOT inherit an already-activated virtualenv, so it
# must be re-activated inside each one. Leave empty ("") to skip this step.
VENV_ACTIVATE="aasvenv/bin/activate"

### ================================================================================================================
### Parse arguments
### ================================================================================================================
GAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --game)
            GAME="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

EXTRA_ARGS=("$@")   # anything left is forwarded to every training run

case "$GAME" in
    staghunt)
        SCRIPT="src/train_staghunt.py"
        GAME_ARGS=(--grid_size "$GRID_SIZE" --forage_qta "$FORAGE_QTA" --forage_reward "$FORAGE_REWARD" \
                   --stag_reward "$STAG_REWARD" --mauling_penalty "$MAULING_PENALTY")
        ;;
    harvest)
        SCRIPT="src/train_harvest.py"
        GAME_ARGS=(--grid_size "$GRID_SIZE" --max_plants "$MAX_PLANTS" --chance_mature "$CHANCE_MATURE" \
                   --chance_die "$CHANCE_DIE" --young_reward "$YOUNG_REWARD" --mature_reward "$MATURE_REWARD")
        ;;
    escalation)
        SCRIPT="src/train_escalation.py"
        GAME_ARGS=(--grid_size "$GRID_SIZE" --punishment_factor "$PUNISHMENT_FACTOR")
        ;;
    *)
        echo "Usage: $0 --game {staghunt|harvest|escalation} [extra training args...]"
        exit 1
        ;;
esac

if [[ ! -f "$SCRIPT" ]]; then
    echo "Error: cannot find $SCRIPT. Run this script from the project root."
    exit 1
fi

COMMON_ARGS=(--max_episodes "$MAX_EPISODES" --max_steps_per_episode "$MAX_STEPS_PER_EPISODE" \
             --discount_factor "$DISCOUNT_FACTOR" --learning_rate "$LEARNING_RATE" --batch_size "$BATCH_SIZE" \
             --epsilon_start "$EPSILON_START" --epsilon_decay "$EPSILON_DECAY" --epsilon_min "$EPSILON_MIN" \
             --target_update_freq "$TARGET_UPDATE_FREQ" --replay_buffer_size "$REPLAY_BUFFER_SIZE" \
             --gae_lambda "$GAE_LAMBDA" --ppo_epochs "$PPO_EPOCHS" --critic_coeff "$CRITIC_COEFF" \
             --entropy_coeff "$ENTROPY_COEFF" --epsilon_clip "$EPSILON_CLIP")

### ================================================================================================================
### Launch: up to MAX_PARALLEL Terminal windows at a time, each running one algorithm directly
### (no log files: each window just runs python3 straight, output goes to that window and nowhere else;
### a small empty marker file per run is used only to know when it's done, so the next one can start)
### ================================================================================================================
PROJECT_DIR="$(pwd)"
MARKER_DIR="$(mktemp -d /tmp/staghunt_run_XXXXXX)"

cleanup() { rm -rf "$MARKER_DIR"; }
trap cleanup EXIT

echo "Game: $GAME | Running ${#ALGOS[@]} algorithms, $MAX_PARALLEL at a time."
echo "A new Terminal window will open for each run."
echo

STARTED=0
for ALGO in "${ALGOS[@]}"; do
    while true; do
        COMPLETED=$(find "$MARKER_DIR" -name '*.done' 2>/dev/null | wc -l | tr -d ' ')
        RUNNING=$(( STARTED - COMPLETED ))
        [[ "$RUNNING" -lt "$MAX_PARALLEL" ]] && break
        sleep 2
    done

    MARKER_FILE="${MARKER_DIR}/${ALGO}.done"
    ACTIVATE_CMD=""
    [[ -n "$VENV_ACTIVATE" ]] && ACTIVATE_CMD="source '${PROJECT_DIR}/${VENV_ACTIVATE}' && "
    CMD="cd '${PROJECT_DIR}' && ${ACTIVATE_CMD}python3 -u '${SCRIPT}' --algorithm '${ALGO}' ${COMMON_ARGS[*]} ${GAME_ARGS[*]} ${EXTRA_ARGS[*]}; touch '${MARKER_FILE}'"

    echo "-> Starting $ALGO"
    osascript -e "tell application \"Terminal\" to do script \"${CMD}\"" > /dev/null

    STARTED=$((STARTED + 1))
done


# Wait for the last batch to finish before cleaning up
while true; do
    COMPLETED=$(find "$MARKER_DIR" -name '*.done' 2>/dev/null | wc -l | tr -d ' ')
    [[ "$COMPLETED" -ge "$STARTED" ]] && break
    sleep 2
done

echo "All runs finished."