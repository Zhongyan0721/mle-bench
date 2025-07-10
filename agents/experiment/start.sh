#!/bin/bash
set -x # Print commands and their arguments as they are executed

cd ${AGENT_DIR}

eval "$(conda shell.bash hook)" # make conda available to the shell
conda activate agent

# Determine hardware available
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/^[ \t]*//' \
    | sed 's/[ \t]*$//' \
    | sort \
    | uniq -c \
    | sed 's/^ *\([0-9]*\) *\(.*\)$/\1 \2/' \
    | paste -sd ', ' -)
else
  HARDWARE="a CPU"
fi
export HARDWARE

# Check GPU availability
echo "Hardware available: $HARDWARE"
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
python -c "import tensorflow as tf; print('TensorFlow GPUs:', len(tf.config.list_physical_devices('GPU')))"

# Convert $TIME_LIMIT_SECS to more readable format for prompt
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}

# Set default time limit if not provided
if [ -z "$TIME_LIMIT_SECS" ]; then
    TIME_LIMIT_SECS=3600  # 1 hour default
fi
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)

# Create experiment workspace structure
mkdir -p ${EXPERIMENT_DIR}/workspace
mkdir -p ${EXPERIMENT_DIR}/data
mkdir -p ${EXPERIMENT_DIR}/models
mkdir -p ${EXPERIMENT_DIR}/results
mkdir -p ${EXPERIMENT_DIR}/notebooks
mkdir -p ${EXPERIMENT_DIR}/scripts
mkdir -p ${EXPERIMENT_DIR}/configs

# Create symbolic links for easy access
ln -sf ${DATA_DIR} ${EXPERIMENT_DIR}/input_data
ln -sf ${CODE_DIR} ${EXPERIMENT_DIR}/code
ln -sf ${LOGS_DIR} ${EXPERIMENT_DIR}/logs
ln -sf ${SUBMISSION_DIR} ${EXPERIMENT_DIR}/submission

# Copy task configuration if it exists
if [ -f "/home/task_config.json" ]; then
    cp /home/task_config.json ${EXPERIMENT_DIR}/task_config.json
fi

# Copy experiment instructions if they exist
if [ -f "/home/experiment_instructions.txt" ]; then
    cp /home/experiment_instructions.txt ${EXPERIMENT_DIR}/instructions.txt
fi

# Set up environment variables for the experiment
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"default_experiment"}
export TASK_TYPE=${TASK_TYPE:-"general"}
export PYTHONPATH="${AGENT_DIR}:${EXPERIMENT_DIR}:${PYTHONPATH}"

# Log environment setup
echo "=== EXPERIMENT ENVIRONMENT SETUP ==="
echo "Agent Directory: ${AGENT_DIR}"
echo "Experiment Directory: ${EXPERIMENT_DIR}"
echo "Data Directory: ${DATA_DIR}"
echo "Code Directory: ${CODE_DIR}"
echo "Logs Directory: ${LOGS_DIR}"
echo "Submission Directory: ${SUBMISSION_DIR}"
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "Task Type: ${TASK_TYPE}"
echo "Time Limit: ${TIME_LIMIT}"
echo "Hardware: ${HARDWARE}"
echo "Python Path: ${PYTHONPATH}"
echo "=================================="

# Start the experiment runner
echo "Starting experiment runner..."
python ${AGENT_DIR}/experiment_runner.py \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --code-dir "${CODE_DIR}" \
    --logs-dir "${LOGS_DIR}" \
    --submission-dir "${SUBMISSION_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --task-type "${TASK_TYPE}" \
    --time-limit "${TIME_LIMIT_SECS}" \
    "$@"

echo "Experiment completed."