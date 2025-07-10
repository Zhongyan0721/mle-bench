FROM mlebench-env

# Environment directories for the experiment agent
ARG SUBMISSION_DIR=/home/submission
ENV SUBMISSION_DIR=${SUBMISSION_DIR}

ARG LOGS_DIR=/home/logs
ENV LOGS_DIR=${LOGS_DIR}

ARG CODE_DIR=/home/code
ENV CODE_DIR=${CODE_DIR}

ARG AGENT_DIR=/home/agent
ENV AGENT_DIR=${AGENT_DIR}

ARG EXPERIMENT_DIR=/home/experiments
ENV EXPERIMENT_DIR=${EXPERIMENT_DIR}

ARG DATA_DIR=/home/data
ENV DATA_DIR=${DATA_DIR}

# Create necessary directories
RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR} ${SUBMISSION_DIR} ${EXPERIMENT_DIR}

ARG CONDA_ENV_NAME=agent
ARG REQUIREMENTS=${AGENT_DIR}/requirements.txt

# Copy requirements first for better Docker layer caching
COPY requirements.txt ${AGENT_DIR}/requirements.txt

# Install additional packages for experimentation
RUN apt-get update && apt-get install -y \
    htop \
    tree \
    jq \
    curl \
    wget \
    git \
    vim \
    nano \
    tmux \
    screen \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for the experiment agent
RUN conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

# Copy agent files
COPY . ${AGENT_DIR}

# Make start script executable
RUN chmod +x ${AGENT_DIR}/start.sh

# Set working directory
WORKDIR ${AGENT_DIR}