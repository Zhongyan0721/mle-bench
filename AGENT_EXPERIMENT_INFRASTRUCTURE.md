# AI Agent Experiment Infrastructure

This document describes the new infrastructure for AI agents to run various kinds of tasks and experiments in the MLE-Bench environment.

## Overview

The AI Agent Experiment Infrastructure extends MLE-Bench with a flexible system for defining, running, and evaluating various types of tasks beyond traditional ML competitions. This allows AI agents like Aide, OpenHands, and others to:

- Run diverse machine learning and data science tasks
- Experiment with different approaches and methodologies
- Access a free environment for code development and experimentation
- Receive automated evaluation and feedback

## Key Components

### 1. Experiment Agent (`agents/experiment/`)

A new agent type specifically designed for general experimentation:

- **Dockerfile**: Extends the base MLE-Bench environment with additional tools
- **Requirements**: Includes packages for experimentation, visualization, and development
- **Start Script**: Sets up the experiment environment and runs tasks
- **Experiment Runner**: Python framework for executing different types of tasks

### 2. Task System (`tasks/`)

A comprehensive task definition and management system:

- **Task Registry**: Centralized management of task definitions
- **Task Runner**: Infrastructure for executing tasks in containers
- **Data Generator**: Utilities for creating synthetic datasets
- **Task Validator**: Automated evaluation of task completion
- **CLI Tools**: Command-line interface for task management

### 3. Task Templates (`tasks/templates/`)

Pre-defined task templates for common scenarios:

- **Classification Tasks**: Binary and multi-class classification
- **Regression Tasks**: Continuous value prediction
- **Data Analysis Tasks**: Exploratory data analysis
- **Time Series Tasks**: Temporal data forecasting
- **Deep Learning Tasks**: Neural network development

### 4. Examples and Documentation (`tasks/examples/`, `tasks/README.md`)

Complete examples and comprehensive documentation for:

- Task implementation patterns
- Agent integration approaches
- Best practices and guidelines
- API reference and usage examples

## Quick Start

### 1. Build the Experiment Agent

```bash
# Build the base environment first
docker build -t mlebench-env -f environment/Dockerfile .

# Build the experiment agent
docker build -t mlebench-experiment -f agents/experiment/Dockerfile agents/experiment/
```

### 2. List Available Tasks

```bash
python -m tasks.cli list
```

### 3. Run a Task

```bash
# Run with Docker (recommended)
python run_task.py classification_basic

# Run locally for development
python run_task.py classification_basic --local

# Run with custom parameters
python run_task.py classification_basic --time-limit 1800 --validate
```

### 4. Create Custom Tasks

```bash
# Interactive task creation
python -m tasks.cli create

# Generate sample data
python -m tasks.cli generate-data classification_basic ./data/

# Validate results
python -m tasks.cli validate classification_basic ./results/
```

## Task Types and Examples

### Machine Learning Tasks

#### Classification
```json
{
  "id": "classification_basic",
  "name": "Basic Classification Task",
  "task_type": "classification",
  "difficulty": "easy",
  "data_requirements": {
    "generate_synthetic": true,
    "dataset_type": "classification",
    "n_samples": 1000,
    "n_features": 10,
    "n_classes": 3
  },
  "evaluation_criteria": {
    "min_accuracy": 0.7
  }
}
```

#### Regression
```json
{
  "id": "regression_basic",
  "name": "Basic Regression Task",
  "task_type": "regression",
  "difficulty": "easy",
  "data_requirements": {
    "generate_synthetic": true,
    "dataset_type": "regression",
    "n_samples": 1000,
    "n_features": 8
  },
  "evaluation_criteria": {
    "min_r2": 0.6
  }
}
```

### Data Science Tasks

#### Exploratory Data Analysis
```json
{
  "id": "data_analysis_exploratory",
  "name": "Exploratory Data Analysis Task",
  "task_type": "data_analysis",
  "difficulty": "medium",
  "evaluation_criteria": {
    "required_visualizations": 5,
    "required_insights": 3
  }
}
```

### Deep Learning Tasks

#### Image Classification
```json
{
  "id": "deep_learning_image_classification",
  "name": "Deep Learning Image Classification",
  "task_type": "deep_learning",
  "difficulty": "hard",
  "data_requirements": {
    "dataset_type": "image_classification",
    "n_samples": 5000,
    "image_size": [32, 32],
    "n_classes": 10
  },
  "evaluation_criteria": {
    "min_accuracy": 0.75,
    "max_training_time": 3600
  }
}
```

## Agent Integration

### Using with Existing Agents

The experiment infrastructure is compatible with existing MLE-Bench agents:

#### Aide Integration
```yaml
# agents/aide/config.yaml
aide/experiment:
  start: aide/start.sh
  dockerfile: aide/Dockerfile
  kwargs:
    agent.task_type: experiment
    agent.experiment_mode: true
  env_vars:
    EXPERIMENT_MODE: "true"
    TASK_TYPE: "general"
```

#### OpenHands Integration
```dockerfile
# agents/opendevin/Dockerfile
FROM mlebench-experiment

# Add OpenHands-specific setup
COPY opendevin_experiment_setup.py ${AGENT_DIR}/
```

### Custom Agent Development

Create new agents for specific task types:

```python
# agents/custom_agent/experiment_handler.py
from tasks import registry, TaskRunner

class CustomExperimentHandler:
    def __init__(self, task_config):
        self.task_config = task_config
        self.task = registry.get_task(task_config['task']['id'])
    
    def run_experiment(self):
        # Custom experiment logic
        pass
```

## Environment Setup

### Directory Structure

The experiment environment provides a structured workspace:

```
/home/experiments/
├── workspace/          # Main working directory
├── data/              # Input data (symlink to /home/data)
├── models/            # Trained models
├── results/           # Output results
├── notebooks/         # Jupyter notebooks
├── scripts/           # Python scripts
├── configs/           # Configuration files
├── code/              # Code directory (symlink to /home/code)
├── logs/              # Logs directory (symlink to /home/logs)
└── submission/        # Submission directory (symlink to /home/submission)
```

### Environment Variables

Key environment variables available to agents:

- `EXPERIMENT_NAME`: Name of the current experiment
- `TASK_TYPE`: Type of task being executed
- `TIME_LIMIT_SECS`: Time limit in seconds
- `EXPERIMENT_DIR`: Path to experiment directory
- `DATA_DIR`: Path to data directory
- `CODE_DIR`: Path to code directory
- `LOGS_DIR`: Path to logs directory
- `SUBMISSION_DIR`: Path to submission directory

## Data Generation

The system includes comprehensive data generation capabilities:

### Synthetic Datasets

```python
from tasks.utils import DataGenerator

generator = DataGenerator()

# Classification data
df = generator.generate_classification_dataset(
    n_samples=1000,
    n_features=10,
    n_classes=3
)

# Time series data
ts_df = generator.generate_time_series_dataset(
    n_samples=1000,
    n_features=5,
    trend=True,
    seasonality=True
)

# Imbalanced data
imb_df = generator.generate_imbalanced_dataset(
    n_samples=1000,
    imbalance_ratio=0.1
)
```

### Multi-modal Data

```python
# Generate multi-modal dataset
multi_modal = generator.generate_multi_modal_dataset(
    n_samples=1000,
    image_size=(28, 28),
    n_text_features=100,
    n_classes=3
)
```

## Task Validation and Evaluation

### Automatic Evaluation

The system provides automatic evaluation for different task types:

```python
from tasks.utils import TaskValidator

validator = TaskValidator()
evaluation = validator.evaluate_task(task, results_dir)

print(f"Overall Score: {evaluation['overall_score']}")
print(f"Success: {evaluation['success']}")
```

### Custom Evaluation

Extend the validator for custom evaluation logic:

```python
class CustomValidator(TaskValidator):
    def _evaluate_custom_task(self, task, results_dir):
        # Custom evaluation logic
        return {
            'criteria_met': {'custom_criterion': True},
            'scores': {'custom_score': 0.85},
            'feedback': ['Custom evaluation completed']
        }
```

## Advanced Features

### Experiment Tracking

Integration with popular experiment tracking tools:

```python
# MLflow integration
import mlflow

mlflow.start_run()
mlflow.log_param("model_type", "random_forest")
mlflow.log_metric("accuracy", 0.85)
mlflow.end_run()

# Weights & Biases integration
import wandb

wandb.init(project="mle-bench-experiments")
wandb.log({"accuracy": 0.85})
```

### Distributed Computing

Support for distributed computing frameworks:

```python
# Ray integration
import ray

@ray.remote
def train_model(config):
    # Distributed training logic
    pass

# Dask integration
import dask
from dask.distributed import Client

client = Client()
# Distributed data processing
```

### Custom Environments

Create specialized environments for specific domains:

```dockerfile
# agents/nlp_agent/Dockerfile
FROM mlebench-experiment

# Install NLP-specific packages
RUN conda run -n agent pip install \
    transformers \
    datasets \
    tokenizers \
    spacy
```

## Best Practices

### Task Design

1. **Clear Objectives**: Define specific, measurable goals
2. **Realistic Constraints**: Set appropriate time limits and resource constraints
3. **Comprehensive Instructions**: Provide detailed task descriptions
4. **Meaningful Evaluation**: Use relevant metrics and validation criteria

### Agent Development

1. **Modular Design**: Create reusable components
2. **Error Handling**: Implement robust error handling and logging
3. **Resource Management**: Monitor memory and compute usage
4. **Documentation**: Document approaches and methodologies

### Experiment Management

1. **Version Control**: Track code and configuration changes
2. **Reproducibility**: Use fixed random seeds and deterministic algorithms
3. **Result Tracking**: Log all experiments and results
4. **Collaboration**: Share findings and insights

## Troubleshooting

### Common Issues

#### Docker Build Failures
```bash
# Clean Docker cache
docker system prune -a

# Rebuild with no cache
docker build --no-cache -t mlebench-experiment agents/experiment/
```

#### Task Execution Errors
```bash
# Check logs
docker logs <container_name>

# Run with verbose logging
python run_task.py task_id --verbose

# Debug locally
python run_task.py task_id --local
```

#### Data Generation Issues
```bash
# Test data generation
python -m tasks.cli generate-data classification_basic ./test_data/

# Validate task configuration
python -m tasks.cli show classification_basic
```

### Performance Optimization

#### Memory Usage
- Monitor memory consumption during experiments
- Use data streaming for large datasets
- Implement garbage collection strategies

#### Compute Efficiency
- Leverage GPU acceleration when available
- Use parallel processing for CPU-intensive tasks
- Optimize algorithm implementations

## Contributing

### Adding New Task Types

1. Create task template in `tasks/templates/`
2. Implement evaluation logic in `TaskValidator`
3. Add data generation support in `DataGenerator`
4. Create example implementation
5. Update documentation

### Extending Agent Capabilities

1. Add new packages to requirements
2. Implement task-specific handlers
3. Create specialized Dockerfiles
4. Add configuration options
5. Test with various task types

### Improving Infrastructure

1. Enhance task validation logic
2. Add new data generation methods
3. Improve error handling and logging
4. Optimize performance
5. Expand documentation

## Future Enhancements

### Planned Features

- **Interactive Notebooks**: Jupyter Lab integration for interactive development
- **Real-time Monitoring**: Live experiment tracking and visualization
- **Collaborative Features**: Multi-agent collaboration capabilities
- **Advanced Evaluation**: More sophisticated evaluation metrics and methods
- **Cloud Integration**: Support for cloud-based execution and storage

### Research Directions

- **Meta-Learning**: Tasks that adapt based on agent performance
- **Curriculum Learning**: Progressive task difficulty
- **Multi-objective Optimization**: Tasks with multiple competing objectives
- **Federated Learning**: Distributed learning across multiple agents

## Support and Resources

### Documentation
- [Task System README](tasks/README.md)
- [Agent Development Guide](agents/README.md)
- [API Reference](docs/api_reference.md)

### Examples
- [Classification Example](tasks/examples/classification_example.py)
- [Data Analysis Example](tasks/examples/data_analysis_example.py)
- [Custom Task Example](tasks/examples/custom_task_example.py)

### Community
- GitHub Issues for bug reports and feature requests
- Discussions for questions and collaboration
- Wiki for community-contributed content

This infrastructure provides a comprehensive foundation for AI agents to explore, experiment, and learn across a wide variety of tasks and domains. The flexible design allows for easy extension and customization while maintaining compatibility with the existing MLE-Bench ecosystem.