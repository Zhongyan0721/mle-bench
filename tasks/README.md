# AI Agent Task System

This directory contains a flexible task definition and management system for AI agents to run various kinds of experiments and tasks in the MLE-Bench environment.

## Overview

The task system provides:

- **Task Registry**: A centralized registry for managing different types of tasks
- **Task Runner**: Infrastructure for executing tasks in containerized environments
- **Data Generation**: Utilities for creating synthetic datasets for various ML tasks
- **Task Validation**: Automated evaluation of task completion and results
- **CLI Tools**: Command-line interface for task management

## Task Types

The system supports several types of tasks:

### Machine Learning Tasks
- **Classification**: Binary and multi-class classification problems
- **Regression**: Continuous value prediction tasks
- **Clustering**: Unsupervised grouping tasks
- **Time Series**: Temporal data forecasting and analysis

### Data Science Tasks
- **Data Analysis**: Exploratory data analysis and insights generation
- **Feature Engineering**: Data preprocessing and feature creation
- **Visualization**: Data visualization and reporting

### Deep Learning Tasks
- **Image Classification**: Computer vision tasks
- **Natural Language Processing**: Text analysis and processing
- **Neural Network Design**: Custom architecture development

### General Tasks
- **Research**: Open-ended research and experimentation
- **Benchmarking**: Performance comparison studies
- **Tool Development**: Creating utilities and tools

## Directory Structure

```
tasks/
├── __init__.py              # Package initialization
├── README.md               # This documentation
├── cli.py                  # Command-line interface
├── task_registry.py        # Task registration system
├── task_runner.py          # Task execution infrastructure
├── templates/              # Pre-defined task templates
│   ├── classification_task.json
│   ├── regression_task.json
│   ├── data_analysis_task.json
│   ├── time_series_task.json
│   └── deep_learning_task.json
├── examples/               # Example task implementations
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── data_generator.py   # Synthetic data generation
│   └── task_validator.py   # Task result validation
```

## Task Definition Format

Tasks are defined using JSON format with the following structure:

```json
{
  "id": "unique_task_identifier",
  "name": "Human-readable task name",
  "description": "Brief description of the task",
  "task_type": "classification|regression|data_analysis|etc",
  "difficulty": "easy|medium|hard",
  "estimated_time": 3600,
  "data_requirements": {
    "generate_synthetic": true,
    "dataset_type": "classification",
    "n_samples": 1000,
    "n_features": 10
  },
  "evaluation_criteria": {
    "min_accuracy": 0.7,
    "required_outputs": ["predictions.csv"]
  },
  "instructions": "Detailed task instructions...",
  "expected_outputs": ["file1.csv", "file2.txt"],
  "metadata": {
    "tags": ["ml", "classification"],
    "learning_objectives": ["model_training"]
  }
}
```

## Using the CLI

The task system includes a command-line interface for managing tasks:

### List Available Tasks
```bash
python -m tasks.cli list
python -m tasks.cli list --type classification
python -m tasks.cli list --difficulty easy
```

### Show Task Details
```bash
python -m tasks.cli show classification_basic
```

### Generate Sample Data
```bash
python -m tasks.cli generate-data classification_basic ./data/
```

### Create Task Configuration
```bash
python -m tasks.cli config classification_basic task_config.json
```

### Validate Task Results
```bash
python -m tasks.cli validate classification_basic ./results/
```

### Create Custom Task
```bash
python -m tasks.cli create
```

## Using with AI Agents

### 1. Experiment Agent Integration

The experiment agent automatically loads task configurations and sets up the appropriate environment:

```bash
# Run experiment agent with a specific task
docker run -v /path/to/data:/home/data \
           -v /path/to/task_config.json:/home/task_config.json \
           -e TASK_TYPE=classification \
           experiment-agent
```

### 2. Task Configuration

Create a task configuration file:

```python
from tasks import registry

# Create configuration for a classification task
config = registry.create_task_config(
    'classification_basic',
    'task_config.json',
    custom_param='value'
)
```

### 3. Running Tasks Programmatically

```python
from tasks import TaskRunner
from pathlib import Path

# Initialize task runner
runner = TaskRunner(
    work_dir=Path('./workspace'),
    data_dir=Path('./data'),
    output_dir=Path('./results')
)

# Run a task
result = runner.run_task(
    'classification_basic',
    ['python', 'my_agent.py'],
    timeout=3600
)

# Evaluate results
evaluation = runner.evaluate_task_results(
    'classification_basic',
    Path('./results')
)
```

## Data Generation

The system includes utilities for generating synthetic datasets:

```python
from tasks.utils import DataGenerator

generator = DataGenerator()

# Generate classification dataset
df = generator.generate_classification_dataset(
    n_samples=1000,
    n_features=10,
    n_classes=3
)

# Generate time series data
ts_df = generator.generate_time_series_dataset(
    n_samples=1000,
    n_features=5,
    trend=True,
    seasonality=True
)
```

## Task Validation

Automatic validation of task completion:

```python
from tasks.utils import TaskValidator
from tasks import registry

validator = TaskValidator()
task = registry.get_task('classification_basic')

# Evaluate task results
evaluation = validator.evaluate_task(task, results_dir)

print(f"Overall Score: {evaluation['overall_score']}")
print(f"Success: {evaluation['success']}")
```

## Creating Custom Tasks

### 1. Define Task JSON

Create a new task definition file in `templates/`:

```json
{
  "id": "my_custom_task",
  "name": "My Custom Task",
  "description": "A custom task for specific requirements",
  "task_type": "custom",
  "difficulty": "medium",
  "estimated_time": 2400,
  "data_requirements": {
    "generate_synthetic": false,
    "copy_files": ["dataset.csv"]
  },
  "evaluation_criteria": {
    "custom_metric": 0.8
  },
  "instructions": "Detailed instructions...",
  "expected_outputs": ["results.json"],
  "metadata": {
    "custom": true
  }
}
```

### 2. Register Task

```python
from tasks import Task, registry

# Load and register custom task
task = Task.load('path/to/custom_task.json')
registry.register_task(task)
```

### 3. Custom Evaluation

Extend the TaskValidator for custom evaluation logic:

```python
from tasks.utils import TaskValidator

class CustomValidator(TaskValidator):
    def _evaluate_custom(self, task, results_dir):
        # Custom evaluation logic
        return {
            'criteria_met': {'custom_criterion': True},
            'scores': {'custom_score': 0.85},
            'feedback': ['Custom evaluation completed']
        }
```

## Integration with MLE-Bench

The task system integrates seamlessly with the existing MLE-Bench infrastructure:

1. **Docker Environment**: Tasks run in the same containerized environment as competitions
2. **Agent Registry**: Compatible with existing agent configurations
3. **Result Collection**: Uses the same result extraction mechanisms
4. **Logging**: Integrates with the existing logging infrastructure

## Best Practices

### Task Design
- Keep tasks focused and well-defined
- Provide clear success criteria
- Include comprehensive instructions
- Specify expected outputs explicitly

### Data Requirements
- Use synthetic data when possible for reproducibility
- Provide realistic data characteristics
- Consider data size and complexity for time limits

### Evaluation
- Define objective, measurable criteria
- Include both quantitative and qualitative assessments
- Provide meaningful feedback for improvement

### Documentation
- Document task objectives clearly
- Include examples and expected approaches
- Specify any domain-specific requirements

## Examples

See the `examples/` directory for complete task implementations and agent code examples.

## Contributing

To contribute new tasks or improvements:

1. Create task definitions following the standard format
2. Test tasks with multiple agents
3. Validate evaluation criteria
4. Update documentation
5. Submit pull request

## Support

For questions or issues with the task system, please refer to the main MLE-Bench documentation or open an issue in the repository.