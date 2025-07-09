# InsightBench

InsightBench is an extension to MLE-bench that allows agents to work on research tasks instead of ML engineering tasks. It provides a framework for evaluating LLM agents on their ability to analyze data and draw conclusions from research papers.

## Directory Structure

InsightBench follows a structured format for organizing research tasks:

```
insightBench/
├── utils/                          # Shared utilities
│   ├── llm_inference.py            # Utilities for LLM inference
│   └── data_loaders.py             # Utilities for loading data
├── benchmark/
│   └── papers/
│       └── paper_example_1/        # A paper
│           ├── instances/
│           │   └── figure_1/       # A specific research task
│           │       ├── instruction.txt    # Natural language instruction
│           │       ├── data/              # Raw datasets
│           │       └── ground_truth.json  # Expected conclusions
```

## Running Research Tasks

To run an agent on a research task, use the `run_agent.py` script with the `--task-type research` flag:

```bash
python run_agent.py --agent-id <agent_id> --task-type research --research-task-id <paper_id>/<instance_id>
```

For example:

```bash
python run_agent.py --agent-id aide --task-type research --research-task-id paper_example_1/figure_1
```

You can also run multiple research tasks by specifying a file with task IDs:

```bash
python run_agent.py --agent-id aide --task-type research --research-task-set <path_to_task_list>
```

## Evaluating Results

To evaluate the results of a research task, use the InsightBench CLI:

```bash
python -m insightBench.cli evaluate <task_id> <path_to_conclusions>
```

For example:

```bash
python -m insightBench.cli evaluate paper_example_1/figure_1 /path/to/conclusions.json
```

## Creating New Research Tasks

To create a new research task:

1. Create a directory structure following the format above
2. Create an `instruction.txt` file with the task instructions
3. Add data files to the `data/` directory
4. Create a `ground_truth.json` file with the expected conclusions

## Integration with MLE-bench

InsightBench is designed to work alongside MLE-bench, allowing you to use the same agents for both ML engineering tasks and research tasks. The `--task-type` flag in `run_agent.py` determines which type of task to run.

## Customizing Agent Behavior

Agents can be configured to handle research tasks by setting the `task_type` field in their configuration. The default is `"mle"`, but you can set it to `"research"` to indicate that the agent is designed for research tasks.

In the agent's `start.sh` script, you can check the `$TASK_TYPE` environment variable to determine whether the agent is running a research task or an MLE task, and adjust the agent's behavior accordingly.