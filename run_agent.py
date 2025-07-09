import argparse
import asyncio
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, Literal

import docker

from agents.registry import Agent
from agents.registry import registry as agent_registry
from agents.run import run_in_container
from environment.defaults import DEFAULT_CONTAINER_CONFIG_PATH
from mlebench.data import is_dataset_prepared
from mlebench.registry import Competition, registry
from mlebench.utils import create_run_dir, get_logger, get_runs_dir, get_timestamp

# Import research task registry if available
try:
    from insightBench.registry import ResearchTask
    from insightBench.registry import registry as research_registry
    RESEARCH_AVAILABLE = True
except ImportError:
    RESEARCH_AVAILABLE = False

logger = get_logger(__name__)


@dataclass(frozen=True)
class Task:
    run_id: str
    seed: int
    image: str
    path_to_run_group: Path
    path_to_run: Path
    agent: Agent
    competition: Optional[Competition] = None
    research_task: Optional["ResearchTask"] = None
    container_config: dict[str, Any] = None
    task_type: Literal["mle", "research"] = "mle"
    
    def __post_init__(self):
        if self.task_type == "mle" and self.competition is None:
            raise ValueError("Competition must be provided for MLE tasks")
        if self.task_type == "research" and self.research_task is None:
            raise ValueError("Research task must be provided for research tasks")


async def worker(
    idx: int,
    queue: asyncio.Queue[Task],
    client: docker.DockerClient,
    tasks_outputs: dict[str, dict[str, Any]],
) -> None:
    while True:
        task = await queue.get()

        # Create logger for the run
        run_logger = get_logger(str(task.path_to_run))
        log_file_handler = logging.FileHandler(task.path_to_run / "run.log")
        log_file_handler.setFormatter(
            logging.getLogger().handlers[0].formatter
        )  # match the formatting we have
        run_logger.addHandler(log_file_handler)
        run_logger.propagate = False

        # Log based on task type
        if task.task_type == "mle":
            task_id = task.competition.id
            run_logger.info(
                f"[Worker {idx}] Running seed {task.seed} for competition {task_id} and agent {task.agent.name}"
            )
        else:  # research
            task_id = task.research_task.id
            run_logger.info(
                f"[Worker {idx}] Running seed {task.seed} for research task {task_id} and agent {task.agent.name}"
            )

        task_output = {}
        try:
            await asyncio.to_thread(
                run_in_container,
                client=client,
                competition=task.competition if task.task_type == "mle" else None,
                agent=task.agent,
                image=task.agent.name,
                container_config=task.container_config,
                retain_container=args.retain,
                run_dir=task.path_to_run,
                logger=run_logger,
                research_task=task.research_task if task.task_type == "research" else None,
            )
            task_output["success"] = True

            # Log completion based on task type
            if task.task_type == "mle":
                task_id = task.competition.id
                run_logger.info(
                    f"[Worker {idx}] Finished running seed {task.seed} for competition {task_id} and agent {task.agent.name}"
                )
            else:  # research
                task_id = task.research_task.id
                run_logger.info(
                    f"[Worker {idx}] Finished running seed {task.seed} for research task {task_id} and agent {task.agent.name}"
                )
        except Exception as e:
            stack_trace = traceback.format_exc()
            run_logger.error(type(e))
            run_logger.error(stack_trace)
            
            # Log error based on task type
            if task.task_type == "mle":
                task_id = task.competition.id
                run_logger.error(
                    f"Run failed for seed {task.seed}, agent {task.agent.id} and competition {task_id}"
                )
            else:  # research
                task_id = task.research_task.id
                run_logger.error(
                    f"Run failed for seed {task.seed}, agent {task.agent.id} and research task {task_id}"
                )
                
            task_output["success"] = False
        finally:
            tasks_outputs[task.run_id] = task_output
            queue.task_done()


async def main(args):
    client = docker.from_env()
    global registry
    registry = registry.set_data_dir(Path(args.data_dir))

    agent = agent_registry.get_agent(args.agent_id)
    if agent.privileged and not (
        os.environ.get("I_ACCEPT_RUNNING_PRIVILEGED_CONTAINERS", "False").lower()
        in ("true", "1", "t")
    ):
        raise ValueError(
            "Agent requires running in a privileged container, but the environment variable `I_ACCEPT_RUNNING_PRIVILEGED_CONTAINERS` is not set to `True`! "
            "Carefully consider if you wish to run this agent before continuing. See agents/README.md for more details."
        )

    run_group = f"{get_timestamp()}_run-group_{agent.name}"

    with open(args.container_config, "r") as f:
        container_config = json.load(f)

    # Create tasks based on task type
    logger.info(f"Launching run group: {run_group}")
    tasks = []
    
    # Determine task type
    task_type = args.task_type or agent.task_type
    
    if task_type == "mle":
        # Load competition ids and check all are prepared
        with open(args.competition_set, "r") as f:
            competition_ids = [line.strip() for line in f.read().splitlines() if line.strip()]
        
        for competition_id in competition_ids:
            competition = registry.get_competition(competition_id)
            if not is_dataset_prepared(competition):
                raise ValueError(
                    f"Dataset for competition `{competition.id}` is not prepared! "
                    f"Please run `mlebench prepare -c {competition.id}` to prepare the dataset."
                )
        
        # Create tasks for each (competition * seed)
        for seed in range(args.n_seeds):
            for competition_id in competition_ids:
                competition = registry.get_competition(competition_id)
                run_dir = create_run_dir(competition.id, agent.id, run_group)
                run_id = run_dir.stem
                task = Task(
                    run_id=run_id,
                    seed=seed,
                    image=agent.name,
                    agent=agent,
                    competition=competition,
                    path_to_run_group=run_dir.parent,
                    path_to_run=run_dir,
                    container_config=container_config,
                    task_type="mle",
                )
                tasks.append(task)
    
    elif task_type == "research":
        # Check if research tasks are available
        if not RESEARCH_AVAILABLE:
            raise ValueError("Research tasks are not available. Please install the insightBench package.")
        
        # Load research task ids
        if args.research_task_set:
            with open(args.research_task_set, "r") as f:
                research_task_ids = [line.strip() for line in f.read().splitlines() if line.strip()]
        elif args.research_task_id:
            research_task_ids = [args.research_task_id]
        else:
            raise ValueError("Either --research-task-set or --research-task-id must be provided for research tasks.")
        
        # Create tasks for each (research task * seed)
        for seed in range(args.n_seeds):
            for task_id in research_task_ids:
                research_task = research_registry.get_task(task_id)
                run_dir = create_run_dir(f"research_{task_id.replace('/', '_')}", agent.id, run_group)
                run_id = run_dir.stem
                task = Task(
                    run_id=run_id,
                    seed=seed,
                    image=agent.name,
                    agent=agent,
                    research_task=research_task,
                    path_to_run_group=run_dir.parent,
                    path_to_run=run_dir,
                    container_config=container_config,
                    task_type="research",
                )
                tasks.append(task)
    
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    logger.info(f"Creating {args.n_workers} workers to serve {len(tasks)} tasks...")

    # Create queue of tasks, and assign workers to run them
    queue = asyncio.Queue()
    for task in tasks:
        queue.put_nowait(task)
    workers = []
    tasks_outputs = {}
    for idx in range(args.n_workers):
        w = asyncio.create_task(worker(idx, queue, client, tasks_outputs))
        workers.append(w)

    # Wait for all tasks to be completed and collect results
    started_at = time.monotonic()
    await queue.join()
    time_taken = time.monotonic() - started_at

    for w in workers:
        w.cancel()  # Cancel all workers now that the queue is empty

    await asyncio.gather(*workers, return_exceptions=True)

    # Generate metadata.json
    metadata = {
        "run_group": run_group,
        "created_at": get_timestamp(),
        "task_type": task_type,
        "runs": tasks_outputs,
    }
    run_group_dir = get_runs_dir() / run_group
    with open(run_group_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=False, default=str)
    logger.info(f"{args.n_workers} workers ran for {time_taken:.2f} seconds in total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an agent on a set of competitions or research tasks in a Docker container."
    )
    parser.add_argument(
        "--agent-id",
        help="Agent ID of the agent to run.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--task-type",
        help="Type of task to run: 'mle' for ML engineering tasks or 'research' for research tasks.",
        type=str,
        choices=["mle", "research"],
        required=False,
        default=None,
    )
    
    # MLE task arguments
    parser.add_argument(
        "--competition-set",
        type=str,
        required=False,
        help="Path to a text file with a single competition ID on each line (for MLE tasks)",
    )
    
    # Research task arguments
    parser.add_argument(
        "--research-task-id",
        type=str,
        required=False,
        help="ID of the research task to run (for research tasks)",
    )
    parser.add_argument(
        "--research-task-set",
        type=str,
        required=False,
        help="Path to a text file with a single research task ID on each line (for research tasks)",
    )
    
    # Common arguments
    parser.add_argument(
        "--n-workers",
        type=int,
        required=False,
        default=1,
        help="Number of workers to run in parallel",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        required=False,
        default=1,
        help="Number of seeds to run for each task",
    )
    parser.add_argument(
        "--container-config",
        help="Path to a JSON file with an environment configuration; these args will be passed to `docker.from_env().containers.create`",
        type=str,
        required=False,
        default=DEFAULT_CONTAINER_CONFIG_PATH,
    )
    parser.add_argument(
        "--retain",
        help="Whether to retain the container after the run instead of removing it.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--run-dir",
        help="Path to the directory where all assets associated with the run are stored.",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--data-dir",
        help="Path to the directory containing the competition data.",
        type=str,
        required=False,
        default=registry.get_data_dir(),
    )
    args = parser.parse_args()
    
    # Validate arguments
    if args.task_type == "mle" and not args.competition_set:
        parser.error("--competition-set is required for MLE tasks")
    
    if args.task_type == "research" and not (args.research_task_id or args.research_task_set):
        parser.error("Either --research-task-id or --research-task-set is required for research tasks")
    
    logger = get_logger(__name__)
    asyncio.run(main(args))
