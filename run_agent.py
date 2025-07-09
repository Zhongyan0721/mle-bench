import argparse
import asyncio
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import docker

from agents.registry import Agent
from agents.registry import registry as agent_registry
from agents.run import run_in_container
from environment.defaults import DEFAULT_CONTAINER_CONFIG_PATH
from mlebench.data import is_dataset_prepared
from mlebench.registry import Competition, registry
from mlebench.utils import create_run_dir, get_logger, get_runs_dir, get_timestamp

logger = get_logger(__name__)


@dataclass(frozen=True)
class Task:
    run_id: str
    seed: int
    image: str
    path_to_run_group: Path
    path_to_run: Path
    agent: Agent
    competition: Competition | None
    container_config: dict[str, Any]
    dataset_dir: Path | None = None
    prompt_file: Path | None = None


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

        if task.competition is not None:
            run_logger.info(
                f"[Worker {idx}] Running seed {task.seed} for {task.competition.id} and agent {task.agent.name}"
            )
        else:
            run_logger.info(
                f"[Worker {idx}] Running seed {task.seed} on research task {task.dataset_dir} and agent {task.agent.name}"
            )

        task_output = {}
        try:
            await asyncio.to_thread(
                run_in_container,
                client=client,
                competition=task.competition,
                agent=task.agent,
                image=task.agent.name,
                container_config=task.container_config,
                retain_container=args.retain,
                run_dir=task.path_to_run,
                logger=run_logger,
                dataset_dir=task.dataset_dir,
                prompt_file=task.prompt_file,
            )
            task_output["success"] = True

            if task.competition is not None:
                run_logger.info(
                    f"[Worker {idx}] Finished running seed {task.seed} for {task.competition.id} and agent {task.agent.name}"
                )
            else:
                run_logger.info(
                    f"[Worker {idx}] Finished running seed {task.seed} for research task and agent {task.agent.name}"
                )
        except Exception as e:
            stack_trace = traceback.format_exc()
            run_logger.error(type(e))
            run_logger.error(stack_trace)
            if task.competition is not None:
                run_logger.error(
                    f"Run failed for seed {task.seed}, agent {task.agent.id} and competition {task.competition.id}"
                )
            else:
                run_logger.error(
                    f"Run failed for seed {task.seed}, agent {task.agent.id} on research task"
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

    # Determine tasks
    if args.task_dir:
        competition_ids = []
    elif args.competition_set:
        with open(args.competition_set, "r") as f:
            competition_ids = [line.strip() for line in f.read().splitlines() if line.strip()]
        for competition_id in competition_ids:
            competition = registry.get_competition(competition_id)
            if not is_dataset_prepared(competition):
                raise ValueError(
                    f"Dataset for competition `{competition.id}` is not prepared! "
                    f"Please run `mlebench prepare -c {competition.id}` to prepare the dataset."
                )
    else:
        raise ValueError("Either --task-dir or --competition-set must be provided")

    with open(args.container_config, "r") as f:
        container_config = json.load(f)

    # Create tasks for each (competition * seed)
    logger.info(f"Launching run group: {run_group}")
    tasks = []
    if args.task_dir:
        dataset_dir = Path(args.task_dir).resolve() / "data"
        prompt_file = Path(args.prompt_file) if args.prompt_file else Path(args.task_dir) / "instruction.txt"
        for seed in range(args.n_seeds):
            run_dir = create_run_dir(Path(args.task_dir).stem, agent.id, run_group)
            run_id = run_dir.stem
            task = Task(
                run_id=run_id,
                seed=seed,
                image=agent.name,
                agent=agent,
                competition=None,
                path_to_run_group=run_dir.parent,
                path_to_run=run_dir,
                container_config=container_config,
                dataset_dir=dataset_dir,
                prompt_file=prompt_file,
            )
            tasks.append(task)
    else:
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
                )
                tasks.append(task)

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
        "runs": tasks_outputs,
    }
    run_group_dir = get_runs_dir() / run_group
    with open(run_group_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=False, default=str)
    logger.info(f"{args.n_workers} workers ran for {time_taken:.2f} seconds in total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an agent on a set of competitions or custom research tasks in a Docker container."
    )
    parser.add_argument(
        "--agent-id",
        help="Agent ID of the agent to run.",
        type=str,
    )
    parser.add_argument(
        "--competition-set",
        type=str,
        required=False,
        help="Path to a text file with a single competition ID on each line",
    )
    parser.add_argument(
        "--task-dir",
        type=str,
        required=False,
        help="Path to a research task directory",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=False,
        help="Optional path to a prompt file. Defaults to <task-dir>/instruction.txt",
    )
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
        help="Number of seeds to run for each competition",
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
    logger = get_logger(__name__)

    asyncio.run(main(args))
