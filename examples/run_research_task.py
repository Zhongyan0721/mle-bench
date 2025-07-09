#!/usr/bin/env python3
"""
Example script for running a research task with InsightBench.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from insightBench.registry import registry as research_registry
from agents.registry import registry as agent_registry
from run_agent import main as run_agent_main


class Args:
    """
    Dummy class to hold arguments for run_agent_main.
    """
    pass


async def run_research_task(agent_id, task_id, n_seeds=1, retain=False):
    """
    Run a research task with the specified agent.
    
    Args:
        agent_id: ID of the agent to run
        task_id: ID of the research task to run
        n_seeds: Number of seeds to run
        retain: Whether to retain the container after the run
    """
    # Create arguments for run_agent_main
    args = Args()
    args.agent_id = agent_id
    args.task_type = "research"
    args.research_task_id = task_id
    args.research_task_set = None
    args.competition_set = None
    args.n_workers = 1
    args.n_seeds = n_seeds
    args.container_config = "environment/config/container_configs/default.json"
    args.retain = retain
    args.run_dir = None
    args.data_dir = None
    
    # Run the agent on the research task
    await run_agent_main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a research task with InsightBench")
    parser.add_argument("agent_id", help="ID of the agent to run")
    parser.add_argument("task_id", help="ID of the research task to run")
    parser.add_argument("--n-seeds", type=int, default=1, help="Number of seeds to run")
    parser.add_argument("--retain", action="store_true", help="Retain the container after the run")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run the research task
    asyncio.run(run_research_task(args.agent_id, args.task_id, args.n_seeds, args.retain))