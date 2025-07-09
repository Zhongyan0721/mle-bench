"""
Command-line interface for InsightBench.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from insightBench.registry import registry
from insightBench.evaluate import evaluate_research_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_task(args):
    """
    Evaluate a research task submission.
    """
    task = registry.get_task(args.task_id)
    
    if not task.ground_truth_path or not task.ground_truth_path.exists():
        logger.error(f"Ground truth not available for task {task.id}")
        return
    
    conclusions_path = Path(args.conclusions_path)
    if not conclusions_path.exists():
        logger.error(f"Conclusions file not found: {conclusions_path}")
        return
    
    metrics = args.metrics.split(",") if args.metrics else None
    
    try:
        scores = evaluate_research_task(task, conclusions_path, metrics)
        
        logger.info(f"Evaluation results for task {task.id}:")
        for metric, score in scores.items():
            logger.info(f"  {metric}: {score:.4f}")
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump({
                    "task_id": task.id,
                    "scores": scores,
                }, f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Error evaluating task: {e}")


def list_tasks(args):
    """
    List available research tasks.
    """
    tasks = registry.get_all_tasks()
    
    if args.paper_id:
        tasks = [task for task in tasks if task.paper_id == args.paper_id]
    
    if not tasks:
        logger.info("No tasks found")
        return
    
    logger.info(f"Found {len(tasks)} tasks:")
    for task in tasks:
        logger.info(f"  {task.id}")
        
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump({
                "tasks": [task.id for task in tasks],
            }, f, indent=2)
        
        logger.info(f"Task list saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="InsightBench CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a research task submission")
    evaluate_parser.add_argument("task_id", help="ID of the research task")
    evaluate_parser.add_argument("conclusions_path", help="Path to the conclusions file")
    evaluate_parser.add_argument("--metrics", help="Comma-separated list of metrics to evaluate")
    evaluate_parser.add_argument("--output", help="Path to save the evaluation results")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available research tasks")
    list_parser.add_argument("--paper-id", help="Filter tasks by paper ID")
    list_parser.add_argument("--output", help="Path to save the task list")
    
    args = parser.parse_args()
    
    if args.command == "evaluate":
        evaluate_task(args)
    elif args.command == "list":
        list_tasks(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()