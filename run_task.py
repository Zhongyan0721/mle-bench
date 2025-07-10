#!/usr/bin/env python3
"""
Task Runner Script

This script provides a convenient way to run AI agent tasks using the experiment infrastructure.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from tasks import registry, TaskRunner
from tasks.utils.data_generator import DataGenerator

logger = logging.getLogger(__name__)


def setup_task_environment(task_id: str, work_dir: Path, data_dir: Path) -> Dict[str, Any]:
    """Set up the environment for running a task."""
    task = registry.get_task(task_id)
    if not task:
        raise ValueError(f"Task '{task_id}' not found")
    
    logger.info(f"Setting up environment for task: {task.name}")
    
    # Create directories
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data if needed
    if task.data_requirements.get('generate_synthetic', False):
        logger.info("Generating synthetic data...")
        generator = DataGenerator()
        
        dataset_type = task.data_requirements.get('dataset_type')
        
        if dataset_type == 'classification':
            df = generator.generate_classification_dataset(
                n_samples=task.data_requirements.get('n_samples', 1000),
                n_features=task.data_requirements.get('n_features', 10),
                n_classes=task.data_requirements.get('n_classes', 2)
            )
            df.to_csv(data_dir / 'classification_data.csv', index=False)
            
        elif dataset_type == 'regression':
            df = generator.generate_regression_dataset(
                n_samples=task.data_requirements.get('n_samples', 1000),
                n_features=task.data_requirements.get('n_features', 10)
            )
            df.to_csv(data_dir / 'regression_data.csv', index=False)
            
        elif dataset_type == 'time_series':
            df = generator.generate_time_series_dataset(
                n_samples=task.data_requirements.get('n_samples', 1000),
                n_features=task.data_requirements.get('n_features', 5)
            )
            df.to_csv(data_dir / 'time_series_data.csv', index=False)
    
    # Create task configuration file
    task_config = {
        'task': task.to_dict(),
        'environment': {
            'work_dir': str(work_dir),
            'data_dir': str(data_dir)
        }
    }
    
    config_path = work_dir / 'task_config.json'
    with open(config_path, 'w') as f:
        json.dump(task_config, f, indent=2)
    
    # Create instructions file
    instructions_path = work_dir / 'experiment_instructions.txt'
    with open(instructions_path, 'w') as f:
        f.write(task.instructions)
    
    return {
        'task': task,
        'config_path': config_path,
        'instructions_path': instructions_path,
        'data_dir': data_dir,
        'work_dir': work_dir
    }


def run_task_with_docker(
    task_id: str,
    agent_id: str = 'experiment',
    work_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    time_limit: Optional[int] = None,
    retain_container: bool = False
) -> Dict[str, Any]:
    """Run a task using Docker with the experiment agent."""
    
    # Set up default directories
    if work_dir is None:
        work_dir = Path.cwd() / 'task_workspace'
    if data_dir is None:
        data_dir = work_dir / 'data'
    
    # Set up task environment
    env_info = setup_task_environment(task_id, work_dir, data_dir)
    task = env_info['task']
    
    # Use task's estimated time if not provided
    if time_limit is None:
        time_limit = task.estimated_time
    
    logger.info(f"Running task '{task.name}' with agent '{agent_id}'")
    logger.info(f"Time limit: {time_limit} seconds")
    logger.info(f"Work directory: {work_dir}")
    logger.info(f"Data directory: {data_dir}")
    
    try:
        import docker
        client = docker.from_env()
        
        # Check if the experiment agent image exists
        agent_image = f"mlebench-{agent_id}"
        try:
            client.images.get(agent_image)
        except docker.errors.ImageNotFound:
            logger.error(f"Docker image '{agent_image}' not found. Please build the experiment agent first.")
            return {'success': False, 'error': f'Image {agent_image} not found'}
        
        # Set up volumes
        volumes = {
            str(data_dir.absolute()): {'bind': '/home/data', 'mode': 'ro'},
            str(work_dir.absolute()): {'bind': '/home/experiments', 'mode': 'rw'},
            str(env_info['config_path'].absolute()): {'bind': '/home/task_config.json', 'mode': 'ro'},
            str(env_info['instructions_path'].absolute()): {'bind': '/home/experiment_instructions.txt', 'mode': 'ro'}
        }
        
        # Set up environment variables
        environment = {
            'TASK_TYPE': task.task_type,
            'EXPERIMENT_NAME': f"task_{task_id}",
            'TIME_LIMIT_SECS': str(time_limit)
        }
        
        # Create and run container
        container_name = f"task-{task_id}-{os.getpid()}"
        
        logger.info(f"Creating container: {container_name}")
        container = client.containers.run(
            agent_image,
            name=container_name,
            volumes=volumes,
            environment=environment,
            detach=True,
            remove=not retain_container
        )
        
        logger.info("Container started, waiting for completion...")
        
        # Wait for container to finish
        result = container.wait(timeout=time_limit + 60)  # Add buffer time
        
        # Get logs
        logs = container.logs().decode('utf-8')
        
        # Clean up if not retaining
        if not retain_container:
            try:
                container.remove()
            except:
                pass  # Container might already be removed
        
        success = result['StatusCode'] == 0
        
        logger.info(f"Task completed. Success: {success}")
        
        return {
            'success': success,
            'exit_code': result['StatusCode'],
            'logs': logs,
            'work_dir': str(work_dir),
            'data_dir': str(data_dir)
        }
        
    except Exception as e:
        logger.error(f"Error running task with Docker: {e}")
        return {'success': False, 'error': str(e)}


def run_task_locally(
    task_id: str,
    work_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    script_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Run a task locally without Docker."""
    
    # Set up default directories
    if work_dir is None:
        work_dir = Path.cwd() / 'task_workspace'
    if data_dir is None:
        data_dir = work_dir / 'data'
    
    # Set up task environment
    env_info = setup_task_environment(task_id, work_dir, data_dir)
    task = env_info['task']
    
    logger.info(f"Running task '{task.name}' locally")
    
    # Use provided script or default example
    if script_path is None:
        if task.task_type == 'classification':
            script_path = Path(__file__).parent / 'tasks' / 'examples' / 'classification_example.py'
        else:
            logger.warning(f"No default script for task type '{task.task_type}'")
            return {'success': False, 'error': f'No script available for task type {task.task_type}'}
    
    if not script_path.exists():
        return {'success': False, 'error': f'Script not found: {script_path}'}
    
    try:
        import subprocess
        
        # Set up environment
        env = os.environ.copy()
        env.update({
            'TASK_TYPE': task.task_type,
            'EXPERIMENT_NAME': f"task_{task_id}",
            'PYTHONPATH': str(Path(__file__).parent)
        })
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=task.estimated_time
        )
        
        success = result.returncode == 0
        
        logger.info(f"Task completed. Success: {success}")
        
        return {
            'success': success,
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'work_dir': str(work_dir),
            'data_dir': str(data_dir)
        }
        
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Task timed out'}
    except Exception as e:
        logger.error(f"Error running task locally: {e}")
        return {'success': False, 'error': str(e)}


def validate_task_results(task_id: str, results_dir: Path) -> Dict[str, Any]:
    """Validate the results of a completed task."""
    from tasks.utils.task_validator import TaskValidator
    
    task = registry.get_task(task_id)
    if not task:
        raise ValueError(f"Task '{task_id}' not found")
    
    validator = TaskValidator()
    evaluation = validator.evaluate_task(task, results_dir)
    
    return evaluation


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run AI agent tasks")
    parser.add_argument('task_id', help='Task ID to run')
    parser.add_argument('--agent', default='experiment', help='Agent to use (default: experiment)')
    parser.add_argument('--work-dir', type=Path, help='Working directory for the task')
    parser.add_argument('--data-dir', type=Path, help='Data directory for the task')
    parser.add_argument('--time-limit', type=int, help='Time limit in seconds')
    parser.add_argument('--local', action='store_true', help='Run locally without Docker')
    parser.add_argument('--script', type=Path, help='Script to run for local execution')
    parser.add_argument('--retain', action='store_true', help='Retain Docker container after completion')
    parser.add_argument('--validate', action='store_true', help='Validate results after completion')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if task exists
    task = registry.get_task(args.task_id)
    if not task:
        print(f"Error: Task '{args.task_id}' not found")
        print("Available tasks:")
        for t in registry.list_tasks():
            print(f"  - {t.id}: {t.name}")
        sys.exit(1)
    
    print(f"Running task: {task.name}")
    print(f"Type: {task.task_type}")
    print(f"Difficulty: {task.difficulty}")
    print(f"Estimated time: {task.estimated_time}s")
    print()
    
    # Run the task
    if args.local:
        result = run_task_locally(
            args.task_id,
            work_dir=args.work_dir,
            data_dir=args.data_dir,
            script_path=args.script
        )
    else:
        result = run_task_with_docker(
            args.task_id,
            agent_id=args.agent,
            work_dir=args.work_dir,
            data_dir=args.data_dir,
            time_limit=args.time_limit,
            retain_container=args.retain
        )
    
    # Print results
    print(f"Task completed. Success: {result['success']}")
    
    if not result['success']:
        print(f"Error: {result.get('error', 'Unknown error')}")
        if 'stderr' in result and result['stderr']:
            print("Error output:")
            print(result['stderr'])
        sys.exit(1)
    
    # Validate results if requested
    if args.validate and result['success']:
        print("\nValidating results...")
        results_dir = Path(result['work_dir']) / 'results'
        
        if results_dir.exists():
            evaluation = validate_task_results(args.task_id, results_dir)
            print(f"Validation score: {evaluation['overall_score']:.2f}")
            print(f"Validation success: {evaluation['success']}")
            
            if evaluation['feedback']:
                print("Feedback:")
                for feedback in evaluation['feedback']:
                    print(f"  - {feedback}")
        else:
            print("No results directory found for validation")
    
    print(f"\nTask workspace: {result['work_dir']}")
    print(f"Data directory: {result['data_dir']}")


if __name__ == "__main__":
    main()