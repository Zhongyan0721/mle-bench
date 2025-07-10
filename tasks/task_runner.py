"""
Task Runner System

This module provides utilities for running tasks in the agent environment.
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .task_registry import Task, registry


logger = logging.getLogger(__name__)


class TaskRunner:
    """Runs tasks in the agent environment."""
    
    def __init__(self, work_dir: Path, data_dir: Path, output_dir: Path):
        self.work_dir = work_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_task_environment(self, task: Task) -> Dict[str, Any]:
        """Prepare the environment for running a task."""
        logger.info(f"Preparing environment for task: {task.name}")
        
        # Create task-specific directories
        task_dir = self.work_dir / f"task_{task.id}"
        task_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (task_dir / 'data').mkdir(exist_ok=True)
        (task_dir / 'code').mkdir(exist_ok=True)
        (task_dir / 'results').mkdir(exist_ok=True)
        (task_dir / 'logs').mkdir(exist_ok=True)
        
        # Save task configuration
        task_config_path = task_dir / 'task_config.json'
        with open(task_config_path, 'w') as f:
            json.dump(task.to_dict(), f, indent=2)
        
        # Create instructions file
        instructions_path = task_dir / 'instructions.txt'
        with open(instructions_path, 'w') as f:
            f.write(task.instructions)
        
        # Prepare data if needed
        self._prepare_task_data(task, task_dir / 'data')
        
        environment = {
            'task_dir': task_dir,
            'task_config_path': task_config_path,
            'instructions_path': instructions_path,
            'data_dir': task_dir / 'data',
            'code_dir': task_dir / 'code',
            'results_dir': task_dir / 'results',
            'logs_dir': task_dir / 'logs'
        }
        
        logger.info(f"Task environment prepared at: {task_dir}")
        return environment
    
    def _prepare_task_data(self, task: Task, data_dir: Path):
        """Prepare data for the task."""
        data_requirements = task.data_requirements
        
        if data_requirements.get('generate_synthetic', False):
            self._generate_synthetic_data(data_requirements, data_dir)
        
        if 'copy_files' in data_requirements:
            self._copy_data_files(data_requirements['copy_files'], data_dir)
    
    def _generate_synthetic_data(self, requirements: Dict[str, Any], data_dir: Path):
        """Generate synthetic data based on requirements."""
        from .utils.data_generator import DataGenerator
        
        generator = DataGenerator()
        
        if requirements.get('dataset_type') == 'classification':
            df = generator.generate_classification_dataset(
                n_samples=requirements.get('n_samples', 1000),
                n_features=requirements.get('n_features', 10),
                n_classes=requirements.get('n_classes', 2)
            )
            df.to_csv(data_dir / 'classification_data.csv', index=False)
        
        elif requirements.get('dataset_type') == 'regression':
            df = generator.generate_regression_dataset(
                n_samples=requirements.get('n_samples', 1000),
                n_features=requirements.get('n_features', 10)
            )
            df.to_csv(data_dir / 'regression_data.csv', index=False)
        
        elif requirements.get('dataset_type') == 'time_series':
            df = generator.generate_time_series_dataset(
                n_samples=requirements.get('n_samples', 1000),
                n_features=requirements.get('n_features', 5)
            )
            df.to_csv(data_dir / 'time_series_data.csv', index=False)
    
    def _copy_data_files(self, file_list: List[str], data_dir: Path):
        """Copy specified data files to the task data directory."""
        for file_path in file_list:
            source_path = self.data_dir / file_path
            if source_path.exists():
                dest_path = data_dir / source_path.name
                dest_path.write_bytes(source_path.read_bytes())
                logger.info(f"Copied {source_path} to {dest_path}")
            else:
                logger.warning(f"Data file not found: {source_path}")
    
    def run_task(self, task_id: str, agent_command: List[str], timeout: Optional[int] = None) -> Dict[str, Any]:
        """Run a task with the specified agent command."""
        task = registry.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        logger.info(f"Running task: {task.name}")
        
        # Prepare environment
        env = self.prepare_task_environment(task)
        
        # Set up environment variables
        env_vars = {
            'TASK_ID': task.id,
            'TASK_TYPE': task.task_type,
            'TASK_DIR': str(env['task_dir']),
            'DATA_DIR': str(env['data_dir']),
            'CODE_DIR': str(env['code_dir']),
            'RESULTS_DIR': str(env['results_dir']),
            'LOGS_DIR': str(env['logs_dir'])
        }
        
        # Run the agent command
        start_time = time.time()
        try:
            result = subprocess.run(
                agent_command,
                cwd=env['task_dir'],
                env={**dict(os.environ), **env_vars},
                capture_output=True,
                text=True,
                timeout=timeout or task.estimated_time
            )
            
            success = result.returncode == 0
            execution_time = time.time() - start_time
            
        except subprocess.TimeoutExpired:
            success = False
            execution_time = timeout or task.estimated_time
            result = type('Result', (), {
                'returncode': -1,
                'stdout': '',
                'stderr': 'Task timed out'
            })()
        
        # Collect results
        task_result = {
            'task_id': task.id,
            'task_name': task.name,
            'success': success,
            'execution_time': execution_time,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'environment': {k: str(v) for k, v in env.items()},
            'timestamp': time.time()
        }
        
        # Save results
        results_path = self.output_dir / f"task_{task.id}_results.json"
        with open(results_path, 'w') as f:
            json.dump(task_result, f, indent=2)
        
        logger.info(f"Task completed. Success: {success}, Time: {execution_time:.2f}s")
        return task_result
    
    def evaluate_task_results(self, task_id: str, results_dir: Path) -> Dict[str, Any]:
        """Evaluate the results of a completed task."""
        task = registry.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        from .utils.task_validator import TaskValidator
        
        validator = TaskValidator()
        evaluation = validator.evaluate_task(task, results_dir)
        
        # Save evaluation
        eval_path = self.output_dir / f"task_{task.id}_evaluation.json"
        with open(eval_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        
        return evaluation