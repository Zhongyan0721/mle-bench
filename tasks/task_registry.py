"""
Task Registry System

This module provides a registry for different types of tasks that AI agents can execute.
"""

import json
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


@dataclass
class Task:
    """Represents a task that can be executed by an AI agent."""
    
    id: str
    name: str
    description: str
    task_type: str
    difficulty: str  # easy, medium, hard
    estimated_time: int  # in seconds
    data_requirements: Dict[str, Any]
    evaluation_criteria: Dict[str, Any]
    instructions: str
    expected_outputs: List[str]
    metadata: Dict[str, Any]
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        return cls(**data)
    
    def save(self, path: Path):
        """Save task to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'Task':
        """Load task from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class TaskRegistry:
    """Registry for managing tasks."""
    
    def __init__(self, tasks_dir: Optional[Path] = None):
        if tasks_dir is None:
            tasks_dir = Path(__file__).parent
        self.tasks_dir = tasks_dir
        self.templates_dir = tasks_dir / 'templates'
        self.examples_dir = tasks_dir / 'examples'
        self._tasks = {}
        self._load_tasks()
    
    def _load_tasks(self):
        """Load all tasks from the tasks directory."""
        # Load template tasks
        if self.templates_dir.exists():
            for task_file in self.templates_dir.glob('*.json'):
                try:
                    task = Task.load(task_file)
                    self._tasks[task.id] = task
                except Exception as e:
                    print(f"Error loading task {task_file}: {e}")
        
        # Load example tasks
        if self.examples_dir.exists():
            for task_file in self.examples_dir.glob('*.json'):
                try:
                    task = Task.load(task_file)
                    self._tasks[task.id] = task
                except Exception as e:
                    print(f"Error loading task {task_file}: {e}")
    
    def register_task(self, task: Task):
        """Register a new task."""
        self._tasks[task.id] = task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def list_tasks(self, task_type: Optional[str] = None, difficulty: Optional[str] = None) -> List[Task]:
        """List tasks with optional filtering."""
        tasks = list(self._tasks.values())
        
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]
        
        if difficulty:
            tasks = [t for t in tasks if t.difficulty == difficulty]
        
        return tasks
    
    def get_task_types(self) -> List[str]:
        """Get all available task types."""
        return list(set(task.task_type for task in self._tasks.values()))
    
    def create_task_config(self, task_id: str, output_path: Path, **kwargs) -> Dict[str, Any]:
        """Create a task configuration file for an agent."""
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        config = {
            'task': task.to_dict(),
            'runtime_config': kwargs,
            'created_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def save_task(self, task: Task, filename: Optional[str] = None):
        """Save a task to the templates directory."""
        if filename is None:
            filename = f"{task.id}.json"
        
        self.templates_dir.mkdir(exist_ok=True)
        task_path = self.templates_dir / filename
        task.save(task_path)
        
        # Update registry
        self._tasks[task.id] = task


# Global task registry instance
registry = TaskRegistry()