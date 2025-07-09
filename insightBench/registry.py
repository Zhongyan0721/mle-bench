"""
Registry for research tasks in InsightBench.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from insightBench import INSIGHT_BENCH_ROOT
from insightBench.utils.data_loaders import get_research_task_structure


@dataclass(frozen=True)
class ResearchTask:
    """
    A research task in InsightBench.
    """
    id: str
    paper_id: str
    instance_id: str
    task_dir: Path
    instruction_path: Path
    data_dir: Path
    ground_truth_path: Optional[Path] = None
    
    @property
    def full_id(self) -> str:
        """
        Get the full ID of the research task.
        """
        return f"{self.paper_id}/{self.instance_id}"


class ResearchTaskRegistry:
    """
    Registry for research tasks in InsightBench.
    """
    def __init__(self, root_dir: Optional[Path] = None):
        """
        Initialize the registry.
        
        Args:
            root_dir: Root directory for InsightBench
        """
        self.root_dir = root_dir or INSIGHT_BENCH_ROOT
        self.benchmark_dir = self.root_dir / "benchmark"
        self.tasks: Dict[str, ResearchTask] = {}
        self._load_tasks()
    
    def _load_tasks(self) -> None:
        """
        Load all research tasks from the benchmark directory.
        """
        papers_dir = self.benchmark_dir / "papers"
        
        for paper_dir in papers_dir.glob("*"):
            if not paper_dir.is_dir():
                continue
                
            paper_id = paper_dir.name
            instances_dir = paper_dir / "instances"
            
            if not instances_dir.exists():
                continue
                
            for instance_dir in instances_dir.glob("*"):
                if not instance_dir.is_dir():
                    continue
                    
                instance_id = instance_dir.name
                task_id = f"{paper_id}/{instance_id}"
                
                try:
                    structure = get_research_task_structure(instance_dir)
                    
                    task = ResearchTask(
                        id=task_id,
                        paper_id=paper_id,
                        instance_id=instance_id,
                        task_dir=instance_dir,
                        instruction_path=structure["instruction"],
                        data_dir=structure["data_dir"],
                        ground_truth_path=structure.get("ground_truth") if structure.get("ground_truth").exists() else None
                    )
                    
                    self.tasks[task_id] = task
                except FileNotFoundError as e:
                    print(f"Error loading task {task_id}: {e}")
    
    def get_task(self, task_id: str) -> ResearchTask:
        """
        Get a research task by ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            The research task
            
        Raises:
            ValueError: If the task is not found
        """
        if task_id not in self.tasks:
            raise ValueError(f"Research task with ID {task_id} not found")
        
        return self.tasks[task_id]
    
    def get_all_tasks(self) -> List[ResearchTask]:
        """
        Get all research tasks.
        
        Returns:
            List of all research tasks
        """
        return list(self.tasks.values())
    
    def get_tasks_by_paper(self, paper_id: str) -> List[ResearchTask]:
        """
        Get all research tasks for a paper.
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            List of research tasks for the paper
        """
        return [task for task in self.tasks.values() if task.paper_id == paper_id]


# Create a global registry instance
registry = ResearchTaskRegistry()