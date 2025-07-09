"""
Evaluation utilities for research tasks.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from insightBench.registry import ResearchTask
from insightBench.utils.llm_inference import load_ground_truth


def load_conclusions(conclusions_path: Path) -> Dict[str, Any]:
    """
    Load conclusions from a JSON file.
    
    Args:
        conclusions_path: Path to the conclusions file
        
    Returns:
        The conclusions data
    """
    with open(conclusions_path, "r") as f:
        return json.load(f)


def evaluate_research_task(
    task: ResearchTask,
    conclusions_path: Path,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate the conclusions for a research task.
    
    Args:
        task: The research task
        conclusions_path: Path to the conclusions file
        metrics: List of metrics to evaluate (default: ["completeness", "correctness"])
        
    Returns:
        Dictionary of metric scores
    """
    if not task.ground_truth_path or not task.ground_truth_path.exists():
        raise ValueError(f"Ground truth not available for task {task.id}")
    
    metrics = metrics or ["completeness", "correctness"]
    ground_truth = load_ground_truth(task.ground_truth_path)
    
    try:
        conclusions = load_conclusions(conclusions_path)
    except Exception as e:
        print(f"Error loading conclusions: {e}")
        return {metric: 0.0 for metric in metrics}
    
    scores = {}
    
    if "completeness" in metrics:
        scores["completeness"] = evaluate_completeness(ground_truth, conclusions)
    
    if "correctness" in metrics:
        scores["correctness"] = evaluate_correctness(ground_truth, conclusions)
    
    return scores


def evaluate_completeness(ground_truth: Dict[str, Any], conclusions: Dict[str, Any]) -> float:
    """
    Evaluate the completeness of conclusions.
    
    Args:
        ground_truth: The ground truth data
        conclusions: The conclusions data
        
    Returns:
        Completeness score (0.0 to 1.0)
    """
    # Check if all expected keys are present
    expected_keys = set(ground_truth.keys())
    actual_keys = set(conclusions.keys())
    
    if not expected_keys:
        return 1.0  # No expected keys means perfect completeness
    
    # Calculate the proportion of expected keys that are present
    completeness = len(actual_keys.intersection(expected_keys)) / len(expected_keys)
    
    return completeness


def evaluate_correctness(ground_truth: Dict[str, Any], conclusions: Dict[str, Any]) -> float:
    """
    Evaluate the correctness of conclusions.
    
    Args:
        ground_truth: The ground truth data
        conclusions: The conclusions data
        
    Returns:
        Correctness score (0.0 to 1.0)
    """
    # This is a simplified implementation
    # In a real-world scenario, you would need more sophisticated evaluation methods
    
    # Check if all expected keys are present
    expected_keys = set(ground_truth.keys())
    actual_keys = set(conclusions.keys())
    
    common_keys = expected_keys.intersection(actual_keys)
    
    if not common_keys:
        return 0.0  # No common keys means zero correctness
    
    # For numerical values, calculate relative error
    # For lists and strings, use simple matching
    correctness_scores = []
    
    for key in common_keys:
        if key not in conclusions:
            correctness_scores.append(0.0)
            continue
        
        gt_value = ground_truth[key]
        conclusion_value = conclusions[key]
        
        if isinstance(gt_value, (int, float)) and isinstance(conclusion_value, (int, float)):
            # For numerical values, use relative error
            if gt_value == 0:
                correctness_scores.append(1.0 if conclusion_value == 0 else 0.0)
            else:
                rel_error = abs(gt_value - conclusion_value) / abs(gt_value)
                correctness_scores.append(max(0.0, 1.0 - min(rel_error, 1.0)))
        
        elif isinstance(gt_value, dict) and isinstance(conclusion_value, dict):
            # For dictionaries, recursively evaluate
            sub_correctness = evaluate_correctness(gt_value, conclusion_value)
            correctness_scores.append(sub_correctness)
        
        elif isinstance(gt_value, list) and isinstance(conclusion_value, list):
            # For lists, calculate overlap
            if not gt_value:
                correctness_scores.append(1.0 if not conclusion_value else 0.0)
            else:
                # This is a simplified approach for list comparison
                # In practice, you would need more sophisticated methods
                correctness_scores.append(min(1.0, len(conclusion_value) / len(gt_value)))
        
        elif isinstance(gt_value, str) and isinstance(conclusion_value, str):
            # For strings, use simple matching
            # In practice, you would use more sophisticated NLP methods
            correctness_scores.append(1.0 if gt_value.lower() == conclusion_value.lower() else 0.0)
        
        else:
            # Different types
            correctness_scores.append(0.0)
    
    # Average the correctness scores
    return sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0