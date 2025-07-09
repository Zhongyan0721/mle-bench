"""
Utility functions for LLM inference in research tasks.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List


def load_instruction(instruction_path: Path) -> str:
    """
    Load instruction from a text file.
    
    Args:
        instruction_path: Path to the instruction file
        
    Returns:
        The instruction text
    """
    with open(instruction_path, "r") as f:
        return f.read().strip()


def load_ground_truth(ground_truth_path: Path) -> Dict[str, Any]:
    """
    Load ground truth from a JSON file.
    
    Args:
        ground_truth_path: Path to the ground truth file
        
    Returns:
        The ground truth data
    """
    with open(ground_truth_path, "r") as f:
        return json.load(f)


def save_conclusions(conclusions: Dict[str, Any], output_path: Path) -> None:
    """
    Save conclusions to a JSON file.
    
    Args:
        conclusions: The conclusions to save
        output_path: Path to save the conclusions to
    """
    with open(output_path, "w") as f:
        json.dump(conclusions, f, indent=2)


def format_research_prompt(instruction: str, data_description: Optional[str] = None) -> str:
    """
    Format a research prompt with instruction and data description.
    
    Args:
        instruction: The research instruction
        data_description: Optional description of the data
        
    Returns:
        The formatted prompt
    """
    prompt = f"# Research Task\n\n{instruction}\n\n"
    
    if data_description:
        prompt += f"# Data Description\n\n{data_description}\n\n"
    
    prompt += "Please analyze the data and provide your conclusions in a structured format."
    
    return prompt