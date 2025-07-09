"""
Utility functions for loading data in research tasks.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple


def get_data_files(data_dir: Path) -> List[Path]:
    """
    Get all data files in a directory.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        List of paths to data files
    """
    return list(data_dir.glob("*"))


def load_data_file(file_path: Path) -> Any:
    """
    Load a data file based on its extension.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        The loaded data
    """
    extension = file_path.suffix.lower()
    
    if extension == ".csv":
        return pd.read_csv(file_path)
    elif extension == ".json":
        with open(file_path, "r") as f:
            return json.load(f)
    elif extension == ".txt":
        with open(file_path, "r") as f:
            return f.read()
    elif extension in [".npy", ".npz"]:
        return np.load(file_path)
    elif extension in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")


def load_research_data(data_dir: Path) -> Dict[str, Any]:
    """
    Load all data files in a research task directory.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        Dictionary mapping file names to loaded data
    """
    data_files = get_data_files(data_dir)
    data = {}
    
    for file_path in data_files:
        try:
            data[file_path.name] = load_data_file(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return data


def get_research_task_structure(task_dir: Path) -> Dict[str, Path]:
    """
    Get the structure of a research task.
    
    Args:
        task_dir: Path to the research task directory
        
    Returns:
        Dictionary with paths to instruction, data, and ground truth
    """
    structure = {
        "instruction": task_dir / "instruction.txt",
        "data_dir": task_dir / "data",
        "ground_truth": task_dir / "ground_truth.json"
    }
    
    # Verify paths exist
    for key, path in structure.items():
        if not path.exists() and key != "ground_truth":  # ground_truth is optional during inference
            raise FileNotFoundError(f"Required path does not exist: {path}")
    
    return structure