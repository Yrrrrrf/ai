"""
Notebook initialization script.
Put this file in the same directory as your notebook.
"""

import sys
from pathlib import Path


# Find project root and add to Python path
def setup_project_path():
    # Find the project root (where pyproject.toml is located)
    project_root = Path().absolute()
    while project_root != project_root.parent:
        if (project_root / "pyproject.toml").exists():
            break
        project_root = project_root.parent

    # Add to path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added {project_root} to Python path")

    return project_root


# Set up project path when module is imported
project_root = setup_project_path()


# * Common imports for AI notebooks
from src.lib.ga import GeneticAlgorithm
from src.lib.utils.asset_manager import AssetType, am
import torch
import numpy as np
import matplotlib.pyplot as plt


# Set seeds for reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


# Call this to initialize everything in one go
seed = set_seeds()
print(f"Set random seeds to {seed}")
