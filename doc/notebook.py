"""
Notebook initialization script.
Put this file in the same directory as your notebook.
"""

# * std imports
import sys
from pathlib import Path


# ^ Project imports should be imported after setting up the path
# Find project root and add to Python path
def setup_project_path() -> Path:
    # Find the project root (where pyproject.toml is located)
    project_root: Path = Path().absolute()
    while project_root != project_root.parent:
        if (project_root / "pyproject.toml").exists():
            break
        project_root = project_root.parent

    # Add to path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added {project_root} to Python path")

    return project_root


project_root = setup_project_path()


# * Main imports

# ^ assets manager
from lib.tools.asset_manager import AssetType, am

# ^ libs for data handling and visualization
import matplotlib.pyplot as plt
import numpy as np
import torch

# ^ src imports
from src.lib.model.ga import GeneticAlgorithm
from src.lib.model.pso import PSO


# Set seeds for reproducibility
def set_seeds(seed: int = None):
    if seed is None:
        seed = 42
        print(
            f"\033[96m[INFO]\033[0m No seed provided, using default seed \033[92m{seed}\033[0m. "
            f"To set a new seed, call \033[93mset_seeds(seed=YOUR_SEED)\033[0m"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


seed = set_seeds()
