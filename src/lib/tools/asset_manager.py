"""
Asset Manager module for AI project.
Provides utilities to access files and resources from anywhere in the project structure.
"""

from pathlib import Path
from enum import Enum
from typing import Union, Optional, Callable
import sys
import importlib
import inspect


class AssetType(Enum):
    """Enum for different types of assets in the project."""

    # Data types
    TEMP = "temp"
    IMG = "img"
    IMG_RAW = "img/raw"
    IMG_PROC = "img/proc"

    # Code modules
    LIB = "lib"
    UTILS = "lib/utils"
    EXAMPLES = "lib/examples"

    # Documentation
    GA = "ga"
    NEURAL_NETWORKS = "neural_networks"
    EVOLUTIVE_ALGORITHMS = "evolutive_algorithms"
    CV = "cv"
    PATTERN_RECOGNITION = "pattern-recognition"
    REPORTS = "reports"


class ProjectPaths:
    """Resolve paths relative to the project root."""

    @staticmethod
    def find_project_root() -> Path:
        """Find the project root by looking for pyproject.toml."""
        # Start with the current working directory
        paths_to_try = [Path.cwd().absolute()]

        # Add the directory of the calling file
        frame = inspect.stack()[1]
        module_path = Path(frame.filename).absolute()
        paths_to_try.append(module_path)

        # Try each path
        for start_path in paths_to_try:
            path = start_path
            while path != path.parent:
                if (path / "pyproject.toml").exists():
                    return path
                path = path.parent

        # If still not found, return current directory with a warning
        print("Warning: Project root not found, using current directory")
        return Path.cwd().absolute()

    # Project root path - call as a function to ensure it's evaluated at runtime
    @classmethod
    def ROOT(cls) -> Path:
        return cls.find_project_root()

    # Main directories
    @classmethod
    def SRC(cls) -> Path:
        return cls.ROOT() / "src"

    @classmethod
    def DOC(cls) -> Path:
        return cls.ROOT() / "doc"

    @classmethod
    def DT(cls) -> Path:
        return cls.ROOT() / "data"

    @classmethod
    def resolve_path(
        cls,
        base_dir_func: Callable[[], Path],
        asset_type: AssetType,
        filename: Optional[str] = None,
    ) -> Path:
        """Resolve a path relative to a base directory."""
        base_dir = base_dir_func()
        path = base_dir / asset_type.value
        if filename:
            path = path / filename
        return path


class AssetManager:
    """Utility for accessing project assets and code from anywhere in the project."""

    @staticmethod
    def get_project_root() -> Path:
        """Get the project root directory."""
        return ProjectPaths.ROOT()

    @staticmethod
    def get_asset_path(
        asset_type: AssetType, filename: Optional[str] = None, absolute: bool = False
    ) -> Union[Path, str]:
        """
        Get the path to an asset.

        Args:
            asset_type: Type of the asset (from AssetType enum)
            filename: Optional filename to append to the path
            absolute: Whether to return absolute path as string

        Returns:
            Path object or string path to the asset
        """
        # Determine which base directory to use based on asset type
        if asset_type.name.startswith(("TEMP", "IMG")):
            base_dir_func = ProjectPaths.DT
        elif asset_type.name.startswith(("LIB", "UTILS", "EXAMPLES")):
            base_dir_func = ProjectPaths.SRC
        else:
            base_dir_func = ProjectPaths.DOC

        path = ProjectPaths.resolve_path(base_dir_func, asset_type, filename)
        return str(path.absolute()) if absolute else path

    @staticmethod
    def get_data(
        asset_type: AssetType, filename: str, absolute: bool = False
    ) -> Union[Path, str]:
        """Get the path to a data asset."""
        return AssetManager.get_asset_path(asset_type, filename, absolute)

    @staticmethod
    def get_img(
        filename: str, subfolder: Optional[str] = None, absolute: bool = False
    ) -> Union[Path, str]:
        """Get the path to an image asset."""
        asset_type = AssetType.IMG
        if subfolder:
            filename = f"{subfolder}/{filename}"
        return AssetManager.get_asset_path(asset_type, filename, absolute)

    @staticmethod
    def get_raw_img(filename: str, absolute: bool = False) -> Union[Path, str]:
        """Get the path to a raw image asset."""
        return AssetManager.get_asset_path(AssetType.IMG_RAW, filename, absolute)

    @staticmethod
    def get_proc_img(filename: str, absolute: bool = False) -> Union[Path, str]:
        """Get the path to a processed image asset."""
        return AssetManager.get_asset_path(AssetType.IMG_PROC, filename, absolute)

    @staticmethod
    def get_temp(filename: str, absolute: bool = False) -> Union[Path, str]:
        """Get the path to a temporary file."""
        return AssetManager.get_asset_path(AssetType.TEMP, filename, absolute)

    @staticmethod
    def ensure_path_exists(path: Path) -> Path:
        """Ensure a directory path exists, creating it if necessary."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def add_project_to_path():
        """Add the project root to sys.path to enable imports."""
        project_root = str(ProjectPaths.ROOT())
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            return True
        return False

    @classmethod
    def setup_notebook_env(cls):
        """
        Set up the environment for Jupyter notebook.
        Adds the project root to path and returns True if successful.
        """
        was_added = cls.add_project_to_path()
        if was_added:
            print(f"Added project root '{ProjectPaths.ROOT()}' to Python path")
        else:
            print(f"Project root '{ProjectPaths.ROOT()}' was already in Python path")
        return was_added

    @staticmethod
    def import_module(module_path: str):
        """
        Import a module from the project using its relative path.

        Args:
            module_path: Relative path to module from project root, e.g., 'src.lib.ga'

        Returns:
            Imported module
        """
        AssetManager.add_project_to_path()
        try:
            return importlib.import_module(module_path)
        except ImportError as e:
            # Try some alternative paths
            alternatives = []
            if module_path.startswith("src."):
                # Try without 'src.' prefix
                alternatives.append(module_path[4:])
            else:
                # Try with 'src.' prefix
                alternatives.append("src." + module_path)

            # Try alternatives
            for alt_path in alternatives:
                try:
                    return importlib.import_module(alt_path)
                except ImportError:
                    continue

            # If we got here, all alternatives failed
            print(f"Error importing {module_path}. Alternatives tried: {alternatives}")
            raise e

    @staticmethod
    def import_from_lib(module_name: str):
        """
        Import a module from src/lib.

        Args:
            module_name: Name of the module without the 'src.lib.' prefix

        Returns:
            Imported module
        """
        # Try different import paths
        for prefix in ["src.lib.", "lib.", ""]:
            try:
                return AssetManager.import_module(f"{prefix}{module_name}")
            except ImportError:
                continue

        # If all attempts failed, try one last time with the original path
        # and let the exception propagate
        return AssetManager.import_module(f"src.lib.{module_name}")


# Create a convenience instance
am = AssetManager()

# Automatically add project root to path when this module is imported
am.add_project_to_path()
