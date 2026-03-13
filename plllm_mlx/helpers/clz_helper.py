"""
Class discovery and path utilities.

This module provides utilities for dynamically discovering and
loading classes from Python files, useful for plugin systems
and extensible architectures.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path
from typing import List, Type, TypeVar

T = TypeVar("T")


def PlRootPath() -> str:
    """
    Get the root path of the project.

    Returns:
        The absolute path to the project root (parent of plllm_mlx directory).
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def PlFindSpecifialSubclass(file_path: str, base_clz: Type[T]) -> List[Type[T]]:
    """
    Find all classes in a file that are subclasses of a base class.

    This function dynamically loads a Python file and inspects it for
    classes that inherit from the specified base class.

    Args:
        file_path: Path to the Python file to inspect.
        base_clz: The base class to search for subclasses of.

    Returns:
        List of class objects that are subclasses of base_clz.
        Returns empty list if the file cannot be loaded.

    Example:
        >>> # Find all loaders in a file
        >>> loaders = PlFindSpecifialSubclass(
        ...     "/path/to/my_loader.py",
        ...     PlModelLoader
        ... )
    """
    module_name = Path(file_path).stem
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return []
        user_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_module)
        classes = [
            cls_obj
            for _, cls_obj in inspect.getmembers(user_module)
            if (
                inspect.isclass(cls_obj)
                and cls_obj.__module__ == module_name
                and issubclass(cls_obj, base_clz)
            )
        ]
        return classes
    except Exception as e:
        from plllm_mlx.logging_config import get_logger

        logger = get_logger(__name__)
        logger.error(f"Failed to load module from {file_path}: {e}")
    return []


def PlUnpackPath(
    path: str, recursive: bool = True, list_dir: bool = False
) -> List[str]:
    """
    Unpack a path into a list of files or directories.

    If the path is a directory, recursively or non-recursively list
    its contents based on the parameters.

    Args:
        path: The path to unpack (file or directory).
        recursive: If True, recursively list subdirectories.
        list_dir: If True, list directories; if False, list files.

    Returns:
        List of absolute paths. Returns empty list if path doesn't exist.

    Example:
        >>> # Get all Python files recursively
        >>> files = PlUnpackPath("/path/to/project", recursive=True, list_dir=False)
        >>> # Get all subdirectories non-recursively
        >>> dirs = PlUnpackPath("/path/to/project", recursive=False, list_dir=True)
    """
    if not os.path.exists(path):
        return []

    if list_dir:
        if os.path.isdir(path):
            return [path]
    else:
        if os.path.isfile(path):
            return [path]

    result = []
    for f in os.listdir(path):
        sub_path = os.path.join(path, f)
        if list_dir:
            if os.path.isdir(sub_path):
                result.append(sub_path)
                if recursive:
                    result.extend(PlUnpackPath(sub_path, recursive, list_dir))
        else:
            if os.path.isfile(sub_path):
                result.append(sub_path)
            else:
                if recursive:
                    result.extend(PlUnpackPath(sub_path, recursive, list_dir))
    return result
