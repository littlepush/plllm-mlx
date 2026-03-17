"""
Model loader implementations.

This directory contains model loader implementations.
Each xxx_loader.py file should define a class that:
1. Inherits from PlModelLoader (from ..loader)
2. Implements all abstract methods
3. Calls PlModelLoader.registerModelLoader() at module level

Base class: plllm_mlx.subprocess.python.loader.PlModelLoader

Available loaders:
- mlx_loader: MLX model loader for text-only models
- mlxvlm_loader: MLX-VLM loader for vision-language models
"""

__all__ = []
