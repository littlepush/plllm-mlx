"""
Python subprocess implementation.

Directory structure:
├── __init__.py           # This file - exports PlModelLoader, PlStepProcessor
├── loader.py             # PlModelLoader base class
├── step_processor.py     # PlStepProcessor base class
├── kv_cache.py           # KV cache implementation
├── special_tokens.py     # Special tokens handling
├── loaders/              # Model loader implementations
│   ├── __init__.py
│   ├── mlx_loader.py     # MLX text model loader
│   └── mlxvlm_loader.py  # MLX vision-language model loader
└── stepps/               # Step processor implementations
    ├── __init__.py
    ├── base_step_processor.py      # Default processor
    ├── thinking_step_processor.py  # Thinking/reasoning processor
    └── gpt_oss_step_processor.py   # GPT-OSS channel processor

Usage:
    from plllm_mlx.subprocess.python import PlModelLoader, PlStepProcessor

    # List available loaders and processors
    PlModelLoader.listModelLoaders()      # ['mlx', 'mlxvlm']
    PlStepProcessor.listStepProcessors()  # ['base', 'thinking', 'gpt_oss']

Adding new loaders:
    1. Create xxx_loader.py in loaders/
    2. Inherit from PlModelLoader (from ..loader import PlModelLoader)
    3. Implement all abstract methods
    4. Call PlModelLoader.registerModelLoader(name, cls) at module level

Adding new step processors:
    1. Create xxx_step_processor.py in stepps/
    2. Inherit from PlStepProcessor (from ..step_processor import PlStepProcessor)
    3. Implement all abstract methods
    4. Call PlStepProcessor.registerStepProcessor(name, cls) at module level
"""

from .loader import PlModelLoader, async_ticker, yield_ticker
from .step_processor import PlStepProcessor

__all__ = [
    "PlModelLoader",
    "PlStepProcessor",
    "async_ticker",
    "yield_ticker",
]
