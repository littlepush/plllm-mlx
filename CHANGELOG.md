# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.1] - 2025-03-16

### Added
- Clear error message when VLM model detected but dependencies not installed
- Installation hint: `pip install 'plllm-mlx[vlm]'`
- VLM support section in README with installation instructions

### Fixed
- Check loader availability before using detected loader
- Add `is_loaded` field to model list response
- Skip re-loading if model already loaded

## [1.3.0] - 2025-03-16

### Added
- Auto-detection of model loader (mlx/mlxvlm) and step processor from local config.json
- CLI `load` command supports `--loader` and `--stpp` options for manual override
- CLI `download` command supports `--loader` and `--stpp` options
- `mlx-vlm` as required dependency for VLM model support
- `path_helper.py` for unified HuggingFace cache path resolution (HF_HUB_CACHE, HF_HOME, HUGGING_FACE_PATH)
- Optional `[vlm]` extra for torch/torchvision dependencies

### Changed
- Unified API endpoints: removed duplicate `/v1/loader/load` and `/v1/loader/unload`
- All model load/unload operations now use `/v1/model/load` and `/v1/model/unload`
- Fixed `qwen3_thinking_step_processor.py` to use standard logging instead of plpybase
- Fixed daemon.py to use local .venv when available instead of uv tool
- Added `is_loaded` field to model list response
- Skip re-loading if model already loaded

## [1.2.0] - 2025-03-16

### Added
- Platform check: now shows clear error message when running on non-macOS systems
- Added macOS-only classifier in package metadata

### Changed
- Removed Huawei Cloud PyPI mirror configuration for open source release

## [1.0.8] - 2025-01-15

### Fixed
- Fixed async generator error: "'async for' requires an object with __aiter__ method, got coroutine"
- Changed `chat_completions_stream_with_isolation` decorator from `@async_ticker` to `@yield_ticker`
- Removed incorrect `yield` statements from abstract method definitions

## [1.0.0] - 2025-01-14

### Added
- Initial release of plllm-mlx
- OpenAI compatible API endpoints
  - `POST /v1/chat/completions` - Chat completions with streaming support
  - `GET /v1/models` - List available models
  - `GET /health` - Health check endpoint
- MLX-based model inference optimized for Apple Silicon
- Prefix KV cache for efficient multi-turn conversations
- Command line interface with configuration support
- Configuration via YAML file and command line arguments
- Streaming response support via Server-Sent Events (SSE)
- Support for popular LLM models (Qwen, Llama, Mistral, etc.)

### Features
- Zero external dependencies (no database, no Redis)
- Standalone operation via `uv tool install`
- Efficient memory management
- Real-time token streaming
- OpenAI API compatibility

### Documentation
- Comprehensive README with installation and usage guide
- API reference documentation
- Configuration guide
- Development setup instructions

[1.0.0]: https://github.com/littlepush/plllm-mlx/releases/tag/v1.0.0