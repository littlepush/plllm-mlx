# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.6] - 2025-03-16

### Changed
- Optimize service status check for faster CLI response (ps: 2s -> 0.2s)
- Add STATUS_FILE for fast port/pid lookup without parsing YAML config
- Use HTTP request directly instead of socket check in client

## [1.3.5] - 2025-03-16

### Fixed
- Increase service start timeout from 30s to 60s for first-time startup on new machines
- Add `PLLLM_START_TIMEOUT` environment variable for custom timeout

## [1.3.4] - 2025-03-16

### Added
- Auto-detect special tokens from tokenizer's `added_tokens_decoder`
- `SpecialTokens` dataclass for storing detected tokens
- `detect_special_tokens()` function for automatic token detection
- `PlGptOssStepProcessor` for GPT-OSS models with channel-based mechanism

### Changed
- `PlStepProcessor` base class now accepts `special_tokens` parameter
- `PlBaseStepProcessor` refactored to support dynamic think/tool_call tokens
- `PlMlxModel` and `PlMlxVlmModel` now auto-detect and pass special tokens to StepProcessor

### Fixed
- Remove hardcoded special tokens in model loaders
- Support for Qwen3 thinking mode with auto-detected tokens

## [1.3.3] - 2025-03-16

### Changed
- `GET /v1/models` now only returns loaded models (OpenAI compatible behavior)
- Chat endpoint auto-loads model with auto-detection when model not loaded
- `GET /v1/model/list` still returns all available models with extended info

## [1.3.2] - 2025-03-16

### Added
- OpenAI compatible endpoint `GET /v1/models` for listing available models

### Fixed
- Remove circular import in models/__init__.py

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