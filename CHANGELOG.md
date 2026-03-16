# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.1] - 2025-03-16

### Fixed
- Increase `load_model` timeout from 5s to 300s for large models

## [1.5.0] - 2025-03-16

### Added
- `plx` command alias for `plllm-mlx` (shorter and easier to type)
- Channel and message token detection for GPT-OSS models

### Changed
- GPT-OSS step processor now uses dynamic `special_tokens.channel_token` and `special_tokens.message_token`
- Chat display format: `[You]:`, `[Assistant]:`, `[Reasoning]:`
- Reasoning content now displays with dim style

### Removed
- Deprecated `openai_step_processor.py` - use `gpt_oss_step_processor.py` instead
- Deprecated `qwen3_thinking_step_processor.py` - use `thinking_step_processor.py` instead

## [1.4.3] - 2025-03-16

### Changed
- Rename `gptoss` to `gpt_oss` step processor for consistency with model_type
- Auto-detect GPT-OSS models and use `gpt_oss` step processor

## [1.4.2] - 2025-03-16

### Changed
- Improve chat command UI: add "Assistant:" label with bold green style
- Remove excessive blank lines in chat output
- Fix timeout error message display

## [1.4.1] - 2025-03-16

### Added
- `thinking` step processor for models with thinking mode (Qwen3, etc.)
- Interactive chat command (`plllm-mlx chat`) for conversations with loaded models

### Changed
- Refactor `base` step processor: filter begin/end tokens, output all as CONTENT
- `thinking` step processor handles begin/end tokens + think tags + tool calls
- Model detector: Qwen3 uses `thinking` processor instead of `qwen3think`
- Fix package build: include `plllm_mlx.commands` module

### Fixed
- Filter out special tokens (`<|im_start|>`, `<|im_end|>`) in output

## [1.4.0] - 2025-03-16

### Added
- Interactive chat command (`plllm-mlx chat`) for conversations with loaded models
- Single prompt mode (`plllm-mlx chat -m model -p "prompt"`)
- System prompt file support (`--system` flag)
- Model selection from loaded models when not specified
- Usage statistics display after each chat round (tokens, first token time, speed)
- Thinking/reasoning content display with dimmed style
- Chat commands: `/quit`, `/help`
- 5-minute timeout for chat requests

### Changed
- Optimize service status check for faster CLI response (ps: 2s -> 0.2s)

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