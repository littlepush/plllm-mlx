# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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