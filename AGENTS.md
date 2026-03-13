# AI Agent Development Guide

> **Important**: This is an open-source project with zero external dependencies. No database, no Redis, no external services required.

## Project Overview

**plllm-mlx** is a standalone MLX-based LLM inference service with OpenAI compatible API, designed specifically for Apple Silicon.

### Key Characteristics

1. **Zero External Dependencies** - No database, no Redis, no message queue
2. **Standalone Operation** - Single binary via `uv tool install`
3. **Open Source** - MIT License, fully transparent
4. **Apple Silicon Native** - Optimized for M-series chips

## Development Guidelines

### Code Style

- Use Python 3.12+ features
- Follow PEP 8 style guide
- Use type hints for all public functions
- Keep functions small and focused

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=plllm_mlx
```

### Commit Guidelines

- Write clear, descriptive commit messages
- Reference issues when applicable
- Keep commits atomic and focused

### Before Submitting

1. Run tests: `uv run pytest`
2. Format code: `uv run ruff format .`
3. Check linting: `uv run ruff check .`
4. Update documentation if needed

## Architecture

```
plllm_mlx/
├── cli.py           # Entry point, argument parsing
├── server.py        # FastAPI application, routes
├── model.py         # Model loading and inference
├── cache.py         # KV cache management
├── config.py        # Configuration handling
└── utils.py         # Helper functions
```

## Important Notes

1. **No External Services** - This project must remain standalone
2. **Memory Efficiency** - Be mindful of memory usage on Apple Silicon
3. **API Compatibility** - Maintain OpenAI API compatibility
4. **Streaming First** - Prefer streaming responses for better UX

## License

MIT License - See LICENSE file for details.