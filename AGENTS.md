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

#### Unit Tests

```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Run with coverage
uv run pytest tests/unit/ -v --cov=plllm_mlx --cov-report=term-missing
```

#### Integration Tests

Integration tests require a real model and will start/stop the server automatically.

```bash
# Run all integration tests
./tests/run_tests.sh

# Run with quiet mode (only final summary)
./tests/run_tests.sh --quiet

# Redirect output to file
./tests/run_tests.sh --redirect test_output.log

# Run specific integration test
bash tests/integration/test_01_health.sh
```

#### Complete Test Suite

```bash
# Run all tests (unit + integration)
./tests/run_tests.sh
```

#### Manual Testing Commands

```bash
# Start the server in background
uv run plx serve

# Stop the server
uv run plx stop

# Check server status
uv run plx status

# List Local Model
uv run plx ls

# Load Model
uv run plx load <model name>

# Test Chat
uv run plx chat -m <model_name> -p "prompt"
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