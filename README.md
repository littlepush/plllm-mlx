# plllm-mlx

A standalone MLX-based LLM inference service with OpenAI compatible API for Apple Silicon.

## Features

- 🚀 **Native Apple Silicon Support** - Optimized for M-series chips using MLX framework
- 🔌 **OpenAI Compatible API** - Drop-in replacement for OpenAI API
- 📦 **Zero External Dependencies** - No database, no Redis, no external services required
- 🎯 **Standalone Operation** - Single binary installation via `uv tool`
- 💾 **Efficient Memory Management** - Smart KV cache with prefix caching
- 🔄 **Streaming Support** - Real-time token streaming via SSE

## Installation

### Using uv tool (Recommended)

```bash
# Install as a standalone tool
uv tool install plllm-mlx

# Run the service
plllm-mlx --config config.yaml
```

### From Source

```bash
# Clone the repository
git clone https://github.com/littlepush/plllm-mlx.git
cd plllm-mlx

# Install with uv
uv sync

# Run the service
uv run plllm-mlx --config config.yaml
```

## Quick Start

1. **Create a configuration file** (`config.yaml`):

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"

model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  max_tokens: 4096
  temperature: 0.7

cache:
  enable_prefix_cache: true
  max_memory_ratio: 0.9
```

2. **Start the service**:

```bash
plllm-mlx --config config.yaml
```

3. **Make API requests**:

```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "stream": true
  }'
```

## Command Line Options

```
usage: plllm-mlx [-h] [--config CONFIG] [--host HOST] [--port PORT] [--log-level LEVEL]

options:
  -h, --help         Show help message and exit
  --config CONFIG    Path to configuration file (default: config.yaml)
  --host HOST        Server host (default: 0.0.0.0)
  --port PORT        Server port (default: 8000)
  --log-level LEVEL  Log level: debug, info, warning, error (default: info)
```

## API Reference

### Chat Completions

**POST** `/v1/chat/completions`

Request body:
```json
{
  "model": "model-name",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 4096,
  "stream": true
}
```

Response (streaming):
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: [DONE]
```

### Models

**GET** `/v1/models`

Returns list of available models.

### Health Check

**GET** `/health`

Returns service health status.

## Configuration

### Server Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `server.host` | string | "0.0.0.0" | Server bind address |
| `server.port` | int | 8000 | Server port |
| `server.log_level` | string | "info" | Log level (debug, info, warning, error) |

### Model Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model.name` | string | required | HuggingFace model name or local path |
| `model.max_tokens` | int | 4096 | Maximum tokens to generate |
| `model.temperature` | float | 0.7 | Sampling temperature |
| `model.top_p` | float | 1.0 | Top-p sampling |
| `model.repetition_penalty` | float | 1.0 | Repetition penalty |

### Cache Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `cache.enable_prefix_cache` | bool | true | Enable prefix KV cache |
| `cache.max_memory_ratio` | float | 0.9 | Maximum memory usage ratio |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PLLLM_MLX_CONFIG` | config.yaml | Default config file path |
| `PLLLM_MLX_HOST` | 0.0.0.0 | Default server host |
| `PLLLM_MLX_PORT` | 8000 | Default server port |
| `PLLLM_MLX_LOG_LEVEL` | info | Default log level |

## Development

### Setup Development Environment

```bash
# Clone and install dev dependencies
git clone https://github.com/littlepush/plllm-mlx.git
cd plllm-mlx
uv sync --extra dev
```

### Run Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run ruff format .
uv run ruff check .
```

## Architecture

```
plllm_mlx/
├── __init__.py      # Package initialization
├── cli.py           # Command line interface
├── server.py        # FastAPI server
├── model.py         # Model loader and inference
├── cache.py         # KV cache management
├── config.py        # Configuration handling
└── utils.py         # Utility functions
```

## Requirements

- Python 3.12+
- macOS with Apple Silicon (M1/M2/M3/M4)
- MLX framework

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - MLX LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework