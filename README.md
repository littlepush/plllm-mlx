# plllm-mlx

A standalone MLX-based LLM inference service with OpenAI compatible API, designed specifically for Apple Silicon.

**[Documentation](docs/README.md)** | **[中文文档](docs/cn/README.md)**

## Overview

**plllm-mlx** is a production-ready inference service that provides:
- Native Apple Silicon optimization via MLX framework
- OpenAI-compatible API for seamless integration
- Zero external dependencies (no database, Redis, or external services)
- Process isolation for stable multi-model serving
- Efficient KV cache with prefix caching
- Real-time streaming support

## Key Features

### 🍎 Apple Silicon Native
- Optimized for M-series chips (M1/M2/M3/M4)
- Leverages MLX framework for efficient inference
- Hardware-accelerated operations

### 🔌 OpenAI Compatible API
- Drop-in replacement for OpenAI API
- Chat completions with streaming support
- Model listing and health check endpoints

### 📦 Zero External Dependencies
- No database required
- No Redis or message queues
- Standalone operation
- Simple deployment

### 🔄 Process Isolation
- Each model runs in separate subprocess
- Isolated memory management
- Fault tolerance and stability
- Clean shutdown handling

### 💾 Intelligent KV Cache
- Prefix-based caching for efficiency
- Message-level cache matching
- Incremental prefill optimization
- Memory-aware eviction

### 🎯 Extensible Architecture
- Pluggable model loaders (MLX-LM, MLX-VLM)
- Customizable step processors
- Easy to extend for new models

## Installation

### Using uv tool (Recommended)

```bash
# Install as a standalone tool
uv tool install plllm-mlx

# Start the service
plllm-mlx serve

# Or with options
plllm-mlx serve --port 8000 --config ~/.plllm-mlx/config.yaml
```

### From Source

```bash
# Clone the repository
git clone https://github.com/littlepush/plllm-mlx.git
cd plllm-mlx

# Install dependencies
uv sync

# Run the service
uv run plllm-mlx serve
```

### VLM Support (Vision Language Models)

For VLM models (e.g., Qwen2.5-VL, Qwen3.5-VL), install with VLM dependencies:

```bash
# Using uv tool
uv tool install 'plllm-mlx[vlm]'

# Using pip
pip install 'plllm-mlx[vlm]'

# If already installed, add VLM support
uv tool install 'plllm-mlx[vlm]' --force
# or
pip install torch torchvision
```

## Quick Start

### 1. Start Service

```bash
# Start service (registers as LaunchAgent)
plllm-mlx serve

# Check status
plllm-mlx status

# Stop service
plllm-mlx stop
```

### 2. Manage Models

```bash
# List loaded models
plllm-mlx ps

# List all local models
plllm-mlx ls

# Search HuggingFace
plllm-mlx search qwen

# Download a model
plllm-mlx download mlx-community/Qwen2.5-7B-8bit

# Load/unload models
plllm-mlx load Qwen2.5-7B-8bit
plllm-mlx unload Qwen2.5-7B-8bit

# Configure model
plllm-mlx config Qwen2.5-7B-8bit temperature=0.8 max_tokens=2048
```

### 3. Use API

```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-8bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### 2. Start Service

```bash
plllm-mlx --config config.yaml
```

### 3. Make API Requests

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'
```

## Architecture

```
plllm_mlx/
├── cli.py                    # Command-line interface
├── config.py                 # Configuration management
├── logging_config.py         # Logging setup
├── exceptions.py             # Custom exceptions
│
├── models/                   # Model loading and inference
│   ├── model_loader.py       # Base loader class
│   ├── mlx_loader.py         # MLX-LM implementation
│   ├── mlxvlm_loader.py      # MLX-VLM implementation
│   ├── model_subprocess.py   # Subprocess execution
│   ├── process_manager.py    # Process lifecycle management
│   ├── kv_cache.py           # KV cache implementation
│   ├── local_models.py       # Model manager
│   ├── model_detector.py     # Model type detection
│   │
│   ├── step_processor.py     # Base processor
│   ├── base_step_processor.py
│   ├── default_step_processor.py
│   ├── openai_step_processor.py
│   └── qwen3_thinking_step_processor.py
│
├── helpers/                  # Utility modules
│   ├── chain_cache.py        # Chain-based cache
│   ├── chat_helper.py        # Chat completion builder
│   ├── chunk_helper.py       # Chunk data structures
│   ├── step_info.py          # Step metadata
│   ├── toolcall_helper.py    # Tool call parsing
│   └── clz_helper.py         # Class registry
│
└── routers/                  # FastAPI endpoints
    ├── chat.py               # Chat completions
    ├── models.py             # Model listing
    ├── loader.py             # Loader management
    ├── stepprocessor.py      # Processor management
    └── model_manager.py      # Model operations
```

## Core Concepts

### Model Loaders

Model loaders handle model loading, inference, and streaming:

- **MLX-LM Loader**: For standard language models
- **MLX-VLM Loader**: For vision-language models

Each loader implements:
```python
class PlModelLoader:
    async def ensure_model_loaded()
    async def stream_generate(session_object)
    async def prepare_prompt(body)
```

### Step Processors

Step processors transform raw generation results:

- **Base**: Basic text generation
- **Default**: Standard processing
- **OpenAI**: OpenAI-compatible formatting
- **Qwen3Thinking**: Qwen3 thinking mode support

Processors handle:
- Token accumulation
- Tool call parsing
- Thinking/reasoning content
- Finish reason detection

### KV Cache

Prefix-based KV cache for efficient inference:

**How it works:**
1. Split prompt into message segments
2. Calculate MD5 hash for each message
3. Match against cached message chains
4. Skip prefill for matched prefix
5. Only process incremental messages

**Benefits:**
- Faster multi-turn conversations
- Reduced memory usage
- Lower latency for repeated prompts

### Process Isolation

Each model runs in a separate subprocess:

```
Main Process
├── API Server
├── Process Manager
│   ├── Model A Subprocess
│   │   ├── Model Loader
│   │   └── KV Cache
│   └── Model B Subprocess
│       ├── Model Loader
│       └── KV Cache
```

**Advantages:**
- Memory isolation
- Fault tolerance
- Clean resource cleanup
- Parallel model serving

## API Reference

### Chat Completions

**POST** `/v1/chat/completions`

Request:
```json
{
  "model": "model-name",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 4096,
  "stream": true
}
```

Streaming response:
```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Hi"}}]}

data: [DONE]
```

### Models

**GET** `/v1/models`

Returns available models.

### Health Check

**GET** `/health`

Returns service status.

### Model Management

**GET** `/api/v1/model/list`
**POST** `/api/v1/model/load`
**POST** `/api/v1/model/unload`

## Configuration

### Server Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `server.host` | string | "0.0.0.0" | Bind address |
| `server.port` | int | 8000 | Server port |
| `server.log_level` | string | "info" | Log level |

### Model Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | string | required | Model identifier |
| `model_id` | string | required | HuggingFace model ID |
| `loader` | string | "mlx" | Loader type (mlx/mlxvlm) |
| `max_tokens` | int | 4096 | Maximum output tokens |
| `temperature` | float | 0.7 | Sampling temperature |
| `enable_prefix_cache` | bool | true | Enable KV cache |

### Cache Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_prefix_cache` | bool | true | Enable prefix cache |
| `max_memory_ratio` | float | 0.9 | Memory threshold |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PLLLM_MLX_CONFIG` | config.yaml | Config file path |
| `PLLLM_MLX_HOST` | 0.0.0.0 | Server host |
| `PLLLM_MLX_PORT` | 8000 | Server port |
| `PLLLM_MLX_LOG_LEVEL` | info | Log level |

## Development

### Setup

```bash
git clone https://github.com/littlepush/plllm-mlx.git
cd plllm-mlx
uv sync --extra dev
```

### Run Tests

```bash
uv run pytest
```

### Code Quality

```bash
uv run ruff format .
uv run ruff check .
```

## Extending plllm-mlx

### Add a New Model Loader

1. Create `models/my_loader.py`
2. Inherit from `PlModelLoader`
3. Implement required methods:
   ```python
   class MyLoader(PlModelLoader):
       @staticmethod
       def model_loader_name() -> str:
           return "my_loader"
       
       async def ensure_model_loaded(self):
           # Load model
       
       async def stream_generate(self, session_object):
           # Generate tokens
   ```

### Add a New Step Processor

1. Create `models/my_step_processor.py`
2. Inherit from `PlStepProcessor`
3. Implement processing logic:
   ```python
   class MyProcessor(PlStepProcessor):
       @staticmethod
       def step_clz_name() -> str:
           return "my_processor"
       
       def step(self, generate_response) -> Optional[PlChunk]:
           # Process token
   ```

## Performance Tips

### Memory Management

- Use process isolation for multiple models
- Monitor memory with `PLLLM_MEMORY_THRESHOLD`
- Adjust `prefill_step_size` for large prompts

### KV Cache Optimization

- Enable prefix cache for multi-turn conversations
- Monitor cache hit rates in logs
- Adjust `kv_bits` for memory/speed tradeoff

### Streaming Performance

- Use streaming (`"stream": true`) for better UX
- Monitor first token latency
- Check KV cache effectiveness

## Troubleshooting

### Model Loading Issues

```bash
# Check model availability
ls ~/.cache/huggingface/hub/

# Verify model format
python -c "from transformers import AutoModel; AutoModel.from_pretrained('model-id')"
```

### Memory Issues

```bash
# Monitor memory
top -l 1 | grep PhysMem

# Reduce memory usage
# - Use quantized models (4bit/8bit)
# - Reduce max_tokens
# - Disable unused models
```

### Streaming Issues

- Check if streaming is enabled in request
- Verify SSE support in client
- Check logs for generation errors

## Requirements

- Python 3.12+
- macOS with Apple Silicon (M1/M2/M3/M4)
- MLX framework (`pip install mlx mlx-lm`)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - MLX language models
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - MLX vision-language models
- [FastAPI](https://fastapi.tiangolo.com/) - Modern async web framework

## Support

- **Issues**: [GitHub Issues](https://github.com/littlepush/plllm-mlx/issues)
- **Discussions**: [GitHub Discussions](https://github.com/littlepush/plllm-mlx/discussions)

---

Made with ❤️ for Apple Silicon