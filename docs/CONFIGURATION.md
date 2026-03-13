# Configuration Guide

This document provides comprehensive configuration options for plllm-mlx.

## Table of Contents

- [Environment Variables](#environment-variables)
- [Model Configuration](#model-configuration)
- [KV Cache Configuration](#kv-cache-configuration)
- [Process Isolation](#process-isolation)
- [Server Configuration](#server-configuration)
- [Redis Configuration](#redis-configuration)
- [Logging Configuration](#logging-configuration)

---

## Environment Variables

### Core Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `0` | Enable debug mode (`0` or `1`) |
| `HUGGING_FACE_PATH` | `~/.cache/huggingface/hub` | HuggingFace cache directory |

### KV Cache Memory Management

| Variable | Default | Description |
|----------|---------|-------------|
| `PLLLM_MEMORY_THRESHOLD` | `0.9` | Memory usage threshold to trigger eviction (0.0-1.0) |
| `PLLLM_MEMORY_LOWBOUND_THRESHOLD` | `0.7` | Target memory level after eviction (0.0-1.0) |
| `PLLLM_CACHE_MIN_ENTRIES` | `3` | Minimum cache entries to keep |

### Example

```bash
# Set in shell
export DEBUG=1
export HUGGING_FACE_PATH="/data/models/huggingface"
export PLLLM_MEMORY_THRESHOLD=0.8
export PLLLM_MEMORY_LOWBOUND_THRESHOLD=0.6
export PLLLM_CACHE_MIN_ENTRIES=5

# Or in .env file
DEBUG=1
HUGGING_FACE_PATH=/data/models/huggingface
PLLLM_MEMORY_THRESHOLD=0.8
PLLLM_MEMORY_LOWBOUND_THRESHOLD=0.6
PLLLM_CACHE_MIN_ENTRIES=5
```

---

## Model Configuration

Model configurations are stored in Redis and can be updated via API.

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_prompt_tokens` | int | dynamic | Maximum prompt tokens (auto-calculated) |
| `max_output_tokens` | int | 16384 | Maximum output tokens |
| `temperature` | float | 0.8 | Sampling temperature (0.0-2.0) |
| `top_p` | float | 0.0 | Nucleus sampling threshold (0.0-1.0) |
| `top_k` | int | 100 | Top-k sampling |
| `min_p` | float | 0.0 | Minimum token probability |
| `repetition_penalty` | float | 1.1 | Repetition penalty (1.0-2.0) |
| `enable_prefix_cache` | bool | true | Enable prefix KV cache |

### MLX-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefill_step_size` | int | 4096 | Prompt processing chunk size |
| `kv_bits` | int | None | KV quantization bits (None = disabled) |
| `kv_group_size` | int | 32 | Quantization group size |
| `quantized_kv_start` | int | 0 | Token threshold to start KV quantization |
| `max_kv_size` | int | None | Maximum KV cache size |

### Prefix Cache Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `begin_tokens` | list | `['<|start|>', '<|im_start|>']` | Message start tokens |
| `end_tokens` | list | `['<|end|>', '<|im_end|>']` | Message end tokens |
| `vision_begin_tokens` | list | `['<|vision_start|>']` | Vision content start tokens |
| `vision_end_tokens` | list | `['<|vision_end|>']` | Vision content end tokens |

### Setting Configuration via API

```bash
# Update temperature
curl -X POST http://localhost:8080/api/v1/model/update/config \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "key": "temperature",
    "value": 0.7
  }'

# Update max tokens
curl -X POST http://localhost:8080/api/v1/model/update/config \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "key": "max_output_tokens",
    "value": 8192
  }'

# Enable KV quantization
curl -X POST http://localhost:8080/api/v1/model/update/config \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "key": "kv_bits",
    "value": 4
  }'
```

### Configuration Best Practices

#### Temperature Settings

```python
# Creative writing
temperature = 0.9  # Higher = more creative

# Code generation
temperature = 0.3  # Lower = more deterministic

# Factual Q&A
temperature = 0.1  # Very low = focused answers
```

#### Memory-Constrained Systems

```python
# Low memory (8GB RAM)
config = {
    "max_output_tokens": 4096,
    "kv_bits": 4,  # Enable 4-bit quantization
    "max_kv_size": 2048,
}
os.environ["PLLLM_MEMORY_THRESHOLD"] = "0.7"
os.environ["PLLLM_CACHE_MIN_ENTRIES"] = "2"

# High memory (32GB+ RAM)
config = {
    "max_output_tokens": 16384,
    "kv_bits": None,  # No quantization
}
os.environ["PLLLM_MEMORY_THRESHOLD"] = "0.9"
os.environ["PLLLM_CACHE_MIN_ENTRIES"] = "10"
```

---

## KV Cache Configuration

### Memory Thresholds

The KV cache uses a two-threshold eviction policy:

```
┌─────────────────────────────────────────────────────────┐
│ Memory Usage                                            │
│                                                         │
│  100% ──────────────────────────────────────────────    │
│   90% ────────────────── ● THRESHOLD ──────────────    │
│   80% ──────────────────│──────────────────────────    │
│   70% ──────────────────│─ ● LOWBOUND ─────────────    │
│   60% ──────────────────│──────────────────────────    │
│   50% ──────────────────│──────────────────────────    │
│                        │                               │
│                        │  Eviction Zone                │
│                        │  (Remove LRU entries)         │
│                        │                               │
└─────────────────────────────────────────────────────────┘

When memory > THRESHOLD: Start evicting
When memory < LOWBOUND: Stop evicting
Always keep at least MIN_ENTRIES
```

### Configuration by Memory Size

#### 8GB RAM

```bash
export PLLLM_MEMORY_THRESHOLD=0.7
export PLLLM_MEMORY_LOWBOUND_THRESHOLD=0.5
export PLLLM_CACHE_MIN_ENTRIES=2
```

#### 16GB RAM

```bash
export PLLLM_MEMORY_THRESHOLD=0.8
export PLLLM_MEMORY_LOWBOUND_THRESHOLD=0.6
export PLLLM_CACHE_MIN_ENTRIES=3
```

#### 32GB+ RAM

```bash
export PLLLM_MEMORY_THRESHOLD=0.9
export PLLLM_MEMORY_LOWBOUND_THRESHOLD=0.7
export PLLLM_CACHE_MIN_ENTRIES=5
```

### KV Quantization

Enable KV cache quantization to reduce memory usage:

```bash
# 4-bit quantization (75% memory reduction)
curl -X POST http://localhost:8080/api/v1/model/update/config \
  -H "Content-Type: application/json" \
  -d '{"model_name": "model-name", "key": "kv_bits", "value": 4}'

# 8-bit quantization (50% memory reduction)
curl -X POST http://localhost:8080/api/v1/model/update/config \
  -H "Content-Type: application/json" \
  -d '{"model_name": "model-name", "key": "kv_bits", "value": 8}'
```

**Memory Impact:**

| Quantization | Memory per 1000 tokens | Quality Impact |
|--------------|------------------------|----------------|
| None (FP16) | ~640MB | Baseline |
| 8-bit | ~320MB | Minimal |
| 4-bit | ~160MB | Slight degradation |

---

## Process Isolation

Process isolation is **always enabled** in production mode. It cannot be disabled.

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│ Main Process (FastAPI)                                  │
│  - Handles HTTP requests                               │
│  - Routes to subprocesses                              │
│  - Never blocks on inference                           │
└────────────────────┬────────────────────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
    ▼                ▼                ▼
┌────────┐      ┌────────┐      ┌────────┐
│ Subproc│      │ Subproc│      │ Subproc│
│ Model A│      │ Model B│      │ Model C│
└────────┘      └────────┘      └────────┘
```

### Startup Configuration

```bash
# Start with all services (LLM + Embedding + Rerank)
./install.sh

# Start LLM only (faster startup, less memory)
./install.sh --llm-only

# Start without preloading models
./install.sh --no-auto-load
```

---

## Server Configuration

### Port and Host

Default configuration in `main.py`:

```python
# Server binds to all interfaces on port 8080
uvicorn.Config(app, host="0.0.0.0", port=8080, reload=False)
```

To change, modify `main.py` or use a reverse proxy.

### CORS

CORS is configured to allow all origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For production, restrict origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
```

### Concurrency

Chat requests are controlled by a semaphore:

```python
# Default: Large queue with 300s timeout
_chat_semaphore = asyncio.Semaphore(1000)
```

Requests wait up to 300 seconds for a slot before returning 503.

---

## Redis Configuration

Redis is used for persistent storage of:
- Category mappings
- Model configurations
- Model metadata

### Connection String

Default connection in `models/category.py` and `models/local_models.py`:

```python
# Category storage
redis_url = "redis://10.10.64.132:6379/3"

# Model info storage
redis_url = "redis://10.10.64.132:6379/3"
```

### Changing Redis URL

Modify the connection strings in:
- `models/category.py` (line 22)
- `models/local_models.py` (line 37)

```python
# Example: Change to local Redis
self._category_storage = PlObjectStorage(
    f"plllm_categories_{PlHostName().lower()}", 
    "redis://localhost:6379/3",  # Change here
    PlCategoryItem, 
    "name"
)
```

### Redis Data Structure

| Key Pattern | Type | Description |
|-------------|------|-------------|
| `plllm_categories_{host}` | Hash | Category → Model mapping |
| `plllm_{host}_local_models_info` | Hash | Model configurations |

---

## Logging Configuration

### Log Levels

Controlled by `DEBUG` environment variable:

```bash
# Production (INFO level)
export DEBUG=0

# Development (DEBUG level)
export DEBUG=1
```

### Querying Logs

Use the built-in log query tool:

```bash
# Recent logs
uv run python query_logs.py

# Error logs only
uv run python query_logs.py -l ERROR

# Filter by keyword
uv run python query_logs.py -k "PlMessageBasedKVCache"

# Filter by category
uv run python query_logs.py -k "category_name"
```

### Log Format

```
[timestamp] [level] [logger] message
```

Example:
```
[2024-03-14 10:30:45] [INFO] [PlMessageBasedKVCache] Cache HIT: matched 4/6 messages
```

---

## Configuration Files

### Project Structure

```
plllm/
├── main.py              # Server entry point
├── install.sh           # Startup script
├── start.sh             # Alternative startup
├── uninstall.sh         # Stop service
├── test.sh              # Test script
├── models/
│   ├── model_loader.py  # Base loader class
│   ├── mlx_loader.py    # MLX implementation
│   └── kv_cache.py      # KV cache manager
├── routers/
│   ├── chat.py          # Chat endpoints
│   └── ...
└── helpers/
    └── ...
```

### Startup Scripts

#### install.sh

Starts the service in background mode:

```bash
#!/bin/bash
# Options:
#   --llm-only      Only load LLM models (skip embedding/rerank)
#   --no-auto-load  Don't preload models at startup
```

#### uninstall.sh

Stops the background service:

```bash
#!/bin/bash
# Kills the process and cleans up
```

#### test.sh

Tests model responses without starting HTTP server:

```bash
#!/bin/bash
# Options:
#   --dry-run  Test all models and exit
```

---

## Example Configurations

### Development Setup

```bash
# .env
DEBUG=1
HUGGING_FACE_PATH=~/models/huggingface
PLLLM_MEMORY_THRESHOLD=0.9
PLLLM_MEMORY_LOWBOUND_THRESHOLD=0.7
PLLLM_CACHE_MIN_ENTRIES=3
```

### Production Setup (16GB RAM)

```bash
# .env
DEBUG=0
HUGGING_FACE_PATH=/data/models/huggingface
PLLLM_MEMORY_THRESHOLD=0.8
PLLLM_MEMORY_LOWBOUND_THRESHOLD=0.6
PLLLM_CACHE_MIN_ENTRIES=5
```

### Production Setup (8GB RAM)

```bash
# .env
DEBUG=0
HUGGING_FACE_PATH=/data/models/huggingface
PLLLM_MEMORY_THRESHOLD=0.7
PLLLM_MEMORY_LOWBOUND_THRESHOLD=0.5
PLLLM_CACHE_MIN_ENTRIES=2
```

### Model Configuration via API

```bash
# Set up a model for creative writing
curl -X POST http://localhost:8080/api/v1/model/update/config \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen2.5-7B-Instruct", "key": "temperature", "value": 0.9}'

curl -X POST http://localhost:8080/api/v1/model/update/config \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen2.5-7B-Instruct", "key": "max_output_tokens", "value": 4096}'

# Set up a model for code generation
curl -X POST http://localhost:8080/api/v1/model/update/config \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen2.5-Coder-7B", "key": "temperature", "value": 0.3}'
```

---

## Troubleshooting

### Configuration Not Taking Effect

1. **Check if model is reloaded**:
   ```bash
   curl -X POST http://localhost:8080/api/v1/model/reload
   ```

2. **Verify configuration**:
   ```bash
   curl http://localhost:8080/api/v1/model/list
   ```

### Memory Issues

1. **Check memory usage**:
   ```bash
   # Query logs for memory info
   uv run python query_logs.py -k "memory"
   ```

2. **Reduce cache size**:
   ```bash
   export PLLLM_MEMORY_THRESHOLD=0.7
   export PLLLM_CACHE_MIN_ENTRIES=2
   ```

3. **Enable KV quantization**:
   ```bash
   curl -X POST http://localhost:8080/api/v1/model/update/config \
     -H "Content-Type: application/json" \
     -d '{"model_name": "model", "key": "kv_bits", "value": 4}'
   ```

### Model Loading Failures

1. **Check HuggingFace path**:
   ```bash
   ls $HUGGING_FACE_PATH
   ```

2. **Use --no-auto-load**:
   ```bash
   ./install.sh --no-auto-load
   ```

3. **Load model manually**:
   ```bash
   curl -X POST http://localhost:8080/api/v1/loader/load \
     -H "Content-Type: application/json" \
     -d '{"model_name": "Qwen/Qwen2.5-7B-Instruct"}'
   ```

---

*Last updated: 2024*