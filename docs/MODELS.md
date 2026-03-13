# Supported Models

This document lists the models supported by plllm-mlx and their configurations.

## Table of Contents

- [Overview](#overview)
- [Text Models](#text-models)
- [Vision Language Models](#vision-language-models)
- [Model Formats](#model-formats)
- [Adding New Models](#adding-new-models)

---

## Overview

plllm-mlx supports models in MLX format, which is optimized for Apple Silicon. Models are loaded from the HuggingFace cache directory.

### Model Loaders

| Loader | Description | File |
|--------|-------------|------|
| `mlx` | Standard text models | `mlx_loader.py` |
| `mlxvlm` | Vision Language Models | `mlxvlm_loader.py` |

### Step Processors

| Processor | Description | Use Case |
|-----------|-------------|----------|
| `base` | Basic text output | General text generation |
| `openai` | OpenAI-style channel format | Function calling |
| `qwen3_thinking` | Qwen3 with thinking | Qwen3 models with reasoning |

---

## Text Models

### Qwen2.5 Series

| Model | Parameters | Context | Recommended Use |
|-------|------------|---------|-----------------|
| Qwen/Qwen2.5-0.5B-Instruct | 0.5B | 32K | Lightweight tasks |
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | 32K | Fast responses |
| Qwen/Qwen2.5-3B-Instruct | 3B | 32K | Balanced performance |
| Qwen/Qwen2.5-7B-Instruct | 7B | 128K | General purpose |
| Qwen/Qwen2.5-14B-Instruct | 14B | 128K | Complex tasks |
| Qwen/Qwen2.5-32B-Instruct | 32B | 128K | High-quality output |
| Qwen/Qwen2.5-72B-Instruct | 72B | 128K | Best quality |

**Recommended Step Processor**: `qwen3_thinking` or `base`

**Example Configuration**:

```bash
# Add category
curl -X POST http://localhost:8080/api/v1/category/add \
  -H "Content-Type: application/json" \
  -d '{
    "name": "qwen2.5-7b",
    "type": "chat",
    "model": "Qwen/Qwen2.5-7B-Instruct"
  }'
```

### Qwen2.5-Coder Series

| Model | Parameters | Context | Recommended Use |
|-------|------------|---------|-----------------|
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 1.5B | 32K | Code completion |
| Qwen/Qwen2.5-Coder-7B-Instruct | 7B | 128K | Code generation |
| Qwen/Qwen2.5-Coder-32B-Instruct | 32B | 128K | Complex code tasks |

**Recommended Step Processor**: `qwen3_thinking` or `base`

### Llama Series

| Model | Parameters | Context | Recommended Use |
|-------|------------|---------|-----------------|
| mlx-community/Llama-3.2-1B-Instruct | 1B | 128K | Lightweight |
| mlx-community/Llama-3.2-3B-Instruct | 3B | 128K | Balanced |
| mlx-community/Meta-Llama-3.1-8B-Instruct | 8B | 128K | General purpose |
| mlx-community/Meta-Llama-3.1-70B-Instruct | 70B | 128K | High quality |

**Recommended Step Processor**: `base`

### Mistral Series

| Model | Parameters | Context | Recommended Use |
|-------|------------|---------|-----------------|
| mlx-community/Mistral-7B-Instruct-v0.3 | 7B | 32K | General purpose |
| mlx-community/Mixtral-8x7B-Instruct-v0.1 | 47B | 32K | Complex tasks |

**Recommended Step Processor**: `base`

### Phi Series

| Model | Parameters | Context | Recommended Use |
|-------|------------|---------|-----------------|
| mlx-community/Phi-3.5-mini-instruct | 3.8B | 128K | Efficient reasoning |

**Recommended Step Processor**: `base`

---

## Vision Language Models

### Qwen2.5-VL Series

| Model | Parameters | Context | Features |
|-------|------------|---------|----------|
| Qwen/Qwen2.5-VL-3B-Instruct | 3B | 32K | Image understanding |
| Qwen/Qwen2.5-VL-7B-Instruct | 7B | 128K | Image + text |

**Recommended Step Processor**: `qwen3_thinking`

**Loader**: `mlxvlm`

**Example Configuration**:

```bash
# Add VLM category
curl -X POST http://localhost:8080/api/v1/category/add \
  -H "Content-Type: application/json" \
  -d '{
    "name": "qwen2.5-vl",
    "type": "chat",
    "model": "Qwen/Qwen2.5-VL-7B-Instruct"
  }'

# Update loader
curl -X POST http://localhost:8080/api/v1/model/update/modelloader \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
    "model_loader": "mlxvlm"
  }'
```

### Qwen2-VL Series

| Model | Parameters | Context | Features |
|-------|------------|---------|----------|
| Qwen/Qwen2-VL-2B-Instruct | 2B | 32K | Image understanding |
| Qwen/Qwen2-VL-7B-Instruct | 7B | 32K | Image + text |

**Recommended Step Processor**: `qwen3_thinking`

**Loader**: `mlxvlm`

---

## Model Formats

### MLX Format

Models must be in MLX format for optimal performance on Apple Silicon. MLX-converted models are available from:

- [mlx-community](https://huggingface.co/mlx-community) on HuggingFace
- Models with `mlx` in the name

### Quantized Models

Quantized models reduce memory usage:

| Quantization | Memory Reduction | Quality Impact |
|--------------|------------------|----------------|
| 4-bit (4bit) | 75% | Slight |
| 8-bit (8bit) | 50% | Minimal |

**Example Models**:

```
mlx-community/Qwen2.5-7B-Instruct-4bit
mlx-community/Qwen2.5-7B-Instruct-8bit
mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
```

### Model Directory Structure

Models are stored in the HuggingFace cache:

```
~/.cache/huggingface/hub/
└── models--Qwen--Qwen2.5-7B-Instruct/
    ├── blobs/
    ├── refs/
    └── snapshots/
        └── <commit-hash>/
            ├── config.json
            ├── model.safetensors
            ├── tokenizer.json
            └── tokenizer_config.json
```

---

## Adding New Models

### Method 1: Download via API

```bash
# Search for MLX models
curl "http://localhost:8080/api/v1/model/search?keyword=qwen"

# Download model
curl -X POST http://localhost:8080/api/v1/model/download \
  -H "Content-Type: application/json" \
  -d '{"model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit"}'

# Check download status
curl "http://localhost:8080/api/v1/model/download/status/{task_id}"
```

### Method 2: Manual Download

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit

# Or using git lfs
cd ~/.cache/huggingface/hub
git lfs install
git clone https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit
```

### Method 3: Convert from HuggingFace

```bash
# Install mlx-lm
pip install mlx-lm

# Convert model
python -m mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-7B-Instruct \
  --mlx-path ./Qwen2.5-7B-Instruct-mlx \
  --quantize  # Optional: add 4-bit quantization

# Move to cache
mv ./Qwen2.5-7B-Instruct-mlx ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-mlx
```

### Configure New Model

```bash
# Reload model list
curl -X POST http://localhost:8080/api/v1/model/reload

# Create category
curl -X POST http://localhost:8080/api/v1/category/add \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-model",
    "type": "chat",
    "model": "Qwen/Qwen2.5-7B-Instruct"
  }'

# Update step processor (optional)
curl -X POST http://localhost:8080/api/v1/model/update/stepprocessor \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "step_processor": "qwen3_thinking"
  }'
```

---

## Model Recommendations

### By Use Case

#### General Chat

| Scenario | Model | Reason |
|----------|-------|--------|
| Fast responses | Qwen2.5-3B | Good balance of speed and quality |
| High quality | Qwen2.5-7B | Better reasoning and knowledge |
| Best quality | Qwen2.5-14B+ | Most capable |

#### Code Generation

| Scenario | Model | Reason |
|----------|-------|--------|
| Code completion | Qwen2.5-Coder-1.5B | Fast, focused |
| Code generation | Qwen2.5-Coder-7B | Better context understanding |
| Complex code | Qwen2.5-Coder-32B | Best for architecture |

#### Image Understanding

| Scenario | Model | Reason |
|----------|-------|--------|
| Image description | Qwen2.5-VL-3B | Efficient |
| Document analysis | Qwen2.5-VL-7B | Better OCR |

### By Hardware

| RAM | Recommended Models |
|-----|-------------------|
| 8GB | Qwen2.5-0.5B, Qwen2.5-1.5B (4-bit) |
| 16GB | Qwen2.5-3B, Qwen2.5-7B (4-bit) |
| 32GB | Qwen2.5-7B, Qwen2.5-14B (4-bit) |
| 64GB+ | Qwen2.5-14B, Qwen2.5-32B, Qwen2.5-72B (4-bit) |

---

## Model Configuration Examples

### High-Performance Configuration

```bash
# Set high temperature for creativity
curl -X POST http://localhost:8080/api/v1/model/update/config \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen2.5-7B-Instruct", "key": "temperature", "value": 0.9}'

# Maximize output length
curl -X POST http://localhost:8080/api/v1/model/update/config \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen2.5-7B-Instruct", "key": "max_output_tokens", "value": 16384}'
```

### Code Generation Configuration

```bash
# Lower temperature for deterministic output
curl -X POST http://localhost:8080/api/v1/model/update/config \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen2.5-Coder-7B-Instruct", "key": "temperature", "value": 0.3}'
```

### Memory-Constrained Configuration

```bash
# Enable KV quantization
curl -X POST http://localhost:8080/api/v1/model/update/config \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen2.5-7B-Instruct", "key": "kv_bits", "value": 4}'

# Reduce max tokens
curl -X POST http://localhost:8080/api/v1/model/update/config \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen2.5-7B-Instruct", "key": "max_output_tokens", "value": 4096}'
```

---

## Troubleshooting

### Model Not Found

```
Error: model Qwen/Qwen2.5-7B-Instruct not found
```

**Solution**:
1. Check HuggingFace cache path
2. Download the model
3. Reload model list

```bash
# Check cache
ls $HUGGING_FACE_PATH

# Download
curl -X POST http://localhost:8080/api/v1/model/download \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Qwen/Qwen2.5-7B-Instruct"}'

# Reload
curl -X POST http://localhost:8080/api/v1/model/reload
```

### Out of Memory

```
Error: Out of memory loading model
```

**Solution**:
1. Use quantized model (4-bit)
2. Enable KV quantization
3. Reduce max tokens
4. Use smaller model

```bash
# Switch to 4-bit model
curl -X POST http://localhost:8080/api/v1/category/chmodel/my-model \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Qwen2.5-7B-Instruct-4bit"}'
```

### Wrong Step Processor

```
Error: Unexpected output format
```

**Solution**: Update step processor

```bash
curl -X POST http://localhost:8080/api/v1/model/update/stepprocessor \
  -H "Content-Type: application/json" \
  -d '{"model_name": "model", "step_processor": "qwen3_thinking"}'
```

---

*Last updated: 2024*