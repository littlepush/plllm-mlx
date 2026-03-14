# Documentation

Welcome to plllm-mlx documentation.

## Getting Started

- [Architecture Overview](ARCHITECTURE.md) - System design and components
- [README](../README.md) - Quick start guide

## Core Concepts

### Model Loaders
- MLX-LM Loader - Text-only language models
- MLX-VLM Loader - Vision-language models

### Processing Pipeline
- Step Processors - Transform generation results
- KV Cache - Efficient prompt caching

### System Design
- Process Isolation - Stable multi-model serving
- Streaming - Real-time token delivery

## API Reference

All endpoints follow OpenAI API specification:

### Chat Completions
```
POST /v1/chat/completions
```

### Models
```
GET /v1/models
```

### Health
```
GET /health
```

## Configuration

Configuration via YAML file or environment variables:

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  
models:
  - name: "default"
    model_id: "Qwen/Qwen2.5-7B-Instruct"
    loader: "mlx"
```

## Examples

### Basic Chat
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Multi-turn Conversation
```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "How are you?"}
]
```

## Performance

### Memory Management
- Process isolation per model
- Configurable cache limits
- LRU eviction policy

### Optimization Tips
- Enable prefix cache for multi-turn
- Use quantized models (4bit/8bit)
- Adjust prefill step size

## Troubleshooting

### Common Issues

**Model not found**
```bash
# Check model cache
ls ~/.cache/huggingface/hub/
```

**Memory errors**
```bash
# Reduce max_tokens or use quantized models
```

**Slow streaming**
- Check if streaming is enabled
- Verify KV cache effectiveness

## More Resources

- [GitHub Repository](https://github.com/littlepush/plllm-mlx)
- [Issue Tracker](https://github.com/littlepush/plllm-mlx/issues)

---

## 中文文档

[点击这里查看中文文档](cn/README.md)