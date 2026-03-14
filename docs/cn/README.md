# 文档

欢迎使用 plllm-mlx 文档。

## 快速开始

- [架构概览](ARCHITECTURE.md) - 系统设计和组件
- [自述文件](../../README.md) - 快速入门指南

## 核心概念

### 模型加载器
- MLX-LM 加载器 - 纯文本语言模型
- MLX-VLM 加载器 - 视觉语言模型

### 处理流程
- Step 处理器 - 转换生成结果
- KV 缓存 - 高效的提示缓存

### 系统设计
- 进程隔离 - 稳定的多模型服务
- 流式传输 - 实时token推送

## API 参考

所有端点遵循 OpenAI API 规范：

### 对话补全
```
POST /v1/chat/completions
```

### 模型列表
```
GET /v1/models
```

### 健康检查
```
GET /health
```

## 配置

通过 YAML 文件或环境变量配置：

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  
models:
  - name: "default"
    model_id: "Qwen/Qwen2.5-7B-Instruct"
    loader: "mlx"
```

## 示例

### 基础对话
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "你好！"}],
    "stream": true
  }'
```

### 多轮对话
```python
messages = [
    {"role": "system", "content": "你是一个助手。"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "你好吗？"}
]
```

## 性能优化

### 内存管理
- 每个模型独立进程
- 可配置缓存限制
- LRU 淘汰策略

### 优化建议
- 多轮对话启用前缀缓存
- 使用量化模型（4bit/8bit）
- 调整预填充步长

## 故障排除

### 常见问题

**模型未找到**
```bash
# 检查模型缓存
ls ~/.cache/huggingface/hub/
```

**内存错误**
```bash
# 减少 max_tokens 或使用量化模型
```

**流式传输慢**
- 检查是否启用了流式
- 验证 KV 缓存效果

## 更多资源

- [GitHub 仓库](https://github.com/littlepush/plllm-mlx)
- [问题追踪](https://github.com/littlepush/plllm-mlx/issues)

---

[English Documentation](../README.md)
