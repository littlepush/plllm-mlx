# 配置详细说明

本文档详细介绍 plllm-mlx 的所有配置选项及其使用方法。

## 目录

- [配置方式概览](#配置方式概览)
- [环境变量](#环境变量)
- [模型配置](#模型配置)
- [采样参数](#采样参数)
- [KV Cache 配置](#kv-cache-配置)
- [Prefix KV Cache 配置](#prefix-kv-cache-配置)
- [服务器配置](#服务器配置)
- [日志配置](#日志配置)
- [配置示例](#配置示例)

---

## 配置方式概览

plllm-mlx 支持以下配置方式：

| 方式 | 用途 | 优先级 |
|------|------|--------|
| 环境变量 | 全局配置、运行时行为 | 低 |
| 配置文件 | 模型列表、服务配置 | 中 |
| API 参数 | 单次请求配置 | 高 |
| set_config() | 运行时模型配置 | 高 |

---

## 环境变量

### 基础环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DEBUG` | `0` | 调试模式（0/1） |
| `HUGGING_FACE_PATH` | `~/.cache/huggingface/hub` | HuggingFace 模型缓存路径 |
| `PLLLM_HOST` | `0.0.0.0` | 服务监听地址 |
| `PLLLM_PORT` | `8080` | 服务监听端口 |

### 内存管理环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `PLLLM_MEMORY_THRESHOLD` | `0.9` | 触发缓存淘汰的内存阈值 |
| `PLLLM_MEMORY_LOWBOUND_THRESHOLD` | `0.7` | 淘汰后的目标内存水位 |
| `PLLLM_CACHE_MIN_ENTRIES` | `3` | 最小保留的缓存条目数 |

### 日志环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `PLLLM_LOG_LEVEL` | `INFO` | 日志级别 |
| `PLLLM_CACHE_DEBUG` | `0` | 启用缓存调试日志 |

### 使用示例

```bash
# 在 shell 中设置
export DEBUG=1
export PLLLM_MEMORY_THRESHOLD=0.8
export HUGGING_FACE_PATH="/data/models"

# 或在启动命令中指定
PLLLM_PORT=9000 python -m plllm_mlx
```

---

## 模型配置

### 通过配置文件

创建 `config.yaml`：

```yaml
# 模型列表
models:
  # LLM 模型
  - name: "qwen2.5-0.5b"
    path: "Qwen/Qwen2.5-0.5B-Instruct"
    type: "llm"
    loader: "mlx"
    default: true
    
  - name: "qwen2.5-7b"
    path: "Qwen/Qwen2.5-7B-Instruct"
    type: "llm"
    loader: "mlx"
    
  # 视觉语言模型
  - name: "qwen2.5-vl-7b"
    path: "Qwen/Qwen2.5-VL-7B-Instruct"
    type: "vlm"
    loader: "mlxvlm"

# Embedding 模型
embedding:
  model: "BAAI/bge-m3"
  
# Rerank 模型
rerank:
  model: "BAAI/bge-reranker-v2-m3"
```

### 通过 API 动态加载

```bash
# 加载模型
curl -X POST http://localhost:8080/ai/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-14b",
    "path": "Qwen/Qwen2.5-14B-Instruct",
    "loader": "mlx"
  }'

# 卸载模型
curl -X POST http://localhost:8080/ai/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-14b"
  }'
```

---

## 采样参数

### 参数说明

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `temperature` | float | 0.8 | 0-2 | 采样温度。值越高输出越随机，值越低输出越确定 |
| `top_p` | float | 0 | 0-1 | 核采样概率。0 表示不使用 |
| `top_k` | int | 100 | 1+ | Top-k 采样，只从概率最高的 k 个 token 中采样 |
| `min_p` | float | 0.0 | 0-1 | 最小 token 概率阈值 |
| `repetition_penalty` | float | 1.1 | 1+ | 重复惩罚。大于 1 时惩罚重复内容 |
| `max_tokens` | int | 16384 | 1+ | 最大输出 token 数 |

### 采样参数详解

#### temperature（温度）

```
temperature = 0.1  → 输出非常确定，适合代码生成
temperature = 0.8  → 平衡创造性和准确性
temperature = 1.5  → 输出非常随机，适合创意写作
```

#### top_p（核采样）

```python
# top_p = 0.9
# 从累积概率达到 90% 的最小 token 集合中采样
# 例如: 如果前 5 个 token 的概率分别是 [0.4, 0.3, 0.15, 0.08, 0.04]
# 累积概率 = 0.97 > 0.9，所以只从前 4 个 token 中采样
```

#### top_k

```python
# top_k = 50
# 每次只从概率最高的 50 个 token 中采样
# 与 top_p 结合使用可以进一步限制候选范围
```

#### repetition_penalty

```python
# repetition_penalty = 1.0  → 不惩罚重复
# repetition_penalty = 1.1  → 轻微惩罚（默认）
# repetition_penalty = 1.5  → 强烈惩罚，避免重复内容
```

### 使用示例

#### API 请求

```bash
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [{"role": "user", "content": "写一个故事"}],
    "temperature": 0.9,
    "top_p": 0.95,
    "max_tokens": 500
  }'
```

#### Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed"
)

# 创意写作
creative_response = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[{"role": "user", "content": "写一首诗"}],
    temperature=1.2,
    top_p=0.9
)

# 代码生成
code_response = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[{"role": "user", "content": "写一个排序算法"}],
    temperature=0.1,
    top_p=0.9
)

# 避免重复
no_repeat_response = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[{"role": "user", "content": "列出 10 个不同的颜色"}],
    repetition_penalty=1.5
)
```

---

## KV Cache 配置

### 基础 KV Cache 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prefill_step_size` | int | 4096 | Prompt 处理块大小 |
| `kv_bits` | int | None | KV 量化位数（4/8） |
| `kv_group_size` | int | 32 | 量化组大小 |
| `quantized_kv_start` | int | 0 | 开始量化的 token 阈值 |
| `max_kv_size` | int | None | 最大 KV 缓存大小 |

### KV Cache 量化

启用 KV Cache 量化可以减少内存占用：

```python
# 通过 set_config 配置
config = {
    "kv_bits": 4,           # 4-bit 量化
    "kv_group_size": 32,    # 量化组大小
    "quantized_kv_start": 1000  # 从第 1000 个 token 开始量化
}
await model.set_config(config)
```

**量化效果对比**：

| 配置 | 内存占用 | 精度损失 |
|------|----------|----------|
| 无量化 (fp16) | 100% | 无 |
| 8-bit 量化 | ~50% | 极小 |
| 4-bit 量化 | ~25% | 可接受 |

---

## Prefix KV Cache 配置

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_prefix_cache` | bool | true | 启用/禁用 Prefix KV Cache |
| `begin_tokens` | list | `["<|im_start|>", "<|start|>"]` | 消息开始 token |
| `end_tokens` | list | `["<|im_end|>", "<|end|>"]` | 消息结束 token |

### 视觉模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vision_begin_tokens` | list | `["<|vision_start|>"]` | 视觉内容开始 token |
| `vision_end_tokens` | list | `["<|vision_end|>"]` | 视觉内容结束 token |

### 缓存管理参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_cache_entries` | int | 100 | 最大缓存条目数 |
| `max_cache_memory_mb` | int | 2048 | 最大缓存内存（MB） |

### 配置示例

```python
# 完整配置示例
config = {
    # 启用 Prefix KV Cache
    "enable_prefix_cache": True,
    
    # 消息分割 token（支持多种格式）
    "begin_tokens": [
        "<|im_start|>",  # Qwen 格式
        "<|start|>",     # 通用格式
    ],
    "end_tokens": [
        "<|im_end|>",    # Qwen 格式
        "<|end|>",       # 通用格式
    ],
    
    # 视觉模型 token
    "vision_begin_tokens": ["<|vision_start|>"],
    "vision_end_tokens": ["<|vision_end|>"],
    
    # 缓存大小限制
    "max_cache_entries": 50,
    "max_cache_memory_mb": 4096,  # 4GB
}

# 应用配置
await model_loader.set_config(config)
```

### 不同模型的 token 配置

#### Qwen 系列

```python
qwen_config = {
    "begin_tokens": ["<|im_start|>"],
    "end_tokens": ["<|im_end|>"],
}
```

#### Llama 系列

```python
llama_config = {
    "begin_tokens": ["<|begin_of_text|>", "<|start_header_id|>"],
    "end_tokens": ["<|end_of_text|>", "<|end_header_id|>"],
}
```

#### 自定义模型

```python
# 需要根据模型的 chat template 确定
custom_config = {
    "begin_tokens": ["[START]"],
    "end_tokens": ["[END]"],
}
```

---

## 服务器配置

### 启动参数

```bash
python -m plllm_mlx --help

# 参数说明
--host TEXT              监听地址 [默认: 0.0.0.0]
--port INTEGER           监听端口 [默认: 8080]
--workers INTEGER        工作进程数 [默认: 1]
--reload                 开发模式，自动重载
--log-level TEXT         日志级别 [默认: info]
```

### Uvicorn 配置

```python
# 在代码中配置
import uvicorn
from plllm_mlx import create_app

app = create_app()

uvicorn.run(
    app,
    host="0.0.0.0",
    port=8080,
    workers=1,
    log_level="info",
    access_log=True,
)
```

---

## 日志配置

### 日志级别

| 级别 | 说明 |
|------|------|
| `DEBUG` | 详细调试信息 |
| `INFO` | 一般信息（默认） |
| `WARNING` | 警告信息 |
| `ERROR` | 错误信息 |
| `CRITICAL` | 严重错误 |

### 配置日志

```bash
# 通过环境变量
export PLLLM_LOG_LEVEL=DEBUG

# 或在启动时指定
python -m plllm_mlx --log-level debug
```

### 查询日志

```bash
# 实时查询日志（需要配置日志上报）
uv run python query_logs.py

# 按级别过滤
uv run python query_logs.py -l ERROR

# 按关键词搜索
uv run python query_logs.py -k "PlMessageBasedKVCache"
```

---

## 配置示例

### 开发环境配置

```bash
# .env.development
DEBUG=1
PLLLM_LOG_LEVEL=DEBUG
PLLLM_MEMORY_THRESHOLD=0.95
PLLLM_CACHE_MIN_ENTRIES=1
```

```yaml
# config.dev.yaml
models:
  - name: "qwen2.5-0.5b"
    path: "Qwen/Qwen2.5-0.5B-Instruct"
    type: "llm"
    loader: "mlx"
    default: true
```

### 生产环境配置

```bash
# .env.production
DEBUG=0
PLLLM_LOG_LEVEL=INFO
PLLLM_MEMORY_THRESHOLD=0.85
PLLLM_MEMORY_LOWBOUND_THRESHOLD=0.6
PLLLM_CACHE_MIN_ENTRIES=5
```

```yaml
# config.prod.yaml
models:
  - name: "qwen2.5-7b"
    path: "Qwen/Qwen2.5-7B-Instruct"
    type: "llm"
    loader: "mlx"
    default: true
    
  - name: "qwen2.5-vl-7b"
    path: "Qwen/Qwen2.5-VL-7B-Instruct"
    type: "vlm"
    loader: "mlxvlm"

embedding:
  model: "BAAI/bge-m3"
  
rerank:
  model: "BAAI/bge-reranker-v2-m3"
```

### 内存受限环境配置

```bash
# 适用于 16GB 内存的 Mac
export PLLLM_MEMORY_THRESHOLD=0.7
export PLLLM_MEMORY_LOWBOUND_THRESHOLD=0.5
export PLLLM_CACHE_MIN_ENTRIES=1
```

```yaml
# 只加载小模型
models:
  - name: "qwen2.5-0.5b"
    path: "Qwen/Qwen2.5-0.5B-Instruct"
    type: "llm"
    loader: "mlx"
```

### 高性能配置

```bash
# 适用于 64GB+ 内存的 Mac
export PLLLM_MEMORY_THRESHOLD=0.9
export PLLLM_MEMORY_LOWBOUND_THRESHOLD=0.8
export PLLLM_CACHE_MIN_ENTRIES=10
```

```python
# 通过 API 调整模型配置
config = {
    "max_output_tokens": 32768,  # 更大的输出
    "max_cache_entries": 20,     # 更多的缓存
    "max_cache_memory_mb": 8192, # 8GB 缓存
}
await model.set_config(config)
```

---

## 配置最佳实践

### 1. 根据内存调整缓存

```python
import psutil

def get_recommended_cache_config():
    """根据系统内存推荐配置"""
    total_mem_gb = psutil.virtual_memory().total / (1024**3)
    
    if total_mem_gb < 16:
        return {
            "max_cache_entries": 3,
            "max_cache_memory_mb": 1024,
            "PLLLM_MEMORY_THRESHOLD": 0.7,
        }
    elif total_mem_gb < 32:
        return {
            "max_cache_entries": 5,
            "max_cache_memory_mb": 2048,
            "PLLLM_MEMORY_THRESHOLD": 0.8,
        }
    else:
        return {
            "max_cache_entries": 10,
            "max_cache_memory_mb": 4096,
            "PLLLM_MEMORY_THRESHOLD": 0.9,
        }
```

### 2. 根据场景选择采样参数

```python
# 代码生成
CODE_GEN_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
}

# 创意写作
CREATIVE_CONFIG = {
    "temperature": 1.0,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
}

# 问答/分析
QA_CONFIG = {
    "temperature": 0.3,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}
```

### 3. 监控和调优

```python
import time
import psutil

def monitor_and_tune():
    """监控内存使用并动态调整"""
    while True:
        mem = psutil.virtual_memory()
        cache_entries = get_cache_count()
        
        print(f"内存: {mem.percent}%")
        print(f"缓存条目: {cache_entries}")
        
        if mem.percent > 85:
            # 触发主动淘汰
            evict_oldest_cache()
        
        time.sleep(60)
```

---

## 故障排查

### 配置未生效

1. **检查优先级**：API 参数 > set_config > 配置文件 > 环境变量
2. **检查重启**：某些配置需要重启服务
3. **检查日志**：查看是否有配置错误提示

```bash
# 查看当前配置
curl http://localhost:8080/ai/v1/models/qwen2.5-7b/config
```

### 内存溢出

```bash
# 降低缓存阈值
export PLLLM_MEMORY_THRESHOLD=0.7
export PLLLM_CACHE_MIN_ENTRIES=1

# 启用 KV 量化
config = {"kv_bits": 4}
```

### 缓存命中率低

1. 检查消息是否一致
2. 检查 begin/end token 配置
3. 启用调试日志查看匹配情况

```bash
export PLLLM_CACHE_DEBUG=1
```