# plllm-mlx

基于 MLX (Apple Silicon GPU) 的 LLM 推理服务，支持本地模型部署和 OpenAI 兼容 API。

## 功能特性

- **多模型加载器支持**: MLX (Apple Silicon), OpenAI 等
- **多种 Step Processor**: 支持不同的模型输出格式解析
- **OpenAI 兼容 API**: 完全兼容 OpenAI Chat Completions 和 Completions 接口
- **流式输出**: 支持 Server-Sent Events (SSE) 流式响应
- **Embedding & Rerank**: 内置 BGE-M3 embedding 模型支持
- **Category 管理**: 支持多模型多 category 动态管理
- **模型热插拔**: 支持运行时切换模型加载器和 step processor
- **Prefix KV Cache**: 基于消息链的 KV Cache 复用，显著提升推理性能

## 系统要求

- macOS (Apple Silicon) 设备
- Python 3.12+
- 50GB+ 存储空间用于模型缓存

## 快速开始

### 1. 安装依赖

```bash
# 使用 pip
pip install -e .

# 或使用 uv
uv pip install -e .
```

### 2. 环境配置

设置 HuggingFace 缓存路径（可选）:

```bash
export HUGGING_FACE_PATH="$HOME/.cache/huggingface/hub"
```

### 3. 启动服务

```bash
# 使用命令行工具
plllm-mlx

# 或直接运行
python -m plllm_mlx

# 或使用启动脚本
./start.sh
```

服务默认监听 `http://0.0.0.0:8080`

#### Dry Run 模式

用于测试所有 LLM 模型是否能正常响应，不会启动 HTTP 服务器：

```bash
./test.sh --dry-run
# 或
./start.sh --dry-run
# 或
python -m plllm_mlx --dry-run
```

启动后会对每个 LLM 模型发送 "hello" 测试请求，确保都能收到响应后退出。

## API 接口

### Chat Completions

```bash
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "你好！"}],
    "stream": true
  }'
```

### Embedding

```bash
curl -X POST http://localhost:8080/ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "你好，世界！"
  }'
```

### 模型列表

```bash
curl http://localhost:8080/ai/v1/models
```

## 项目结构

```
plllm-mlx/
├── plllm_mlx/
│   ├── __init__.py
│   ├── cli.py                # 命令行入口
│   ├── models/
│   │   ├── base_step_processor.py    # 基础 step processor
│   │   ├── model_loader.py           # 模型加载器基类
│   │   ├── mlx_loader.py             # MLX 模型加载器
│   │   ├── openai_step_processor.py  # OpenAI 格式输出处理器
│   │   └── step_processor.py         # Step processor 基类
│   ├── routers/
│   │   ├── chat.py           # Chat completions 路由
│   │   ├── embedding.py      # Embedding 路由
│   │   └── rerank.py         # Rerank 路由
│   └── helpers/
│       ├── chat_helper.py    # Chat 辅助工具
│       ├── chunk_helper.py   # Chunk 处理工具
│       └── chain_cache.py    # 消息链缓存
├── docs/
│   ├── cn/                   # 中文文档
│   │   ├── README.md         # 总览
│   │   ├── ARCHITECTURE.md   # 架构设计
│   │   ├── API.md            # API 文档
│   │   ├── KV_CACHE.md       # KV Cache 原理
│   │   ├── CONFIGURATION.md  # 配置说明
│   │   └── EXAMPLES.md       # 使用示例
│   └── en/                   # 英文文档
├── tests/                    # 测试用例
└── pyproject.toml            # 项目配置
```

## 主要组件

### Model Loader（模型加载器）

| 加载器 | 描述 |
|--------|------|
| mlx | Apple Silicon GPU 加速推理 |
| openai | OpenAI API 兼容模型 |
| mlxvlm | 视觉语言模型（支持图像输入） |

### Step Processor（步骤处理器）

| 处理器 | 描述 |
|--------|------|
| base | 基础文本输出 |
| openai | OpenAI XML 格式输出（支持 tool_call） |
| qwen3_thinking | Qwen3 思维链输出（支持 reasoning） |

### Prefix KV Cache（前缀 KV 缓存）

基于消息链的 Prefix KV Cache，使用消息内容（含 token）的 MD5 作为唯一标识，显著提升重复对话场景的推理性能。

**核心优势**：
- 支持多轮对话的缓存复用
- 自动增量 prefill，减少计算量
- 智能内存管理，自动淘汰旧缓存

详见 [KV_CACHE.md](./KV_CACHE.md)。

## 文档索引

- [架构设计](./ARCHITECTURE.md) - 了解系统架构和核心组件
- [API 文档](./API.md) - 完整的 API 接口说明
- [KV Cache 原理](./KV_CACHE.md) - 深入理解缓存机制
- [配置说明](./CONFIGURATION.md) - 详细的配置参数
- [使用示例](./EXAMPLES.md) - 丰富的代码示例

## License

MIT