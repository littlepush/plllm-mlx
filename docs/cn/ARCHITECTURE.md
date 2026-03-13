# 架构设计文档

本文档详细介绍 plllm-mlx 的系统架构和核心组件。

## 系统架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        路由层 (Router Layer)                      │
│  (routers/chat.py - /ai/v1/chat/completions 端点)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PlModelLoader                               │
│  (抽象基类 - model_loader.py)                                   │
│  - chat_completions_stream()    流式聊天完成                     │
│  - chat_completions_restful()  阻塞式聊天完成                    │
│  - completions_stream()        流式文本完成                      │
│  - completions_restful()       阻塞式文本完成                    │
└──────────┬───────────────────────────┬─────────────────────────┘
           │                           │
           ▼                           ▼
┌─────────────────────────┐     ┌─────────────────────────────────┐
│   PlMlxModel            │     │   其他加载器                    │
│   (mlx_loader.py)       │     │   (未来可扩展)                  │
│   - MLX 专用实现        │     │                                 │
└──────────┬──────────────┘     └─────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step Processors (PlStepProcessor)                 │
│  - PlBaseStepProcessor         标准处理                         │
│  - PlOpenAIStepProcessor       OpenAI 格式                      │
│  - Qwen3ThinkingStepProcessor  Qwen3 思维链模型                 │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   数据类型 (PlChunk)                           │
│  - REASONING  思维内容（来自 Qwen3 等模型）                     │
│  - CONTENT    常规生成文本                                      │
│  - TOOLCALL   函数调用                                          │
│  - NONE       控制信号（如 finish_reason）                      │
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. PlModelLoader（模型加载器基类）

`model_loader.py` 定义了所有模型加载器的抽象基类，负责处理 chat/completion 请求的高级编排，包括 SSE 流式响应和 RESTful 响应。

#### 1.1 模型加载器注册表

```python
class PlModelLoader(ABC):
    __LOADER_MAP__ = {}  # 类级别注册表
```

- 使用静态映射管理已注册的模型加载器
- 提供 `registerModelLoader()` 用于加载器注册
- 提供 `createModel()` 工厂方法实例化加载器
- 提供 `listModelLoaders()` 枚举可用加载器

#### 1.2 抽象方法（必须重写）

| 方法 | 用途 |
|------|------|
| `model_loader_name()` | 返回唯一标识符（如 "mlx"） |
| `ensure_model_loaded()` | 异步加载模型到内存 |
| `ensure_model_unloaded()` | 异步卸载/清理模型 |
| `set_config(config)` | 应用运行时配置 |
| `get_config()` | 获取当前配置 |
| `prepare_prompt(body)` | 将请求转换为模型输入 |
| `stream_generate(session)` | 异步 token 生成 |
| `completion_stream_generate(session)` | 用于 /completions 端点 |

#### 1.3 核心逻辑流程

**请求到流式响应流程（进程隔离）：**

```
客户端请求 ──► chat_completions_stream() ──► stream_generate_via_process()
                                                    │
                                                    ▼
                                          子进程 (PlModelSubprocess)
                                                    │
                                          prepare_prompt() ──► stream_generate()
                                                          │
                                                          ▼
                                                  ┌───────────────┐
                                                  │ PlStepProcessor│
                                                  │   .step()      │  ◄── 每个 token
                                                  └───────┬───────┘
                                                          │
                                                          ▼
                                                  ┌───────────────┐
                                                  │ PlChunk       │
                                                  │ (REASONING/   │
                                                  │  CONTENT/     │
                                                  │  TOOLCALL)    │
                                                  └───────┬───────┘
                                                          │ 通过 Queue
                                                          ▼
                                                  ┌───────────────┐
                                                  │ PlChatCompl   │
                                                  │ Helper        │
                                                  └───────┬───────┘
                                                          │
                                                          ▼
                                                  SSE chunk ──► 客户端
```

> 当进程隔离禁用时（仅开发环境），`stream_generate()` 直接在主进程中运行。

**关键方法说明：**

1. **`chat_completions_stream()`**：主流式入口
   - 创建 `PlChatCompletionHelper` 用于构建 SSE
   - 迭代 `stream_generate()` 生成的 token
   - 按 `PlChunkDataType` 分类处理：
     - `REASONING` → `helper.update_reason_step()`（思维链模型）
     - `CONTENT` → `helper.update_content_step()`（带缓冲）
     - `TOOLCALL` → `helper.update_tool_step()`（函数调用）
   - 处理 `finish_reason`（stop, length, tool_calls）
   - 支持 `include_usage` 显示 token 使用统计

2. **`chat_completions_restful()`**：阻塞响应模式
   - 内部消费 `chat_completions_stream()`
   - 累积 thinking、content 和 tool_calls
   - 返回完整 JSON 响应
   - 存在 tool_calls 时设置 `content: null`（遵循 OpenAI 规范）

#### 1.4 内容缓冲

`__block_count_helper__` 函数控制 SSE 块频率：

```python
def __block_count_helper__(total_tokens: int):
    return next(v for t, v in [(300, 3), (800, 5), (1500, 8), (2000, 10)]
               if total_tokens < t) if total_tokens < 2000 else 15
```

- 300 tokens 时：每 3 个 token 输出一次
- 800 tokens 时：每 5 个 token 输出一次
- 1500 tokens 时：每 8 个 token 输出一次
- 2000+ tokens 时：每 15 个 token 输出一次

这平衡了网络开销与响应性。

#### 1.5 自动注册

模块导入时，自动扫描所有 `_loader.py` 文件：

```python
# 当前脚本导入时加载所有模型加载器
_all_python_codes = PlUnpackPath(os.path.join(PlRootPath(), 'models'),
                                  recursive=False)
for script in _all_python_codes:
    if script.endswith("_loader.py"):
        model_loader_clz = PlFindSpecifialSubclass(script, PlModelLoader)
        for clz in model_loader_clz:
            PlModelLoader.registerModelLoader(clz.model_loader_name(), clz)
```

这实现了新加载器实现的自动发现。

---

### 2. 进程隔离

> **注意**：进程隔离现在是默认架构。所有模型推理都在子进程中运行，防止阻塞 FastAPI 事件循环。

进程隔离层包括：

- **PlProcessManager** (`process_manager.py`)：单例管理器，处理子进程生命周期
- **PlModelSubprocess** (`model_subprocess.py`)：每个模型的子进程，负责加载和运行推理
- **通信**：使用 `multiprocessing.Queue` 进行请求/响应传递

进程隔离启用时：
1. 主进程永不阻塞在模型推理上
2. 每个模型运行在独立的 Python 进程中
3. 模型在启动时预热，加快首次请求响应

---

### 3. PlMlxModel（MLX 加载器）

`mlx_loader.py` 为 Apple Silicon (M 系列) Mac 提供专用实现，使用 MLX 框架进行高效的本地推理和 Metal 加速。

#### 3.1 模型加载

```python
async def ensure_model_loaded(self):
    self._model, self._tokenizer, model_config = load(self.model_name,
                                                        return_config=True)
    self._model.eval()
```

- 使用 `mlx_lm.load()` 加载模型和 tokenizer
- 自动从 `max_position_embeddings` 检测模型上下文限制
- 使用 asyncio 锁实现线程安全加载

#### 3.2 Prompt 处理

- 通过 tokenizer 应用 chat 模板
- 处理 tools/tool_choice 用于函数调用

#### 3.3 采样配置

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `temperature` | 0.8 | 采样温度 |
| `top_p` | 0 | 核采样阈值 |
| `top_k` | 100 | Top-k 采样 |
| `min_p` | 0.0 | 最小 token 概率 |
| `repetition_penalty` | 1.1 | 重复惩罚 |
| `max_output_tokens` | 16K | 最大输出 token 数 |

#### 3.4 KVCache 优化

```python
self._prefill_step_size = 4096     # Prompt 处理块大小
self._kv_bits = None               # KV 量化位数
self._kv_group_size = 32           # 量化组大小
self._quantized_kv_start = 0       # 量化开始的 token 阈值
self._max_kv_size = None           # 最大 KV 缓存大小
```

#### 3.5 基于消息的 Prefix KV Cache

> **特性**：Prefix KV Cache 通过缓存和复用已计算的 KV 值，显著提升重复/相似 prompt 的性能。

基于消息链的 Prefix KV Cache 实现，使用消息内容（含 token）的 MD5 作为唯一标识。

**核心组件**：
- `helpers/chain_cache.py`：
  - **PlChain**：消息链对象，包含 node_ids（消息 ID 列表）、cache_item（KV 缓存）、temp_cache_item（临时缓存）
  - **PlChainCache**：基于 OrderedDict 实现的 LRU 缓存，支持最长链搜索
- `models/kv_cache.py`：
  - **PlMessageBasedKVCache**：主缓存管理器，封装消息分割、ID 生成、缓存查询、淘汰逻辑
  - **PlKVCacheMessage**：消息数据模型

**实现原理**：
1. **消息分割**：使用 begin/end tokens（如 `<|im_start|>` / `<|im_end|>`）分割 prompt
2. **ID 生成**：使用消息完整内容（含 tokens）的 MD5 作为 msg_id
3. **链式缓存**：将消息链 (node_ids) 的 MD5 作为 cache_key
4. **缓存查询**：查找最长匹配的消息链，支持完全匹配和部分匹配
5. **增量 prefill**：仅对未缓存的消息进行 prefill

**缓存匹配策略**：
- **跳过首轮**：消息数 < 3 时不缓存（新对话场景）
- **角色顺序校验**：2 条消息仅允许 (system+user)，3 条消息允许 (sys+usr+usr), (sys+asst+usr), (usr+asst+usr)
- **完全匹配**：所有消息 ID 完全匹配时返回缓存；若用户重试（retry），使用前 N-1 条消息的缓存
- **部分匹配**：返回前缀消息的缓存，待 prefill 增量部分

**缓存升级机制**：
- 临时缓存 (temp_cache_item)：存储 assistant 回复前的中间状态
- 当用户发送下一轮消息时，升级临时缓存为正式缓存

详见 [KV_CACHE.md](./KV_CACHE.md)。

#### 3.6 Session 管理

```python
class PlMlxSessionStorage:
    """保存每次请求的会话数据"""
    - prompt_cache     # Prompt 的 KV 缓存
    - prompt           # 格式化的 prompt 字符串
    - sampler          # 采样函数
    - logits_processors # 重复处理
    - max_tokens       # 最大输出 token 数
    - tools            # 函数定义
    - tool_choice      # 强制工具选择
```

#### 3.7 流式生成流程

```python
async def stream_generate(self, session_object):
    # 1. 从 MLX 获取 token（阻塞，在 executor 中运行）
    tokens = await loop.run_in_executor(None, run_in_thread)

    # 2. 通过 step processor 处理
    stpp = self.step_processor_clz()  # 如 Qwen3ThinkingStepProcessor

    # 3. 输出块到客户端
    for gr in tokens:
        chunk = stpp.step(gr)
        if chunk is not None:
            yield chunk

    # 4. 提取工具调用（如有）
    tool_calls = stpp.tool_calls()
    for tc in tool_calls:
        yield tc

    # 5. 发送完成信号
    yield stpp.finish()
```

#### 3.8 线程安全

- 使用 `asyncio.Lock()` 序列化推理（MLX Metal 限制）
- 直接在 executor 中运行阻塞的 MLX 推理
- 允许在一个推理运行时进行并发请求准备

---

### 4. PlMlxVlmModel（视觉语言模型加载器）

`mlxvlm_loader.py` 为视觉语言模型 (VLM) 提供专用实现，使用 mlx-vlm 包支持图像理解。

#### 4.1 关键特性

- 使用 `mlx_vlm.load()` 加载模型和处理器
- 处理文本和图像
- 与 PlMlxModel 相同的采样和 KVCache 优化
- 兼容 step processors（通常使用 `Qwen3ThinkingStepProcessor`）

#### 4.2 统一的 Prefix KV Cache

`PlMlxModel` (mlx_loader) 和 `PlMlxVlmModel` (mlxvlm_loader) 都使用 `kv_cache.py` 中统一的 `PlMessageBasedKVCache` 实现，支持基于链的前缀缓存。

#### 4.3 视觉支持

`PlMlxVlmModel` 通过 mlx-vlm 包支持多模态输入（图像），使 Qwen2.5-VL 等视觉语言模型能够理解图像和文本。

**图像输入格式**：

```python
# 单张图像
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }
]

# 多张图像（base64）
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "比较这两张图片"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
        ]
    }
]
```

**视觉 Token**：

模型使用特殊 token 表示图像：
- `<|vision_start|>` - 视觉内容开始
- `<|image_pad|>` - 图像占位符（根据图像大小重复）
- `<|vision_end|>` - 视觉内容结束

这些 token 由 HuggingFace 模型配置的 chat 模板自动生成。

#### 4.4 带视觉的 Prefix KV Cache

涉及图像时，prefix KV cache 的工作方式：

1. **视觉 Token 检测**：缓存解析 prompt 以找到 `<|vision_start|>/<|vision_end|>` 对并计算 `vision_count`

2. **视觉感知匹配**：缓存命中时，检查未匹配部分是否有视觉 token：
   - 如果当前请求有图像但缓存部分没有：清除图像，使用缓存的 KV
   - 如果视觉计数不匹配：修剪图像以匹配未匹配部分

3. **图像列表调整**：

```python
if len(images) > 0:
    if left_vision_count == 0:
        images = []  # 未匹配部分无视觉内容
    elif left_vision_count != len(images):
        images = images[-left_vision_count:]  # 匹配计数
```

---

### 5. Step Processors（步骤处理器）

#### PlStepProcessor（抽象基类）

位于 `step_processor.py`，定义：
- `step(generate_response)`：处理每个 token，返回 `PlChunk`
- `tool_calls()`：从缓冲区提取函数调用
- `finish()`：生成带有 `finish_reason` 的最终块

#### 实现

| 处理器 | 用途 |
|--------|------|
| `PlBaseStepProcessor` | 标准内容生成 |
| `PlOpenAIStepProcessor` | OpenAI 兼容输出 |
| `Qwen3ThinkingStepProcessor` | Qwen3 带内部思维 |

#### PlChunk 数据类型

```python
class PlChunkDataType(Enum):
    NONE = 0       # 仅控制信号
    REASONING = 1  # 思维内容（Qwen3）
    CONTENT = 2    # 常规文本输出
    TOOLCALL = 3   # 函数调用
```

---

### 6. 辅助类

#### PlChatCompletionHelper (`helpers/chat_helper.py`)

遵循 OpenAI 格式构建 SSE 块：
- 第一个块：包含 `role: "assistant"`
- 后续块：省略 `role`
- 最终块：空 delta `{}`
- 处理思维模型的 `reasoning` 字段
- 当 `finish_reason="tool_calls"` 时在 delta 中保留 `tool_calls`

#### PlStepUsage / PlStepHelper (`helpers/step_info.py`)

跟踪 token 使用和性能指标：
- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `prompt_tps`, `generation_tps`
- `prompt_process`（毫秒）, `first_token`（秒）

---

### 7. 配置参数

#### 模型级别（通过 set_config）

| 键 | 默认值 | 描述 |
|----|--------|------|
| `max_prompt_tokens` | 动态 | 最大 prompt token（计算方式：max_model_tokens - max_output_tokens） |
| `max_output_tokens` | 16K | 最大输出 token |
| `temperature` | 0.8 | 采样温度 |
| `top_p` | 0 | 核采样 |
| `top_k` | 100 | Top-k 采样 |
| `repetition_penalty` | 1.1 | 重复惩罚 |

#### MLX 专用（KVCache）

| 键 | 默认值 | 描述 |
|----|--------|------|
| `prefill_step_size` | 4096 | Prompt 处理块大小 |
| `kv_bits` | None | KV 量化（None=关闭） |
| `kv_group_size` | 32 | 量化组大小 |
| `quantized_kv_start` | 0 | 开始 KV 量化的 token 阈值 |
| `max_kv_size` | None | 最大 KV 缓存大小 |

#### MLXVLM 专用（Prefix KV Cache）

| 键 | 默认值 | 描述 |
|----|--------|------|
| `enable_prefix_cache` | true | 启用/禁用基于消息的前缀缓存 |
| `prefix_cache_min_update_length` | 50 | 触发缓存更新的最小字符数 |
| `max_cache_entries` | 3 | 最大缓存对话数 |
| `max_cache_memory_mb` | 2048 | 最大总缓存内存（MB） |
| `vision_begin_tokens` | ['<\|vision_start\|>'] | 缓存的视觉开始 token |
| `vision_end_tokens` | ['<\|vision_end\|>'] | 缓存的视觉结束 token |

---

## 时序图：聊天完成请求

```
┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────────┐
│  客户端   │───►│ Chat Router  │───►│ PlMlxModel  │───►│ prepare_prompt  │
└──────────┘    └──────────────┘    └─────────────┘    └────────┬────────┘
                                                                   │
                                                                   ▼
                                               ┌─────────────────┐
                                               │ process_manager │
                                               │ .submit_request │
                                               └────────┬────────┘
                                                        │
                                                        ▼ multiprocessing Queue
┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────────┐
│  客户端   │◄───│ SSE Stream   │◄───│  Queue      │◄───│ PlModelSubprocess│
│          │    │              │    │  (chunks)   │    │  (推理)          │
└──────────┘    └──────┬───────┘    └──────┬──────┘    └────────┬────────┘
                        │                    │                    │
                        │                    │           prepare_prompt() &
                        │                    │           stream_generate()
                        │                    │                    │
                        ▼                    ▼                    ▼
                 ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
                 │ PlChunk     │    │ PlStepProcessor │    │ mlx_lm.gen()    │
                 │ (via Queue) │◄───│ .step(token)    │◄───│ (MLX 推理)      │
                 └─────────────┘    └─────────────────┘    └─────────────────┘
                        │
                        ▼
                 ┌─────────────────┐
                 │ PlChatCompl     │
                 │ Helper          │
                 │ .build_chunk()  │
                 └─────────────────┘
```

---

## 扩展：添加新的模型加载器

添加对新后端（如 vLLM、llama.cpp）的支持：

```python
# 1. 在 models/ 目录创建 new_loader.py
from models.model_loader import PlModelLoader

class PlVllmModel(PlModelLoader):
    def __init__(self, model_name: str, step_processor_clz):
        super().__init__(model_name, step_processor_clz)
        # 初始化 vLLM 引擎

    @staticmethod
    def model_loader_name():
        return "vllm"

    # 2. 实现所有抽象方法...
    async def ensure_model_loaded(self): ...
    async def ensure_model_unloaded(self): ...
    def set_config(self, config: dict): ...
    def get_config(self): ...
    def prepare_prompt(self, body: dict): ...
    async def stream_generate(self, session): ...
    async def completion_stream_generate(self, session): ...
```

加载器会在导入时自动注册。