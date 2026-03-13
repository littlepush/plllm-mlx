# API 详细文档

本文档详细描述 plllm-mlx 提供的所有 API 接口，完全兼容 OpenAI API 格式。

## 目录

- [基础信息](#基础信息)
- [Chat Completions](#chat-completions)
- [Completions](#completions)
- [Embeddings](#embeddings)
- [Rerank](#rerank)
- [Models](#models)
- [错误处理](#错误处理)

---

## 基础信息

### 基础 URL

```
http://localhost:8080/ai/v1
```

### 认证

当前版本不需要 API Key 认证。

### 请求头

```http
Content-Type: application/json
```

---

## Chat Completions

聊天补全接口，支持多轮对话、流式输出和函数调用。

### 端点

```
POST /ai/v1/chat/completions
```

### 请求参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `model` | string | 是 | 模型名称，如 `qwen2.5-0.5b` |
| `messages` | array | 是 | 消息数组 |
| `stream` | boolean | 否 | 是否流式输出，默认 `false` |
| `temperature` | number | 否 | 采样温度，默认 `0.8`，范围 `0-2` |
| `top_p` | number | 否 | 核采样概率，默认 `0`，范围 `0-1` |
| `top_k` | integer | 否 | Top-k 采样，默认 `100` |
| `max_tokens` | integer | 否 | 最大输出 token 数，默认 `16384` |
| `repetition_penalty` | number | 否 | 重复惩罚，默认 `1.1` |
| `tools` | array | 否 | 函数定义数组 |
| `tool_choice` | string/object | 否 | 工具选择策略 |
| `stop` | array | 否 | 停止词列表 |
| `user` | string | 否 | 用户标识 |

### Messages 格式

#### 基础消息

```json
{
  "role": "system" | "user" | "assistant" | "tool",
  "content": "消息内容"
}
```

#### 多模态消息（支持图像）

```json
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "描述这张图片"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "https://example.com/image.jpg"
      }
    }
  ]
}
```

### 示例

#### 基础请求

```bash
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手。"},
      {"role": "user", "content": "你好，请介绍一下自己。"}
    ]
  }'
```

#### 流式输出

```bash
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "写一首关于春天的诗"}],
    "stream": true
  }'
```

**SSE 响应格式**：

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"春"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"风"},"finish_reason":null}]}

data: [DONE]
```

#### 函数调用

```bash
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "获取指定城市的天气信息",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {
                "type": "string",
                "description": "城市名称"
              }
            },
            "required": ["city"]
          }
        }
      }
    ]
  }'
```

**函数调用响应**：

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_xxx",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"北京\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

#### 多模态请求（图像理解）

```bash
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-vl-7b",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "这张图片里有什么？"},
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/image.jpg"
            }
          }
        ]
      }
    ]
  }'
```

#### Base64 图像

```bash
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-vl-7b",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "分析这张截图"},
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,iVBORw0KGgo..."
            }
          }
        ]
      }
    ]
  }'
```

### 响应格式

#### 非流式响应

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "qwen2.5-0.5b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "你好！我是你的AI助手..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 50,
    "total_tokens": 75
  }
}
```

#### 思维链模型响应（Qwen3）

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "最终回答内容",
      "reasoning_content": "思维链内容..."
    },
    "finish_reason": "stop"
  }]
}
```

---

## Completions

文本补全接口（非聊天模式）。

### 端点

```
POST /ai/v1/completions
```

### 请求参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `model` | string | 是 | 模型名称 |
| `prompt` | string/array | 是 | 输入提示 |
| `stream` | boolean | 否 | 是否流式输出 |
| `max_tokens` | integer | 否 | 最大输出 token 数 |
| `temperature` | number | 否 | 采样温度 |
| `top_p` | number | 否 | 核采样概率 |
| `stop` | array | 否 | 停止词列表 |

### 示例

```bash
curl -X POST http://localhost:8080/ai/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "prompt": "从前有座山，",
    "max_tokens": 100
  }'
```

---

## Embeddings

文本嵌入接口，用于生成文本向量表示。

### 端点

```
POST /ai/v1/embeddings
```

### 请求参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `input` | string/array | 是 | 输入文本 |
| `model` | string | 否 | 嵌入模型，默认 `bge-m3` |
| `encoding_format` | string | 否 | 编码格式：`float` 或 `base64` |

### 示例

#### 单文本嵌入

```bash
curl -X POST http://localhost:8080/ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "你好，世界！"
  }'
```

#### 批量嵌入

```bash
curl -X POST http://localhost:8080/ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["你好", "世界", "你好，世界！"]
  }'
```

### 响应格式

```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "index": 0,
    "embedding": [0.123, -0.456, 0.789, ...]
  }],
  "model": "bge-m3",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

---

## Rerank

重排序接口，用于对文档进行相关性排序。

### 端点

```
POST /ai/v1/rerank
```

### 请求参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `query` | string | 是 | 查询文本 |
| `documents` | array | 是 | 待排序的文档列表 |
| `top_n` | integer | 否 | 返回前 N 个结果 |

### 示例

```bash
curl -X POST http://localhost:8080/ai/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是机器学习？",
    "documents": [
      "机器学习是人工智能的一个分支...",
      "今天天气很好...",
      "深度学习是机器学习的一个子领域..."
    ],
    "top_n": 2
  }'
```

### 响应格式

```json
{
  "results": [{
    "index": 0,
    "relevance_score": 0.95
  }, {
    "index": 2,
    "relevance_score": 0.87
  }]
}
```

---

## Models

列出可用模型。

### 端点

```
GET /ai/v1/models
```

### 示例

```bash
curl http://localhost:8080/ai/v1/models
```

### 响应格式

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen2.5-0.5b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local"
    },
    {
      "id": "qwen2.5-vl-7b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local"
    }
  ]
}
```

---

## 错误处理

### 错误响应格式

```json
{
  "error": {
    "message": "错误描述",
    "type": "invalid_request_error",
    "code": "invalid_api_key"
  }
}
```

### 常见错误码

| HTTP 状态码 | 错误类型 | 描述 |
|-------------|----------|------|
| 400 | `invalid_request_error` | 请求参数错误 |
| 404 | `not_found_error` | 资源不存在 |
| 429 | `rate_limit_error` | 请求过于频繁 |
| 500 | `api_error` | 服务器内部错误 |
| 503 | `model_not_ready` | 模型正在加载 |

### 错误示例

```json
{
  "error": {
    "message": "Model 'unknown-model' not found",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

---

## SDK 使用示例

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed"  # 本地部署无需 API Key
)

# 基础对话
response = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[
        {"role": "user", "content": "你好！"}
    ]
)
print(response.choices[0].message.content)

# 流式输出
stream = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# 函数调用
response = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "北京天气"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }]
)
```

### JavaScript (fetch)

```javascript
// 基础对话
const response = await fetch('http://localhost:8080/ai/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'qwen2.5-0.5b',
    messages: [{ role: 'user', content: '你好！' }]
  })
});
const data = await response.json();
console.log(data.choices[0].message.content);

// 流式输出
const streamResponse = await fetch('http://localhost:8080/ai/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'qwen2.5-0.5b',
    messages: [{ role: 'user', content: '写一首诗' }],
    stream: true
  })
});
const reader = streamResponse.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value);
  console.log(chunk);
}
```

---

## 使用流选项 (stream_options)

支持 OpenAI 的 `stream_options` 参数：

```bash
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true,
    "stream_options": {
      "include_usage": true
    }
  }'
```

**带 usage 的流式响应**：

```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"role":"assistant"},"index":0}]}
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"你好"},"index":0}]}
...
data: {"id":"chatcmpl-xxx","choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}
data: [DONE]
```