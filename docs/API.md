# API Reference

This document provides detailed API reference for all endpoints in plllm-mlx.

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Chat Completions API](#chat-completions-api)
- [Completions API](#completions-api)
- [Embeddings API](#embeddings-api)
- [Rerank API](#rerank-api)
- [Models API](#models-api)
- [Category API](#category-api)
- [Loader API](#loader-api)
- [Step Processor API](#step-processor-api)
- [Model Manager API](#model-manager-api)
- [Error Handling](#error-handling)

---

## Overview

### Base URL

```
http://localhost:8080
```

### API Prefixes

| Prefix | Description |
|--------|-------------|
| `/ai/v1` | OpenAI-compatible endpoints |
| `/api/v1` | Management endpoints |

### Content Types

- Request: `application/json`
- Streaming Response: `text/event-stream`
- Non-streaming Response: `application/json`

---

## Authentication

Currently, no authentication is required. All endpoints are open.

> **Note**: In production deployments, consider adding authentication via a reverse proxy or API gateway.

---

## Chat Completions API

OpenAI-compatible chat completions endpoint.

### Create Chat Completion

```http
POST /ai/v1/chat/completions
```

**Request Body:**

```json
{
  "model": "string (required)",
  "messages": [
    {
      "role": "system | user | assistant | tool",
      "content": "string | array",
      "name": "string (optional)",
      "tool_calls": "array (optional)",
      "tool_call_id": "string (optional)"
    }
  ],
  "stream": "boolean (default: false)",
  "stream_options": {
    "include_usage": "boolean (default: false)"
  },
  "temperature": "number (default: 0.8)",
  "top_p": "number (default: 0)",
  "top_k": "number (default: 100)",
  "max_tokens": "number (default: 16384)",
  "repetition_penalty": "number (default: 1.1)",
  "tools": "array (optional)",
  "tool_choice": "string | object (optional)"
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Category name or model ID |
| `messages` | array | required | Conversation messages |
| `stream` | boolean | false | Enable streaming response |
| `stream_options` | object | null | Streaming options |
| `temperature` | number | 0.8 | Sampling temperature (0-2) |
| `top_p` | number | 0 | Nucleus sampling threshold |
| `top_k` | number | 100 | Top-k sampling |
| `max_tokens` | number | 16384 | Maximum output tokens |
| `repetition_penalty` | number | 1.1 | Repetition penalty |
| `tools` | array | null | Tool definitions for function calling |
| `tool_choice` | string/object | null | Tool selection mode |

**Non-streaming Response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat_completion",
  "created": 1234567890,
  "model": "qwen2.5-7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?",
        "reasoning": null,
        "tool_calls": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

**Streaming Response (SSE):**

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"qwen2.5-7b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"qwen2.5-7b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"qwen2.5-7b","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: [DONE]
```

**Example - Basic Chat:**

```bash
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'
```

**Example - Streaming:**

```bash
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "messages": [
      {"role": "user", "content": "Tell me a joke."}
    ],
    "stream": true
  }'
```

**Example - Vision (with image):**

```bash
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
      }
    ]
  }'
```

**Example - Function Calling:**

```bash
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "messages": [
      {"role": "user", "content": "What is the weather in Tokyo?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

### Response Fields

| Field | Description |
|-------|-------------|
| `id` | Unique completion ID |
| `object` | Object type (`chat_completion` or `chat.completion.chunk`) |
| `created` | Unix timestamp |
| `model` | Model name used |
| `choices` | Array of completion choices |
| `choices[].message` | The generated message |
| `choices[].message.content` | Generated text (null if tool_calls) |
| `choices[].message.reasoning` | Thinking content (for Qwen3 models) |
| `choices[].message.tool_calls` | Function calls (if any) |
| `choices[].finish_reason` | `stop`, `length`, or `tool_calls` |
| `usage` | Token usage statistics |

---

## Completions API

Legacy completions endpoint (non-chat).

### Create Completion

```http
POST /ai/v1/completions
```

**Request Body:**

```json
{
  "model": "string (required)",
  "prompt": "string (required)",
  "stream": "boolean (default: false)",
  "max_tokens": "number (default: 16384)",
  "temperature": "number (default: 0.8)",
  "top_p": "number (default: 0)",
  "top_k": "number (default: 100)"
}
```

**Response:**

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1234567890,
  "model": "qwen2.5",
  "choices": [
    {
      "index": 0,
      "text": "generated text here",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/ai/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "prompt": "Once upon a time",
    "max_tokens": 100
  }'
```

---

## Embeddings API

Generate text embeddings using BGE-M3 model.

### Create Embedding

```http
POST /ai/v1/embeddings
```

**Request Body:**

```json
{
  "input": "string | array of strings",
  "model": "string (optional)"
}
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1, 0.2, 0.3, ...]
    }
  ],
  "model": "bge-m3"
}
```

**Example - Single Text:**

```bash
curl -X POST http://localhost:8080/ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world"
  }'
```

**Example - Batch:**

```bash
curl -X POST http://localhost:8080/ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello", "World", "Test"]
  }'
```

### Single Embed Endpoint

```http
POST /ai/v1/embed
```

**Request Body:**

```json
{
  "prompt": "string (required)",
  "model": "string (optional)"
}
```

**Response:**

```json
{
  "embedding": [0.1, 0.2, 0.3, ...],
  "model": "bge-m3"
}
```

### Classify Endpoint

Classify text against labels using embeddings.

```http
POST /ai/v1/classify
```

**Request Body:**

```json
{
  "text": "string (required)",
  "labels": ["label1", "label2", ...],
  "model": "string (optional)"
}
```

**Response:**

```json
{
  "model": "bge-m3",
  "data": {
    "label1": 0.8,
    "label2": 0.2
  }
}
```

---

## Rerank API

Rerank documents by relevance to a query.

### Rerank Documents

```http
POST /ai/v1/rerank
```

**Request Body:**

```json
{
  "query": "string (required)",
  "documents": ["doc1", "doc2", ...],
  "model": "string (optional)"
}
```

**Response:**

```json
{
  "model": "bge-reranker",
  "data": {
    "scores": [0.9, 0.5, 0.3]
  }
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/ai/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a branch of AI.",
      "Python is a programming language.",
      "Deep learning uses neural networks."
    ]
  }'
```

---

## Models API

Manage models and their configurations.

### List Models

```http
GET /ai/v1/models
```

**Response:**

```json
{
  "data": [
    {
      "id": "qwen2.5",
      "object": "model",
      "owned_by": "plllm"
    }
  ],
  "object": "list"
}
```

### List Local Models

```http
GET /api/v1/model/list
```

**Response:**

```json
{
  "data": [
    {
      "model_name": "Qwen/Qwen2.5-7B-Instruct",
      "model_loader": "mlx",
      "step_processor": "qwen3_thinking",
      "config": {
        "temperature": 0.8,
        "max_tokens": 16384
      }
    }
  ]
}
```

### Reload Models

Scan HuggingFace cache and reload model list.

```http
POST /api/v1/model/reload
```

**Response:**

```json
{
  "status": "OK"
}
```

### Load Model

Load a specific model into memory.

```http
POST /api/v1/model/load
```

**Request Body:**

```json
{
  "model_name": "string (required)"
}
```

**Response:**

```json
{
  "status": "OK"
}
```

### Unload Model

Unload a model from memory.

```http
POST /api/v1/model/unload
```

**Request Body:**

```json
{
  "model_name": "string (required)"
}
```

**Response:**

```json
{
  "status": "OK"
}
```

### Update Model Config

Update a configuration parameter for a model.

```http
POST /api/v1/model/update/config
```

**Request Body:**

```json
{
  "model_name": "string (required)",
  "key": "string (required)",
  "value": "any (required)"
}
```

**Response:**

```json
{
  "status": "OK"
}
```

### Update Step Processor

Change the step processor for a model.

```http
POST /api/v1/model/update/stepprocessor
```

**Request Body:**

```json
{
  "model_name": "string (required)",
  "step_processor": "string (required)"
}
```

**Response:**

```json
{
  "status": "OK"
}
```

### Update Model Loader

Change the model loader for a model.

```http
POST /api/v1/model/update/modelloader
```

**Request Body:**

```json
{
  "model_name": "string (required)",
  "model_loader": "string (required)"
}
```

**Response:**

```json
{
  "status": "OK"
}
```

---

## Category API

Manage categories (model-to-endpoint mappings).

### List Categories

```http
GET /api/v1/category/list
```

**Response:**

```json
{
  "data": [
    {
      "name": "qwen2.5",
      "type": "chat",
      "model": "Qwen/Qwen2.5-7B-Instruct"
    }
  ]
}
```

### Add Category

```http
POST /api/v1/category/add
```

**Request Body:**

```json
{
  "name": "string (required)",
  "type": "string (required)",
  "model": "string (required)"
}
```

**Response:**

```json
{
  "status": "OK"
}
```

### Delete Category

```http
POST /api/v1/category/delete/{category_name}
```

**Response:**

```json
{
  "status": "OK"
}
```

### Change Category Model

```http
POST /api/v1/category/chmodel/{category_name}
```

**Request Body:**

```json
{
  "model": "string (required)"
}
```

**Response:**

```json
{
  "status": "OK"
}
```

### Get Category Status

```http
GET /api/v1/category/status
```

**Response:**

```json
{
  "data": [
    {
      "name": "qwen2.5",
      "type": "chat",
      "model_name": "Qwen/Qwen2.5-7B-Instruct",
      "model_loader": "mlx",
      "step_processor": "qwen3_thinking",
      "is_loaded": true,
      "verbose": false
    }
  ]
}
```

---

## Loader API

Control model loaders.

### List Loaders

```http
GET /api/v1/loader/list
```

**Response:**

```json
{
  "data": ["mlx", "mlxvlm"]
}
```

### Load Model via Loader

```http
POST /api/v1/loader/load
```

**Request Body:**

```json
{
  "model_name": "string (required)"
}
```

**Response:**

```json
{
  "status": "OK",
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "is_loaded": true
}
```

### Unload Model via Loader

```http
POST /api/v1/loader/unload
```

**Request Body:**

```json
{
  "model_name": "string (required)"
}
```

**Response:**

```json
{
  "status": "OK",
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "is_loaded": false
}
```

---

## Step Processor API

List available step processors.

### List Step Processors

```http
GET /api/v1/stepprocessor/list
```

**Response:**

```json
{
  "data": ["base", "openai", "qwen3_thinking"]
}
```

---

## Model Manager API

Search, download, and manage models from HuggingFace.

### Search Models

```http
GET /api/v1/model/search?keyword=string
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keyword` | string | "mlx" | Search keyword |

**Response:**

```json
{
  "data": [
    {
      "id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
      "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
      "downloads": 50000,
      "likes": 200,
      "tags": ["mlx", "qwen", "text-generation"],
      "author": "mlx-community",
      "pipeline_tag": "text-generation"
    }
  ],
  "total": 50
}
```

### Download Model

```http
POST /api/v1/model/download
```

**Request Body:**

```json
{
  "model_id": "string (required)",
  "model_loader": "string (default: mlx)"
}
```

**Response:**

```json
{
  "task_id": "uuid-string",
  "status": "pending",
  "message": "Download task created for Qwen/Qwen2.5-7B-Instruct"
}
```

### Check Download Status

```http
GET /api/v1/model/download/status/{task_id}
```

**Response:**

```json
{
  "task_id": "uuid-string",
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "status": "completed",
  "message": "Model downloaded successfully",
  "model_name": "Qwen/Qwen2.5-7B-Instruct"
}
```

**Status Values:**

- `pending` - Task created, not started
- `downloading` - Download in progress
- `completed` - Download finished
- `failed` - Download failed

### Delete Model

```http
POST /api/v1/model/delete
```

**Request Body:**

```json
{
  "model_name": "string (required)"
}
```

**Response:**

```json
{
  "status": "OK",
  "message": "Model Qwen/Qwen2.5-7B-Instruct deleted successfully"
}
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message description"
}
```

### Common HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (missing/invalid parameters) |
| 404 | Not Found (model/category not found) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (server busy) |

### Error Examples

**Missing Model:**

```json
{
  "detail": "category qwen3 not found"
}
```

**Invalid Request:**

```json
{
  "detail": "model field is required"
}
```

**Server Busy:**

```json
{
  "detail": "Server is busy, please try again later"
}
```

---

## Usage Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed"  # No auth required
)

# Chat completion
response = client.chat.completions.create(
    model="qwen2.5",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen2.5",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### JavaScript (fetch)

```javascript
// Chat completion
const response = await fetch('http://localhost:8080/ai/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'qwen2.5',
    messages: [{ role: 'user', content: 'Hello!' }],
    stream: false
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);

// Streaming
const streamResponse = await fetch('http://localhost:8080/ai/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'qwen2.5',
    messages: [{ role: 'user', content: 'Hello!' }],
    stream: true
  })
});

const reader = streamResponse.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n').filter(line => line.startsWith('data: '));
  
  for (const line of lines) {
    const data = line.slice(6);
    if (data === '[DONE]') continue;
    
    const parsed = JSON.parse(data);
    const content = parsed.choices[0]?.delta?.content;
    if (content) console.log(content);
  }
}
```

### cURL

```bash
# Chat completion
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5", "messages": [{"role": "user", "content": "Hello!"}]}'

# Streaming
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'

# Embedding
curl -X POST http://localhost:8080/ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world"}'

# List models
curl http://localhost:8080/ai/v1/models
```

---

*Last updated: 2024*