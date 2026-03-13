# Usage Examples

This document provides practical, runnable code examples for using plllm-mlx.

## Table of Contents

- [Quick Start](#quick-start)
- [Chat Completions](#chat-completions)
- [Streaming Responses](#streaming-responses)
- [Function Calling](#function-calling)
- [Vision Language Models](#vision-language-models)
- [Embeddings](#embeddings)
- [Reranking](#reranking)
- [Model Management](#model-management)
- [Advanced Patterns](#advanced-patterns)

---

## Quick Start

### Start the Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start server (all services)
./install.sh

# Or start LLM only (faster, less memory)
./install.sh --llm-only

# Check health
curl http://localhost:8080/healthz
```

### First Request

```bash
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Chat Completions

### Python (OpenAI SDK)

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed"  # No authentication required
)

# Simple chat
response = client.chat.completions.create(
    model="qwen2.5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
```

### Python (httpx)

```python
import httpx
import json

async def chat(message: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8080/ai/v1/chat/completions",
            json={
                "model": "qwen2.5",
                "messages": [{"role": "user", "content": message}]
            },
            timeout=60.0
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]

# Usage
import asyncio
result = asyncio.run(chat("Hello!"))
print(result)
```

### JavaScript (fetch)

```javascript
async function chat(message) {
  const response = await fetch('http://localhost:8080/ai/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'qwen2.5',
      messages: [{ role: 'user', content: message }]
    })
  });
  
  const data = await response.json();
  return data.choices[0].message.content;
}

// Usage
chat('Hello!').then(console.log);
```

### cURL

```bash
# Basic chat
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in one sentence."}
    ]
  }'

# With parameters
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "messages": [{"role": "user", "content": "Write a haiku about coding."}],
    "temperature": 0.9,
    "max_tokens": 100
  }'
```

---

## Streaming Responses

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed"
)

# Stream chat
stream = client.chat.completions.create(
    model="qwen2.5",
    messages=[{"role": "user", "content": "Write a short story about a robot."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()  # Newline at end
```

### Python (httpx with SSE)

```python
import httpx
import json

async def stream_chat(message: str):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8080/ai/v1/chat/completions",
            json={
                "model": "qwen2.5",
                "messages": [{"role": "user", "content": message}],
                "stream": True
            },
            timeout=120.0
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    content = chunk["choices"][0].get("delta", {}).get("content")
                    if content:
                        print(content, end="", flush=True)
    print()

# Usage
import asyncio
asyncio.run(stream_chat("Tell me a joke."))
```

### JavaScript (Streaming)

```javascript
async function streamChat(message) {
  const response = await fetch('http://localhost:8080/ai/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'qwen2.5',
      messages: [{ role: 'user', content: message }],
      stream: true
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n').filter(line => line.startsWith('data: '));

    for (const line of lines) {
      const data = line.slice(6);
      if (data === '[DONE]') continue;

      try {
        const parsed = JSON.parse(data);
        const content = parsed.choices[0]?.delta?.content;
        if (content) {
          process.stdout.write(content);
        }
      } catch (e) {
        // Skip invalid JSON
      }
    }
  }
}

// Usage
streamChat('Write a poem about the sea.');
```

### cURL (Streaming)

```bash
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "messages": [{"role": "user", "content": "Count from 1 to 10."}],
    "stream": true
  }'
```

---

## Function Calling

### Define Tools and Call

```python
from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed"
)

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Make request
response = client.chat.completions.create(
    model="qwen2.5",
    messages=[
        {"role": "user", "content": "What's the weather like in Tokyo?"}
    ],
    tools=tools
)

# Check if model wants to call a function
message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
        
        # Parse and execute the function
        args = json.loads(tool_call.function.arguments)
        if tool_call.function.name == "get_weather":
            result = get_weather(args["location"])
            print(f"Result: {result}")

def get_weather(location: str) -> dict:
    """Mock weather function."""
    # In real code, call a weather API
    return {
        "location": location,
        "temperature": 22,
        "condition": "Sunny"
    }
```

### Multi-Turn Function Calling

```python
def run_conversation():
    messages = [
        {"role": "user", "content": "What's the weather in Paris and London?"}
    ]
    
    # First call - model decides to call function
    response = client.chat.completions.create(
        model="qwen2.5",
        messages=messages,
        tools=tools
    )
    
    message = response.choices[0].message
    messages.append(message)
    
    # Process tool calls
    if message.tool_calls:
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = get_weather(args["location"])
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
    
    # Continue conversation
    response = client.chat.completions.create(
        model="qwen2.5",
        messages=messages
    )
    
    return response.choices[0].message.content

print(run_conversation())
```

---

## Vision Language Models

### Image Analysis

```python
from openAI import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed"
)

# Analyze image from URL
response = client.chat.completions.create(
    model="qwen2.5-vl",  # Vision model category
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

### Base64 Image

```python
import base64

# Read and encode image
with open("image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="qwen2.5-vl",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

### Multiple Images

```python
response = client.chat.completions.create(
    model="qwen2.5-vl",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image1.jpg"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image2.jpg"}
                }
            ]
        }
    ]
)
```

---

## Embeddings

### Single Text Embedding

```python
import httpx

async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8080/ai/v1/embeddings",
            json={"input": text}
        )
        data = response.json()
        return data["data"][0]["embedding"]

# Usage
import asyncio
embedding = asyncio.run(get_embedding("Hello world"))
print(f"Embedding dimension: {len(embedding)}")
```

### Batch Embeddings

```python
import httpx

async def get_embeddings(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8080/ai/v1/embeddings",
            json={"input": texts}
        )
        data = response.json()
        return [item["embedding"] for item in data["data"]]

# Usage
texts = ["Hello", "World", "Test"]
embeddings = asyncio.run(get_embeddings(texts))
print(f"Got {len(embeddings)} embeddings")
```

### Semantic Similarity

```python
import httpx
import numpy as np

async def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def find_similar(query: str, documents: list[str]) -> list[tuple[str, float]]:
    """Find documents similar to query."""
    async with httpx.AsyncClient() as client:
        # Get embeddings for query and documents
        response = await client.post(
            "http://localhost:8080/ai/v1/embeddings",
            json={"input": [query] + documents}
        )
        data = response.json()
        
        embeddings = [item["embedding"] for item in data["data"]]
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]
        
        # Calculate similarities
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            sim = await cosine_similarity(query_embedding, doc_emb)
            similarities.append((documents[i], sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

# Usage
async def main():
    query = "What is machine learning?"
    documents = [
        "Machine learning is a branch of artificial intelligence.",
        "Python is a popular programming language.",
        "Deep learning uses neural networks.",
        "The weather is nice today."
    ]
    
    results = await find_similar(query, documents)
    for doc, sim in results:
        print(f"{sim:.3f}: {doc}")

asyncio.run(main())
```

---

## Reranking

### Rerank Search Results

```python
import httpx

async def rerank_documents(query: str, documents: list[str]) -> list[tuple[str, float]]:
    """Rerank documents by relevance to query."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8080/ai/v1/rerank",
            json={
                "query": query,
                "documents": documents
            }
        )
        data = response.json()
        scores = data["data"]["scores"]
        
        # Pair documents with scores
        ranked = list(zip(documents, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

# Usage
async def main():
    query = "What causes rain?"
    documents = [
        "Rain occurs when water vapor condenses in clouds.",
        "The sun rises in the east.",
        "Precipitation happens when clouds become saturated.",
        "Cars run on gasoline or electricity."
    ]
    
    results = await rerank_documents(query, documents)
    for doc, score in results:
        print(f"{score:.3f}: {doc[:50]}...")

asyncio.run(main())
```

---

## Model Management

### List Available Models

```python
import httpx

async def list_models() -> list[dict]:
    """List all available models."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8080/api/v1/model/list"
        )
        return response.json()["data"]

# Usage
models = asyncio.run(list_models())
for model in models:
    print(f"- {model['model_name']} (loader: {model['model_loader']})")
```

### Download New Model

```python
import httpx
import asyncio

async def download_model(model_id: str) -> str:
    """Start model download and return task ID."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8080/api/v1/model/download",
            json={"model_id": model_id}
        )
        return response.json()["task_id"]

async def check_download_status(task_id: str) -> dict:
    """Check download status."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8080/api/v1/model/download/status/{task_id}"
        )
        return response.json()

async def download_and_wait(model_id: str):
    """Download model and wait for completion."""
    task_id = await download_model(model_id)
    print(f"Started download: {task_id}")
    
    while True:
        status = await check_download_status(task_id)
        print(f"Status: {status['status']} - {status['message']}")
        
        if status["status"] in ["completed", "failed"]:
            return status
        
        await asyncio.sleep(5)

# Usage
asyncio.run(download_and_wait("mlx-community/Qwen2.5-7B-Instruct-4bit"))
```

### Manage Categories

```python
import httpx

async def create_category(name: str, model: str):
    """Create a new category."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8080/api/v1/category/add",
            json={
                "name": name,
                "type": "chat",
                "model": model
            }
        )
        return response.json()

async def list_categories() -> list[dict]:
    """List all categories."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8080/api/v1/category/list"
        )
        return response.json()["data"]

# Usage
async def main():
    # Create category
    await create_category("my-model", "Qwen/Qwen2.5-7B-Instruct")
    
    # List categories
    categories = await list_categories()
    for cat in categories:
        print(f"- {cat['name']} -> {cat['model']}")

asyncio.run(main())
```

### Update Model Configuration

```python
import httpx

async def update_config(model_name: str, key: str, value):
    """Update model configuration."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8080/api/v1/model/update/config",
            json={
                "model_name": model_name,
                "key": key,
                "value": value
            }
        )
        return response.json()

# Usage
async def main():
    # Set temperature
    await update_config("Qwen/Qwen2.5-7B-Instruct", "temperature", 0.7)
    
    # Set max tokens
    await update_config("Qwen/Qwen2.5-7B-Instruct", "max_output_tokens", 4096)
    
    # Enable KV quantization
    await update_config("Qwen/Qwen2.5-7B-Instruct", "kv_bits", 4)
    
    print("Configuration updated")

asyncio.run(main())
```

---

## Advanced Patterns

### Conversation with Context

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed"
)

class Conversation:
    def __init__(self, system_prompt: str = None):
        self.messages = []
        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })
    
    def chat(self, user_input: str) -> str:
        # Add user message
        self.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Get response
        response = client.chat.completions.create(
            model="qwen2.5",
            messages=self.messages
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add to history
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def clear(self):
        """Clear conversation history except system prompt."""
        self.messages = [m for m in self.messages if m["role"] == "system"]

# Usage
conv = Conversation("You are a helpful coding assistant.")
print(conv.chat("How do I read a file in Python?"))
print(conv.chat("Can you show me how to write to a file too?"))
```

### Retry with Backoff

```python
import httpx
import asyncio
import random

async def chat_with_retry(
    message: str,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> str:
    """Chat with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8080/ai/v1/chat/completions",
                    json={
                        "model": "qwen2.5",
                        "messages": [{"role": "user", "content": message}]
                    },
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                elif response.status_code == 503:
                    # Server busy, retry
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Server busy, retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    raise Exception(f"Error: {response.status_code}")
                    
        except httpx.TimeoutException:
            delay = base_delay * (2 ** attempt)
            print(f"Timeout, retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)
    
    raise Exception("Max retries exceeded")

# Usage
result = asyncio.run(chat_with_retry("Hello!"))
print(result)
```

### Async Batch Processing

```python
import asyncio
import httpx

async def process_single(client: httpx.AsyncClient, text: str) -> str:
    """Process a single text."""
    response = await client.post(
        "http://localhost:8080/ai/v1/chat/completions",
        json={
            "model": "qwen2.5",
            "messages": [{"role": "user", "content": f"Summarize: {text}"}],
            "max_tokens": 100
        }
    )
    return response.json()["choices"][0]["message"]["content"]

async def process_batch(texts: list[str], max_concurrent: int = 5) -> list[str]:
    """Process multiple texts concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_process(client: httpx.AsyncClient, text: str) -> str:
        async with semaphore:
            return await process_single(client, text)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [limited_process(client, text) for text in texts]
        return await asyncio.gather(*tasks)

# Usage
texts = [
    "Long text 1 about machine learning...",
    "Long text 2 about natural language processing...",
    "Long text 3 about computer vision...",
]

summaries = asyncio.run(process_batch(texts, max_concurrent=3))
for i, summary in enumerate(summaries):
    print(f"Summary {i+1}: {summary}")
```

### Custom Session with Headers

```python
from openai import OpenAI
import httpx

# Create custom httpx client with timeout
http_client = httpx.Client(
    timeout=httpx.Timeout(120.0, connect=30.0),
    limits=httpx.Limits(max_connections=100)
)

client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed",
    http_client=http_client
)

# Use the client
response = client.chat.completions.create(
    model="qwen2.5",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)

# Clean up
http_client.close()
```

---

*Last updated: 2024*