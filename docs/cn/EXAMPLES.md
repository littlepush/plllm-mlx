# 使用示例集合

本文档提供 plllm-mlx 的丰富使用示例，涵盖各种常见场景。

## 目录

- [基础对话](#基础对话)
- [流式输出](#流式输出)
- [多轮对话](#多轮对话)
- [函数调用](#函数调用)
- [图像理解](#图像理解)
- [文本嵌入](#文本嵌入)
- [重排序](#重排序)
- [高级用法](#高级用法)
- [错误处理](#错误处理)
- [性能优化](#性能优化)

---

## 基础对话

### cURL 示例

```bash
# 最简单的对话请求
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下自己"}
    ]
  }'
```

### Python 示例

```python
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed"  # 本地部署无需 API Key
)

# 发送请求
response = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[
        {"role": "user", "content": "你好，请介绍一下自己"}
    ]
)

# 输出结果
print(response.choices[0].message.content)
```

### JavaScript 示例

```javascript
// 使用 fetch API
async function chat(message) {
  const response = await fetch('http://localhost:8080/ai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'qwen2.5-7b',
      messages: [{ role: 'user', content: message }],
    }),
  });
  
  const data = await response.json();
  return data.choices[0].message.content;
}

// 调用示例
chat('你好，请介绍一下自己').then(console.log);
```

### 带 System Prompt

```python
response = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[
        {
            "role": "system",
            "content": "你是一个专业的 Python 开发者，擅长编写简洁、高效的代码。"
        },
        {
            "role": "user",
            "content": "如何读取 JSON 文件？"
        }
    ]
)

print(response.choices[0].message.content)
```

---

## 流式输出

### Python 流式示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed"
)

# 流式请求
stream = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[{"role": "user", "content": "写一首关于春天的诗"}],
    stream=True  # 启用流式输出
)

# 逐块输出
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()  # 换行
```

### 带进度显示的流式

```python
import time

def stream_with_progress(prompt: str):
    """带进度显示的流式输出"""
    start_time = time.time()
    token_count = 0
    
    stream = client.chat.completions.create(
        model="qwen2.5-7b",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    print("生成中: ", end="", flush=True)
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            token_count += 1
    
    elapsed = time.time() - start_time
    print(f"\n\n生成完成: {token_count} tokens, {elapsed:.2f}s, {token_count/elapsed:.1f} tokens/s")

# 使用示例
stream_with_progress("详细介绍一下 Transformer 架构")
```

### JavaScript 流式示例

```javascript
async function streamChat(message) {
  const response = await fetch('http://localhost:8080/ai/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'qwen2.5-7b',
      messages: [{ role: 'user', content: message }],
      stream: true,
    }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let result = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    // 解析 SSE 格式
    const lines = chunk.split('\n').filter(line => line.startsWith('data: '));
    
    for (const line of lines) {
      const data = line.slice(6); // 移除 'data: ' 前缀
      if (data === '[DONE]') continue;
      
      try {
        const json = JSON.parse(data);
        const content = json.choices[0]?.delta?.content;
        if (content) {
          result += content;
          process.stdout.write(content); // 实时输出
        }
      } catch (e) {
        // 忽略解析错误
      }
    }
  }

  return result;
}
```

### cURL 流式示例

```bash
curl -X POST http://localhost:8080/ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [{"role": "user", "content": "讲一个故事"}],
    "stream": true
  }' \
  --no-buffer
```

---

## 多轮对话

### 完整多轮对话示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed"
)

def multi_turn_conversation():
    """多轮对话示例"""
    messages = [
        {
            "role": "system",
            "content": "你是一个有帮助的助手，专门回答技术问题。"
        }
    ]
    
    while True:
        # 获取用户输入
        user_input = input("\n用户: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("再见！")
            break
        
        # 添加用户消息
        messages.append({"role": "user", "content": user_input})
        
        # 获取模型回复
        response = client.chat.completions.create(
            model="qwen2.5-7b",
            messages=messages
        )
        
        assistant_message = response.choices[0].message.content
        print(f"助手: {assistant_message}")
        
        # 添加助手回复到历史
        messages.append({"role": "assistant", "content": assistant_message})

# 运行对话
multi_turn_conversation()
```

### 对话历史管理

```python
class ConversationManager:
    """对话历史管理器"""
    
    def __init__(self, system_prompt: str = None, max_history: int = 10):
        self.messages = []
        self.max_history = max_history
        
        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })
    
    def add_user_message(self, content: str):
        """添加用户消息"""
        self.messages.append({"role": "user", "content": content})
        self._trim_history()
    
    def add_assistant_message(self, content: str):
        """添加助手消息"""
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()
    
    def _trim_history(self):
        """修剪历史消息，保留最近的 N 条"""
        if len(self.messages) > self.max_history * 2 + 1:  # +1 for system
            # 保留 system 消息
            system_msg = self.messages[0] if self.messages[0]["role"] == "system" else None
            
            # 裁剪消息
            start = 1 if system_msg else 0
            self.messages = self.messages[start:]
            self.messages = self.messages[-(self.max_history * 2):]
            
            if system_msg:
                self.messages.insert(0, system_msg)
    
    def get_messages(self):
        return self.messages.copy()
    
    def clear(self):
        """清空对话历史（保留 system）"""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []


# 使用示例
conv = ConversationManager(
    system_prompt="你是一个代码助手，帮助用户解决编程问题。",
    max_history=5
)

# 第一轮
conv.add_user_message("如何在 Python 中读取文件？")
response = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=conv.get_messages()
)
conv.add_assistant_message(response.choices[0].message.content)

# 第二轮
conv.add_user_message("如果文件很大怎么办？")
response = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=conv.get_messages()
)
conv.add_assistant_message(response.choices[0].message.content)
```

---

## 函数调用

### 定义函数并调用

```python
from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed"
)

# 定义可用函数
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，默认为摄氏度"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "在网络上搜索信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# 发送请求
response = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[
        {"role": "user", "content": "北京和上海今天天气怎么样？"}
    ],
    tools=tools
)

# 检查是否有函数调用
message = response.choices[0].message

if message.tool_calls:
    print("模型决定调用函数:")
    for tool_call in message.tool_calls:
        print(f"  函数名: {tool_call.function.name}")
        print(f"  参数: {tool_call.function.arguments}")
        
        # 解析参数
        args = json.loads(tool_call.function.arguments)
        print(f"  解析后的参数: {args}")
```

### 完整的函数执行流程

```python
import json
from typing import Callable, Dict, Any

class FunctionCaller:
    """函数调用处理器"""
    
    def __init__(self, client):
        self.client = client
        self.functions: Dict[str, Callable] = {}
    
    def register(self, name: str, func: Callable):
        """注册函数"""
        self.functions[name] = func
    
    def execute(self, name: str, arguments: str) -> Any:
        """执行函数"""
        if name not in self.functions:
            return {"error": f"Unknown function: {name}"}
        
        try:
            args = json.loads(arguments)
            return self.functions[name](**args)
        except Exception as e:
            return {"error": str(e)}
    
    def chat_with_tools(
        self,
        messages: list,
        tools: list,
        max_iterations: int = 5
    ) -> str:
        """带函数调用的对话"""
        
        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model="qwen2.5-7b",
                messages=messages,
                tools=tools
            )
            
            message = response.choices[0].message
            
            # 如果没有函数调用，返回结果
            if not message.tool_calls:
                return message.content
            
            # 添加助手消息（包含函数调用）
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })
            
            # 执行函数并添加结果
            for tool_call in message.tool_calls:
                result = self.execute(
                    tool_call.function.name,
                    tool_call.function.arguments
                )
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })
        
        return "达到最大迭代次数"


# 使用示例
def get_weather(city: str, unit: str = "celsius") -> dict:
    """模拟天气 API"""
    # 实际应用中这里会调用真实的天气 API
    weather_data = {
        "北京": {"temp": 25, "condition": "晴"},
        "上海": {"temp": 28, "condition": "多云"},
    }
    
    data = weather_data.get(city, {"temp": 20, "condition": "未知"})
    
    if unit == "fahrenheit":
        data["temp"] = data["temp"] * 9/5 + 32
    
    return {
        "city": city,
        "temperature": data["temp"],
        "condition": data["condition"],
        "unit": unit
    }


# 初始化
caller = FunctionCaller(client)
caller.register("get_weather", get_weather)

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]

# 执行对话
messages = [{"role": "user", "content": "北京今天天气怎么样？适合户外活动吗？"}]
result = caller.chat_with_tools(messages, tools)
print(result)
```

---

## 图像理解

### 单张图像理解

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed"
)

# 使用图像 URL
response = client.chat.completions.create(
    model="qwen2.5-vl-7b",  # 使用视觉模型
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "描述这张图片中的内容"
                },
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

### Base64 图像

```python
import base64

def encode_image(image_path: str) -> str:
    """将图像编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 读取并编码图像
image_base64 = encode_image("screenshot.png")

response = client.chat.completions.create(
    model="qwen2.5-vl-7b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这个截图显示什么内容？"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

### 多图像对比

```python
# 对比两张图像
response = client.chat.completions.create(
    model="qwen2.5-vl-7b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "比较这两张图片的差异"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/before.jpg"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/after.jpg"}
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

### OCR 文字识别

```python
# 识别图像中的文字
response = client.chat.completions.create(
    model="qwen2.5-vl-7b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请识别并提取这张图片中的所有文字，保持原有格式"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/document.png"}
                }
            ]
        }
    ]
)

print("识别结果:")
print(response.choices[0].message.content)
```

---

## 文本嵌入

### 单文本嵌入

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed"
)

# 获取文本嵌入向量
response = client.embeddings.create(
    model="bge-m3",
    input="你好，世界！"
)

embedding = response.data[0].embedding
print(f"向量维度: {len(embedding)}")
print(f"向量前 5 个值: {embedding[:5]}")
```

### 批量嵌入

```python
# 批量获取嵌入
texts = [
    "机器学习是人工智能的一个分支",
    "深度学习使用神经网络进行学习",
    "自然语言处理是 AI 的重要应用"
]

response = client.embeddings.create(
    model="bge-m3",
    input=texts
)

for i, item in enumerate(response.data):
    print(f"文本 {i+1}: {texts[i][:20]}...")
    print(f"  向量维度: {len(item.embedding)}")
```

### 语义相似度计算

```python
import numpy as np

def cosine_similarity(vec1: list, vec2: list) -> float:
    """计算余弦相似度"""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_most_similar(query: str, documents: list) -> tuple:
    """找出与查询最相似的文档"""
    # 获取所有嵌入
    all_texts = [query] + documents
    response = client.embeddings.create(
        model="bge-m3",
        input=all_texts
    )
    
    query_embedding = response.data[0].embedding
    doc_embeddings = [d.embedding for d in response.data[1:]]
    
    # 计算相似度
    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]
    
    # 找出最相似的
    best_idx = np.argmax(similarities)
    return documents[best_idx], similarities[best_idx]


# 使用示例
query = "如何学习编程？"
documents = [
    "Python 是一门适合初学者的编程语言",
    "今天的天气很好",
    "编程入门建议从基础语法开始学习",
    "美食烹饪需要掌握火候"
]

best_doc, score = find_most_similar(query, documents)
print(f"查询: {query}")
print(f"最相似文档: {best_doc}")
print(f"相似度: {score:.4f}")
```

### 向量数据库集成示例

```python
# 简单的内存向量库示例
class SimpleVectorStore:
    """简单的向量存储"""
    
    def __init__(self, client):
        self.client = client
        self.documents = []
        self.embeddings = []
    
    def add(self, documents: list):
        """添加文档"""
        response = self.client.embeddings.create(
            model="bge-m3",
            input=documents
        )
        
        self.documents.extend(documents)
        self.embeddings.extend([d.embedding for d in response.data])
    
    def search(self, query: str, top_k: int = 3) -> list:
        """搜索相似文档"""
        # 获取查询向量
        response = self.client.embeddings.create(
            model="bge-m3",
            input=[query]
        )
        query_vec = response.data[0].embedding
        
        # 计算相似度
        similarities = [
            cosine_similarity(query_vec, doc_vec)
            for doc_vec in self.embeddings
        ]
        
        # 获取 top_k
        indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            {
                "document": self.documents[i],
                "score": similarities[i]
            }
            for i in indices
        ]


# 使用示例
store = SimpleVectorStore(client)

# 添加文档
store.add([
    "Python 是一种高级编程语言",
    "机器学习使用算法从数据中学习",
    "数据库用于存储和管理数据",
    "前端开发关注用户界面设计"
])

# 搜索
results = store.search("编程语言有哪些")
for r in results:
    print(f"文档: {r['document']}")
    print(f"相似度: {r['score']:.4f}\n")
```

---

## 重排序

### 基础重排序

```python
import requests

def rerank_documents(query: str, documents: list, top_n: int = 3) -> list:
    """重排序文档"""
    response = requests.post(
        "http://localhost:8080/ai/v1/rerank",
        json={
            "query": query,
            "documents": documents,
            "top_n": top_n
        }
    )
    
    results = response.json()["results"]
    
    return [
        {
            "document": documents[r["index"]],
            "score": r["relevance_score"]
        }
        for r in results
    ]


# 使用示例
query = "如何提高编程技能？"
documents = [
    "编程需要多练习，熟能生巧",
    "今天股市大涨",
    "阅读优秀的代码可以提高编程水平",
    "运动有助于健康"
]

ranked = rerank_documents(query, documents)
print(f"查询: {query}\n")
for i, r in enumerate(ranked, 1):
    print(f"{i}. {r['document']}")
    print(f"   相关性得分: {r['score']:.4f}\n")
```

### RAG 检索增强示例

```python
def rag_query(query: str, knowledge_base: list, top_k: int = 3) -> str:
    """RAG 检索增强生成"""
    
    # 1. 先用嵌入检索相关文档
    all_embeddings = client.embeddings.create(
        model="bge-m3",
        input=[query] + knowledge_base
    )
    
    query_vec = all_embeddings.data[0].embedding
    doc_vecs = [d.embedding for d in all_embeddings.data[1:]]
    
    similarities = [
        cosine_similarity(query_vec, v) for v in doc_vecs
    ]
    
    # 取 top_k * 2 作为候选
    candidate_indices = np.argsort(similarities)[::-1][:top_k * 2]
    candidates = [knowledge_base[i] for i in candidate_indices]
    
    # 2. 用 rerank 精排
    rerank_response = requests.post(
        "http://localhost:8080/ai/v1/rerank",
        json={
            "query": query,
            "documents": candidates,
            "top_n": top_k
        }
    )
    
    top_docs = [
        candidates[r["index"]]
        for r in rerank_response.json()["results"]
    ]
    
    # 3. 构建上下文并生成回答
    context = "\n\n".join(top_docs)
    
    response = client.chat.completions.create(
        model="qwen2.5-7b",
        messages=[
            {
                "role": "system",
                "content": "根据以下参考资料回答问题。如果参考资料中没有相关信息，请说明。"
            },
            {
                "role": "user",
                "content": f"参考资料：\n{context}\n\n问题：{query}"
            }
        ]
    )
    
    return response.choices[0].message.content


# 使用示例
knowledge_base = [
    "Python 由 Guido van Rossum 于 1991 年创建",
    "Python 支持面向对象、函数式和过程式编程",
    "Python 广泛应用于 Web 开发、数据科学和人工智能",
    "Java 是一种静态类型的编程语言",
    "JavaScript 主要用于前端开发"
]

answer = rag_query("Python 有什么特点？", knowledge_base)
print(answer)
```

---

## 高级用法

### 自定义采样参数

```python
def generate_with_params(prompt: str, preset: str = "default"):
    """根据预设配置生成"""
    
    presets = {
        "creative": {
            "temperature": 1.2,
            "top_p": 0.95,
            "repetition_penalty": 1.2
        },
        "precise": {
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 1.0
        },
        "balanced": {
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    }
    
    params = presets.get(preset, presets["balanced"])
    
    response = client.chat.completions.create(
        model="qwen2.5-7b",
        messages=[{"role": "user", "content": prompt}],
        **params
    )
    
    return response.choices[0].message.content


# 使用示例
print("创意模式:")
print(generate_with_params("写一个科幻故事开头", "creative"))

print("\n精确模式:")
print(generate_with_params("什么是 Python 的 GIL？", "precise"))
```

### 思维链模型（Qwen3）

```python
# Qwen3 支持思维链输出
response = client.chat.completions.create(
    model="qwen3-8b",
    messages=[
        {"role": "user", "content": "计算 123 * 456"}
    ]
)

message = response.choices[0].message

print("思维过程:")
if hasattr(message, 'reasoning_content') and message.reasoning_content:
    print(message.reasoning_content)

print("\n最终答案:")
print(message.content)
```

### 获取 Token 使用统计

```python
# 非流式请求获取 usage
response = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[{"role": "user", "content": "介绍一下 Python"}]
)

usage = response.usage
print(f"Prompt tokens: {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens: {usage.total_tokens}")
```

```python
# 流式请求获取 usage
stream = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[{"role": "user", "content": "介绍一下 Python"}],
    stream=True,
    stream_options={"include_usage": True}  # 启用 usage 统计
)

for chunk in stream:
    # 最后一个 chunk 包含 usage
    if hasattr(chunk, 'usage') and chunk.usage:
        print(f"\nToken 使用统计:")
        print(f"  Prompt: {chunk.usage.prompt_tokens}")
        print(f"  Completion: {chunk.usage.completion_tokens}")
        print(f"  Total: {chunk.usage.total_tokens}")
    elif chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## 错误处理

### 完善的错误处理

```python
from openai import APIError, APIConnectionError, RateLimitError
import time

def safe_chat(messages: list, max_retries: int = 3) -> str:
    """带错误处理的安全对话"""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="qwen2.5-7b",
                messages=messages
            )
            return response.choices[0].message.content
            
        except RateLimitError:
            wait_time = (attempt + 1) * 2
            print(f"请求过于频繁，等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
            
        except APIConnectionError:
            print(f"连接失败，尝试重新连接... ({attempt + 1}/{max_retries})")
            time.sleep(1)
            
        except APIError as e:
            print(f"API 错误: {e}")
            if attempt == max_retries - 1:
                return f"抱歉，服务暂时不可用: {e}"
            time.sleep(1)
    
    return "抱歉，多次尝试后仍然失败"


# 使用示例
result = safe_chat([{"role": "user", "content": "你好"}])
print(result)
```

### 超时处理

```python
import httpx

# 设置超时
client = OpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed",
    timeout=httpx.Timeout(60.0, connect=5.0)  # 连接超时 5s，总超时 60s
)

try:
    response = client.chat.completions.create(
        model="qwen2.5-7b",
        messages=[{"role": "user", "content": "写一个长故事"}],
        timeout=120.0  # 本次请求超时 120s
    )
except Exception as e:
    print(f"请求超时: {e}")
```

---

## 性能优化

### 并发请求

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    base_url="http://localhost:8080/ai/v1",
    api_key="not-needed"
)

async def async_chat(prompt: str) -> str:
    """异步对话"""
    response = await async_client.chat.completions.create(
        model="qwen2.5-7b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

async def batch_process(prompts: list) -> list:
    """批量处理"""
    tasks = [async_chat(p) for p in prompts]
    return await asyncio.gather(*tasks)

# 使用示例
prompts = [
    "什么是 Python？",
    "什么是 Java？",
    "什么是 JavaScript？"
]

results = asyncio.run(batch_process(prompts))
for p, r in zip(prompts, results):
    print(f"问题: {p}")
    print(f"回答: {r[:100]}...\n")
```

### 缓存对话历史

```python
from functools import lru_cache
import hashlib

def cache_key(messages: list) -> str:
    """生成缓存 key"""
    content = str(messages)
    return hashlib.md5(content.encode()).hexdigest()

# 简单的内存缓存
response_cache = {}

def cached_chat(messages: list) -> str:
    """带缓存的对话"""
    key = cache_key(messages)
    
    if key in response_cache:
        print("命中缓存")
        return response_cache[key]
    
    print("未命中缓存，调用 API")
    response = client.chat.completions.create(
        model="qwen2.5-7b",
        messages=messages
    )
    
    result = response.choices[0].message.content
    response_cache[key] = result
    return result
```

### Prefix KV Cache 最佳实践

```python
# 利用 Prefix KV Cache 的最佳实践
# 1. 保持 system prompt 一致
# 2. 有序添加消息

SYSTEM_PROMPT = "你是一个专业的技术顾问"

def efficient_multi_turn_chat():
    """高效的多轮对话"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    while True:
        user_input = input("用户: ")
        if user_input == "quit":
            break
        
        # 添加用户消息
        messages.append({"role": "user", "content": user_input})
        
        # 由于 system prompt 相同，Prefix KV Cache 会自动复用
        response = client.chat.completions.create(
            model="qwen2.5-7b",
            messages=messages
        )
        
        assistant_reply = response.choices[0].message.content
        print(f"助手: {assistant_reply}")
        
        # 添加助手回复
        messages.append({"role": "assistant", "content": assistant_reply})


# 运行
efficient_multi_turn_chat()
```

---

## 总结

本示例集涵盖了 plllm-mlx 的主要功能：

1. **基础对话**：简单易用的聊天接口
2. **流式输出**：实时响应，提升用户体验
3. **多轮对话**：支持上下文连续对话
4. **函数调用**：扩展模型能力，执行实际操作
5. **图像理解**：多模态输入支持
6. **文本嵌入**：语义搜索、相似度计算
7. **重排序**：精准检索，提升 RAG 效果
8. **高级用法**：自定义参数、思维链、并发等

更多问题请参考：
- [API 文档](./API.md)
- [架构设计](./ARCHITECTURE.md)
- [KV Cache 原理](./KV_CACHE.md)
- [配置说明](./CONFIGURATION.md)