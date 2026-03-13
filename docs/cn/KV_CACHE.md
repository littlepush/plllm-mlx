# KV Cache 原理详解

本文档深入解析 plllm-mlx 的 Prefix KV Cache 实现原理，包含多个实际场景示例。

## 目录

- [什么是 KV Cache](#什么是-kv-cache)
- [为什么需要 Prefix KV Cache](#为什么需要-prefix-kv-cache)
- [核心组件](#核心组件)
- [实现原理](#实现原理)
- [缓存匹配策略](#缓存匹配策略)
- [内存管理](#内存管理)
- [实际场景示例](#实际场景示例)
- [配置参数](#配置参数)
- [性能调优](#性能调优)
- [故障排查](#故障排查)

---

## 什么是 KV Cache

### 背景：Transformer 推理过程

在 Transformer 模型的推理过程中，每次生成一个新 token 时，都需要重新计算所有之前 token 的注意力（Attention）：

```
输入: "今天天气很好"
生成第1个token: 计算 [今, 天, 天, 气, 很, 好] 的注意力 → "适合"
生成第2个token: 计算 [今, 天, 天, 气, 很, 好, 适, 合] 的注意力 → "出门"
生成第3个token: 计算 [今, 天, 天, 气, 很, 好, 适, 合, 出, 门] 的注意力 → "散步"
```

每次都要重新计算之前的所有注意力，这是巨大的计算浪费。

### KV Cache 的作用

KV Cache 存储每层的 Key 和 Value 矩阵，避免重复计算：

```python
# 没有 KV Cache
for token in generate_tokens():
    # 每次都要重新计算整个序列的 K, V
    k, v = compute_attention(all_previous_tokens)
    new_token = decode(k, v)

# 有 KV Cache
cached_k, cached_v = kv_cache
for token in generate_tokens():
    # 只计算新 token 的 K, V
    new_k, new_v = compute_attention(new_token)
    # 追加到缓存
    cached_k = concat(cached_k, new_k)
    cached_v = concat(cached_v, new_v)
    new_token = decode(cached_k, cached_v)
```

### 内存估算

以 Qwen2.5-7B 为例：

| 参数 | 值 |
|------|-----|
| 层数 | 28 层 |
| 隐藏维度 | 3584 |
| 注意力头数 | 28 |
| 每个 token 每层 KV 大小 | 2 × 3584 × 2 bytes (fp16) ≈ 14KB |
| 4096 token 的 KV Cache | 28 × 14KB × 4096 ≈ 1.6GB |
| 16384 token 的 KV Cache | ≈ 6.4GB |

---

## 为什么需要 Prefix KV Cache

### 问题场景

考虑一个多轮对话场景：

```
第1轮：
用户: "请帮我写一篇关于人工智能的文章"
助手: [生成文章...]

第2轮：
用户: "把文章改得更简短一些"
助手: [修改文章...]

第3轮：
用户: "把语言改得更正式一些"
助手: [再次修改...]
```

**没有 Prefix KV Cache 时**：
- 每轮对话都要重新计算所有历史消息的 prefill
- 第3轮需要 prefill 第1轮、第2轮的所有内容
- 大量重复计算，延迟累积

**有 Prefix KV Cache 时**：
- 第1轮：完整 prefill，缓存结果
- 第2轮：复用第1轮的缓存，只 prefill 新消息
- 第3轮：复用第1轮+第2轮的缓存，只 prefill 新消息
- 计算量大幅减少

### 性能对比

| 场景 | 无缓存 | 有缓存 | 提升 |
|------|--------|--------|------|
| 10轮对话，每轮500 token | 5000 token prefill | 500 token prefill | 10x |
| System prompt 2000 token | 每次重新计算 | 只计算一次 | ∞ |

---

## 核心组件

### PlChain（消息链对象）

```python
class PlChain:
    """
    消息链对象，表示一个完整的对话上下文
    
    属性:
        node_ids: List[str] - 消息ID列表，按对话顺序排列
        cache_item: KV Cache - 正式缓存，用于消息链匹配
        temp_cache_item: KV Cache - 临时缓存，存储assistant回复
    """
    def __init__(self):
        self.node_ids: List[str] = []
        self.cache_item: Optional[PlKVCache] = None
        self.temp_cache_item: Optional[PlKVCache] = None
```

**消息ID的生成**：

```python
import hashlib

def generate_msg_id(content: str, begin_token: str, end_token: str) -> str:
    """
    使用消息完整内容的MD5作为唯一ID
    
    Args:
        content: 消息内容（不含begin/end tokens）
        begin_token: 消息开始token，如 "<|im_start|>"
        end_token: 消息结束token，如 "<|im_end|>"
    
    Returns:
        32位MD5哈希字符串
    """
    full_content = f"{begin_token}{content}{end_token}"
    return hashlib.md5(full_content.encode()).hexdigest()
```

### PlChainCache（缓存容器）

```python
from collections import OrderedDict

class PlChainCache:
    """
    基于OrderedDict实现的LRU缓存容器
    
    特性:
        - O(1) 时间复杂度的查找、插入、删除
        - 支持最长链搜索（部分匹配）
        - 自动内存管理
    """
    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[str, PlChain] = OrderedDict()
        self._max_size = max_size
    
    def get(self, chain_key: str) -> Optional[PlChain]:
        """获取缓存，并更新访问顺序"""
        if chain_key in self._cache:
            # 移动到末尾（最近使用）
            self._cache.move_to_end(chain_key)
            return self._cache[chain_key]
        return None
    
    def put(self, chain_key: str, chain: PlChain):
        """添加缓存，超出容量时淘汰最旧的"""
        if chain_key in self._cache:
            self._cache.move_to_end(chain_key)
        else:
            self._cache[chain_key] = chain
            if len(self._cache) > self._max_size:
                # 删除最旧的（队列头部）
                self._cache.popitem(last=False)
    
    def find_longest_prefix(self, msg_ids: List[str]) -> Optional[PlChain]:
        """
        查找最长匹配的消息链
        
        例如:
            当前消息链: [msg1, msg2, msg3, msg4]
            缓存中有: [msg1, msg2, msg3]
            返回: 匹配3条消息的链
        """
        for i in range(len(msg_ids), 0, -1):
            prefix_key = self._make_key(msg_ids[:i])
            if prefix_key in self._cache:
                return self._cache[prefix_key]
        return None
```

### PlMessageBasedKVCache（主缓存管理器）

```python
class PlMessageBasedKVCache:
    """
    基于消息的Prefix KV Cache管理器
    
    职责:
        - 消息分割：将prompt按begin/end tokens分割
        - ID生成：为每条消息生成唯一ID
        - 缓存查询：查找最长匹配链
        - 增量prefill：只处理未缓存的消息
        - 内存管理：自动淘汰旧缓存
    """
    
    def __init__(self, config: dict = None):
        self.enable_prefix_cache = config.get('enable_prefix_cache', True)
        self.begin_tokens = config.get('begin_tokens', ['<|im_start|>'])
        self.end_tokens = config.get('end_tokens', ['<|im_end|>'])
        self.chain_cache = PlChainCache()
        
    def split_prompt_by_messages(self, prompt: str) -> List[str]:
        """将prompt分割为消息列表"""
        # 实现细节...
        
    def lookup_cache(self, prompt: str) -> Tuple[Optional[PlChain], List[str]]:
        """
        查询缓存
        
        Returns:
            matched_chain: 匹配的消息链（可能为None）
            remaining_messages: 未匹配的消息列表
        """
        # 实现细节...
```

---

## 实现原理

### 1. 消息分割

将完整的 prompt 字符串分割为独立的消息：

```
原始 prompt:
"<|im_start|>system\n你是一个助手<|im_end|><|im_start|>user\n你好<|im_end|><|im_start|>assistant\n你好！<|im_end|>"

分割结果:
[
    "system\n你是一个助手",
    "user\n你好",
    "assistant\n你好！"
]
```

**分割算法**：

```python
def split_prompt_by_messages(self, prompt: str) -> List[str]:
    """
    使用begin/end tokens分割prompt
    
    支持多种token格式，如:
        - Qwen格式: <|im_start|>, <|im_end|>
        - 通用格式: <|start|>, <|end|>
    """
    messages = []
    current_start = 0
    
    # 找到所有begin token的位置
    for begin_token in self.begin_tokens:
        idx = 0
        while True:
            start = prompt.find(begin_token, idx)
            if start == -1:
                break
            
            # 找到对应的end token
            for end_token in self.end_tokens:
                end = prompt.find(end_token, start + len(begin_token))
                if end != -1:
                    # 提取消息内容
                    content = prompt[start + len(begin_token):end]
                    messages.append(content)
                    break
            
            idx = end + len(end_token) if end != -1 else start + 1
    
    return messages
```

### 2. 消息ID生成

```python
def generate_chain_ids(self, messages: List[str]) -> List[str]:
    """
    为消息列表生成ID列表
    
    示例:
        输入: ["system\n你是助手", "user\n你好"]
        输出: ["a1b2c3d4", "e5f6g7h8"]
    """
    ids = []
    for msg in messages:
        # 使用所有支持的begin/end token组合生成ID
        for begin_token in self.begin_tokens:
            for end_token in self.end_tokens:
                full_msg = f"{begin_token}{msg}{end_token}"
                msg_id = hashlib.md5(full_msg.encode()).hexdigest()
                ids.append(msg_id)
                break
            break
    return ids
```

### 3. 链式匹配

```python
def find_best_match(self, msg_ids: List[str]) -> Tuple[Optional[PlChain], int]:
    """
    查找最佳匹配
    
    Returns:
        chain: 匹配的链（可能为None）
        matched_count: 匹配的消息数量
    """
    # 从最长到最短搜索
    for i in range(len(msg_ids), 0, -1):
        prefix_ids = msg_ids[:i]
        chain_key = hashlib.md5(str(prefix_ids).encode()).hexdigest()
        
        chain = self.chain_cache.get(chain_key)
        if chain is not None:
            return chain, i
    
    return None, 0
```

### 4. 增量 Prefill

```python
async def generate_with_cache(self, prompt: str, model, tokenizer):
    """
    使用缓存的生成流程
    """
    # 1. 分割消息
    messages = self.split_prompt_by_messages(prompt)
    msg_ids = self.generate_chain_ids(messages)
    
    # 2. 查找缓存
    matched_chain, matched_count = self.find_best_match(msg_ids)
    
    if matched_chain is not None:
        # 3a. 缓存命中：只prefill未匹配部分
        print(f"缓存命中: {matched_count}/{len(msg_ids)} 条消息")
        
        remaining_messages = messages[matched_count:]
        remaining_prompt = "".join([
            f"{self.begin_tokens[0]}{msg}{self.end_tokens[0]}"
            for msg in remaining_messages
        ])
        
        # 使用已有KV Cache
        kv_cache = matched_chain.cache_item
        
    else:
        # 3b. 缓存未命中：完整prefill
        print("缓存未命中: 完整prefill")
        remaining_prompt = prompt
        kv_cache = None
    
    # 4. 执行prefill和生成
    return await self._stream_generate(remaining_prompt, kv_cache)
```

---

## 缓存匹配策略

### 跳过首轮策略

**原因**：首轮对话通常是新对话，缓存无意义，且可能误匹配。

```python
def should_skip_cache(self, msg_ids: List[str]) -> bool:
    """
    判断是否跳过缓存
    
    规则:
        - 消息数 < 3: 跳过（新对话场景）
        - 消息数 >= 3: 尝试缓存
    """
    return len(msg_ids) < 3
```

**示例**：

```
场景: 用户开始新对话
消息: [system消息, user消息]
消息数: 2
策略: 跳过缓存查找，直接prefill

原因: 
    1. 没有历史缓存可用
    2. 避免误匹配其他对话的缓存
```

### 角色顺序校验

```python
def validate_role_sequence(self, messages: List[str]) -> bool:
    """
    校验消息角色顺序是否合理
    
    合法模式:
        2条消息: (system, user)
        3条消息: (system, user, user)
                 (system, assistant, user)
                 (user, assistant, user)
    """
    roles = [self._extract_role(msg) for msg in messages]
    
    if len(roles) == 2:
        return roles == ['system', 'user']
    
    if len(roles) == 3:
        valid_patterns = [
            ['system', 'user', 'user'],
            ['system', 'assistant', 'user'],
            ['user', 'assistant', 'user']
        ]
        return roles in valid_patterns
    
    return True  # 更多消息时不校验
```

### 完全匹配

```python
def full_match(self, msg_ids: List[str], cached_chain: PlChain) -> bool:
    """
    完全匹配：所有消息ID完全相同
    """
    return msg_ids == cached_chain.node_ids
```

### 部分匹配（Retry场景）

```python
def partial_match_for_retry(self, msg_ids: List[str], cached_chain: PlChain) -> bool:
    """
    Retry场景：用户重试上一条消息
    
    例如:
        缓存: [msg1, msg2, msg3]
        当前: [msg1, msg2, msg3, msg4]  # msg4是新的user消息
        
    策略: 使用前N-1条消息的缓存
    """
    if len(msg_ids) == len(cached_chain.node_ids) + 1:
        return msg_ids[:-1] == cached_chain.node_ids
    return False
```

---

## 内存管理

### 内存估算

```python
def estimate_cache_memory(self, chain: PlChain, num_layers: int) -> float:
    """
    估算缓存占用内存
    
    Args:
        chain: 消息链
        num_layers: 模型层数
    
    Returns:
        估算内存（MB）
    """
    if chain.cache_item is None:
        return 0
    
    # 每个token每层的KV大小（假设fp16）
    bytes_per_token_per_layer = 2 * hidden_dim * 2  # 2 for K+V, 2 for fp16
    
    total_tokens = sum(len(msg) for msg in chain.message_splits)
    total_bytes = total_tokens * num_layers * bytes_per_token_per_layer
    
    return total_bytes / (1024 * 1024)  # 转换为MB
```

### 淘汰策略

```python
import psutil

class MemoryManager:
    def __init__(self):
        self.memory_threshold = float(os.getenv('PLLLM_MEMORY_THRESHOLD', '0.9'))
        self.memory_lowbound = float(os.getenv('PLLLM_MEMORY_LOWBOUND_THRESHOLD', '0.7'))
        self.min_entries = int(os.getenv('PLLLM_CACHE_MIN_ENTRIES', '3'))
    
    def check_and_evict(self, chain_cache: PlChainCache):
        """
        检查内存使用，必要时淘汰缓存
        """
        memory_percent = psutil.virtual_memory().percent / 100
        
        if memory_percent < self.memory_threshold:
            return  # 内存充足，无需淘汰
        
        print(f"内存使用超过阈值 ({memory_percent:.1%})，开始淘汰缓存")
        
        while (memory_percent > self.memory_lowbound and 
               len(chain_cache) > self.min_entries):
            # 淘汰最旧的缓存
            evicted = chain_cache.evict_oldest()
            print(f"淘汰缓存: {evicted.node_ids}")
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            memory_percent = psutil.virtual_memory().percent / 100
        
        print(f"淘汰完成，当前内存使用: {memory_percent:.1%}")
```

### 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PLLLM_MEMORY_THRESHOLD` | 0.9 | 触发淘汰的内存阈值 |
| `PLLLM_MEMORY_LOWBOUND_THRESHOLD` | 0.7 | 淘汰后的目标内存水位 |
| `PLLLM_CACHE_MIN_ENTRIES` | 3 | 最小保留的缓存条目数 |

---

## 实际场景示例

### 场景1：多轮对话

```python
"""
用户进行多轮对话，每轮追加新消息
"""

# 第1轮
messages_1 = [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "介绍一下Python"}
]
# 缓存状态: 空
# 操作: 完整prefill，缓存 [sys_msg, user_msg_1]

# 第2轮
messages_2 = [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "介绍一下Python"},
    {"role": "assistant", "content": "Python是一种编程语言..."},
    {"role": "user", "content": "它有什么优点？"}
]
# 缓存状态: [sys_msg, user_msg_1]
# 匹配结果: 2条消息命中
# 操作: 增量prefill [asst_msg_1, user_msg_2]，升级缓存

# 第3轮
messages_3 = [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "介绍一下Python"},
    {"role": "assistant", "content": "Python是一种编程语言..."},
    {"role": "user", "content": "它有什么优点？"},
    {"role": "assistant", "content": "Python的优点包括..."},
    {"role": "user", "content": "举个例子"}
]
# 缓存状态: [sys_msg, user_msg_1, asst_msg_1, user_msg_2]
# 匹配结果: 4条消息命中
# 操作: 增量prefill [asst_msg_2, user_msg_3]
```

**性能数据**：

| 轮次 | 总Token数 | 缓存命中 | Prefill Token | 节省比例 |
|------|-----------|----------|---------------|----------|
| 1 | 500 | 0 | 500 | 0% |
| 2 | 1000 | 500 | 500 | 50% |
| 3 | 1500 | 1000 | 500 | 67% |
| 4 | 2000 | 1500 | 500 | 75% |

### 场景2：长System Prompt

```python
"""
应用有很长的system prompt（如角色扮演、RAG上下文）
"""

SYSTEM_PROMPT = """
你是一个专业的法律顾问，专门处理合同纠纷...

[5000 token的法律知识库]

回答问题时请遵循以下原则：
1. ...
2. ...
"""

# 用户A的对话
messages_a = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "合同违约怎么处理？"}
]
# Prefill: 5200 token
# 缓存: [sys_msg_a, user_msg_a]

# 用户A继续对话
messages_a_2 = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "合同违约怎么处理？"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "赔偿金额怎么算？"}
]
# 匹配: 2条消息（sys + user_1）
# Prefill: 200 token（只处理新消息）
# 节省: 5000 token

# 用户B开始新对话（相同system prompt）
messages_b = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "如何解除合同？"}
]
# 匹配: 1条消息（sys）
# Prefill: 200 token（只处理user消息）
# 节省: 5000 token
```

**关键优势**：
- 长System Prompt只需prefill一次
- 不同用户共享System Prompt缓存
- 极大降低首Token延迟

### 场景3：函数调用（Tool Call）

```python
"""
多轮函数调用场景
"""

# 第1轮：模型决定调用函数
messages_1 = [
    {"role": "system", "content": "你是一个助手，可以调用get_weather函数"},
    {"role": "user", "content": "北京天气怎么样？"}
]
# 模型响应: tool_call(get_weather, city="北京")
# 缓存: [sys, user_1, asst_1(tool_call)]

# 第2轮：返回函数结果
messages_2 = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "北京天气怎么样？"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "content": "北京今天晴，25度"}
]
# 匹配: 3条消息
# Prefill: tool消息
# 模型生成最终回答

# 第3轮：继续对话
messages_3 = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "北京天气怎么样？"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "content": "北京今天晴，25度"},
    {"role": "assistant", "content": "北京今天天气晴朗..."},
    {"role": "user", "content": "上海呢？"}
]
# 匹配: 5条消息
# Prefill: 新的user消息
# 模型调用get_weather(city="上海")
```

### 场景4：用户重试（Retry）

```python
"""
用户对回答不满意，点击"重新生成"
"""

# 原始对话
messages_original = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "写一首诗"},
    {"role": "assistant", "content": "春风吹..."}  # 用户不满意
]
# 缓存: [sys, user, asst]

# 用户重试
messages_retry = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "写一首诗"}
]
# 匹配策略: 检测到retry模式（最后一条消息变化）
# 使用: 前2条消息的缓存 [sys, user]
# Prefill: 空（使用缓存直接生成）
# 结果: 生成不同的诗
```

### 场景5：多图像对话（VLM）

```python
"""
视觉语言模型，多图像对话
"""

# 第1轮：用户上传图片
messages_1 = [
    {"role": "user", "content": [
        {"type": "text", "text": "描述这张图片"},
        {"type": "image_url", "image_url": {"url": "image1.jpg"}}
    ]}
]
# 缓存: [user_msg_with_image1]

# 第2轮：追问
messages_2 = [
    {"role": "user", "content": [...]},
    {"role": "assistant", "content": "这是一张..."},
    {"role": "user", "content": "图片里有人吗？"}
]
# 匹配: 1条消息（包含image1）
# Prefill: [asst, user_msg]
# 注意: 图像token已缓存，无需重新处理

# 第3轮：上传新图片
messages_3 = [
    {"role": "user", "content": [...]},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "比较这两张图"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": [
        {"type": "text", "text": "和这张比呢？"},
        {"type": "image_url", "image_url": {"url": "image2.jpg"}}
    ]}
]
# 视觉token检测: 当前有1张新图
# 策略: 保留文本部分缓存，处理新图像
```

### 场景6：并发用户

```python
"""
多个用户同时使用服务
"""

# 用户A的对话
user_a_messages = [
    {"role": "system", "content": "你是客服助手"},
    {"role": "user", "content": "我想退货"}
]

# 用户B的对话（相同system）
user_b_messages = [
    {"role": "system", "content": "你是客服助手"},
    {"role": "user", "content": "如何修改地址"}
]

# 用户C的对话（相同system）
user_c_messages = [
    {"role": "system", "content": "你是客服助手"},
    {"role": "user", "content": "订单什么时候发货"}
]

# 缓存状态:
# - 用户A: 缓存 [sys, user_a] 
# - 用户B: 命中 sys，prefill user_b，缓存 [sys, user_b]
# - 用户C: 命中 sys，prefill user_c

# 节省: 每个用户只需prefill自己的user消息
# System prompt的KV被所有用户共享
```

### 场景7：RAG应用

```python
"""
RAG应用，每轮对话带有检索到的文档
"""

def build_rag_messages(query: str, documents: List[str]):
    """构建RAG消息"""
    context = "\n\n".join(documents)
    return [
        {"role": "system", "content": "根据以下文档回答问题:\n" + context},
        {"role": "user", "content": query}
    ]

# 查询1
docs_1 = ["文档1内容...", "文档2内容..."]  # 3000 token
messages_1 = build_rag_messages("什么是X?", docs_1)
# Prefill: 3000 + 50 token
# 缓存: [sys_with_docs, user_1]

# 查询2（相同文档，不同问题）
docs_2 = docs_1  # 检索结果相同
messages_2 = build_rag_messages("X怎么用?", docs_2)
# 匹配: sys消息完全相同
# Prefill: 50 token（只处理user消息）
# 节省: 3000 token

# 查询3（部分文档变化）
docs_3 = ["文档1内容...", "文档3内容..."]  # 文档2变文档3
messages_3 = build_rag_messages("比较X和Y", docs_3)
# 匹配: 部分匹配（sys消息ID不同）
# Prefill: 完整prefill
# 注意: 可通过文档级缓存进一步优化
```

---

## 配置参数

### 通过 set_config 配置

```python
config = {
    # 基础配置
    "enable_prefix_cache": True,  # 启用/禁用Prefix KV Cache
    
    # Token配置
    "begin_tokens": ["<|im_start|>", "<|start|>"],  # 消息开始token
    "end_tokens": ["<|im_end|>", "<|end|>"],        # 消息结束token
    
    # 视觉模型配置
    "vision_begin_tokens": ["<|vision_start|>"],    # 视觉开始token
    "vision_end_tokens": ["<|vision_end|>"],        # 视觉结束token
}

await model_loader.set_config(config)
```

### 环境变量

```bash
# 内存管理
export PLLLM_MEMORY_THRESHOLD=0.9        # 触发淘汰的内存阈值
export PLLLM_MEMORY_LOWBOUND_THRESHOLD=0.7  # 淘汰后目标水位
export PLLLM_CACHE_MIN_ENTRIES=3         # 最小保留条目数

# 日志级别
export PLLLM_CACHE_DEBUG=1               # 启用缓存调试日志
```

---

## 性能调优

### 监控缓存命中率

```python
# 查询日志
# uv run python query_logs.py -k "PlMessageBasedKVCache"

# 日志示例
[PlMessageBasedKVCache] Cache HIT: matched 4/6 messages
[PlMessageBasedKVCache] Cache HIT: full match for 5 messages
[PlMessageBasedKVCache] Cache MISS: no matching chain found
[PlMessageBasedKVCache] Added cache: chain_ids=[msg1, msg2], est_memory=640.0MB
```

### 优化建议

1. **保持System Prompt一致**
   - 相同应用使用相同的System Prompt
   - 避免动态变化的System Prompt

2. **合理设置缓存大小**
   ```bash
   # 内存充足时增加缓存条目
   export PLLLM_CACHE_MIN_ENTRIES=10
   
   # 内存紧张时减少
   export PLLLM_CACHE_MIN_ENTRIES=1
   ```

3. **预热常用对话**
   ```python
   # 预先缓存常见对话模式
   async def warmup_cache():
       common_messages = [
           [{"role": "system", "content": SYS_PROMPT}],
           # 常见用户问题...
       ]
       for msgs in common_messages:
           await model.chat_completions_stream(msgs, stream=False)
   ```

4. **监控内存使用**
   ```python
   import psutil
   
   def check_memory():
       mem = psutil.virtual_memory()
       print(f"内存使用: {mem.percent}%")
       print(f"缓存条目: {len(chain_cache)}")
   ```

---

## 故障排查

### 缓存未命中

**症状**：每次请求都完整prefill

**排查步骤**：
1. 检查消息ID是否一致
   ```python
   # 启用调试日志
   export PLLLM_CACHE_DEBUG=1
   
   # 查看消息ID
   # 日志会显示每条消息的ID
   ```

2. 检查begin/end token配置
   ```python
   # 确保token与模型匹配
   config = model.get_config()
   print(f"begin_tokens: {config['begin_tokens']}")
   print(f"end_tokens: {config['end_tokens']}")
   ```

3. 检查消息数
   ```
   规则: 消息数 >= 3 才会缓存
   ```

### 内存溢出

**症状**：服务崩溃，内存不足

**排查步骤**：
1. 检查缓存大小
   ```python
   # 查看缓存统计
   print(f"缓存条目: {len(chain_cache)}")
   for chain in chain_cache.values():
       print(f"Chain: {chain.node_ids}, Memory: {estimate_memory(chain)}MB")
   ```

2. 调整淘汰参数
   ```bash
   # 更激进的淘汰
   export PLLLM_MEMORY_THRESHOLD=0.8
   export PLLLM_MEMORY_LOWBOUND_THRESHOLD=0.5
   ```

3. 限制缓存大小
   ```python
   config = {
       "max_cache_entries": 5,  # 最多缓存5个对话
   }
   ```

### 缓存污染

**症状**：返回错误的回复

**可能原因**：
- 消息ID碰撞（极罕见）
- 并发写入问题

**解决方案**：
```python
# 清空缓存
chain_cache.clear()

# 或重启服务
```

---

## 总结

Prefix KV Cache 通过智能复用已计算的注意力矩阵，显著提升了多轮对话和重复场景的推理性能。关键要点：

1. **消息链标识**：使用MD5唯一标识每条消息
2. **最长匹配**：查找可复用的最大前缀
3. **增量计算**：只prefill未缓存部分
4. **内存管理**：自动淘汰，防止OOM
5. **透明使用**：对API调用者完全透明

合理利用Prefix KV Cache，可以在保证响应质量的同时，大幅降低计算成本和响应延迟。