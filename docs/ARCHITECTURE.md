# Architecture Overview

This document describes the architecture and design decisions of plllm-mlx.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          主进程 (FastAPI)                            │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │   Chat       │  │   Models     │  │   Subprocess Manager     │   │
│  │   Router     │  │   Router     │  │   - 发现/启动/监控子进程   │   │
│  │              │  │              │  │   - 健康检查轮询 (1s)     │   │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘   │
│         │                 │                       │                  │
│         │        ┌────────┴─────────┐             │                  │
│         │        │  PlLocalModel    │             │                  │
│         │        │  Manager         │             │                  │
│         │        │  (模型文件管理)   │             │                  │
│         │        └────────┬─────────┘             │                  │
└─────────┼─────────────────┼───────────────────────┼──────────────────┘
          │                 │                       │
          │                 │   HTTP over UDS       │
          │                 │                       │
          ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ~/.plllm-mlx/subprocess/                         │
│                                                                      │
│   a1b2c3d4.sock              e5f6g7h8.sock                          │
│   ┌─────────────────┐        ┌─────────────────┐                    │
│   │  子进程服务      │        │  子进程服务      │                    │
│   │  (独立进程)      │        │  (独立进程)      │                    │
│   │                 │        │                 │                    │
│   │  ┌───────────┐  │        │  ┌───────────┐  │                    │
│   │  │ API 线程   │  │        │  │ API 线程   │  │                    │
│   │  │ (FastAPI) │  │        │  │ (FastAPI) │  │                    │
│   │  └─────┬─────┘  │        │  └─────┬─────┘  │                    │
│   │        │ Queue  │        │        │ Queue  │                    │
│   │  ┌─────▼─────┐  │        │  ┌─────▼─────┐  │                    │
│   │  │ 推理线程   │  │        │  │ 推理线程   │  │                    │
│   │  │ + Model   │  │        │  │ + Model   │  │                    │
│   │  │ + Cache   │  │        │  │ + Cache   │  │                    │
│   │  └───────────┘  │        │  └───────────┘  │                    │
│   └─────────────────┘        └─────────────────┘                    │
│        Model A                     Model B                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Model Loaders

**Purpose**: Handle model lifecycle and inference

**Key Classes**:
- `PlModelLoader` (abstract base)
  - Defines interface for all loaders
  - Common functionality (config, stream generation)
  - Chat completion methods
  
- `PlMlxModel` (MLX-LM)
  - Text-only models
  - Uses `mlx_lm.stream_generate`
  - Standard token-by-token generation
  
- `PlMlxVlmModel` (MLX-VLM)
  - Vision-language models
  - Uses `mlx_vlm.generate.stream_generate`
  - Supports image inputs

**Design Pattern**: Strategy Pattern
- Interchangeable loaders
- Common interface
- Runtime selection

### 2. Step Processors

**Purpose**: Transform raw generation results into API responses

**Processing Pipeline**:
```
Generation Result → Step Processor → PlChunk → SSE Response
```

**Key Classes**:
- `PlStepProcessor` (abstract base)
  - Token accumulation
  - State management
  - Finish reason detection
  
- `PlBaseStepProcessor`
  - Basic text processing
  - Tool call detection
  
- `PlOpenAIStepProcessor`
  - OpenAI-compatible formatting
  - Usage statistics
  
- `PlQwen3ThinkingStepProcessor`
  - Thinking mode support
  - Reasoning content handling

### 3. KV Cache

**Purpose**: Efficient prompt reuse via prefix caching

**Implementation**:
```
Message Chain: [msg_id_1, msg_id_2, msg_id_3, ...]
                ↓
            Cache Lookup
                ↓
         Partial Match Found
                ↓
      Reuse prefix, prefill remainder
```

**Key Classes**:
- `PlMessageBasedKVCache`
  - Message-level caching
  - MD5-based message IDs
  - LRU eviction
  
- `PlChain`
  - Message chain representation
  - Cache item reference
  
- `PlChainCache`
  - OrderedDict-based storage
  - Longest prefix matching

### 4. Process Isolation

**Purpose**: Stable multi-model serving

**Architecture**:
```
Main Process (FastAPI Server)
    │
    ├─► Subprocess 1 (Model A) - HTTP over UDS
    │   └─► FastAPI + Model + KV Cache
    │
    └─► Subprocess 2 (Model B) - HTTP over UDS
        └─► FastAPI + Model + KV Cache
```

**Communication**:
- HTTP over Unix Domain Socket: Main ↔ Subprocess
- Socket path: `~/.plllm-mlx/subprocess/{hash(model_name)}.sock`
- Endpoints: `/health`, `/status`, `/load`, `/unload`, `/config`, `/infer`

**Thread Model (Subprocess)**:
- API Thread: Handles HTTP requests (FastAPI/uvicorn)
- Inference Thread: Handles model loading/inference
- Shared state protected by `threading.Lock`

**Benefits**:
- Memory isolation
- Fault tolerance
- Clean resource cleanup
- Language-agnostic (subprocess can be Python, C++, Rust, etc.)
- Independent process lifecycle
- Health check polling (1 second interval)

## Data Flow

### Chat Completion Request

```
 1. Client Request
    ↓
 2. Router (chat.py)
    ↓
 3. Local Model Manager → PlModelProxy
    ↓
 4. Subprocess Manager → get_or_create()
    ↓
 5. HTTP over UDS → Subprocess Server
    ↓
 6. Subprocess: Model Loader
    ├─► prepare_prompt()
    │   └─► KV Cache lookup
    ├─► stream_generate()
    │   ├─► MLX inference
    │   └─► Step Processor
    └─► Response chunks (SSE)
    ↓
 7. SSE Stream → Client Response
```

### KV Cache Flow

```
1. prepare_prompt(body)
   ├─► Split messages
   ├─► Calculate msg_ids
   └─► Cache lookup
       ├─► HIT: Return cached prefix
       └─► MISS: Prepare for prefill
   ↓
2. stream_generate(session)
   ├─► Check cache_item
   ├─► Skip cached prefix
   └─► Prefill remainder
   ↓
3. Generation complete
   └─► Update cache
```

## Design Decisions

### Why Process Isolation?

**Problem**: Multiple models in same process
- Memory fragmentation
- Resource conflicts
- Crash propagation

**Solution**: Separate subprocesses
- Clean memory management
- Fault isolation
- Independent lifecycles

### Why Message-based KV Cache?

**Problem**: Prompt caching challenges
- Granularity: Token vs Message?
- Matching: Exact vs Prefix?
- Efficiency: Storage vs Speed?

**Solution**: Message-level prefix cache
- Natural boundary (messages)
- Prefix matching for multi-turn
- MD5 for fast comparison
- LRU for memory management

### Why Streaming First?

**Problem**: Long generation times
- Poor user experience
- Connection timeouts
- Resource waste

**Solution**: Real-time streaming
- Immediate feedback
- Better UX
- Efficient resource use

## Performance Considerations

### Memory Management

**Strategies**:
1. Process isolation
   - Each model: separate process
   - Clean shutdown: full cleanup
    
2. KV cache eviction
   - LRU policy
   - Memory threshold
   - Minimum entries
    
3. Quantization
   - 4-bit/8-bit models
   - KV cache quantization
   - Reduced memory footprint

## Subprocess API

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (fast, non-blocking) |
| `/status` | GET | Get model status |
| `/load` | POST | Load model |
| `/unload` | POST | Unload model |
| `/config` | GET/PUT | Get/update configuration |
| `/infer` | POST | Inference request (streaming) |

### Socket Naming

```
~/.plllm-mlx/subprocess/{hash(model_name)}.sock
```

The hash is the first 8 characters of MD5(model_name).

### CLI Commands

```bash
# Start subprocess server
plllm-mlx subprocess serve --socket /path/to/socket.sock

# Check subprocess status
plllm-mlx subprocess status --model mlx-community/Qwen2.5-7B

# List all subprocesses
plllm-mlx subprocess list

# Stop a subprocess
plllm-mlx subprocess stop --model mlx-community/Qwen2.5-7B
```