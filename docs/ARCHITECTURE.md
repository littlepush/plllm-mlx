# Architecture Overview

This document describes the architecture and design decisions of plllm-mlx.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FastAPI Server                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │  Chat    │  │  Models  │  │  Management APIs     │  │
│  │  Router  │  │  Router  │  │  (Loader/Processor)  │  │
│  └────┬─────┘  └────┬─────┘  └──────────┬───────────┘  │
└───────┼─────────────┼────────────────────┼──────────────┘
        │             │                    │
        └─────────────┼────────────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Local Model Manager     │
        │  ┌──────────────────────┐ │
        │  │  Model Registry      │ │
        │  │  - Model Loaders     │ │
        │  │  - Step Processors   │ │
        │  └──────────┬───────────┘ │
        └─────────────┼─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Process Manager         │
        │  ┌──────────────────────┐ │
        │  │  Subprocess Pool     │ │
        │  │  ┌────────────────┐  │ │
        │  │  │ Model Process  │  │ │
        │  │  │ ┌────────────┐ │  │ │
        │  │  │ │   Model    │ │  │ │
        │  │  │ │   Loader   │ │  │ │
        │  │  │ ├────────────┤ │  │ │
        │  │  │ │  KV Cache  │ │  │ │
        │  │  │ └────────────┘ │  │ │
        │  │  └────────────────┘  │ │
        │  └──────────────────────┘ │
        └───────────────────────────┘
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
Main Process (API Server)
    │
    ├─► Subprocess 1 (Model A)
    │   └─► Model + KV Cache
    │
    └─► Subprocess 2 (Model B)
        └─► Model + KV Cache
```

**Communication**:
- Request Queue: Main → Subprocess
- Response Queue: Subprocess → Main
- Async bridge: Queue → AsyncGenerator

**Benefits**:
- Memory isolation
- Fault tolerance
- Clean resource cleanup
- Parallel serving

## Data Flow

### Chat Completion Request

```
1. Client Request
   ↓
2. Router (chat.py)
   ↓
3. Local Model Manager
   ↓
4. Process Manager
   ↓
5. Subprocess: Model Loader
   ├─► prepare_prompt()
   │   └─► KV Cache lookup
   ├─► stream_generate()
   │   ├─► MLX inference
   │   └─► Step Processor
   └─► Response chunks
   ↓
6. SSE Stream
   ↓
7. Client Response
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