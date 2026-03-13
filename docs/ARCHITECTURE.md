# Architecture Overview

This document describes the high-level architecture of plllm-mlx, a high-performance LLM inference server optimized for Apple Silicon.

## Table of Contents

- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Request Flow](#request-flow)
- [Process Isolation](#process-isolation)
- [Data Models](#data-models)
- [Extensibility](#extensibility)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Client Layer                                  │
│                    (OpenAI SDK, HTTP Clients)                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ HTTP/SSE
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          FastAPI Server                                  │
│                              (main.py)                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │   /chat     │ │  /models    │ │ /embedding  │ │  /rerank    │       │
│  │  router     │ │   router    │ │   router    │ │   router    │       │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘       │
└─────────┼───────────────┼───────────────┼───────────────┼──────────────┘
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Model Layer                                     │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                     PlModelLoader (Abstract)                    │    │
│  │  ┌──────────────────┐  ┌──────────────────┐                    │    │
│  │  │   PlMlxModel     │  │  PlMlxVlmModel   │  ...               │    │
│  │  │  (mlx_loader.py) │  │ (mlxvlm_loader)  │                    │    │
│  │  └──────────────────┘  └──────────────────┘                    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                   PlStepProcessor (Abstract)                    │    │
│  │  ┌───────────────┐ ┌───────────────┐ ┌────────────────────┐   │    │
│  │  │ PlBaseStep    │ │ PlOpenAIStep  │ │ Qwen3ThinkingStep  │   │    │
│  │  │ Processor     │ │ Processor     │ │ Processor          │   │    │
│  │  └───────────────┘ └───────────────┘ └────────────────────┘   │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Process Isolation Layer                           │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    PlProcessManager                             │    │
│  │  - Manages subprocess lifecycle                                │    │
│  │  - Routes requests to subprocesses                              │    │
│  │  - Handles client disconnect                                    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐  │
│  │  PlModelSubprocess │ │  PlModelSubprocess │ │  PlModelSubprocess │  │
│  │   (Model A)        │ │   (Model B)        │ │   (Model C)        │  │
│  └────────────────────┘ └────────────────────┘ └────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          MLX Framework                                   │
│                    (Apple Silicon GPU Acceleration)                      │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐        │
│  │   Model Weights  │ │   KV Cache       │ │   Tokenizer      │        │
│  │   (.safetensors) │ │   (Metal Memory) │ │   (tiktoken)     │        │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. FastAPI Server (`main.py`)

The entry point that sets up the HTTP server and routes.

```python
# Key responsibilities:
- Lifespan management (startup/shutdown)
- Route registration
- CORS middleware
- Process isolation initialization
- Model prewarming
```

**Routes:**
| Prefix | Router | Description |
|--------|--------|-------------|
| `/ai/v1` | chat | OpenAI-compatible chat/completions endpoints |
| `/ai/v1` | embedding | Text embedding endpoints |
| `/ai/v1` | rerank | Document reranking endpoints |
| `/api/v1` | models | Model management endpoints |
| `/api/v1` | category | Category management endpoints |
| `/api/v1` | loader | Model loader control |

### 2. Model Loader (`models/model_loader.py`)

Abstract base class defining the interface for all model loaders.

```python
class PlModelLoader(ABC):
    """Abstract base class for model loaders."""
    
    # Abstract methods (must override)
    @staticmethod
    @abstractmethod
    def model_loader_name() -> str:
        """Return unique loader identifier."""
        pass
    
    @abstractmethod
    async def ensure_model_loaded(self):
        """Load model into memory."""
        pass
    
    @abstractmethod
    async def ensure_model_unloaded(self):
        """Unload model from memory."""
        pass
    
    @abstractmethod
    def set_config(self, config: dict):
        """Apply runtime configuration."""
        pass
    
    @abstractmethod
    def get_config(self) -> dict:
        """Get current configuration."""
        pass
    
    @abstractmethod
    def prepare_prompt(self, body: dict) -> object:
        """Convert request to model input."""
        pass
    
    @abstractmethod
    async def stream_generate(self, session: object):
        """Generate tokens streamingly."""
        pass
```

**Implemented Loaders:**

| Loader | File | Description |
|--------|------|-------------|
| `mlx` | `mlx_loader.py` | Standard MLX models (text-only) |
| `mlxvlm` | `mlxvlm_loader.py` | Vision Language Models |

### 3. Step Processor (`models/step_processor.py`)

Processes generated tokens into structured output chunks.

```python
class PlStepProcessor(ABC):
    """Abstract base class for step processors."""
    
    @abstractmethod
    def step(self, generate_response) -> Optional[PlChunk]:
        """Process a single generation step."""
        pass
    
    @abstractmethod
    def tool_calls(self) -> List[PlChunk]:
        """Extract tool calls from buffer."""
        pass
    
    @abstractmethod
    def finish(self) -> PlChunk:
        """Generate final chunk."""
        pass
```

**Implemented Processors:**

| Processor | Description |
|-----------|-------------|
| `base` | Standard text generation |
| `openai` | OpenAI-style output with channels |
| `qwen3_thinking` | Qwen3 with thinking/reasoning |

### 4. Process Manager (`models/process_manager.py`)

Manages subprocess isolation for non-blocking inference.

```python
class PlProcessManager:
    """Singleton manager for model subprocesses."""
    
    def enable(self):
        """Enable process isolation mode."""
        pass
    
    async def get_or_create_subprocess(
        self, 
        model_name: str,
        loader_name: str,
        processor_name: str,
        config: dict
    ) -> PlModelSubprocess:
        """Get or create a subprocess for the model."""
        pass
    
    async def submit_request(
        self,
        model_name: str,
        loader_name: str,
        processor_name: str,
        config: dict,
        body: dict,
        cancel_event: asyncio.Event
    ) -> asyncio.Queue:
        """Submit request to subprocess, return chunk queue."""
        pass
    
    async def shutdown(self):
        """Shutdown all subprocesses."""
        pass
```

### 5. KV Cache Manager (`models/kv_cache.py`)

Manages prefix KV cache for optimization.

See [KV_CACHE.md](./KV_CACHE.md) for detailed documentation.

### 6. Category Manager (`models/category.py`)

Maps API endpoints to specific models.

```python
class PlCategory:
    """Manages model-to-endpoint mapping."""
    
    async def add_category(
        self, 
        name: str, 
        type: str, 
        model: str
    ) -> bool:
        """Add a new category."""
        pass
    
    async def get_category(self, name: str) -> PlCategoryItem:
        """Get category by name."""
        pass
    
    async def change_category_model(
        self, 
        name: str, 
        model: str
    ) -> bool:
        """Change the model for a category."""
        pass
```

---

## Request Flow

### Chat Completion Request (Streaming)

```
1. Client Request
   POST /ai/v1/chat/completions
   { "model": "qwen2.5", "messages": [...], "stream": true }
   │
   ▼
2. Router (chat.py)
   - Validate request
   - Look up category → model mapping
   - Acquire semaphore (concurrency control)
   │
   ▼
3. Model Loader (model_loader.py)
   - Check process isolation
   │
   ├─ [Process Isolation ON]
   │  │
   │  ▼
   │  Process Manager
   │  - Get/create subprocess
   │  - Submit request via Queue
   │  - Receive chunks via Queue
   │  │
   │  ▼
   │  Subprocess (model_subprocess.py)
   │  - Load model in isolated process
   │  - Run prepare_prompt()
   │  - Run stream_generate()
   │  - Send PlChunks via Queue
   │
   └─ [Process Isolation OFF - Development Only]
      - Direct call to prepare_prompt()
      - Direct call to stream_generate()
   │
   ▼
4. Step Processor
   - Process each token via step()
   - Classify as CONTENT/REASONING/TOOLCALL
   - Buffer content
   │
   ▼
5. Chat Completion Helper
   - Build SSE chunks
   - Manage content buffering
   - Track token usage
   │
   ▼
6. StreamingResponse
   - Yield SSE chunks to client
   - Handle client disconnect
   - Release semaphore
```

### Sequence Diagram

```
┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐
│ Client │     │ Router │     │ Loader │     │Process │     │  MLX   │
└───┬────┘     └───┬────┘     └───┬────┘     │Manager │     └───┬────┘
    │              │              │          └───┬────┘         │
    │ POST /chat   │              │              │              │
    │─────────────►│              │              │              │
    │              │              │              │              │
    │              │ get_category │              │              │
    │              │─────────────►│              │              │
    │              │              │              │              │
    │              │              │ submit_req   │              │
    │              │              │─────────────►│              │
    │              │              │              │              │
    │              │              │              │ load model   │
    │              │              │              │─────────────►│
    │              │              │              │              │
    │              │              │              │ prepare      │
    │              │              │              │─────────────►│
    │              │              │              │              │
    │              │              │              │ generate     │
    │              │              │              │◄─────────────│
    │              │              │              │              │
    │ SSE chunk    │              │              │              │
    │◄─────────────│◄─────────────│◄─────────────│              │
    │              │              │              │              │
    │ SSE chunk    │              │              │              │
    │◄─────────────│◄─────────────│◄─────────────│              │
    │              │              │              │              │
    │ [DONE]       │              │              │              │
    │◄─────────────│              │              │              │
    │              │              │              │              │
```

---

## Process Isolation

### Why Process Isolation?

MLX inference runs on the Metal GPU and can block the Python event loop. To prevent blocking the FastAPI server, each model runs in its own subprocess.

```
Without Process Isolation:
┌──────────────────────────────────────────┐
│ Main Process                              │
│  ┌─────────────┐    ┌─────────────┐      │
│  │ FastAPI     │    │ MLX Model   │      │
│  │ Event Loop  │◄───│ (BLOCKING)  │      │
│  └─────────────┘    └─────────────┘      │
│        ▲                                  │
│        │ BLOCKED during inference        │
└──────────────────────────────────────────┘

With Process Isolation:
┌──────────────────────────┐  ┌──────────────────────┐
│ Main Process              │  │ Subprocess           │
│  ┌─────────────┐         │  │  ┌─────────────┐    │
│  │ FastAPI     │  Queue  │  │  │ MLX Model   │    │
│  │ Event Loop  │◄────────┼──┼──│ (Blocking)  │    │
│  └─────────────┘         │  │  └─────────────┘    │
│        ▲                 │  │                      │
│        │ NOT BLOCKED     │  │  Isolated process    │
└──────────────────────────┘  └──────────────────────┘
```

### Communication

```python
# Main process → Subprocess
request_queue.put({
    "type": "generate",
    "body": request_body,
    "config": model_config
})

# Subprocess → Main process
response_queue.put(PlChunk(...))

# Client disconnect
cancel_event.set()  # Signal subprocess to stop
```

### Subprocess Lifecycle

```
1. Startup
   - PlProcessManager.get_or_create_subprocess()
   - Create multiprocessing.Process
   - Create request/response Queues
   - Start process

2. Running
   - Subprocess loads model
   - Waits for requests on queue
   - Processes and sends responses

3. Shutdown
   - PlProcessManager.shutdown()
   - Send shutdown signal
   - Join process
   - Cleanup queues
```

---

## Data Models

### PlChunk

The fundamental data unit passed through the generation pipeline.

```python
class PlChunk:
    """A chunk of generated content."""
    
    data: str                    # The text content
    data_type: PlChunkDataType   # Type classification
    step: PlStepUsage           # Token usage info
    finish_reason: Optional[str] # "stop", "length", "tool_calls"

class PlChunkDataType(Enum):
    NONE = 0       # Control signal
    REASONING = 1  # Thinking content (Qwen3)
    CONTENT = 2    # Regular text
    TOOLCALL = 3   # Function call
```

### PlStepUsage

Token usage and performance metrics.

```python
class PlStepUsage:
    """Token usage information."""
    
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tps: float       # Prompt processing speed
    generation_tps: float   # Generation speed
    prompt_process: int     # Prompt time (ms)
    first_token: float      # Time to first token (s)
```

### PlKVCacheMessage

Message representation for KV cache.

```python
class PlKVCacheMessage(BaseModel):
    """A message in the KV cache."""
    
    msg_id: str           # MD5 hash of content
    role: str             # system, user, assistant
    vision_count: int     # Number of images
    full_content: str     # Complete message text
```

---

## Extensibility

### Adding a New Model Loader

1. Create a new file `models/my_loader.py`:

```python
from models.model_loader import PlModelLoader

class PlMyModel(PlModelLoader):
    @staticmethod
    def model_loader_name() -> str:
        return "my_loader"
    
    async def ensure_model_loaded(self):
        # Load your model
        pass
    
    async def ensure_model_unloaded(self):
        # Cleanup
        pass
    
    def set_config(self, config: dict):
        # Apply configuration
        pass
    
    def get_config(self) -> dict:
        return {}
    
    def prepare_prompt(self, body: dict):
        # Convert request to model input
        pass
    
    async def stream_generate(self, session):
        # Generate tokens
        pass
    
    async def completion_stream_generate(self, session):
        # For /completions endpoint
        pass
```

2. The loader is automatically registered on import.

### Adding a New Step Processor

1. Create a new file `models/my_step_processor.py`:

```python
from models.step_processor import PlStepProcessor

class PlMyStepProcessor(PlStepProcessor):
    @staticmethod
    def step_clz_name() -> str:
        return "my_processor"
    
    def step(self, generate_response) -> Optional[PlChunk]:
        # Process each token
        pass
    
    def tool_calls(self) -> List[PlChunk]:
        # Extract tool calls
        return []
    
    def finish(self) -> PlChunk:
        # Return final chunk
        return PlChunk(finish_reason="stop")
```

2. The processor is automatically registered on import.

### Adding a New Router

1. Create a new file `routers/my_router.py`:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["My API"])

@router.get("/my-endpoint")
async def my_endpoint():
    return {"status": "ok"}
```

2. Register in `main.py`:

```python
from routers import my_router
app.include_router(my_router.router)
```

---

## Design Principles

### 1. Separation of Concerns

- **Router**: HTTP handling, validation
- **Loader**: Model management, inference
- **Processor**: Output formatting
- **Helper**: SSE building, buffering

### 2. Plugin Architecture

- Model loaders and step processors are discovered automatically
- No hardcoded model types
- Easy to extend with new models/formats

### 3. Non-Blocking by Default

- Process isolation prevents blocking
- Async/await throughout
- Proper resource cleanup

### 4. OpenAI Compatibility

- Standard API endpoints
- Standard request/response formats
- Works with existing OpenAI SDKs

---

## Performance Considerations

### Memory Management

- KV cache eviction based on memory thresholds
- LRU policy for cache entries
- Configurable limits

### Concurrency

- Semaphore for request limiting
- Queue-based communication
- Non-blocking I/O

### GPU Utilization

- Metal acceleration via MLX
- Batch processing where possible
- Efficient KV cache reuse

---

*Last updated: 2024*