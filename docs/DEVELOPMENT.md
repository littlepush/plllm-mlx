# Development Guide

This document provides guidance for developers who want to contribute to or extend plllm-mlx.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Conventions](#code-conventions)
- [Adding Model Loaders](#adding-model-loaders)
- [Adding Step Processors](#adding-step-processors)
- [Testing](#testing)
- [Debugging](#debugging)
- [Contributing](#contributing)

---

## Development Setup

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- 50GB+ free disk space for models

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd plllm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest ruff
```

### Running in Development Mode

```bash
# Enable debug logging
export DEBUG=1

# Run server directly
python main.py

# Or use the start script
./start.sh

# Run without auto-loading models (faster startup)
./install.sh --no-auto-load
```

### Dry Run Testing

Test all LLM models without starting the HTTP server:

```bash
./test.sh --dry-run
```

---

## Project Structure

```
plllm/
├── main.py                 # FastAPI application entry point
├── install.sh              # Service startup script
├── start.sh                # Alternative startup script
├── uninstall.sh            # Service shutdown script
├── test.sh                 # Test script
├── query_logs.py           # Log query utility
│
├── models/                 # Model layer
│   ├── __init__.py
│   ├── model_loader.py     # Base model loader class
│   ├── mlx_loader.py       # MLX implementation
│   ├── mlxvlm_loader.py    # VLM implementation
│   ├── step_processor.py   # Base step processor class
│   ├── base_step_processor.py
│   ├── openai_step_processor.py
│   ├── qwen3_thinking_step_processor.py
│   ├── kv_cache.py         # KV cache manager
│   ├── category.py         # Category manager
│   ├── local_models.py     # Local model registry
│   ├── process_manager.py  # Process isolation manager
│   └── model_subprocess.py # Subprocess worker
│
├── routers/                # API routes
│   ├── chat.py             # Chat completions
│   ├── embedding.py        # Embeddings
│   ├── rerank.py           # Reranking
│   ├── category.py         # Category management
│   ├── models.py           # Model management
│   ├── loader.py           # Loader control
│   ├── stepprocessor.py    # Step processor list
│   └── model_manager.py    # Model download/delete
│
├── helpers/                # Helper utilities
│   ├── chat_helper.py      # Chat completion builder
│   ├── chain_cache.py      # Chain cache implementation
│   ├── chunk_helper.py     # Chunk utilities
│   ├── step_info.py        # Step usage tracking
│   └── ...
│
├── tests/                  # Test files
│   ├── test_*.py
│   └── ...
│
├── pl_embedding.py         # Embedding service
├── pl_rerank.py           # Rerank service
│
└── requirements.txt        # Dependencies
```

---

## Code Conventions

### Naming Conventions

- **Files**: Snake case (`model_loader.py`)
- **Classes**: Pascal case with `Pl` prefix (`PlModelLoader`)
- **Functions**: Snake case (`get_kv_cache`)
- **Private methods**: Leading underscore (`_generate_msg_id`)
- **Constants**: Upper snake case (`_DEFAULT_MEMORY_THRESHOLD`)

### File Naming Patterns

| Pattern | Description | Auto-discovered |
|---------|-------------|-----------------|
| `*_loader.py` | Model loader | Yes |
| `*_step_processor.py` | Step processor | Yes |

### Documentation

Use docstrings for all public classes and methods:

```python
class PlModelLoader(ABC):
    """
    Abstract base class for model loaders.
    
    Provides the interface for loading, unloading, and running inference
    with different model backends.
    
    Attributes:
        model_name: The name/ID of the model.
        is_loaded: Whether the model is currently loaded.
    """
    
    @abstractmethod
    async def ensure_model_loaded(self):
        """
        Load the model into memory.
        
        This method should:
        1. Download the model if not present
        2. Load weights into memory
        3. Initialize the tokenizer
        
        Raises:
            RuntimeError: If the model fails to load.
        """
        pass
```

### Type Hints

Use type hints for better IDE support:

```python
from typing import Optional, List, Dict, Any

def prepare_prompt(self, body: Dict[str, Any]) -> PlMlxSessionStorage:
    """
    Prepare the prompt for generation.
    
    Args:
        body: The request body containing messages and parameters.
        
    Returns:
        A session storage object with prepared prompt and sampler.
    """
    pass
```

### Async Patterns

Use `async/await` consistently:

```python
# Good
async def load_model(self):
    async with self._lock:
        await self.ensure_model_loaded()

# Bad
def load_model(self):
    # Blocking call in async context
    self.ensure_model_loaded()
```

---

## Adding Model Loaders

### Step 1: Create the Loader File

Create `models/my_loader.py`:

```python
from models.model_loader import PlModelLoader
from typing import Optional, Dict, Any

class PlMyModel(PlModelLoader):
    """
    Custom model loader for [your backend].
    """
    
    def __init__(self, model_name: str, step_processor_clz):
        super().__init__(model_name, step_processor_clz)
        # Initialize your specific attributes
        self._model = None
        self._tokenizer = None
    
    @staticmethod
    def model_loader_name() -> str:
        """Return unique loader identifier."""
        return "my_loader"
    
    async def ensure_model_loaded(self):
        """Load model into memory."""
        if self._model is not None:
            return
        
        # Your model loading logic here
        # Example:
        # self._model = load_my_model(self._model_name)
        # self._tokenizer = load_my_tokenizer(self._model_name)
        
        self._is_loaded = True
    
    async def ensure_model_unloaded(self):
        """Unload model from memory."""
        if self._model is None:
            return
        
        # Your cleanup logic here
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
    
    def set_config(self, config: Dict[str, Any]):
        """Apply runtime configuration."""
        for key, value in config.items():
            setattr(self, f"_{key}", value)
    
    def get_config(self) -> Dict[str, Any]:
        """Return current configuration."""
        return {
            # Return your config values
        }
    
    def prepare_prompt(self, body: Dict[str, Any]) -> Any:
        """
        Prepare the prompt for generation.
        
        This method should:
        1. Extract messages from the request body
        2. Apply chat template
        3. Create sampling parameters
        4. Return a session object for stream_generate
        """
        messages = body.get("messages", [])
        
        # Your prompt preparation logic
        prompt = self._tokenizer.apply_chat_template(messages)
        
        # Return session object
        return MySessionStorage(
            prompt=prompt,
            max_tokens=body.get("max_tokens", 16384),
            temperature=body.get("temperature", 0.8),
            # ... other parameters
        )
    
    async def stream_generate(self, session: Any):
        """
        Generate tokens streamingly.
        
        Yields:
            PlChunk objects for each generation step.
        """
        # Your generation logic
        for token in self._model.generate(session.prompt):
            chunk = PlChunk(
                data=token.text,
                data_type=PlChunkDataType.CONTENT,
                step=PlStepUsage(...)
            )
            yield chunk
        
        # Final chunk with finish_reason
        yield PlChunk(finish_reason="stop")
    
    async def completion_stream_generate(self, session: Any):
        """For /completions endpoint."""
        # Similar to stream_generate but without chat formatting
        pass
```

### Step 2: Register the Loader

The loader is automatically registered when the file is imported. No manual registration needed.

### Step 3: Use the Loader

```bash
# Update model to use your loader
curl -X POST http://localhost:8080/api/v1/model/update/modelloader \
  -H "Content-Type: application/json" \
  -d '{"model_name": "my-model", "model_loader": "my_loader"}'
```

---

## Adding Step Processors

### Step 1: Create the Processor File

Create `models/my_step_processor.py`:

```python
from models.step_processor import PlStepProcessor
from helpers.chunk_helper import PlChunk, PlChunkDataType, PlStepUsage
from typing import Optional, List, Any

class PlMyStepProcessor(PlStepProcessor):
    """
    Custom step processor for [your output format].
    """
    
    def __init__(self):
        super().__init__()
        self._buffer = ""
        self._finish_reason = None
    
    @staticmethod
    def step_clz_name() -> str:
        """Return unique processor identifier."""
        return "my_processor"
    
    def step(self, generate_response: Any) -> Optional[PlChunk]:
        """
        Process a single generation step.
        
        Args:
            generate_response: The token/response from the model.
            
        Returns:
            PlChunk if there's content to yield, None otherwise.
        """
        self.total_tokens += 1
        token = generate_response.text
        
        # Check for finish
        if generate_response.finish_reason:
            self._finish_reason = generate_response.finish_reason
            self.stop()
            return None
        
        # Your processing logic
        # Example: detect special markers, parse structured output, etc.
        
        self._buffer += token
        
        # Return a content chunk
        return PlChunk(
            data=token,
            data_type=PlChunkDataType.CONTENT,
            step=PlStepUsage(
                prompt_tokens=generate_response.prompt_tokens,
                completion_tokens=generate_response.generation_tokens,
                total_tokens=generate_response.prompt_tokens + generate_response.generation_tokens,
                prompt_tps=generate_response.prompt_tps,
                generation_tps=generate_response.generation_tps,
            )
        )
    
    def tool_calls(self) -> List[PlChunk]:
        """
        Extract tool calls from the buffer.
        
        Called after generation stops.
        
        Returns:
            List of PlChunk with tool call data.
        """
        # Parse your tool call format
        # Example: JSON extraction, XML parsing, etc.
        return []
    
    def finish(self) -> PlChunk:
        """
        Return the final chunk.
        
        Called after all generation is complete.
        
        Returns:
            PlChunk with finish_reason set.
        """
        return PlChunk(
            finish_reason=self._finish_reason or "stop"
        )
```

### Step 2: Use the Processor

```bash
# Update model to use your processor
curl -X POST http://localhost:8080/api/v1/model/update/stepprocessor \
  -H "Content-Type: application/json" \
  -d '{"model_name": "my-model", "step_processor": "my_processor"}'
```

---

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_kv_cache.py -v

# Run with coverage
uv run pytest --cov=models --cov=helpers
```

### Test Patterns

```python
# tests/test_my_feature.py
import pytest
from models.kv_cache import PlMessageBasedKVCache

@pytest.fixture
def cache():
    return PlMessageBasedKVCache()

def test_split_prompt(cache):
    """Test message splitting."""
    prompt = "<|im_start|>user\nHello<|im_end|>"
    messages = cache.split_prompt_by_messages(prompt)
    
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert "Hello" in messages[0].full_content

@pytest.mark.asyncio
async def test_async_operation(cache):
    """Test async operations."""
    result = await some_async_function()
    assert result is not None
```

### Integration Testing

```python
# tests/test_integration.py
import httpx
import pytest

@pytest.fixture
async def client():
    async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
        yield client

@pytest.mark.asyncio
async def test_chat_completion(client):
    """Test chat completion endpoint."""
    response = await client.post(
        "/ai/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
```

---

## Debugging

### Enable Debug Logging

```bash
# Set environment variable
export DEBUG=1

# Or in Python
import os
os.environ["DEBUG"] = "1"
```

### Query Logs

```bash
# All recent logs
uv run python query_logs.py

# Only errors
uv run python query_logs.py -l ERROR

# Filter by keyword
uv run python query_logs.py -k "PlMessageBasedKVCache"

# Filter by category
uv run python query_logs.py -k "qwen2.5"
```

### Debugging in Code

```python
from plpybase import pl_log

# Different log levels
pl_log.debug("Detailed debug information")
pl_log.info("General information")
pl_log.warning("Warning message")
pl_log.error("Error message")
```

### Debugging Process Isolation

```python
# In model_loader.py
# Temporarily disable process isolation for debugging

# Comment out the process isolation check:
async def chat_completions_stream(self, body, ...):
    # if self.is_process_isolation_enabled():
    #     async for chunk in self.chat_completions_stream_with_isolation(...):
    #         yield chunk
    #     return
    
    # Direct mode for debugging
    helper = PlChatCompletionHelper(...)
    session = self.prepare_prompt(body)
    # ...
```

---

## Contributing

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Make your changes**
   - Follow code conventions
   - Add tests
   - Update documentation
4. **Run tests**
   ```bash
   uv run pytest
   ```
5. **Commit with clear message**
   ```bash
   git commit -m "feat: add support for new model format"
   ```
6. **Push and create PR**

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Testing
- `chore`: Maintenance

**Examples**:
```
feat(loader): add support for GGUF format

fix(kv-cache): resolve memory leak in eviction logic

docs(api): update endpoint documentation
```

### Code Review Checklist

- [ ] Code follows naming conventions
- [ ] Type hints are added
- [ ] Docstrings are present
- [ ] Tests are added/updated
- [ ] Documentation is updated
- [ ] No hardcoded values
- [ ] Error handling is proper
- [ ] Async patterns are correct

---

## Architecture Decisions

### Why Process Isolation?

MLX inference blocks the Python event loop. Running inference in a subprocess:
- Keeps the FastAPI server responsive
- Allows concurrent request handling
- Isolates model crashes

### Why Prefix KV Cache?

Traditional KV cache stores all past tokens. Message-based prefix cache:
- Enables partial cache hits
- Supports conversation branching
- Reduces memory with smart eviction

### Why Custom Step Processors?

Different models have different output formats:
- Qwen3: `<think>...</think>` reasoning tags
- OpenAI-style: Channel-based structured output
- Custom models: Various other formats

Step processors abstract this complexity from the main loader.

---

## Common Tasks

### Adding a New API Endpoint

1. Create route in `routers/my_router.py`:

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

### Adding Configuration Parameter

1. Add to model loader's `set_config`:

```python
def set_config(self, config: dict):
    if "my_new_param" in config:
        self._my_new_param = config["my_new_param"]
```

2. Update API:

```bash
curl -X POST http://localhost:8080/api/v1/model/update/config \
  -d '{"model_name": "...", "key": "my_new_param", "value": 42}'
```

### Adding Environment Variable

1. Read in the relevant module:

```python
import os

MY_VAR = os.environ.get("PLLLM_MY_VAR", "default_value")
```

2. Document in `docs/CONFIGURATION.md`

---

*Last updated: 2024*