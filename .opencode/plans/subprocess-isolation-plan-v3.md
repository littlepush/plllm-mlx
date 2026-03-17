# 子进程独立服务架构设计 v3

## 实现状态

| Phase | 状态 | 说明 |
|-------|------|------|
| Phase 1: 目录重组 | ✅ 完成 | `subprocess/python/` 结构已创建 |
| Phase 2: 子进程服务 | ✅ 完成 | `server.py`, CLI 命令已实现 |
| Phase 3: 主进程客户端 | ✅ 完成 | `client.py`, `manager.py`, `proxy.py` 已实现 |
| Phase 4: 集成改造 | ✅ 完成 | import 路径已更新，服务正常运行 |
| Phase 5: 测试与文档 | ✅ 完成 | 文档已更新，功能已验证 |

## 设计目标

每个模型由一个**独立的子进程服务**承载，通过 Unix Domain Socket (UDS) 对外提供 HTTP 服务。

**核心原则**：
- 主进程与子进程**完全隔离**，通过 HTTP over UDS 通信
- 子进程可由**任意语言实现**（Python、C++、Rust 等）
- 主进程通过**外部命令**启动子进程，不依赖 Python import

---

## 架构总览

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

---

## 核心设计决策

| 决策项 | 选择 | 说明 |
|--------|------|------|
| 通信协议 | HTTP over UDS | 标准、易调试、可独立测试、语言无关 |
| 启动方式 | 外部命令 | `plllm-mlx subprocess serve`，支持任意语言实现 |
| Socket命名 | `hash(model_name).sock` | 确定性的文件名，便于发现 |
| 绑定关系 | 1:1 | 一个子进程 ↔ 一个模型 |
| 线程模型 | API线程 + 推理线程 | 互不阻塞，健康检查稳定 |
| 健康检查 | 主进程轮询 (1s) | 检测僵尸 socket，自动重启 |

---

## 一、目录结构

```
plllm_mlx/
├── models/                          # 本地模型文件管理（仅主进程）
│   ├── __init__.py
│   ├── local_models.py             # 模型查询/下载/删除
│   └── model_detector.py           # 模型类型检测
│
├── subprocess/                      # 子进程服务架构
│   ├── __init__.py
│   ├── manager.py                  # PlSubprocessManager（主进程）
│   ├── client.py                   # PlSubprocessHandle（主进程）
│   ├── proxy.py                    # PlModelProxy（主进程）
│   │
│   └── python/                     # Python 子进程实现
│       ├── __init__.py
│       ├── server.py               # FastAPI 服务
│       ├── loader.py               # PlModelLoader 基类
│       ├── mlx_loader.py           # MLX 加载器
│       ├── mlxvlm_loader.py        # VLM 加载器
│       ├── kv_cache.py             # KV 缓存
│       ├── step_processor.py       # Step processor 基类
│       ├── base_step_processor.py
│       ├── thinking_step_processor.py
│       ├── gpt_oss_step_processor.py
│       └── special_tokens.py
│
├── routers/                         # API 路由（主进程）
│   ├── chat.py
│   └── models.py
│
├── cli.py                          # 主 CLI
├── server.py                       # FastAPI 应用
├── config.py
└── ...
```

**未来扩展**：
```
subprocess/
├── python/         # 当前实现
└── cxx/            # 未来 C++ 实现
    ├── CMakeLists.txt
    ├── server.cpp
    └── ...
```

---

## 二、模块职责

| 模块 | 运行位置 | 职责 |
|------|----------|------|
| `models/local_models.py` | 主进程 | 本地模型文件管理（查询、下载、删除） |
| `subprocess/manager.py` | 主进程 | 子进程生命周期管理（发现、启动、监控、重启） |
| `subprocess/client.py` | 主进程 | HTTP over UDS 客户端封装 |
| `subprocess/proxy.py` | 主进程 | 模型代理，提供给 Router 使用 |
| `subprocess/python/server.py` | 子进程 | FastAPI 服务，提供 HTTP API |
| `subprocess/python/loader.py` | 子进程 | 模型加载基类 |
| `subprocess/python/mlx_loader.py` | 子进程 | MLX 模型加载实现 |
| `subprocess/python/kv_cache.py` | 子进程 | KV 缓存管理 |

---

## 三、子进程 API 设计

### 3.1 端点定义

```
Socket: $HOME/.plllm-mlx/subprocess/{hash(model_name)}.sock
```

| 端点 | 方法 | 用途 | 线程 |
|------|------|------|------|
| `/health` | GET | 健康检查 | API 线程 |
| `/status` | GET | 获取模型状态 | API 线程 |
| `/load` | POST | 加载模型 | API 线程 → 推理线程 |
| `/unload` | POST | 卸载模型 | API 线程 → 推理线程 |
| `/config` | GET | 获取配置 | API 线程 |
| `/config` | PUT | 更新配置 | API 线程 |
| `/infer` | POST | 推理请求 | API 线程 → 推理线程 |

### 3.2 API 详细定义

#### GET /health

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "mlx-community/Qwen2.5-7B-Instruct"
}
```

#### GET /status

```json
{
  "model_name": "mlx-community/Qwen2.5-7B-Instruct",
  "loader": "mlx",
  "step_processor": "base",
  "is_loaded": true,
  "config": {
    "temperature": 0.8,
    "max_tokens": 4096,
    "top_p": 0.9,
    "kv_bits": null,
    "enable_prefix_cache": true
  },
  "pid": 12345,
  "uptime_seconds": 3600,
  "inferencing": false
}
```

#### POST /load

```json
// Request
{
  "model_name": "mlx-community/Qwen2.5-7B-Instruct",
  "loader": "mlx",
  "step_processor": "base",
  "config": {
    "temperature": 0.8,
    "max_tokens": 4096
  }
}

// Response
{
  "success": true,
  "model_name": "mlx-community/Qwen2.5-7B-Instruct"
}
```

#### POST /unload

```json
// Response
{
  "success": true
}
```

#### GET /config

```json
{
  "temperature": 0.8,
  "max_tokens": 4096,
  "top_p": 0.9,
  "top_k": 40,
  "repetition_penalty": 1.0,
  "enable_prefix_cache": true,
  "kv_bits": null,
  "kv_group_size": 64,
  "max_kv_size": null,
  "prefill_step_size": 512
}
```

#### PUT /config

**路由到 `loader.set_config()`**：

```json
// Request - 部分更新
{
  "temperature": 0.9,
  "max_tokens": 2048
}

// Response
{
  "success": true,
  "config": { ... }
}
```

支持的所有配置参数（见 `mlx_loader.py:212-266`）：
- `temperature`, `top_p`, `top_k`, `min_p`
- `repetition_penalty`, `repetition_context_size`
- `max_model_tokens`, `max_prompt_tokens`, `max_output_tokens`
- `xtc_probability`, `xtc_threshold`, `logit_bias`, `logprobs`
- `prefill_step_size`, `kv_bits`, `kv_group_size`, `quantized_kv_start`, `max_kv_size`
- `enable_prefix_cache`, `begin_tokens`, `end_tokens`

#### POST /infer

```json
// Request (OpenAI 兼容)
{
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true,
  "max_tokens": 1024,
  "temperature": 0.7
}

// Response (流式 SSE)
data: {"id": "chatcmpl-xxx", "choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}
data: {"id": "chatcmpl-xxx", "choices": [{"delta": {"content": "!"}, "finish_reason": null}]}
data: {"id": "chatcmpl-xxx", "choices": [{"delta": {}, "finish_reason": "stop"}]}
data: [DONE]
```

---

## 四、子进程线程隔离

### 4.1 线程模型

```
┌─────────────────────────────────────────────────────────────┐
│                       子进程                                  │
│                                                              │
│  ┌─────────────────────┐      ┌─────────────────────────┐   │
│  │    API 线程          │      │     推理线程             │   │
│  │    (FastAPI)        │      │     (独立线程)           │   │
│  │                     │      │                         │   │
│  │  /health  ✓ 快速    │      │  model.generate()       │   │
│  │  /status  ✓ 快速    │      │  stream_generate()      │   │
│  │  /config  ✓ 快速    │      │                         │   │
│  │  /load    → Queue   │      │                         │   │
│  │  /unload  → Queue   │      │                         │   │
│  │  /infer   → Queue   │──────►  从 Queue 取任务        │   │
│  │                     │      │  执行推理                │   │
│  └─────────────────────┘      │  结果放回 Queue          │   │
│                               └─────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              共享状态 (threading.Lock)               │    │
│  │  - model_name, loader, step_processor               │    │
│  │  - is_loaded, config                                │    │
│  │  - inferencing (当前是否在推理)                      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 关键实现

```python
# subprocess/python/server.py

import threading
import queue
from typing import Optional

_infer_queue: queue.Queue = queue.Queue()
_result_queues: Dict[str, queue.Queue] = {}
_loader: Optional["PlModelLoader"] = None
_model_name: str = ""
_state_lock = threading.Lock()
_inferencing: bool = False

def _infer_worker():
    """推理线程 - 独立运行，不阻塞 API 线程"""
    global _inferencing
    while True:
        task = _infer_queue.get()
        if task is None:
            break
        
        request_id, endpoint, payload = task
        
        with _state_lock:
            _inferencing = True
        
        try:
            if endpoint == "load":
                # 加载模型
                _handle_load(payload)
                _result_queues[request_id].put({"success": True})
            elif endpoint == "unload":
                # 卸载模型
                _handle_unload()
                _result_queues[request_id].put({"success": True})
            elif endpoint == "infer":
                # 推理
                for chunk in _handle_infer(payload):
                    _result_queues[request_id].put(chunk)
                _result_queues[request_id].put(None)  # 结束标记
        except Exception as e:
            _result_queues[request_id].put({"error": str(e)})
        finally:
            with _state_lock:
                _inferencing = False

# 启动推理线程
_infer_thread = threading.Thread(target=_infer_worker, daemon=True)
_infer_thread.start()

# FastAPI 路由
@app.get("/health")
async def health():
    """健康检查 - 直接读取共享状态"""
    with _state_lock:
        return {
            "status": "healthy",
            "model_loaded": _loader is not None and _loader.is_loaded,
            "model_name": _model_name,
            "inferencing": _inferencing,
        }

@app.put("/config")
async def update_config(config: dict):
    """更新配置 - 直接调用 loader.set_config"""
    with _state_lock:
        if _loader is None:
            raise HTTPException(400, "Model not loaded")
        _loader.set_config(config)
        return {"success": True, "config": _loader.get_config()}
```

---

## 五、主进程子进程管理

### 5.1 PlSubprocessManager

```python
# subprocess/manager.py

class PlSubprocessManager:
    """子进程管理器 - 仅主进程使用"""
    
    _instance: Optional["PlSubprocessManager"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subprocess_dir = Path.home() / ".plllm-mlx" / "subprocess"
            cls._instance._subprocesses: Dict[str, PlSubprocessHandle] = {}
            cls._instance._health_check_interval = 1.0
            cls._instance._health_check_task: Optional[asyncio.Task] = None
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "PlSubprocessManager":
        return cls()
    
    def socket_path(self, model_name: str) -> Path:
        """获取模型对应的 socket 路径"""
        model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        return self._subprocess_dir / f"{model_hash}.sock"
    
    async def start_health_check_loop(self):
        """启动健康检查轮询"""
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self):
        """每秒检查所有子进程健康状态"""
        while True:
            await asyncio.sleep(self._health_check_interval)
            
            for model_name, handle in list(self._subprocesses.items()):
                try:
                    healthy = await handle.health_check(timeout=0.5)
                    if not healthy:
                        logger.warning(f"Subprocess {model_name} unhealthy")
                        await self._restart_subprocess(model_name)
                except Exception as e:
                    logger.error(f"Health check failed for {model_name}: {e}")
                    await self._handle_dead_subprocess(model_name)
    
    async def get_or_create(
        self,
        model_name: str,
        loader: str = "mlx",
        step_processor: str = "base",
        config: dict | None = None,
    ) -> PlSubprocessHandle:
        """获取或创建子进程"""
        socket_path = self.socket_path(model_name)
        
        # 1. 尝试连接已存在的子进程
        if socket_path.exists():
            handle = await self._try_connect(socket_path)
            if handle:
                return handle
            # 连接失败 → 僵尸 socket，删除
            socket_path.unlink(missing_ok=True)
        
        # 2. 启动新的子进程
        handle = await self._start_subprocess(model_name, socket_path)
        
        # 3. 加载模型
        await handle.load_model(model_name, loader, step_processor, config or {})
        
        self._subprocesses[model_name] = handle
        return handle
    
    async def _try_connect(self, socket_path: Path) -> Optional[PlSubprocessHandle]:
        """尝试连接已存在的子进程"""
        handle = PlSubprocessHandle(socket_path)
        try:
            if await handle.connect():
                return handle
        except Exception:
            pass
        return None
    
    async def _start_subprocess(
        self, model_name: str, socket_path: Path
    ) -> PlSubprocessHandle:
        """通过外部命令启动子进程"""
        
        # 确保目录存在
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建启动命令
        cmd = [
            "plllm-mlx", "subprocess", "serve",
            "--socket", str(socket_path),
        ]
        
        logger.info(f"Starting subprocess: {' '.join(cmd)}")
        
        # 启动子进程
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        # 等待 socket 文件创建
        await self._wait_for_socket(socket_path, timeout=30)
        
        # 创建句柄并连接
        handle = PlSubprocessHandle(socket_path, process)
        await handle.connect()
        
        return handle
    
    async def _wait_for_socket(self, socket_path: Path, timeout: float = 30):
        """等待 socket 文件创建"""
        start = time.time()
        while time.time() - start < timeout:
            if socket_path.exists():
                # 等待一小段时间确保服务端已绑定
                await asyncio.sleep(0.1)
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Socket not created: {socket_path}")
    
    async def _handle_dead_subprocess(self, model_name: str):
        """处理死亡的子进程"""
        handle = self._subprocesses.pop(model_name, None)
        if handle:
            await handle.cleanup()
            # 清理僵尸 socket
            socket_path = self.socket_path(model_name)
            socket_path.unlink(missing_ok=True)
```

### 5.2 PlSubprocessHandle

```python
# subprocess/client.py

class PlSubprocessHandle:
    """子进程句柄 - HTTP over UDS 客户端"""
    
    def __init__(
        self,
        socket_path: Path,
        process: asyncio.subprocess.Process | None = None,
    ):
        self._socket_path = socket_path
        self._process = process
        self._client: httpx.AsyncClient | None = None
    
    async def connect(self) -> bool:
        """连接到子进程"""
        self._client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=str(self._socket_path)),
            base_url="http://localhost",
            timeout=30.0,
        )
        return await self.health_check()
    
    async def health_check(self, timeout: float = 1.0) -> bool:
        """健康检查"""
        try:
            resp = await self._client.get("/health", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False
    
    async def load_model(
        self,
        model_name: str,
        loader: str,
        step_processor: str,
        config: dict,
    ) -> bool:
        """加载模型"""
        resp = await self._client.post("/load", json={
            "model_name": model_name,
            "loader": loader,
            "step_processor": step_processor,
            "config": config,
        })
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"Failed to load model: {data}")
        return True
    
    async def unload_model(self) -> bool:
        """卸载模型"""
        resp = await self._client.post("/unload")
        return resp.json().get("success", False)
    
    async def get_config(self) -> dict:
        """获取配置"""
        resp = await self._client.get("/config")
        return resp.json()
    
    async def update_config(self, config: dict) -> bool:
        """更新配置"""
        resp = await self._client.put("/config", json=config)
        return resp.json().get("success", False)
    
    async def status(self) -> dict:
        """获取状态"""
        resp = await self._client.get("/status")
        return resp.json()
    
    async def infer(self, body: dict) -> AsyncIterator[str]:
        """推理请求 - 流式响应"""
        async with self._client.stream("POST", "/infer", json=body) as resp:
            async for line in resp.aiter_lines():
                if line:
                    yield line
    
    async def cleanup(self):
        """清理资源"""
        if self._client:
            await self._client.aclose()
            self._client = None
        
        # 如果是主进程启动的，终止子进程
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
```

### 5.3 PlModelProxy

```python
# subprocess/proxy.py

class PlModelProxy:
    """模型代理 - 主进程中的代理对象，供 Router 使用"""
    
    def __init__(
        self,
        model_name: str,
        loader: str,
        step_processor: str,
        subprocess_manager: PlSubprocessManager,
    ):
        self._model_name = model_name
        self._loader = loader
        self._step_processor = step_processor
        self._subprocess_manager = subprocess_manager
        self._handle: PlSubprocessHandle | None = None
        self._config: dict = {}
        self._is_loaded = False
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    @property
    def step_processor_clz(self):
        from plllm_mlx.subprocess.python.step_processor import PlStepProcessor
        return PlStepProcessor.findStepProcessor(self._step_processor)
    
    async def _ensure_handle(self) -> PlSubprocessHandle:
        """确保子进程已启动并连接"""
        if self._handle is None:
            self._handle = await self._subprocess_manager.get_or_create(
                self._model_name,
                self._loader,
                self._step_processor,
                self._config,
            )
        return self._handle
    
    async def load_model(self) -> None:
        """加载模型"""
        handle = await self._ensure_handle()
        await handle.load_model(
            self._model_name,
            self._loader,
            self._step_processor,
            self._config,
        )
        self._is_loaded = True
    
    async def unload_model(self) -> None:
        """卸载模型"""
        if self._handle:
            await self._handle.unload_model()
            self._is_loaded = False
    
    def set_config(self, config: dict) -> None:
        """设置配置（同步方法，异步更新到子进程）"""
        self._config.update(config)
        if self._handle and self._is_loaded:
            asyncio.create_task(self._handle.update_config(config))
    
    def get_config(self) -> dict:
        return self._config.copy()
    
    async def chat_completions_stream(
        self,
        body: dict,
        alias_name: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """推理请求"""
        handle = await self._ensure_handle()
        async for chunk in handle.infer(body):
            yield chunk
```

---

## 六、CLI 命令

### 6.1 子进程命令

```bash
# 启动子进程服务
plllm-mlx subprocess serve --socket /path/to/socket.sock

# 查看子进程状态
plllm-mlx subprocess status --model mlx-community/Qwen2.5-7B-Instruct

# 停止子进程
plllm-mlx subprocess stop --model mlx-community/Qwen2.5-7B-Instruct

# 列出所有子进程
plllm-mlx subprocess list
```

### 6.2 CLI 实现

```python
# cli.py

@app_cli.command("subprocess")
def subprocess_cli():
    """子进程管理命令"""
    pass

@subprocess_cli.command("serve")
@click.option("--socket", required=True, help="Unix socket path")
@click.option("--model", help="Model name to load on startup")
def serve(socket: str, model: str | None):
    """启动子进程服务"""
    from plllm_mlx.subprocess.python.server import run_server
    run_server(socket_path=socket, model_name=model)

@subprocess_cli.command("status")
@click.option("--model", required=True, help="Model name")
def status(model: str):
    """查看子进程状态"""
    import httpx
    import hashlib
    
    socket_path = Path.home() / ".plllm-mlx" / "subprocess" / f"{hashlib.md5(model.encode()).hexdigest()[:8]}.sock"
    
    if not socket_path.exists():
        print(f"Subprocess for {model} not found")
        return
    
    with httpx.Client(transport=httpx.HTTPTransport(uds=str(socket_path))) as client:
        resp = client.get("http://localhost/status")
        print(json.dumps(resp.json(), indent=2))
```

---

## 七、实现步骤

### Phase 1: 目录重组 (预计 30 分钟)

| 步骤 | 操作 | 验收标准 |
|------|------|----------|
| 1.1 | 创建 `subprocess/python/` 目录结构 | 目录存在 |
| 1.2 | 移动 `models/model_loader.py` → `subprocess/python/loader.py` | 文件已移动 |
| 1.3 | 移动 `models/mlx_loader.py` → `subprocess/python/` | 文件已移动 |
| 1.4 | 移动 `models/mlxvlm_loader.py` → `subprocess/python/` | 文件已移动 |
| 1.5 | 移动 `models/kv_cache.py` → `subprocess/python/` | 文件已移动 |
| 1.6 | 移动所有 `*_step_processor.py` → `subprocess/python/` | 文件已移动 |
| 1.7 | 移动 `models/special_tokens.py` → `subprocess/python/` | 文件已移动 |
| 1.8 | 更新所有 import 路径 | 测试通过 |

### Phase 2: 子进程服务实现 (预计 2 小时)

| 步骤 | 操作 | 验收标准 |
|------|------|----------|
| 2.1 | 实现 `subprocess/python/server.py` | 文件存在 |
| 2.2 | 实现 `/health` 端点 | `curl --unix-socket` 返回正确 |
| 2.3 | 实现 `/status` 端点 | 返回完整状态信息 |
| 2.4 | 实现 `/load` 端点 | 能加载模型 |
| 2.5 | 实现 `/unload` 端点 | 能卸载模型，内存释放 |
| 2.6 | 实现 `/config` GET/PUT 端点 | 能获取/更新配置 |
| 2.7 | 实现 `/infer` 端点 | 能完成推理 |
| 2.8 | 实现线程隔离 | 健康检查不阻塞推理 |
| 2.9 | 实现 CLI `subprocess serve` 命令 | 能独立启动子进程 |

### Phase 3: 主进程客户端实现 (预计 2 小时)

| 步骤 | 操作 | 验收标准 |
|------|------|----------|
| 3.1 | 实现 `subprocess/client.py` | 文件存在 |
| 3.2 | 实现 HTTP over UDS 连接 | 能连接到子进程 |
| 3.3 | 实现健康检查 | 能检测子进程状态 |
| 3.4 | 实现僵尸 socket 检测 | 能清理无效 socket |
| 3.5 | 实现 `subprocess/manager.py` | 文件存在 |
| 3.6 | 实现子进程启动（外部命令） | 能启动子进程 |
| 3.7 | 实现健康检查轮询 | 每秒检查，能自动重启 |
| 3.8 | 实现 `subprocess/proxy.py` | 文件存在 |

### Phase 4: 集成改造 (预计 1.5 小时)

| 步骤 | 操作 | 验收标准 |
|------|------|----------|
| 4.1 | 改造 `local_models.py` | 使用 PlModelProxy |
| 4.2 | 改造 `routers/chat.py` | 通过 proxy 调用推理 |
| 4.3 | 改造 `routers/models.py` | 适配新接口 |
| 4.4 | 更新 `__init__.py` | 启动时初始化 manager |
| 4.5 | 删除旧代码 | 移除 `process_manager.py`, `model_subprocess.py` |

### Phase 5: 测试与文档 (预计 1 小时)

| 步骤 | 操作 | 验收标准 |
|------|------|----------|
| 5.1 | 单元测试 | 覆盖核心功能 |
| 5.2 | 集成测试 | 端到端推理正常 |
| 5.3 | 性能测试 | 延迟开销 < 5% |
| 5.4 | 更新 README | 文档更新 |
| 5.5 | 更新 ARCHITECTURE.md | 架构文档更新 |

---

## 八、验收标准

### 功能验收

| 编号 | 测试项 | 验收标准 |
|------|--------|----------|
| F1 | 独立启动子进程 | `plllm-mlx subprocess serve --socket xxx.sock` 成功启动 |
| F2 | 健康检查 | `curl --unix-socket xxx.sock http://localhost/health` 返回 200 |
| F3 | 加载模型 | `/load` 接口能加载模型，内存增加 |
| F4 | 卸载模型 | `/unload` 接口能卸载模型，内存释放 |
| F5 | 配置更新 | `/config` 接口能更新配置，立即生效 |
| F6 | 推理功能 | `/infer` 接口能完成推理，结果正确 |
| F7 | 主进程连接 | 主进程能发现并连接已运行的子进程 |
| F8 | 主进程启动 | 主进程能通过外部命令启动子进程 |
| F9 | 僵尸 socket 清理 | 连接失败时自动删除僵尸 socket |
| F10 | 健康检查轮询 | 主进程每秒检查，子进程崩溃能自动重启 |

### 性能验收

| 编号 | 测试项 | 验收标准 |
|------|--------|----------|
| P1 | 连接延迟 | < 10ms |
| P2 | 健康检查延迟 | < 5ms |
| P3 | 推理延迟开销 | < 5% (与直接调用相比) |
| P4 | 内存隔离 | 卸载模型后内存完全释放 |

### 稳定性验收

| 编号 | 测试项 | 验收标准 |
|------|--------|----------|
| S1 | 子进程崩溃隔离 | 不影响主进程和其他子进程 |
| S2 | 主进程重启 | 能重新连接已运行的子进程 |
| S3 | 并发推理 | 多个推理请求正常处理 |
| S4 | 线程隔离 | 推理过程中健康检查正常响应 |
| S5 | 信号处理 | SIGTERM/SIGINT 正确清理资源 |

---

## 九、风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| Socket 文件残留 | 无法启动新子进程 | 中 | 连接时检测并清理僵尸 socket |
| HTTP over UDS 性能 | 推理延迟增加 | 低 | 流式传输，减少序列化开销 |
| 子进程僵尸进程 | 资源泄漏 | 中 | 主进程监控并回收，设置超时 |
| 配置同步问题 | 配置不一致 | 低 | 主进程管理配置，同步给子进程 |
| 线程竞争 | 状态不一致 | 中 | 使用 threading.Lock 保护共享状态 |
| 推理线程阻塞 | 健康检查无响应 | 低 | API 线程和推理线程完全隔离 |

---

## 十、后续优化

1. **子进程池**：预启动多个空子进程，加速首次加载
2. **优雅降级**：子进程崩溃时，等待中的请求可重试
3. **负载均衡**：相同模型的多个实例，分散请求
4. **热更新**：子进程升级不中断服务
5. **C++ 实现**：`subprocess/cxx/` 提供更高性能实现
6. **资源限制**：子进程 CPU/内存限制，防止资源耗尽