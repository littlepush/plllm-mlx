# 子进程独立改造计划 v2

## 原始需求

**设计目标**：每个模型必须在独立子进程中加载和执行，这是核心设计原则，不是可选项。

根据 `docs/ARCHITECTURE.md`：
```
主进程 (API服务器)
    │
    ├─► 子进程1 (模型A)
    │   └─► 模型 + KV缓存
    │
    └─► 子进程2 (模型B)
        └─► 模型 + KV缓存
```

---

## 当前问题诊断

### 1. 进程隔离从未启用

| 问题点 | 位置 | 说明 |
|--------|------|------|
| `PlProcessManager.enable()` | process_manager.py:300 | 从未被调用 |
| `is_process_isolation_enabled()` | model_loader.py:198 | 永远返回 False |
| `_enabled` 类变量 | process_manager.py:290 | 初始化为 False，从未设为 True |

### 2. 模型在主进程中加载

```python
# local_models.py:128-138
local_model = PlModelLoader.createModel(...)
self._models_in_memory[model_info.name] = local_model  # 直接存实例！
```

**错误**：`_models_in_memory` 存储的是真实的 `PlModelLoader` 实例，而不是子进程代理。

### 3. 推理直接在主进程执行

```python
# model_loader.py:441-447
if self.is_process_isolation_enabled():  # 永远 False
    async for chunk in self.chat_completions_stream_with_isolation(...):
        yield chunk
    return

# 实际执行路径：直接在主进程推理
```

### 4. 子进程通信不完整

当前 `PlModelSubprocess` 只支持：
- 推理请求（INFER）

缺少：
- 加载模型（LOAD）
- 卸载模型（UNLOAD）
- 更新配置（UPDATE_CONFIG）
- 健康检查（PING）

---

## 改造方案

### 核心原则

1. **主进程**：只持有模型元信息和子进程代理
2. **子进程**：负责模型的完整生命周期（加载、推理、卸载）
3. **通信**：通过 multiprocessing.Queue 实现请求/响应

### 文件改造清单

#### 1. 新建文件

| 文件 | 用途 |
|------|------|
| `subprocess_protocol.py` | 子进程通信协议（请求类型、消息格式） |
| `model_proxy.py` | 子进程模型代理类 |

#### 2. 修改文件

| 文件 | 改造内容 |
|------|----------|
| `process_manager.py` | 改造子进程管理，支持完整的生命周期请求 |
| `model_subprocess.py` | 改造子进程入口，支持动态加载/卸载 |
| `local_models.py` | 使用代理类替代直接实例化 |
| `model_loader.py` | 移除进程隔离的条件判断（始终使用子进程） |
| `routers/models.py` | 适配新的模型管理接口 |
| `__init__.py` | 启动时初始化进程管理器 |

---

## 详细设计

### 1. 子进程通信协议

```python
# subprocess_protocol.py

from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional

class RequestType(Enum):
    """子进程请求类型"""
    LOAD = "load"           # 加载模型
    UNLOAD = "unload"       # 卸载模型
    INFER = "infer"         # 推理请求
    UPDATE_CONFIG = "update_config"  # 更新配置
    PING = "ping"           # 健康检查
    SHUTDOWN = "shutdown"   # 关闭子进程

@dataclass
class SubprocessRequest:
    """子进程请求"""
    request_id: str
    request_type: RequestType
    payload: Any = None

@dataclass
class SubprocessResponse:
    """子进程响应"""
    request_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
```

### 2. 子进程入口改造

```python
# model_subprocess.py

def run_subprocess(
    model_id: str,
    loader_name: str,
    step_processor_name: str,
    request_queue: Queue,
    response_queue: Queue,
) -> None:
    """子进程主循环"""
    
    loader = None
    model_config = {}
    is_loaded = False
    
    # 主循环：处理所有请求类型
    while True:
        request = request_queue.get()
        
        if request is None or request.request_type == RequestType.SHUTDOWN:
            # 清理并退出
            if loader:
                loader.ensure_model_unloaded()
            break
            
        if request.request_type == RequestType.LOAD:
            # 创建并加载模型
            loader = create_loader(model_id, loader_name, step_processor_name)
            if model_config:
                loader.set_config(model_config)
            loader.ensure_model_loaded()
            is_loaded = True
            response_queue.put(SubprocessResponse(request.request_id, True))
            
        elif request.request_type == RequestType.UNLOAD:
            # 卸载模型
            if loader:
                loader.ensure_model_unloaded()
                is_loaded = False
            response_queue.put(SubprocessResponse(request.request_id, True))
            
        elif request.request_type == RequestType.UPDATE_CONFIG:
            # 更新配置
            model_config = request.payload
            if loader:
                loader.set_config(model_config)
            response_queue.put(SubprocessResponse(request.request_id, True))
            
        elif request.request_type == RequestType.INFER:
            # 推理
            if not is_loaded:
                response_queue.put(SubprocessResponse(
                    request.request_id, False, error="Model not loaded"
                ))
                continue
            # 流式推理...
            
        elif request.request_type == RequestType.PING:
            # 健康检查
            response_queue.put(SubprocessResponse(request.request_id, True))
```

### 3. 模型代理类

```python
# model_proxy.py

class PlModelSubprocessProxy:
    """子进程模型代理 - 主进程中的代理对象"""
    
    def __init__(
        self,
        model_id: str,
        loader_name: str,
        step_processor_name: str,
        subprocess: PlModelSubprocess,
    ):
        self._model_id = model_id
        self._loader_name = loader_name
        self._step_processor_name = step_processor_name
        self._subprocess = subprocess
        self._is_loaded = False
        self._config = {}
        
    @property
    def model_name(self) -> str:
        return self._model_id
        
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
        
    @property
    def step_processor_clz(self):
        # 返回 step processor 类（用于 API 响应）
        return PlStepProcessor.findStepProcessor(self._step_processor_name)
        
    async def load_model(self) -> None:
        """通过子进程加载模型"""
        response = await self._subprocess.send_request(RequestType.LOAD)
        if not response.success:
            raise RuntimeError(f"Failed to load model: {response.error}")
        self._is_loaded = True
        
    async def unload_model(self) -> None:
        """通过子进程卸载模型"""
        response = await self._subprocess.send_request(RequestType.UNLOAD)
        self._is_loaded = False
        
    def set_config(self, config: dict) -> None:
        """更新配置（异步传递给子进程）"""
        self._config = config
        # 配置会在加载时传递，或通过 UPDATE_CONFIG 请求更新
        
    def get_config(self) -> dict:
        return self._config.copy()
        
    async def chat_completions_stream(self, body: dict, ...):
        """通过子进程进行推理"""
        return self._subprocess.infer(body, ...)
```

### 4. PlLocalModelManager 改造

```python
# local_models.py

class PlLocalModelManager:
    def __init__(self) -> None:
        self._model_proxies: Dict[str, PlModelSubprocessProxy] = {}
        self._model_configs: Dict[str, PlLocalModelInfo] = {}
        self._process_manager: Optional[PlProcessManager] = None
        
    def set_process_manager(self, pm: PlProcessManager) -> None:
        self._process_manager = pm
        
    def find_model(self, model_id_or_name: str) -> Optional[PlModelSubprocessProxy]:
        """查找模型 - 返回代理对象"""
        return self._model_proxies.get(model_id_or_name)
        
    async def load_model(self, model_name: str) -> None:
        """加载模型到子进程"""
        proxy = self._model_proxies.get(model_name)
        if proxy is None:
            raise ValueError(f"Model {model_name} not found")
        await proxy.load_model()
```

### 5. 启动流程

```python
# __init__.py

def create_app(...) -> FastAPI:
    # ... 现有代码 ...
    
    # 初始化进程管理器（必须）
    from plllm_mlx.models.process_manager import PlProcessManager
    from plllm_mlx.models.local_models import get_local_model_manager
    
    pm = PlProcessManager.get_instance()
    local_mgr = get_local_model_manager()
    local_mgr.set_process_manager(pm)
    
    logger.info("Process isolation initialized")
```

---

## 实施步骤

### Phase 1: 协议层（无破坏性）

1. 创建 `subprocess_protocol.py`
2. 创建 `model_proxy.py`（基础框架）

### Phase 2: 子进程改造

3. 改造 `model_subprocess.py` - 支持所有请求类型
4. 改造 `process_manager.py` - 支持发送各类请求

### Phase 3: 管理层改造

5. 改造 `local_models.py` - 使用代理类
6. 改造 `routers/models.py` - 适配新接口

### Phase 4: 清理

7. 移除 `model_loader.py` 中的条件判断
8. 移除 `PlProcessManager.enable()/disable()` 方法
9. 更新 `__init__.py` 启动流程

### Phase 5: 测试

10. 单元测试
11. 集成测试
12. 更新文档

---

## 关键接口变更

### 移除的接口

| 接口 | 原因 |
|------|------|
| `PlProcessManager.enable()` | 进程隔离不是可选项 |
| `PlProcessManager.disable()` | 同上 |
| `PlProcessManager.is_enabled()` | 始终为 True |
| `PlModelLoader.is_process_isolation_enabled()` | 始终为 True |

### 变更的接口

| 原接口 | 新接口 | 说明 |
|--------|--------|------|
| `PlLocalModelManager._models_in_memory: Dict[str, PlModelLoader]` | `_model_proxies: Dict[str, PlModelSubprocessProxy]` | 存储代理而非实例 |

### 新增的接口

| 接口 | 用途 |
|------|------|
| `PlModelSubprocessProxy` | 子进程模型代理 |
| `SubprocessRequest/Response` | 通信协议 |
| `RequestType.LOAD/UNLOAD/...` | 请求类型枚举 |

---

## 验证标准

改造完成后，必须满足：

1. **进程隔离**：`ps aux | grep plllm` 显示主进程 + N 个子进程（N=已加载模型数）
2. **内存隔离**：卸载模型后，子进程退出，内存完全释放
3. **功能完整**：所有 API 功能正常（load/unload/infer）
4. **故障隔离**：子进程崩溃不影响主进程和其他模型