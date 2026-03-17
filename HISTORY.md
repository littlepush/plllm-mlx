# History

## 2025-03-18: Comprehensive Test Suite Implementation

### Task
Create a complete test suite for plllm-mlx including unit tests (UT) and integration tests, supporting `uv run pytest` to run all tests.

### Implementation

**Unit Tests (217 test cases)**
- `test_client.py`: PlClient API tests
- `test_config.py`: Configuration loading/merging tests
- `test_local_models.py`: Local model management tests
- `test_model_detector.py`: Model detection tests
- `test_utils.py`: Utility function tests
- `test_helpers/`: Helper module tests
  - `test_chain_cache.py`: Chain cache tests
  - `test_chat_helper.py`: Chat completion helper tests
  - `test_chunk_helper.py`: Chunk data structure tests
  - `test_path_helper.py`: Path handling tests
  - `test_step_info.py`: Step info tests
  - `test_toolcall_helper.py`: Tool call parsing tests

**Integration Tests (7 bash scripts)**
- `test_01_health.sh`: Health check endpoints
- `test_02_model_list.sh`: Model listing API
- `test_03_model_load_unload.sh`: Model lifecycle
- `test_04_chat_api.sh`: Chat completion API
- `test_05_chat_cli.sh`: CLI chat command
- `test_06_full_flow.sh`: Full service workflow
- `test_07_subprocess.sh`: Subprocess management

**Features**
- `run_tests.sh` with `--quiet` (summary only) and `--redirect FILE` options
- macOS-compatible timeout using perl
- QUIET mode for cleaner CI output
- Real model testing with automatic service start/stop

### Key Learnings
1. macOS `pgrep -c` syntax differs from Linux; use `pgrep -f "pattern" | wc -l`
2. Process detection patterns: `plllm-mlx run-server` for main, `subprocess/python/main.py` for subprocess
3. `jq` outputs JSON without spaces: `"is_loaded":true` not `"is_loaded": true`
4. macOS lacks `timeout` command; use `perl -e 'alarm shift; exec @ARGV'` as alternative

## 2025-03-18: v1.5.9 - Fix subprocess isolation implementation

### Problem
Subprocess isolation was designed but not working correctly:
1. Model loading failed: `plllm-mlx` command not found when starting subprocess
2. SSE streaming format was broken: missing `\n\n` suffix on data lines
3. Non-streaming chat returned 500 error: `PlModelProxy` missing `chat_completions_restful` method

### Root Cause Analysis
1. **Subprocess startup**: Used `plllm-mlx subprocess serve` command, but `plllm-mlx` wasn't in PATH when service started via LaunchAgent
2. **SSE format**: `httpx.aiter_lines()` strips newline characters, but we didn't add them back
3. **Missing method**: `PlModelProxy` was a wrapper but didn't implement all methods that `PlModelLoader` had

### Solution
1. **Fix subprocess startup**: Use `sys.executable` to run `subprocess/python/main.py` directly
   - Main process discovers Python interpreter path via `sys.executable`
   - Creates command: `<python> <project>/plllm_mlx/subprocess/python/main.py --socket <path>`
   - Works regardless of how main process was started
2. **Fix SSE format**: Add `\n\n` suffix after iterating lines
3. **Add missing method**: Implement `chat_completions_restful` in `PlModelProxy` that accumulates streaming response
4. **Filter kwargs**: Remove non-serializable kwargs like `cancel_event` before sending to subprocess

### Key Learnings
1. When spawning child processes, always use `sys.executable` to ensure same Python interpreter
2. SSE format requires `\n\n` after each data line; `iter_lines()` strips them
3. Proxy objects need to implement all methods of the wrapped object

## 2025-03-17: v1.5.7 - Fix KV cache not working in multi-turn tool call conversations

### Problem
KV cache was not working in multi-turn conversations with tool calls:
1. First request: cache was stored as temp_cache
2. Second request: cache was not found, full prefill happened
3. No speedup in subsequent requests

### Root Cause Analysis
1. **Cache lookup logic order was wrong**: The code checked temp_cache upgrade condition before checking full match
2. **Tool call response handling**: After yielding tool call, the consumer returned immediately without consuming the rest of the generator, so cache add logic never executed
3. **Logging config issue**: Config file's `logging.level` was ignored, command line default was used instead

### Solution
1. **Fix cache lookup order**: Check full match first, then handle temp_cache upgrade
   - Full match with temp_cache only: search shorter chain (retry scenario)
   - Shorter match with temp_cache: check upgrade condition (+2 messages for normal dialog)
2. **Fix tool call handling**: After yielding tool call, continue consuming generator but don't yield to client
3. **Fix logging config**: Read log level from `logging.level` in config file
4. **Remove per-token debug logs**: Removed excessive debug logs that impacted performance

### Key Learnings
1. When implementing cache upgrade logic, full match must be handled before partial match
2. Async generators must be fully consumed for cleanup code to run
3. Debug logs should be conditional, not per-token

## 2025-03-17: v1.5.6 - Fix thinking_step_processor infinite loop

### Problem
After multiple rounds of chat, the service would get stuck with 99% CPU usage.

### Root Cause Analysis
1. `sample` command showed `builtin_any` + `PyUnicode_Contains` consuming most CPU
2. This corresponded to `any(bt in step_text_to_process for bt in tokens.begin_tokens)`
3. The issue: when model outputs tokens one by one (e.g., `<|im_start|>`, `assistant`, `\n`, `think`), the processor accumulates text while waiting for complete `think_start_token`
4. `step_text_to_process` grows unbounded, causing O(n) string searches on each step
5. After many rounds, accumulated text could be thousands of characters, causing performance degradation

### Solution
1. Added `MAX_ACCUMULATE_CHARS = 100` limit
2. When accumulated text exceeds limit, assume model is outputting content without thinking tags
3. Also handle case when no `begin_token` is found (output as content directly)

### Stress Test
- 100 iterations, 15 rounds per iteration
- All passed with normal CPU usage (< 1%)
- No infinite loops detected

### Key Learnings
1. When processing streaming tokens, always have a fallback for unbounded accumulation
2. The `any()` function with generator expression can be expensive if the search string is large
3. Compare with original working code to understand the correct logic flow