# C++ Implementation Plan for plllm-mlx

## Overview

This document outlines the technical plan for implementing a C++ version of plllm-mlx's model_loader and kv_cache components, with support for continuous batching and both MLX-LM and MLX-VLM models.

**Goals:**
1. Implement model_loader in C++ using MLX C++ API
2. Support MLX-VLM (vision-language models)
3. Implement continuous batching
4. Implement advanced KV cache management

**Priority Order:** model_loader > mlx-vlm > batching > kv_cache

---

## Technical Background

### MLX C++ API Capabilities

| Feature | Python API | C++ API | Notes |
|---------|-----------|---------|-------|
| Array operations | ✅ | ✅ | Full support |
| safetensors load/save | ✅ | ✅ | `mx::load_safetensors()` |
| Metal GPU acceleration | ✅ | ✅ | Automatic on macOS |
| nn.Module | ✅ | ❌ | **Must implement manually** |
| stream_generate | ✅ | ❌ | **Must implement manually** |
| KV Cache | ✅ | ❌ | **Must implement manually** |

**Key Limitation:** MLX C++ API is a low-level tensor library without high-level neural network modules. All model architectures must be implemented from scratch using basic operations.

### Target Model: Qwen2.5-7B/14B

**Architecture Highlights:**
- **GQA (Grouped Query Attention)**: 28 Q heads / 4 KV heads (7B) - 85%+ KV cache memory savings
- **RoPE with large base**: θ = 1,000,000 for 128K context support
- **Attention bias**: Q/K/V projections have bias (unlike Llama)
- **SwiGLU activation**: gate_proj × SiLU(up_proj)
- **RMSNorm**: Pre-normalization architecture

**Model Parameters (7B):**
```
Layers: 28
Hidden size: 3584
Intermediate size: 18944
Attention heads: 28
KV heads: 4
Vocab size: 152064
RoPE theta: 1,000,000
```

---

## Phase 1: Model Loader (Core)

**Duration:** 2-3 weeks

### 1.1 Project Setup

```
plllm-mlx-cpp/
├── CMakeLists.txt
├── src/
│   ├── model/
│   │   ├── qwen2.hpp          # Qwen2 model definition
│   │   ├── qwen2.cpp
│   │   ├── attention.hpp       # Attention layer with GQA
│   │   ├── attention.cpp
│   │   ├── mlp.hpp             # SwiGLU MLP
│   │   ├── mlp.cpp
│   │   ├── rms_norm.hpp        # RMSNorm layer
│   │   ├── rms_norm.cpp
│   │   └── model_loader.hpp    # Weight loading utilities
│   ├── tokenizer/
│   │   ├── tokenizer.hpp       # Tokenizer interface
│   │   └── tokenizers_cpp.hpp  # tokenizers-cpp wrapper
│   ├── inference/
│   │   ├── generator.hpp       # Streaming generation
│   │   └── sampler.hpp         # Sampling strategies
│   └── main.cpp
├── third_party/
│   ├── tokenizers-cpp/         # HuggingFace tokenizers
│   └── nlohmann-json/
└── tests/
```

### 1.2 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.27)
project(plllm-mlx-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MLX
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# Sources
add_library(plllm-mlx-cpp
  src/model/qwen2.cpp
  src/model/attention.cpp
  src/model/mlp.cpp
  src/model/rms_norm.cpp
  src/inference/generator.cpp
  src/inference/sampler.cpp
)

target_link_libraries(plllm-mlx-cpp PRIVATE mlx)
target_include_directories(plllm-mlx-cpp PUBLIC ${CMAKE_SOURCE_DIR}/src)

# Main executable
add_executable(plllm-cli src/main.cpp)
target_link_libraries(plllm-cli PRIVATE plllm-mlx-cpp)
```

### 1.3 Core Components Implementation

#### 1.3.1 RMSNorm

```cpp
// src/model/rms_norm.hpp
#pragma once
#include "mlx/mlx.h"

namespace plllm::model {

class RMSNorm {
public:
    RMSNorm(int hidden_size, float eps = 1e-6f);
    
    mx::array forward(const mx::array& x);
    
    void load_weights(const mx::array& weight);
    
private:
    int hidden_size_;
    float eps_;
    mx::array weight_;
};

} // namespace plllm::model
```

```cpp
// src/model/rms_norm.cpp
#include "rms_norm.hpp"

namespace plllm::model {

RMSNorm::RMSNorm(int hidden_size, float eps)
    : hidden_size_(hidden_size), eps_(eps) {
    weight_ = mx::ones({hidden_size}, mx::float32);
}

mx::array RMSNorm::forward(const mx::array& x) {
    // RMS = sqrt(mean(x^2) + eps)
    auto x_sq = mx::square(x);
    auto mean_sq = mx::mean(x_sq, -1, true);
    auto rms = mx::rsqrt(mean_sq + eps_);
    
    // Normalize and scale
    return x * rms * weight_;
}

void RMSNorm::load_weights(const mx::array& weight) {
    weight_ = weight;
}

} // namespace plllm::model
```

#### 1.3.2 SwiGLU MLP

```cpp
// src/model/mlp.hpp
#pragma once
#include "mlx/mlx.h"

namespace plllm::model {

class SwiGLUMLP {
public:
    SwiGLUMLP(int hidden_size, int intermediate_size);
    
    mx::array forward(const mx::array& x);
    
    void load_weights(
        const mx::array& gate_proj_weight,
        const mx::array& up_proj_weight,
        const mx::array& down_proj_weight
    );
    
private:
    int hidden_size_;
    int intermediate_size_;
    mx::array gate_proj_weight_;  // [intermediate_size, hidden_size]
    mx::array up_proj_weight_;    // [intermediate_size, hidden_size]
    mx::array down_proj_weight_;  // [hidden_size, intermediate_size]
};

} // namespace plllm::model
```

```cpp
// src/model/mlp.cpp
#include "mlp.hpp"

namespace plllm::model {

SwiGLUMLP::SwiGLUMLP(int hidden_size, int intermediate_size)
    : hidden_size_(hidden_size), intermediate_size_(intermediate_size) {}

mx::array SwiGLUMLP::forward(const mx::array& x) {
    // gate = SiLU(x @ gate_proj.T)
    auto gate = mx::matmul(x, gate_proj_weight_.T());
    gate = mx::silu(gate);
    
    // up = x @ up_proj.T
    auto up = mx::matmul(x, up_proj_weight_.T());
    
    // output = (gate * up) @ down_proj.T
    auto hidden = gate * up;
    return mx::matmul(hidden, down_proj_weight_.T());
}

void SwiGLUMLP::load_weights(
    const mx::array& gate_proj_weight,
    const mx::array& up_proj_weight,
    const mx::array& down_proj_weight
) {
    gate_proj_weight_ = gate_proj_weight;
    up_proj_weight_ = up_proj_weight;
    down_proj_weight_ = down_proj_weight;
}

} // namespace plllm::model
```

#### 1.3.3 Attention with GQA and RoPE

```cpp
// src/model/attention.hpp
#pragma once
#include "mlx/mlx.h"
#include <optional>

namespace plllm::model {

struct KVCache {
    mx::array key;
    mx::array value;
    int offset = 0;
};

class Attention {
public:
    Attention(
        int hidden_size,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        float rope_base = 1000000.0f
    );
    
    mx::array forward(
        const mx::array& x,
        std::optional<KVCache>& cache,
        const mx::array& positions
    );
    
    void load_weights(
        const mx::array& q_proj_weight,
        const mx::array& q_proj_bias,
        const mx::array& k_proj_weight,
        const mx::array& k_proj_bias,
        const mx::array& v_proj_weight,
        const mx::array& v_proj_bias,
        const mx::array& o_proj_weight
    );

private:
    mx::array apply_rope(
        const mx::array& x,
        const mx::array& positions
    );
    
    mx::array repeat_kv(const mx::array& x, int n);

private:
    int hidden_size_;
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    int num_kv_groups_;
    float rope_base_;
    float scale_;
    
    mx::array q_proj_weight_;
    mx::array q_proj_bias_;
    mx::array k_proj_weight_;
    mx::array k_proj_bias_;
    mx::array v_proj_weight_;
    mx::array v_proj_bias_;
    mx::array o_proj_weight_;
    
    mx::array inv_freq_;  // Precomputed RoPE frequencies
};

} // namespace plllm::model
```

```cpp
// src/model/attention.cpp
#include "attention.hpp"
#include <cmath>

namespace plllm::model {

Attention::Attention(
    int hidden_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float rope_base
) : hidden_size_(hidden_size),
    num_heads_(num_heads),
    num_kv_heads_(num_kv_heads),
    head_dim_(head_dim),
    rope_base_(rope_base),
    scale_(1.0f / std::sqrt(static_cast<float>(head_dim))) {
    
    num_kv_groups_ = num_heads / num_kv_heads;
    
    // Precompute RoPE inverse frequencies
    std::vector<float> freqs(head_dim / 2);
    for (int i = 0; i < head_dim / 2; ++i) {
        freqs[i] = 1.0f / std::pow(rope_base, 
            static_cast<float>(2 * i) / head_dim);
    }
    inv_freq_ = mx::array(freqs.data(), {head_dim / 2}, mx::float32);
}

mx::array Attention::apply_rope(
    const mx::array& x,
    const mx::array& positions
) {
    // x: [batch, seq, heads, head_dim]
    // positions: [seq]
    
    // Compute frequencies: [seq, head_dim/2]
    auto freqs = mx::matmul(
        positions.astype(mx::float32).reshape({-1, 1}),
        inv_freq_.reshape({1, -1})
    );
    
    // Create cos/sin: [seq, head_dim]
    auto cos = mx::cos(freqs);
    auto sin = mx::sin(freqs);
    
    // Split x into two halves
    auto x1 = x[mx::slice(0, x.shape(-1) / 2)];
    auto x2 = x[mx::slice(x.shape(-1) / 2, x.shape(-1))];
    
    // Apply rotation
    auto rotated = mx::concatenate({-x2 * sin + x1 * cos, x1 * sin + x2 * cos}, -1);
    
    return rotated;
}

mx::array Attention::repeat_kv(const mx::array& x, int n) {
    // x: [batch, seq, kv_heads, head_dim]
    // output: [batch, seq, kv_heads * n, head_dim]
    auto shape = x.shape();
    auto new_shape = std::vector<int>{shape[0], shape[1], shape[2], n, shape[3]};
    return mx::broadcast_to(x.reshape(new_shape), 
        {shape[0], shape[1], shape[2] * n, shape[3]});
}

mx::array Attention::forward(
    const mx::array& x,
    std::optional<KVCache>& cache,
    const mx::array& positions
) {
    int batch = x.shape(0);
    int seq = x.shape(1);
    
    // QKV projections with bias
    auto q = mx::addmm(q_proj_bias_, x, q_proj_weight_.T());
    auto k = mx::addmm(k_proj_bias_, x, k_proj_weight_.T());
    auto v = mx::addmm(v_proj_bias_, x, v_proj_weight_.T());
    
    // Reshape to [batch, seq, heads, head_dim]
    q = q.reshape({batch, seq, num_heads_, head_dim_});
    k = k.reshape({batch, seq, num_kv_heads_, head_dim_});
    v = v.reshape({batch, seq, num_kv_heads_, head_dim_});
    
    // Apply RoPE to Q and K
    q = apply_rope(q, positions);
    k = apply_rope(k, positions);
    
    // Update KV cache
    if (cache.has_value()) {
        if (cache->offset == 0) {
            cache->key = k;
            cache->value = v;
        } else {
            cache->key = mx::concatenate({cache->key, k}, 1);
            cache->value = mx::concatenate({cache->value, v}, 1);
        }
        cache->offset += seq;
        
        k = cache->key;
        v = cache->value;
    }
    
    // GQA: Repeat KV heads to match Q heads
    k = repeat_kv(k, num_kv_groups_);
    v = repeat_kv(v, num_kv_groups_);
    
    // Scaled dot-product attention
    // [batch, heads, seq_q, head_dim] @ [batch, heads, head_dim, seq_k]
    auto scores = mx::matmul(
        q.transpose(0, 2, 1, 3),  // [batch, heads, seq, head_dim]
        k.transpose(0, 2, 3, 1)   // [batch, heads, head_dim, seq]
    ) * scale_;
    
    // Apply causal mask
    if (cache.has_value()) {
        // Causal mask for generation
        auto mask = mx::triu(mx::ones({q.shape(2), k.shape(2)}), cache->offset - seq + 1);
        scores = scores - mask * 1e9f;
    } else {
        // Full causal mask for prefill
        auto mask = mx::triu(mx::ones({seq, seq}), 1);
        scores = scores - mask * 1e9f;
    }
    
    // Softmax
    auto attn_weights = mx::softmax(scores, -1);
    
    // Apply attention to values
    auto output = mx::matmul(attn_weights, 
        v.transpose(0, 2, 1, 3));  // [batch, heads, seq, head_dim]
    
    // Reshape back
    output = output.transpose(0, 2, 1, 3).reshape({batch, seq, hidden_size_});
    
    // Output projection
    return mx::matmul(output, o_proj_weight_.T());
}

void Attention::load_weights(
    const mx::array& q_proj_weight,
    const mx::array& q_proj_bias,
    const mx::array& k_proj_weight,
    const mx::array& k_proj_bias,
    const mx::array& v_proj_weight,
    const mx::array& v_proj_bias,
    const mx::array& o_proj_weight
) {
    q_proj_weight_ = q_proj_weight;
    q_proj_bias_ = q_proj_bias;
    k_proj_weight_ = k_proj_weight;
    k_proj_bias_ = k_proj_bias;
    v_proj_weight_ = v_proj_weight;
    v_proj_bias_ = v_proj_bias;
    o_proj_weight_ = o_proj_weight;
}

} // namespace plllm::model
```

### 1.4 Tokenizer Integration

**Recommendation:** Use **tokenizers-cpp** for Qwen2.5 support

```cpp
// src/tokenizer/tokenizer.hpp
#pragma once
#include <string>
#include <vector>
#include <memory>

namespace plllm::tokenizer {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    
    virtual std::vector<int> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<int>& ids) = 0;
    
    virtual int bos_token_id() const = 0;
    virtual int eos_token_id() const = 0;
    virtual int pad_token_id() const = 0;
};

std::unique_ptr<Tokenizer> create_tokenizer(const std::string& tokenizer_path);

} // namespace plllm::tokenizer
```

**Dependencies:**
- Rust toolchain (for compiling tokenizers-cpp)
- Or pre-built tokenizers-cpp library

### 1.5 Streaming Generator

```cpp
// src/inference/generator.hpp
#pragma once
#include "mlx/mlx.h"
#include <functional>
#include <vector>
#include <queue>

namespace plllm::inference {

class StreamingGenerator {
public:
    using TokenCallback = std::function<void(int token, const std::string& text)>;
    
    void generate(
        const std::vector<int>& prompt,
        int max_tokens,
        float temperature,
        TokenCallback callback
    );
    
private:
    int sample_token(const mx::array& logits, float temperature);
};

} // namespace plllm::inference
```

### 1.6 Milestones

| Week | Tasks | Deliverable |
|------|-------|-------------|
| 1 | Project setup, RMSNorm, MLP | Basic layers working |
| 2 | Attention with GQA, RoPE | Single layer forward pass |
| 3 | Full model, weight loading, tokenizer | End-to-end generation |

---

## Phase 2: MLX-VLM Support

**Duration:** 1-2 weeks

### 2.1 Vision Encoder

```cpp
// src/model/vision_encoder.hpp
#pragma once
#include "mlx/mlx.h"

namespace plllm::model {

class VisionEncoder {
public:
    VisionEncoder(int hidden_size, int num_layers, int num_heads);
    
    mx::array forward(const mx::array& pixel_values);
    
    void load_weights(const std::unordered_map<std::string, mx::array>& weights);
    
private:
    // ViT or SigLIP encoder
    int hidden_size_;
    std::vector<std::unique_ptr<Attention>> layers_;
};

} // namespace plllm::model
```

### 2.2 Vision-Language Model

```cpp
// src/model/vlm.hpp
#pragma once
#include "mlx/mlx.h"
#include "qwen2.hpp"
#include "vision_encoder.hpp"

namespace plllm::model {

class Qwen2VLM {
public:
    Qwen2VLM(const ModelConfig& config);
    
    mx::array forward(
        const mx::array& input_ids,
        const mx::array& pixel_values,
        const mx::array& image_sizes
    );
    
private:
    std::unique_ptr<VisionEncoder> vision_encoder_;
    std::unique_ptr<Qwen2ForCausalLM> language_model_;
    mx::array projector_;  // Vision-to-language projection
};

} // namespace plllm::model
```

---

## Phase 3: Continuous Batching

**Duration:** 2-3 weeks

### 3.1 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Scheduler                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Request Queue                                           │ │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                        │ │
│  │  │ R1  │ │ R2  │ │ R3  │ │ R4  │                        │ │
│  │  └─────┘ └─────┘ └─────┘ └─────┘                        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Batch Manager                                           │ │
│  │  - Prefill phase: Process prompts                        │ │
│  │  - Decode phase: Generate tokens                         │ │
│  │  - Dynamic add/remove requests                           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  KV Cache Pool                                           │ │
│  │  - Block-based allocation                                │ │
│  │  - Prefix sharing                                        │ │
│  │  - Memory management                                     │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Scheduler

```cpp
// src/scheduler/scheduler.hpp
#pragma once
#include <queue>
#include <vector>
#include <memory>

namespace plllm::scheduler {

struct Request {
    int id;
    std::vector<int> prompt;
    int max_tokens;
    int generated_tokens = 0;
    bool finished = false;
};

class ContinuousBatchingScheduler {
public:
    void add_request(const Request& request);
    void step();  // One generation step
    std::vector<int> get_finished_requests();
    
private:
    std::queue<Request> waiting_queue_;
    std::vector<Request> running_batch_;
    int max_batch_size_ = 32;
};

} // namespace plllm::scheduler
```

### 3.3 Block-based KV Cache

```cpp
// src/cache/block_cache.hpp
#pragma once
#include "mlx/mlx.h"
#include <unordered_map>
#include <list>

namespace plllm::cache {

struct Block {
    mx::array key;
    mx::array value;
    int ref_count = 1;
};

class BlockCache {
public:
    BlockCache(int block_size, int num_layers, int num_heads, int head_dim);
    
    int allocate_block();
    void free_block(int block_id);
    void increment_ref(int block_id);
    
    Block& get_block(int block_id);
    
private:
    int block_size_;
    std::unordered_map<int, Block> blocks_;
    std::list<int> free_list_;
    int next_block_id_ = 0;
};

} // namespace plllm::cache
```

---

## Phase 4: Advanced KV Cache

**Duration:** 1-2 weeks

### 4.1 Paged Attention Style Cache

```cpp
// src/cache/paged_cache.hpp
#pragma once
#include "block_cache.hpp"
#include <vector>

namespace plllm::cache {

struct SequenceCache {
    std::vector<int> block_ids;  // Block table
    int length = 0;
    int prefix_hash = 0;
};

class PagedCacheManager {
public:
    SequenceCache create_sequence(const std::vector<int>& prefix_block_ids);
    void append_token(SequenceCache& seq, const mx::array& key, const mx::array& value);
    void free_sequence(SequenceCache& seq);
    
    // Prefix sharing
    std::optional<SequenceCache> find_prefix(const std::vector<int>& tokens);
    
private:
    BlockCache block_cache_;
    std::unordered_map<int, SequenceCache> sequences_;
    std::unordered_map<int, int> prefix_to_sequence_;  // For prefix sharing
};

} // namespace plllm::cache
```

---

## Dependencies

### Required
- **MLX C++** (via Python package or source build)
- **CMake** >= 3.27
- **C++20** compiler
- **tokenizers-cpp** (requires Rust)

### Optional
- **nlohmann/json** - JSON parsing
- **spdlog** - Logging
- **Catch2** - Testing

---

## Build Instructions

```bash
# Install MLX
pip install mlx>=0.22

# Install Rust (for tokenizers-cpp)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build
cd plllm-mlx-cpp
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)

# Run
./plllm-cli --model ~/models/Qwen2.5-7B-Instruct --prompt "Hello"
```

---

## Testing Strategy

### Unit Tests
- RMSNorm forward pass
- SwiGLU MLP forward pass
- Attention with GQA
- RoPE position encoding
- Tokenizer encode/decode

### Integration Tests
- Full model forward pass
- Text generation
- KV cache operations
- Continuous batching

### Benchmarks
- Time to first token (TTFT)
- Tokens per second (TPS)
- Memory usage
- Batch throughput

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| MLX C++ API limitations | High | Manual implementation of all components |
| Tokenizer complexity | Medium | Use tokenizers-cpp, fallback to Python |
| Performance issues | Medium | Profile early, optimize hot paths |
| Continuous batching bugs | High | Extensive testing, start with simple batching |

---

## Success Criteria

### Phase 1 (Model Loader)
- [ ] Can load Qwen2.5-7B weights from safetensors
- [ ] Can generate text with streaming output
- [ ] Tokenization works correctly

### Phase 2 (VLM)
- [ ] Can process images with vision encoder
- [ ] Can generate responses to image+text prompts

### Phase 3 (Batching)
- [ ] Can handle multiple concurrent requests
- [ ] Throughput improves with batch size

### Phase 4 (KV Cache)
- [ ] Block-based cache reduces memory usage
- [ ] Prefix sharing works correctly

---

## References

1. [MLX C++ Documentation](https://ml-explore.github.io/mlx/build/html/cpp/ops.html)
2. [Qwen2.5 Architecture](../docs/QWEN2.5_ARCHITECTURE.md)
3. [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)
4. [vLLM PagedAttention](https://github.com/vllm-project/vllm)
5. [Continuous Batching Paper](https://arxiv.org/abs/2309.06180)

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-03-15 | 1.0 | Initial plan |