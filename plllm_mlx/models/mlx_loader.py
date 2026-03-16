"""
MLX model loader for Apple Silicon.

This module provides the MLX model loader for running LLM inference
on Apple Silicon using the MLX framework.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models import cache as mlx_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from plllm_mlx.helpers import PlChain
from plllm_mlx.logging_config import get_logger

from .base_step_processor import PlStepProcessor
from .kv_cache import PlMessageBasedKVCache
from .model_loader import PlModelLoader, async_ticker
from .special_tokens import SpecialTokens, detect_special_tokens

logger = get_logger(__name__)


class PlMlxSessionStorage:
    """Session storage for MLX inference."""

    pass


def _block_count_helper(total_tokens: int) -> int:
    """Helper function to determine block count for batching."""
    if total_tokens < 300:
        return 3
    elif total_tokens < 800:
        return 5
    elif total_tokens < 1500:
        return 8
    elif total_tokens < 2000:
        return 10
    else:
        return 15


class PlMlxModel(PlModelLoader):
    """
    MLX model loader for Apple Silicon.

    This loader uses the MLX framework for efficient inference on Apple Silicon,
    with support for KV cache optimization and prefix caching.
    """

    def __init__(
        self, model_id: str, step_processor_clz: type[PlStepProcessor]
    ) -> None:
        """
        Initialize the MLX model loader.

        Args:
            model_id: The model name or path.
            step_processor_clz: The step processor class.
        """
        super().__init__(model_id, step_processor_clz)
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()
        self._special_tokens: Optional[SpecialTokens] = None
        self._end_tokens = ["<|end|>"]

        self._max_prompt_tokens = 32 * 1024
        self._max_output_tokens = 16 * 1024
        self._max_model_tokens = 128 * 1024
        self._use_model_config_max_tokens = False

        # Generation parameters
        self._top_p = 0
        self._top_k = 100
        self._min_p = 0.0
        self._repetition_penalty = 1.1
        self._repetition_context_size = 20
        self._xtc_probability = 0.0
        self._xtc_threshold = 0.0
        self._logit_bias = None
        self._logprobs = -1
        self._temperature = 0.8

        self._support_system_role = True

        # KVCache optimization parameters
        self._prefill_step_size = 4096
        self._kv_bits = None
        self._kv_group_size = 32
        self._quantized_kv_start = 0
        self._max_kv_size = None

        # Prefix KV cache
        self._enable_prefix_cache = True
        self._prompt_cache: Optional[PlMessageBasedKVCache] = None
        self._num_layers = 40

        # Message boundary tokens
        self._begin_tokens = ["<|start|>", "<|im_start|>"]
        self._end_tokens = ["<|end|>", "<|im_end|>"]

    @staticmethod
    def model_loader_name() -> str:
        return "mlx"

    def _build_prompt_cache(self):
        """Build a new prompt cache."""
        if self._max_kv_size is None or self._max_kv_size == 0:
            return mlx_cache.make_prompt_cache(self._model, max_kv_size=None)
        else:
            return mlx_cache.make_prompt_cache(
                self._model, max_kv_size=self._max_kv_size
            )

    def _make_sampler(self, parameters: dict = None):
        """Create a sampler from parameters."""
        params = parameters or {}
        return make_sampler(
            params.get("temperature", self._temperature),
            top_p=params.get("top_p", self._top_p),
            top_k=params.get("top_k", self._top_k),
            min_p=params.get("min_p", self._min_p),
            xtc_probability=params.get("xtc_probability", self._xtc_probability),
            xtc_threshold=params.get("xtc_threshold", self._xtc_threshold),
            xtc_special_tokens=[
                self._tokenizer.eos_token_id,
                self._tokenizer.encode("\n"),
            ],
        )

    def _make_logits_processors(self, parameters: dict = None):
        """Create logits processors from parameters."""
        params = parameters or {}
        return make_logits_processors(
            params.get("logit_bias", self._logit_bias),
            params.get(
                "repetition_penalty",
                params.get("presence_penalty", self._repetition_penalty),
            ),
            params.get("repetition_context_size", self._repetition_context_size),
        )

    @async_ticker("PlMlxModel")
    async def ensure_model_loaded(self) -> None:
        """Load the model if not already loaded."""
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return

            loop = asyncio.get_event_loop()

            def _sync_load():
                return load(self.model_name, return_config=True)

            self._model, self._tokenizer, model_config = await loop.run_in_executor(
                None, _sync_load
            )

            def _sync_eval():
                self._model.eval()

            await loop.run_in_executor(None, _sync_eval)

            self._special_tokens = detect_special_tokens(self._tokenizer)
            self._begin_tokens = self._special_tokens.begin_tokens
            self._end_tokens = self._special_tokens.end_tokens

            max_model_tokens = model_config.get("max_position_embeddings", None)
            if max_model_tokens:
                self._use_model_config_max_tokens = True
                self._max_model_tokens = max_model_tokens
                logger.info(
                    f"Model {self.model_name} supports max_position_embeddings={max_model_tokens}"
                )

            self._num_layers = model_config.get("num_hidden_layers", 40)

            if self._enable_prefix_cache:
                self._prompt_cache = PlMessageBasedKVCache(
                    begin_tokens=self._begin_tokens, end_tokens=self._end_tokens
                )
                self._prompt_cache.set_num_layers(self._num_layers)
                logger.info(
                    f"[mlx] Initialized PlMessageBasedKVCache, num_layers={self._num_layers}"
                )

    @async_ticker("PlMlxModel")
    async def ensure_model_unloaded(self) -> None:
        """Unload the model."""
        async with self._lock:
            if self._model is not None:
                del self._model
            if self._tokenizer is not None:
                del self._tokenizer
            mx.metal.clear_cache()
            self._model = None
            self._tokenizer = None

            if self._prompt_cache:
                self._prompt_cache.clear()
                logger.info("[mlx] Cleared PlMessageBasedKVCache")

    def set_config(self, model_config: dict) -> None:
        """Set model configuration."""
        if not self._use_model_config_max_tokens:
            if "max_model_tokens" in model_config:
                self._max_model_tokens = int(model_config["max_model_tokens"])
            if "max_prompt_tokens" in model_config:
                self._max_prompt_tokens = int(model_config["max_prompt_tokens"])
            if "max_output_tokens" in model_config:
                self._max_output_tokens = int(model_config["max_output_tokens"])

        if "top_p" in model_config:
            self._top_p = float(model_config["top_p"])
        if "top_k" in model_config:
            self._top_k = int(model_config["top_k"])
        if "min_p" in model_config:
            self._min_p = float(model_config["min_p"])
        if "repetition_penalty" in model_config:
            self._repetition_penalty = float(model_config["repetition_penalty"])
        if "repetition_context_size" in model_config:
            self._repetition_context_size = int(model_config["repetition_context_size"])
        if "xtc_probability" in model_config:
            self._xtc_probability = float(model_config["xtc_probability"])
        if "xtc_threshold" in model_config:
            self._xtc_threshold = float(model_config["xtc_threshold"])
        if "logit_bias" in model_config:
            self._logit_bias = model_config["logit_bias"]
        if "logprobs" in model_config:
            self._logprobs = int(model_config["logprobs"])
        if "temperature" in model_config:
            self._temperature = float(model_config["temperature"])
        if "support_system_role" in model_config:
            self._support_system_role = model_config["support_system_role"]

        # KVCache parameters
        if "prefill_step_size" in model_config:
            self._prefill_step_size = int(model_config["prefill_step_size"])
        if "kv_bits" in model_config:
            kv_bits = model_config["kv_bits"]
            self._kv_bits = int(kv_bits) if kv_bits is not None else None
        if "kv_group_size" in model_config:
            self._kv_group_size = int(model_config["kv_group_size"])
        if "quantized_kv_start" in model_config:
            self._quantized_kv_start = int(model_config["quantized_kv_start"])
        if "max_kv_size" in model_config:
            max_kv = model_config["max_kv_size"]
            self._max_kv_size = int(max_kv) if max_kv is not None else None

        # Prefix cache
        if "enable_prefix_cache" in model_config:
            self._enable_prefix_cache = bool(model_config["enable_prefix_cache"])
        if "begin_tokens" in model_config:
            self._begin_tokens = model_config["begin_tokens"]
        if "end_tokens" in model_config:
            self._end_tokens = model_config["end_tokens"]

    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            "max_model_tokens": self._max_model_tokens,
            "max_prompt_tokens": self._max_prompt_tokens,
            "max_output_tokens": self._max_output_tokens,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "min_p": self._min_p,
            "repetition_penalty": self._repetition_penalty,
            "repetition_context_size": self._repetition_context_size,
            "xtc_probability": self._xtc_probability,
            "xtc_threshold": self._xtc_threshold,
            "prefill_step_size": self._prefill_step_size,
            "kv_bits": self._kv_bits,
            "kv_group_size": self._kv_group_size,
            "quantized_kv_start": self._quantized_kv_start,
            "max_kv_size": self._max_kv_size,
            "logit_bias": self._logit_bias,
            "logprobs": self._logprobs,
            "temperature": self._temperature,
            "support_system_role": self._support_system_role,
            "enable_prefix_cache": self._enable_prefix_cache,
            "begin_tokens": self._begin_tokens,
            "end_tokens": self._end_tokens,
        }

    def prepare_prompt(self, body: dict) -> PlMlxSessionStorage:
        """Prepare prompt from request body."""
        input_question = body.get("prompt", body.get("messages", None))
        sampler = self._make_sampler(body)
        logits_processors = self._make_logits_processors(body)

        # Calculate max tokens
        requested_max_tokens = body.get("max_tokens", -1)
        if requested_max_tokens == -1:
            requested_max_tokens = body.get("max_completion_tokens", -1)
        if requested_max_tokens <= 0:
            requested_max_tokens = self._max_output_tokens

        dynamic_max_prompt_tokens = self._max_model_tokens - requested_max_tokens
        if dynamic_max_prompt_tokens < 4096:
            dynamic_max_prompt_tokens = 4096
            requested_max_tokens = self._max_model_tokens - dynamic_max_prompt_tokens

        logger.debug(
            f"Token allocation: model_total={self._max_model_tokens}, "
            f"output={requested_max_tokens}, prompt={dynamic_max_prompt_tokens}"
        )

        # Parse messages
        if isinstance(input_question, list):
            msgs = []
            for msg in input_question:
                content = msg.get("content", None)
                role = msg.get("role", "user")
                if role == "system" and not self._support_system_role:
                    role = "assistant"
                if isinstance(content, str):
                    msgs.append({"role": role, "content": content})
                elif isinstance(content, dict):
                    content_type = content.get("type", "non-text")
                    if content_type == "text":
                        msgs.append({"role": role, "content": content.get("text")})
                elif isinstance(content, list):
                    temp_msg = []
                    for c in content:
                        if isinstance(c, str):
                            temp_msg.append(c)
                        elif isinstance(c, dict):
                            content_type = c.get("type", "non-text")
                            if content_type == "text":
                                temp_msg.append(c.get("text", ""))
                    msgs.append({"role": role, "content": "\n".join(temp_msg).strip()})
        else:
            msgs = [{"role": "user", "content": input_question}]

        tools = body.get("tools", None)
        tool_choice = body.get("tool_choice", None)

        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=False,
            tokenize=False,
            tools=tools,
            tool_choice=tool_choice,
        )

        gen_prompt = prompt
        matched_chain = None
        message_splits = (
            self._prompt_cache.split_prompt_by_messages(prompt)
            if self._prompt_cache
            else []
        )

        if self._enable_prefix_cache and self._prompt_cache:
            matched_chain = self._prompt_cache.get_kv_cache(message_splits)
            if matched_chain is None:
                matched_chain = PlChain([], cache_item=self._build_prompt_cache())
            gen_prompt = "\n".join(
                [m.full_content for m in message_splits[len(matched_chain.node_ids) :]]
            )
        else:
            matched_chain = PlChain([m.msg_id for m in message_splits])

        logger.debug(f"[mlx] Using prompt: {len(gen_prompt)} chars")

        session_storage = PlMlxSessionStorage()
        setattr(session_storage, "matched_chain", matched_chain)
        setattr(session_storage, "prompt", gen_prompt)
        setattr(session_storage, "sampler", sampler)
        setattr(session_storage, "logits_processors", logits_processors)
        setattr(session_storage, "max_tokens", requested_max_tokens)
        setattr(session_storage, "message_splits", message_splits)

        return session_storage

    async def stream_generate(self, session_object: PlMlxSessionStorage):
        """Stream generate tokens."""
        loop = asyncio.get_event_loop()
        matched_chain = session_object.matched_chain

        def _sync_stream_generate():
            for token in stream_generate(
                self._model,
                self._tokenizer,
                prompt=session_object.prompt,
                max_tokens=session_object.max_tokens,
                prompt_cache=matched_chain.cache_item,
                sampler=session_object.sampler,
                logits_processors=session_object.logits_processors,
                prefill_step_size=self._prefill_step_size,
                kv_group_size=self._kv_group_size,
                kv_bits=self._kv_bits,
                quantized_kv_start=self._quantized_kv_start,
                draft_model=None,
            ):
                yield token

        stpp = self.step_processor_clz(self._special_tokens)

        for gr in await loop.run_in_executor(None, lambda: _sync_stream_generate()):
            chunk = stpp.step(gr)
            if stpp.total_tokens >= self._max_output_tokens:
                logger.info("reach the max output token size, force to stop!")
                break
            if chunk is not None:
                yield chunk
            if not stpp.is_running:
                break

        while stpp.unprocessed_text != "":
            chunk = stpp.step(None)
            if chunk is not None:
                yield chunk

        tool_calls = stpp.tool_calls()
        for tc in tool_calls:
            yield tc

        if self._enable_prefix_cache and self._prompt_cache:
            self._prompt_cache.add_kv_cache(
                [m.msg_id for m in session_object.message_splits],
                matched_chain.cache_item,
                is_resp_cache=True,
            )

        finish_chunk = stpp.finish()
        yield finish_chunk

    async def completion_stream_generate(self, session_object: PlMlxSessionStorage):
        """Stream generate for completion mode."""
        async for chunk in self.stream_generate(session_object):
            yield chunk


# Register the MLX model loader
PlModelLoader.registerModelLoader("mlx", PlMlxModel)

logger.debug("Registered MLX model loader")
