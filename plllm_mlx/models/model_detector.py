"""
HuggingFace model auto-detection utility.

This module provides utilities for automatically detecting model characteristics
from HuggingFace model names or local directories, including:
1. Which loader to use (mlx vs mlxvlm)
2. Which step processor to use (base vs qwen3_thinking vs openai)
3. Model configuration parameters
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from plllm_mlx.helpers import get_model_snapshot_path
from plllm_mlx.logging_config import get_logger

logger = get_logger(__name__)


class PlModelDetector:
    """HuggingFace model auto-detection utility."""

    @staticmethod
    def detect_from_local(model_name_or_path: str) -> Dict[str, Any]:
        """
        Detect model information from local directory.

        Args:
            model_name_or_path: HuggingFace model name (e.g., "Qwen/Qwen3-8B")
                               or local path to model directory.

        Returns:
            Dictionary containing loader, step_processor, and config info.
        """
        try:
            model_path = Path(model_name_or_path)
            if not model_path.is_absolute():
                snapshot_path = get_model_snapshot_path(model_name_or_path)
                if snapshot_path is None:
                    logger.warning(f"Model not found in cache: {model_name_or_path}")
                    return {
                        "model_name": model_name_or_path,
                        "error": "Model not found in cache",
                    }
                model_path = snapshot_path

            config_path = model_path / "config.json"
            if not config_path.exists():
                logger.warning(f"Config not found: {config_path}")
                return {
                    "model_name": model_name_or_path,
                    "error": "config.json not found",
                }

            with open(config_path, encoding="utf-8") as f:
                config_data = json.load(f)

            result: Dict[str, Any] = {
                "model_name": model_name_or_path,
                "model_type": config_data.get("model_type", "unknown"),
                "architectures": config_data.get("architectures", []),
                "loader": "mlx",
                "step_processor": "base",
                "is_vlm": False,
                "is_qwen3": False,
                "config": {},
            }

            if "vision_config" in config_data:
                result["is_vlm"] = True
                result["loader"] = "mlxvlm"
                logger.info(
                    f"[ModelDetector] Detected VLM from local config, using loader: mlxvlm"
                )

            model_type = config_data.get("model_type", "").lower()
            if "qwen3" in model_type:
                result["is_qwen3"] = True
                result["step_processor"] = "thinking"
                logger.info(
                    f"[ModelDetector] Detected Qwen3 from local config, using step_processor: thinking"
                )
            elif "gpt_oss" in model_type:
                result["step_processor"] = "gpt_oss"
                logger.info(
                    f"[ModelDetector] Detected GPT-OSS from local config, using step_processor: gpt_oss"
                )

            max_pos_emb = config_data.get("max_position_embeddings")
            result["config"] = {
                "max_position_embeddings": max_pos_emb,
                "vocab_size": config_data.get("vocab_size"),
                "hidden_size": config_data.get("hidden_size"),
                "num_hidden_layers": config_data.get("num_hidden_layers"),
                "eos_token_id": config_data.get("eos_token_id"),
                "bos_token_id": config_data.get("bos_token_id"),
            }

            if max_pos_emb:
                result["config"]["recommended_max_output_tokens"] = min(
                    max_pos_emb // 4, 16384
                )
                result["config"]["recommended_max_prompt_tokens"] = (
                    max_pos_emb - result["config"]["recommended_max_output_tokens"]
                )

            logger.info(
                f"[ModelDetector] Local detection complete: loader={result['loader']}, "
                f"step_processor={result['step_processor']}"
            )

            return result

        except Exception as e:
            logger.error(f"[ModelDetector] Failed to detect from local: {e}")
            return {"model_name": model_name_or_path, "error": str(e)}

    @staticmethod
    def detect(model_name: str, trust_remote_code: bool = True) -> Dict[str, Any]:
        """
        Auto-detect model information from HuggingFace model name.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen3-8B")
            trust_remote_code: Whether to trust remote code, default True

        Returns:
            Dictionary containing:
                - loader: "mlx" | "mlxvlm"
                - step_processor: "base" | "qwen3think" | "openai"
                - is_vlm: bool
                - is_qwen3: bool
                - thinking_mode: bool
                - config: dict with model parameters
        """
        try:
            logger.info(f"[ModelDetector] Detecting model: {model_name}")

            from transformers import AutoConfig, AutoTokenizer

            config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )

            result: Dict[str, Any] = {
                "model_name": model_name,
                "model_type": config.model_type,
                "architectures": config.architectures,
                "loader": "mlx",
                "step_processor": "base",
                "is_vlm": False,
                "is_qwen3": False,
                "thinking_mode": False,
                "config": {},
            }

            # 1. Detect if VLM (vision-language model)
            if hasattr(config, "vision_config"):
                result["is_vlm"] = True
                result["loader"] = "mlxvlm"
                logger.info(f"[ModelDetector] Detected VLM model, using loader: mlxvlm")

            # 2. Detect if Qwen3
            if "qwen3" in config.model_type.lower():
                result["is_qwen3"] = True
                result["step_processor"] = "thinking"
                logger.info(
                    f"[ModelDetector] Detected Qwen3 model, using step_processor: thinking"
                )
            elif "gpt_oss" in config.model_type.lower():
                result["step_processor"] = "gpt_oss"
                logger.info(
                    f"[ModelDetector] Detected GPT-OSS model, using step_processor: gpt_oss"
                )

            # 3. Detect thinking mode
            if tokenizer.chat_template and "think" in tokenizer.chat_template.lower():
                result["thinking_mode"] = True
                logger.info(f"[ModelDetector] Detected thinking mode support")

            # 4. Get configuration parameters
            max_pos_emb = getattr(config, "max_position_embeddings", None)
            result["config"] = {
                "max_position_embeddings": max_pos_emb,
                "vocab_size": getattr(config, "vocab_size", None),
                "hidden_size": getattr(config, "hidden_size", None),
                "num_hidden_layers": getattr(config, "num_hidden_layers", None),
                "eos_token_id": getattr(config, "eos_token_id", None),
                "bos_token_id": getattr(config, "bos_token_id", None),
            }

            # 5. Recommended configuration
            if max_pos_emb:
                # Recommend max_output_tokens as 1/4 of max_position_embeddings, max 16K
                result["config"]["recommended_max_output_tokens"] = min(
                    max_pos_emb // 4, 16384
                )
                result["config"]["recommended_max_prompt_tokens"] = (
                    max_pos_emb - result["config"]["recommended_max_output_tokens"]
                )

            # Recommend different configurations based on model type
            if result["is_qwen3"]:
                result["config"]["recommended_temperature"] = 0.7
                result["config"]["recommended_top_p"] = 0.9
                result["config"]["recommended_top_k"] = 100
                result["config"]["recommended_repetition_penalty"] = 1.1
            elif result["is_vlm"]:
                result["config"]["recommended_temperature"] = 0.7
                result["config"]["recommended_top_p"] = 0.9
                result["config"]["recommended_top_k"] = 100
                result["config"]["recommended_repetition_penalty"] = 1.1
            else:
                result["config"]["recommended_temperature"] = 0.8
                result["config"]["recommended_top_p"] = 0.9
                result["config"]["recommended_top_k"] = 100
                result["config"]["recommended_repetition_penalty"] = 1.1

            logger.info(
                f"[ModelDetector] Detection complete: loader={result['loader']}, "
                f"step_processor={result['step_processor']}"
            )

            return result

        except Exception as e:
            logger.error(f"[ModelDetector] Failed to detect model {model_name}: {e}")
            return {"model_name": model_name, "error": str(e)}

    @staticmethod
    def detect_loader(model_name: str) -> str:
        """
        Detect which loader to use.

        Args:
            model_name: HuggingFace model name

        Returns:
            "mlx" or "mlxvlm"
        """
        result = PlModelDetector.detect(model_name)
        return result.get("loader", "mlx")

    @staticmethod
    def detect_step_processor(model_name: str) -> str:
        """
        Detect which step processor to use.

        Args:
            model_name: HuggingFace model name

        Returns:
            "base" or "qwen3think" or "openai"
        """
        result = PlModelDetector.detect(model_name)
        return result.get("step_processor", "base")

    @staticmethod
    def is_vlm(model_name: str) -> bool:
        """
        Detect if the model is a vision-language model.

        Args:
            model_name: HuggingFace model name

        Returns:
            True if VLM, False otherwise
        """
        result = PlModelDetector.detect(model_name)
        return result.get("is_vlm", False)

    @staticmethod
    def supports_thinking(model_name: str) -> bool:
        """
        Detect if the model supports thinking mode.

        Args:
            model_name: HuggingFace model name

        Returns:
            True if supports thinking, False otherwise
        """
        result = PlModelDetector.detect(model_name)
        return result.get("thinking_mode", False)
