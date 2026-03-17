"""Tests for model detector module."""

import json
from pathlib import Path
from unittest import mock


from plllm_mlx.models.model_detector import PlModelDetector


class TestPlModelDetector:
    """Tests for PlModelDetector class."""

    def test_detect_from_local_not_found(self, tmp_path: Path):
        """Test detect_from_local when model not in cache."""
        with mock.patch(
            "plllm_mlx.models.model_detector.get_model_snapshot_path",
            return_value=None,
        ):
            result = PlModelDetector.detect_from_local("nonexistent/model")
            assert "error" in result
            assert result["model_name"] == "nonexistent/model"

    def test_detect_from_local_with_config(self, tmp_path: Path):
        """Test detect_from_local with config.json."""
        config_data = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "max_position_embeddings": 32768,
            "vocab_size": 152064,
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "eos_token_id": 151645,
            "bos_token_id": 151643,
        }
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with mock.patch(
            "plllm_mlx.models.model_detector.get_model_snapshot_path",
            return_value=tmp_path,
        ):
            result = PlModelDetector.detect_from_local("test/model")
            assert result["model_name"] == "test/model"
            assert result["model_type"] == "qwen3"
            assert result["loader"] == "mlx"
            assert result["step_processor"] == "thinking"
            assert result["is_qwen3"] is True
            assert result["config"]["max_position_embeddings"] == 32768

    def test_detect_from_local_vlm(self, tmp_path: Path):
        """Test detect_from_local detects VLM model."""
        config_data = {
            "model_type": "qwen2_vl",
            "vision_config": {"hidden_size": 1280},
        }
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with mock.patch(
            "plllm_mlx.models.model_detector.get_model_snapshot_path",
            return_value=tmp_path,
        ):
            result = PlModelDetector.detect_from_local("test/vlm-model")
            assert result["is_vlm"] is True
            assert result["loader"] == "mlxvlm"

    def test_detect_from_local_gpt_oss(self, tmp_path: Path):
        """Test detect_from_local detects GPT-OSS model."""
        config_data = {
            "model_type": "gpt_oss",
        }
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with mock.patch(
            "plllm_mlx.models.model_detector.get_model_snapshot_path",
            return_value=tmp_path,
        ):
            result = PlModelDetector.detect_from_local("test/gpt-oss")
            assert result["step_processor"] == "gpt_oss"

    def test_detect_from_local_no_config(self, tmp_path: Path):
        """Test detect_from_local when config.json not found."""
        with mock.patch(
            "plllm_mlx.models.model_detector.get_model_snapshot_path",
            return_value=tmp_path,
        ):
            result = PlModelDetector.detect_from_local("test/model")
            assert "error" in result

    def test_detect_from_local_recommended_config(self, tmp_path: Path):
        """Test detect_from_local calculates recommended config."""
        config_data = {
            "model_type": "llama",
            "max_position_embeddings": 8192,
        }
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with mock.patch(
            "plllm_mlx.models.model_detector.get_model_snapshot_path",
            return_value=tmp_path,
        ):
            result = PlModelDetector.detect_from_local("test/model")
            assert result["config"]["recommended_max_output_tokens"] == 2048
            assert result["config"]["recommended_max_prompt_tokens"] == 6144

    def test_detect_from_local_large_context(self, tmp_path: Path):
        """Test detect_from_local with large context model."""
        config_data = {
            "model_type": "llama",
            "max_position_embeddings": 128000,
        }
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with mock.patch(
            "plllm_mlx.models.model_detector.get_model_snapshot_path",
            return_value=tmp_path,
        ):
            result = PlModelDetector.detect_from_local("test/model")
            assert result["config"]["recommended_max_output_tokens"] == 16384

    def test_detect_with_mocked_transformers(self):
        """Test detect method with mocked transformers."""
        mock_config = mock.MagicMock()
        mock_config.model_type = "qwen3"
        mock_config.architectures = ["Qwen3ForCausalLM"]
        mock_config.max_position_embeddings = 8192
        mock_config.vocab_size = 152064
        mock_config.hidden_size = 3584
        mock_config.num_hidden_layers = 28
        mock_config.eos_token_id = 151645
        mock_config.bos_token_id = 151643
        del mock_config.vision_config

        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.chat_template = None

        with (
            mock.patch(
                "transformers.AutoConfig.from_pretrained", return_value=mock_config
            ),
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            result = PlModelDetector.detect("Qwen/Qwen3-8B")
            assert result["model_name"] == "Qwen/Qwen3-8B"
            assert result["loader"] == "mlx"
            assert result["step_processor"] == "thinking"
            assert result["is_qwen3"] is True

    def test_detect_with_vlm_mocked(self):
        """Test detect method detects VLM."""
        mock_config = mock.MagicMock()
        mock_config.model_type = "qwen2_vl"
        mock_config.architectures = ["Qwen2VLForConditionalGeneration"]
        mock_config.max_position_embeddings = 32768
        mock_config.vocab_size = 152064
        mock_config.vision_config = mock.MagicMock()

        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.chat_template = None

        with (
            mock.patch(
                "transformers.AutoConfig.from_pretrained", return_value=mock_config
            ),
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            result = PlModelDetector.detect("Qwen/Qwen2-VL-7B")
            assert result["is_vlm"] is True
            assert result["loader"] == "mlxvlm"

    def test_detect_error_handling(self):
        """Test detect method handles errors."""
        with mock.patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=Exception("Connection error"),
        ):
            result = PlModelDetector.detect("nonexistent/model")
            assert "error" in result

    def test_detect_loader(self):
        """Test detect_loader method."""
        with mock.patch.object(
            PlModelDetector, "detect", return_value={"loader": "mlxvlm"}
        ):
            result = PlModelDetector.detect_loader("test/model")
            assert result == "mlxvlm"

    def test_detect_step_processor(self):
        """Test detect_step_processor method."""
        with mock.patch.object(
            PlModelDetector, "detect", return_value={"step_processor": "thinking"}
        ):
            result = PlModelDetector.detect_step_processor("test/model")
            assert result == "thinking"

    def test_is_vlm(self):
        """Test is_vlm method."""
        with mock.patch.object(
            PlModelDetector, "detect", return_value={"is_vlm": True}
        ):
            assert PlModelDetector.is_vlm("test/model") is True

    def test_supports_thinking(self):
        """Test supports_thinking method."""
        with mock.patch.object(
            PlModelDetector, "detect", return_value={"thinking_mode": True}
        ):
            assert PlModelDetector.supports_thinking("test/model") is True
