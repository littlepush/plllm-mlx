"""Tests for path helper module."""

import os
from pathlib import Path
from unittest import mock


from plllm_mlx.helpers import (
    HF_HUB_CACHE,
    get_hf_cache_dir,
    get_model_cache_path,
    get_model_snapshot_path,
)


class TestGetHfCacheDir:
    """Tests for get_hf_cache_dir function."""

    def test_default_path(self):
        """Test default HuggingFace cache path."""
        with mock.patch.dict(os.environ, {}, clear=True):
            if "HF_HUB_CACHE" in os.environ:
                del os.environ["HF_HUB_CACHE"]
            if "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]
            if "HUGGING_FACE_PATH" in os.environ:
                del os.environ["HUGGING_FACE_PATH"]
            result = get_hf_cache_dir()
            expected = f"{Path.home()}/.cache/huggingface/hub"
            assert result == expected

    def test_hf_hub_cache_env(self):
        """Test HF_HUB_CACHE environment variable takes priority."""
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": "/custom/cache"}):
            result = get_hf_cache_dir()
            assert result == "/custom/cache"

    def test_hf_home_env(self):
        """Test HF_HOME environment variable."""
        with mock.patch.dict(os.environ, {"HF_HOME": "/hf_home"}, clear=True):
            if "HF_HUB_CACHE" in os.environ:
                del os.environ["HF_HUB_CACHE"]
            result = get_hf_cache_dir()
            assert result == "/hf_home/hub"

    def test_hugging_face_path_env(self):
        """Test HUGGING_FACE_PATH environment variable."""
        with mock.patch.dict(
            os.environ, {"HUGGING_FACE_PATH": "/legacy/path"}, clear=True
        ):
            if "HF_HUB_CACHE" in os.environ:
                del os.environ["HF_HUB_CACHE"]
            if "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]
            result = get_hf_cache_dir()
            assert result == "/legacy/path"

    def test_priority_order(self):
        """Test environment variable priority order."""
        with mock.patch.dict(
            os.environ,
            {
                "HF_HUB_CACHE": "/first",
                "HF_HOME": "/second",
                "HUGGING_FACE_PATH": "/third",
            },
        ):
            result = get_hf_cache_dir()
            assert result == "/first"


class TestHFHubCache:
    """Tests for HF_HUB_CACHE constant."""

    def test_is_string(self):
        """Test HF_HUB_CACHE is a string."""
        assert isinstance(HF_HUB_CACHE, str)

    def test_matches_get_hf_cache_dir(self):
        """Test HF_HUB_CACHE matches get_hf_cache_dir result."""
        assert HF_HUB_CACHE == get_hf_cache_dir()


class TestGetModelCachePath:
    """Tests for get_model_cache_path function."""

    def test_model_not_found(self, tmp_path: Path):
        """Test getting cache path for non-existent model."""
        with mock.patch(
            "plllm_mlx.helpers.path_helper.get_hf_cache_dir",
            return_value=str(tmp_path),
        ):
            result = get_model_cache_path("nonexistent/model")
            assert result is None

    def test_model_found(self, tmp_path: Path):
        """Test getting cache path for existing model."""
        model_dir = tmp_path / "models--org--model-name"
        model_dir.mkdir()

        with mock.patch(
            "plllm_mlx.helpers.path_helper.get_hf_cache_dir",
            return_value=str(tmp_path),
        ):
            result = get_model_cache_path("org/model-name")
            assert result == model_dir

    def test_model_name_with_slash(self, tmp_path: Path):
        """Test model name with slash is converted correctly."""
        model_dir = tmp_path / "models--Qwen--Qwen2-7B"
        model_dir.mkdir()

        with mock.patch(
            "plllm_mlx.helpers.path_helper.get_hf_cache_dir",
            return_value=str(tmp_path),
        ):
            result = get_model_cache_path("Qwen/Qwen2-7B")
            assert result == model_dir


class TestGetModelSnapshotPath:
    """Tests for get_model_snapshot_path function."""

    def test_model_not_found(self, tmp_path: Path):
        """Test getting snapshot path for non-existent model."""
        with mock.patch(
            "plllm_mlx.helpers.path_helper.get_hf_cache_dir",
            return_value=str(tmp_path),
        ):
            result = get_model_snapshot_path("nonexistent/model")
            assert result is None

    def test_model_no_snapshots(self, tmp_path: Path):
        """Test getting snapshot path when no snapshots exist."""
        model_dir = tmp_path / "models--org--model"
        model_dir.mkdir()
        snapshots_dir = model_dir / "snapshots"
        snapshots_dir.mkdir()

        with mock.patch(
            "plllm_mlx.helpers.path_helper.get_hf_cache_dir",
            return_value=str(tmp_path),
        ):
            result = get_model_snapshot_path("org/model")
            assert result is None

    def test_model_with_snapshot(self, tmp_path: Path):
        """Test getting snapshot path for model with snapshots."""
        model_dir = tmp_path / "models--org--model"
        model_dir.mkdir()
        snapshots_dir = model_dir / "snapshots"
        snapshots_dir.mkdir()
        snapshot_dir = snapshots_dir / "abc123"
        snapshot_dir.mkdir()

        with mock.patch(
            "plllm_mlx.helpers.path_helper.get_hf_cache_dir",
            return_value=str(tmp_path),
        ):
            result = get_model_snapshot_path("org/model")
            assert result == snapshot_dir

    def test_model_multiple_snapshots(self, tmp_path: Path):
        """Test getting snapshot path returns first snapshot."""
        model_dir = tmp_path / "models--org--model"
        model_dir.mkdir()
        snapshots_dir = model_dir / "snapshots"
        snapshots_dir.mkdir()
        (snapshots_dir / "abc123").mkdir()
        (snapshots_dir / "def456").mkdir()

        with mock.patch(
            "plllm_mlx.helpers.path_helper.get_hf_cache_dir",
            return_value=str(tmp_path),
        ):
            result = get_model_snapshot_path("org/model")
            assert result is not None
            assert result.parent == snapshots_dir

    def test_no_snapshots_dir(self, tmp_path: Path):
        """Test when snapshots directory doesn't exist."""
        model_dir = tmp_path / "models--org--model"
        model_dir.mkdir()

        with mock.patch(
            "plllm_mlx.helpers.path_helper.get_hf_cache_dir",
            return_value=str(tmp_path),
        ):
            result = get_model_snapshot_path("org/model")
            assert result is None
