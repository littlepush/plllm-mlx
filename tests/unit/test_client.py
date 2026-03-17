"""Tests for HTTP client module."""

import json
from unittest import mock

import httpx

from plllm_mlx.client import PlClient


class TestPlClient:
    """Tests for PlClient class."""

    def test_init_with_url(self):
        """Test initialization with explicit URL."""
        client = PlClient(url="http://localhost:9000")
        assert client._url == "http://localhost:9000"

    def test_init_with_timeout(self):
        """Test initialization with custom timeout."""
        client = PlClient(timeout=10.0)
        assert client._timeout == 10.0

    def test_discover_service_url_from_status_file(self, tmp_path, monkeypatch):
        """Test URL discovery from status file."""
        status_file = tmp_path / "service.status"
        status_file.write_text(json.dumps({"port": 9000, "pid": 12345}))

        with (
            mock.patch("plllm_mlx.client.STATUS_FILE", status_file),
            mock.patch.object(httpx.Client, "get") as mock_get,
        ):
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            client = PlClient()
            url = client._discover_service_url()
            assert url == "http://localhost:9000"

    def test_health_check(self):
        """Test health_check method."""
        with mock.patch.object(httpx.Client, "get") as mock_get:
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            mock_get.return_value = mock_response

            client = PlClient(url="http://localhost:8000")
            result = client.health_check()
            assert result["status"] == "healthy"
            mock_get.assert_called_with("http://localhost:8000/health")

    def test_list_models(self):
        """Test list_models method."""
        with mock.patch.object(httpx.Client, "get") as mock_get:
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"model_name": "model1", "is_loaded": True},
                    {"model_name": "model2", "is_loaded": False},
                ]
            }
            mock_get.return_value = mock_response

            client = PlClient(url="http://localhost:8000")
            result = client.list_models()
            assert len(result) == 2
            mock_get.assert_called_with("http://localhost:8000/v1/model/list")

    def test_list_models_loaded_only(self):
        """Test list_models with loaded_only filter."""
        with mock.patch.object(httpx.Client, "get") as mock_get:
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"model_name": "model1", "is_loaded": True},
                    {"model_name": "model2", "is_loaded": False},
                ]
            }
            mock_get.return_value = mock_response

            client = PlClient(url="http://localhost:8000")
            result = client.list_models(loaded_only=True)
            assert len(result) == 1
            assert result[0]["model_name"] == "model1"

    def test_search_models(self):
        """Test search_models method."""
        with mock.patch.object(httpx.Client, "get") as mock_get:
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"model_id": "model1", "downloads": 1000},
                    {"model_id": "model2", "downloads": 500},
                ]
            }
            mock_get.return_value = mock_response

            client = PlClient(url="http://localhost:8000")
            result = client.search_models("qwen", limit=10)
            assert len(result) == 2
            mock_get.assert_called_with(
                "http://localhost:8000/v1/model/search",
                params={"keyword": "qwen"},
            )

    def test_load_model(self):
        """Test load_model method."""
        with mock.patch.object(httpx.Client, "post") as mock_post:
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "model_name": "test/model",
                "loader": "mlx",
                "step_processor": "base",
            }
            mock_post.return_value = mock_response

            client = PlClient(url="http://localhost:8000")
            result = client.load_model("test/model")
            assert result["model_name"] == "test/model"
            mock_post.assert_called_with(
                "http://localhost:8000/v1/model/load",
                json={"model_name": "test/model"},
                timeout=300.0,
            )

    def test_load_model_with_options(self):
        """Test load_model with loader and step_processor options."""
        with mock.patch.object(httpx.Client, "post") as mock_post:
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "model_name": "test/model",
                "loader": "mlxvlm",
                "step_processor": "thinking",
            }
            mock_post.return_value = mock_response

            client = PlClient(url="http://localhost:8000")
            result = client.load_model(
                "test/model", loader="mlxvlm", step_processor="thinking"
            )
            assert result["loader"] == "mlxvlm"
            mock_post.assert_called_with(
                "http://localhost:8000/v1/model/load",
                json={
                    "model_name": "test/model",
                    "loader": "mlxvlm",
                    "step_processor": "thinking",
                },
                timeout=300.0,
            )

    def test_unload_model(self):
        """Test unload_model method."""
        with mock.patch.object(httpx.Client, "post") as mock_post:
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "OK",
                "model_name": "test/model",
            }
            mock_post.return_value = mock_response

            client = PlClient(url="http://localhost:8000")
            result = client.unload_model("test/model")
            assert result["status"] == "OK"
            mock_post.assert_called_with(
                "http://localhost:8000/v1/model/unload",
                json={"model_name": "test/model"},
            )

    def test_download_model(self):
        """Test download_model method."""
        with mock.patch.object(httpx.Client, "post") as mock_post:
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "task_id": "task-123",
                "status": "started",
            }
            mock_post.return_value = mock_response

            client = PlClient(url="http://localhost:8000")
            result = client.download_model("Qwen/Qwen2-7B")
            assert result["task_id"] == "task-123"
            mock_post.assert_called_with(
                "http://localhost:8000/v1/model/download",
                json={"model_id": "Qwen/Qwen2-7B"},
            )

    def test_get_download_status(self):
        """Test get_download_status method."""
        with mock.patch.object(httpx.Client, "get") as mock_get:
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "task_id": "task-123",
                "status": "completed",
            }
            mock_get.return_value = mock_response

            client = PlClient(url="http://localhost:8000")
            result = client.get_download_status("task-123")
            assert result["status"] == "completed"
            mock_get.assert_called_with(
                "http://localhost:8000/v1/model/download/status/task-123"
            )

    def test_delete_model(self):
        """Test delete_model method."""
        with mock.patch.object(httpx.Client, "post") as mock_post:
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "OK"}
            mock_post.return_value = mock_response

            client = PlClient(url="http://localhost:8000")
            result = client.delete_model("test/model")
            assert result["status"] == "OK"
            mock_post.assert_called_with(
                "http://localhost:8000/v1/model/delete",
                json={"model_name": "test/model"},
            )

    def test_update_config(self):
        """Test update_config method."""
        with mock.patch.object(httpx.Client, "post") as mock_post:
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "OK"}
            mock_post.return_value = mock_response

            client = PlClient(url="http://localhost:8000")
            result = client.update_config("test/model", "temperature", 0.5)
            assert result["status"] == "OK"
            mock_post.assert_called_with(
                "http://localhost:8000/v1/model/update/config",
                json={"model_name": "test/model", "key": "temperature", "value": 0.5},
            )

    def test_close(self):
        """Test close method."""
        client = PlClient(url="http://localhost:8000")
        _ = client.client
        client.close()
        assert client._client.is_closed
