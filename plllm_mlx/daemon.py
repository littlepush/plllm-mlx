"""
LaunchAgent management for macOS.

This module handles service registration and lifecycle management
using macOS LaunchAgent for background service execution.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from plllm_mlx.logging_config import get_logger

logger = get_logger(__name__)

# Constants
LABEL = "cc.impush.plllm-mlx"
PLIST_NAME = f"{LABEL}.plist"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / PLIST_NAME
CONFIG_DIR = Path.home() / ".plllm-mlx"
LOG_DIR = CONFIG_DIR / "logs"
LOG_FILE = LOG_DIR / "service.log"
DEFAULT_CONFIG = CONFIG_DIR / "config.yaml"
STATUS_FILE = CONFIG_DIR / "service.status"  # Fast status file for port/pid


def create_directories() -> None:
    """Create necessary directories for plllm-mlx."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created directories: {CONFIG_DIR}, {LOG_DIR}")


def generate_plist(config_path: Path, port: int = 8000, log_level: str = "info") -> str:
    """
    Generate LaunchAgent plist content.

    Args:
        config_path: Path to configuration file.
        port: Service port number.
        log_level: Logging level.

    Returns:
        Plist XML content as string.
    """
    import sys

    project_root = Path(__file__).resolve().parent.parent
    venv_bin = project_root / ".venv" / "bin" / "plllm-mlx"
    if venv_bin.exists():
        executable = str(venv_bin)
    else:
        executable = sys.executable.rsplit("/", 1)[0] + "/plllm-mlx"
        if not Path(executable).exists():
            uv_path = Path.home() / ".local" / "bin" / "uv"
            if uv_path.exists():
                executable = f"{uv_path} run plllm-mlx"

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LABEL}</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{executable}</string>
        <string>run-server</string>
        <string>--config</string>
        <string>{config_path}</string>
        <string>--port</string>
        <string>{port}</string>
        <string>--log-level</string>
        <string>{log_level}</string>
    </array>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <true/>
    
    <key>StandardOutPath</key>
    <string>{LOG_FILE}</string>
    
    <key>StandardErrorPath</key>
    <string>{LOG_FILE}</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PLLLM_MLX_CONFIG</key>
        <string>{config_path}</string>
        <key>TOKENIZERS_PARALLELISM</key>
        <string>true</string>
        <key>HUGGING_FACE_PATH</key>
        <string>{Path.home()}/.cache/huggingface/hub</string>
    </dict>
</dict>
</plist>"""

    return plist_content


def _check_port_open(port: int) -> bool:
    """Check if a port is open (fast method)."""
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except Exception:
        return False


def _write_status_file(port: int, pid: int) -> None:
    """Write service status file for fast lookup."""
    import json

    STATUS_FILE.write_text(json.dumps({"port": port, "pid": pid}))


def _read_status_file() -> Optional[dict]:
    """Read service status file."""
    import json

    if not STATUS_FILE.exists():
        return None
    try:
        return json.loads(STATUS_FILE.read_text())
    except Exception:
        return None


def _remove_status_file() -> None:
    """Remove service status file."""
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()


def is_service_running() -> bool:
    """
    Check if the LaunchAgent service is running.

    Returns:
        True if service is running, False otherwise.
    """
    if not PLIST_PATH.exists():
        return False

    # Try status file first (fastest)
    status = _read_status_file()
    if status and status.get("port"):
        if _check_port_open(status["port"]):
            return True

    # Fallback to config
    port = _get_port_from_config()
    return _check_port_open(port) if port else False


def _get_port_from_config() -> Optional[int]:
    """Get port from config file without checking service status."""
    if DEFAULT_CONFIG.exists():
        try:
            import yaml

            with DEFAULT_CONFIG.open() as f:
                config = yaml.safe_load(f)
                return config.get("server", {}).get("port", 8000)
        except Exception:
            pass
    return 8000


def get_service_port() -> Optional[int]:
    """
    Get the port the service is running on.

    Returns:
        Port number if service is running, None otherwise.
    """
    port = _get_port_from_config()
    if port and _check_port_open(port):
        return port
    return None


def get_service_url() -> Optional[str]:
    """
    Get the service URL if running.

    Returns:
        Service URL if running, None otherwise.
    """
    if not PLIST_PATH.exists():
        return None

    port = _get_port_from_config()
    if port and _check_port_open(port):
        return f"http://localhost:{port}"
    return None


def wait_for_service(port: int, timeout: int = 60) -> bool:
    """
    Wait for service to start accepting connections.

    Args:
        port: Port number to check.
        timeout: Maximum time to wait in seconds.

    Returns:
        True if service started, False if timeout.
    """
    import socket
    import os

    # Allow override via environment variable
    timeout = int(os.environ.get("PLLLM_START_TIMEOUT", timeout))

    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", port))
            sock.close()

            if result == 0:
                return True
        except Exception:
            pass

        time.sleep(0.5)

    return False


def start_service(
    config_path: Optional[Path] = None, port: int = 8000, log_level: str = "info"
) -> bool:
    """
    Start the service by registering LaunchAgent.

    Args:
        config_path: Path to configuration file.
        port: Service port number.
        log_level: Logging level.

    Returns:
        True if service started successfully, False otherwise.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG

    # Check if already running
    if is_service_running():
        logger.warning("Service already running")
        return True

    # Create directories
    create_directories()

    # Create default config if not exists
    if not config_path.exists():
        create_default_config(config_path)

    # Generate plist
    plist_content = generate_plist(config_path, port, log_level)

    # Write plist file
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(plist_content)
    logger.info(f"Generated plist at {PLIST_PATH}")

    # Load launchagent
    result = subprocess.run(
        ["launchctl", "load", str(PLIST_PATH)], capture_output=True, text=True
    )

    if result.returncode != 0:
        logger.error(f"Failed to load LaunchAgent: {result.stderr}")
        return False

    # Wait for service to start
    if not wait_for_service(port):
        logger.error("Service failed to start within timeout")
        return False

    # Write status file for fast lookup
    _write_status_file(port, os.getpid())

    logger.info(f"Service started on port {port}")
    return True


def stop_service() -> bool:
    """
    Stop the service by unloading LaunchAgent.

    Returns:
        True if service stopped successfully, False otherwise.
    """
    if not PLIST_PATH.exists():
        logger.warning("Service not running (plist not found)")
        return True

    # Unload launchagent
    result = subprocess.run(
        ["launchctl", "unload", str(PLIST_PATH)], capture_output=True, text=True
    )

    if result.returncode != 0:
        logger.warning(f"Failed to unload LaunchAgent: {result.stderr}")

    # Remove plist file
    try:
        PLIST_PATH.unlink()
        logger.info(f"Removed plist file {PLIST_PATH}")
    except Exception as e:
        logger.warning(f"Failed to remove plist: {e}")

    # Remove status file
    _remove_status_file()

    logger.info("Service stopped")
    return True


def create_default_config(config_path: Path) -> None:
    """
    Create a default configuration file.

    Args:
        config_path: Path where to create the config file.
    """
    default_config = {
        "server": {"host": "0.0.0.0", "port": 8000},
        "models": [],
        "logging": {"level": "info"},
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)

    import yaml

    with config_path.open("w") as f:
        yaml.dump(default_config, f, default_flow_style=False)

    logger.info(f"Created default config at {config_path}")
