"""
Command-line interface for plllm-mlx.

This module provides a comprehensive CLI tool for managing plllm-mlx service.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from plllm_mlx import create_app
from plllm_mlx.client import PlClient
from plllm_mlx.config import PlConfig
from plllm_mlx.daemon import (
    DEFAULT_CONFIG,
    LOG_FILE,
    is_service_running,
    start_service,
    stop_service,
)
from plllm_mlx.logging_config import setup_logging
from plllm_mlx.utils import format_config, format_number, parse_value

app = typer.Typer(
    name="plllm-mlx",
    help="Standalone MLX-based LLM inference service with OpenAI compatible API",
    add_completion=False,
)
console = Console()


# ==================== Serve Command ====================


@app.command()
def serve(
    config: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="Configuration file path"
    ),
    port: int = typer.Option(8000, "--port", "-p", help="Service port"),
    log_level: str = typer.Option("info", "--log-level", "-l", help="Logging level"),
):
    """Start the service as LaunchAgent."""
    console.print("[bold green]Starting plllm-mlx service...[/bold green]")

    if is_service_running():
        console.print("[yellow]Service already running[/yellow]")
        return

    success = start_service(config, port, log_level)

    if success:
        console.print(f"[green]✓[/green] Service started on port {port}")
        console.print(f"  Config: {config}")
        console.print(f"  Log: {LOG_FILE}")
        console.print(f"\nTo view logs: [bold]tail -f {LOG_FILE}[/bold]")
        console.print("To stop: [bold]plllm-mlx stop[/bold]")
    else:
        console.print("[red]✗ Failed to start service[/red]")
        sys.exit(1)


# ==================== Stop Command ====================


@app.command()
def stop():
    """Stop the service."""
    console.print("[bold]Stopping plllm-mlx service...[/bold]")

    if not is_service_running():
        console.print("[yellow]Service not running[/yellow]")
        return

    success = stop_service()

    if success:
        console.print("[green]✓[/green] Service stopped")
    else:
        console.print("[red]✗ Failed to stop service[/red]")
        sys.exit(1)


# ==================== Restart Command ====================


@app.command()
def restart():
    """Restart the service."""
    console.print("[bold]Restarting plllm-mlx service...[/bold]")

    stop()

    import time

    time.sleep(2)

    success = start_service(DEFAULT_CONFIG, 8000, "info")

    if success:
        console.print("[green]✓[/green] Service started on port 8000")
        console.print(f"  Config: {DEFAULT_CONFIG}")
        console.print(f"  Log: {LOG_FILE}")
    else:
        console.print("[red]✗ Failed to start service[/red]")
        sys.exit(1)


# ==================== Run-Server Command (Internal) ====================


@app.command("run-server", hidden=True)
def run_server(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    port: int = typer.Option(8000, "--port", "-p"),
    log_level: str = typer.Option("info", "--log-level", "-l"),
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
):
    """Run the server (internal command, used by LaunchAgent)."""
    # Load configuration first to get logging level
    if config.exists():
        pl_config = PlConfig.from_yaml(config)
    else:
        pl_config = PlConfig()

    # Use logging level from config file
    actual_log_level = pl_config.logging.level if pl_config.logging else log_level
    logger = setup_logging(level=actual_log_level, config=pl_config.logging)

    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Config: {config}")
    logger.debug(f"Log level: {actual_log_level}")

    # Create app
    app_instance = create_app(
        config=pl_config, host=host, port=port, log_level=actual_log_level
    )

    # Run server
    import uvicorn

    uvicorn.run(app_instance, host=host, port=port, log_level=actual_log_level)


# ==================== PS Command ====================


@app.command()
def ps(json_output: bool = typer.Option(False, "--json", help="Output as JSON")):
    """List loaded models."""
    client = PlClient()
    models = client.list_models(loaded_only=True)

    if json_output:
        print(json.dumps(models, indent=2))
        return

    if not models:
        console.print("[yellow]No models loaded[/yellow]")
        return

    table = Table(title="Loaded Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Loader", style="green")
    table.add_column("Status", style="yellow")

    for m in models:
        table.add_row(m.get("model_name", ""), m.get("model_loader", "mlx"), "loaded")

    console.print(table)


# ==================== LS Command ====================


@app.command("ls")
def list_models(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List all local models."""
    client = PlClient()
    models = client.list_models(loaded_only=False)

    if json_output:
        print(json.dumps(models, indent=2))
        return

    if not models:
        console.print("[yellow]No models found[/yellow]")
        return

    table = Table(title="Local Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Loader", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Config", style="dim")

    for m in models:
        status = "loaded" if m.get("is_loaded") else "unloaded"
        config = format_config(m.get("config", {}))
        table.add_row(
            m.get("model_name", ""), m.get("model_loader", "mlx"), status, config
        )

    console.print(table)


# ==================== Search Command ====================


@app.command()
def search(
    keyword: str = typer.Argument("", help="Search keyword"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Search models on HuggingFace."""
    client = PlClient()

    console.print(f"[bold]Searching for '{keyword or 'mlx'}'...[/bold]")
    models = client.search_models(keyword, limit)

    if json_output:
        print(json.dumps(models, indent=2))
        return

    if not models:
        console.print("[yellow]No models found[/yellow]")
        return

    table = Table(title=f"Search Results ({len(models)} found)")
    table.add_column("Model ID", style="cyan")
    table.add_column("Downloads", style="green")
    table.add_column("Likes", style="yellow")

    for m in models:
        table.add_row(
            m.get("model_id", ""),
            format_number(m.get("downloads", 0)),
            format_number(m.get("likes", 0)),
        )

    console.print(table)


# ==================== Chat Command ====================

from plllm_mlx.commands.chat import chat_app

app.add_typer(chat_app, name="chat")


# ==================== Load Command ====================


@app.command()
def load(
    model_name: str = typer.Argument(..., help="Model name to load"),
    loader: Optional[str] = typer.Option(
        None,
        "--loader",
        "-l",
        help="Model loader (mlx/mlxvlm), auto-detect if not specified",
    ),
    stpp: Optional[str] = typer.Option(
        None,
        "--stpp",
        help="Step processor (base/qwen3think/openai), auto-detect if not specified",
    ),
):
    """Load a model.

    If --loader and --stpp are not specified, they will be auto-detected.
    """
    client = PlClient()

    console.print(f"[bold]Loading model: {model_name}[/bold]")
    if loader:
        console.print(f"  Loader: [cyan]{loader}[/cyan]")
    if stpp:
        console.print(f"  Step Processor: [cyan]{stpp}[/cyan]")

    try:
        result = client.load_model(model_name, loader=loader, step_processor=stpp)
        console.print(f"[green]✓[/green] Model {model_name} loaded")
        console.print(f"  Loader: [cyan]{result.get('loader', 'unknown')}[/cyan]")
        console.print(
            f"  Step Processor: [cyan]{result.get('step_processor', 'unknown')}[/cyan]"
        )
    except Exception as e:
        console.print(f"[red]✗ Failed to load model: {e}[/red]")
        sys.exit(1)


# ==================== Unload Command ====================


@app.command()
def unload(
    model_name: str = typer.Argument(..., help="Model name to unload"),
    kill_subprocess: bool = typer.Option(
        False,
        "--kill-subprocess",
        "-k",
        help="Also kill the subprocess for this model",
    ),
):
    """Unload a model."""
    client = PlClient()

    console.print(f"[bold]Unloading model: {model_name}[/bold]")

    try:
        result = client.unload_model(model_name)
        console.print(f"[green]✓[/green] Model {model_name} unloaded")

        if kill_subprocess:
            _kill_subprocess_for_model(model_name)

    except Exception as e:
        console.print(f"[red]✗ Failed to unload model: {e}[/red]")
        sys.exit(1)


def _kill_subprocess_for_model(model_name: str) -> None:
    """Kill the subprocess for a model."""
    import hashlib
    import signal

    subprocess_dir = Path.home() / ".plllm-mlx" / "subprocess"
    model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
    socket_path = subprocess_dir / f"{model_hash}.sock"

    if socket_path.exists():
        try:
            socket_path.unlink()
            console.print(f"[dim]Removed socket: {socket_path}[/dim]")
        except Exception:
            pass

    # Kill the subprocess process
    try:
        import subprocess as sp

        result = sp.run(
            ["pgrep", "-f", f"subprocess.*{model_hash}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for pid_str in result.stdout.strip().split("\n"):
                if pid_str.strip():
                    try:
                        import os

                        os.kill(int(pid_str.strip()), signal.SIGTERM)
                        console.print(f"[dim]Terminated subprocess: {pid_str}[/dim]")
                    except Exception:
                        pass
    except Exception as e:
        console.print(f"[yellow]Warning: Could not kill subprocess: {e}[/yellow]")


# ==================== Reload Command ====================


@app.command()
def reload(model_name: str = typer.Argument(..., help="Model name to reload")):
    """Reload a model."""
    client = PlClient()

    console.print(f"[bold]Reloading model: {model_name}[/bold]")

    try:
        client.unload_model(model_name)
        client.load_model(model_name)
        console.print(f"[green]✓[/green] Model {model_name} reloaded")
    except Exception as e:
        console.print(f"[red]✗ Failed to reload model: {e}[/red]")
        sys.exit(1)


# ==================== Download Command ====================


@app.command()
def download(
    model_id: str = typer.Argument(..., help="HuggingFace model ID"),
    loader: Optional[str] = typer.Option(
        None,
        "--loader",
        "-l",
        help="Model loader (mlx/mlxvlm), auto-detect if not specified",
    ),
    stpp: Optional[str] = typer.Option(
        None,
        "--stpp",
        help="Step processor (base/qwen3think/openai), auto-detect if not specified",
    ),
):
    """Download a model from HuggingFace.

    If --loader and --stpp are not specified, they will be auto-detected after download.
    """
    client = PlClient()

    console.print(f"[bold]Downloading: {model_id}[/bold]")
    if loader:
        console.print(f"  Loader: [cyan]{loader}[/cyan]")
    if stpp:
        console.print(f"  Step Processor: [cyan]{stpp}[/cyan]")

    try:
        result = client.download_model(model_id, loader=loader, step_processor=stpp)
        task_id = result.get("task_id")
        console.print("[green]✓[/green] Download started")
        console.print(f"  Task ID: {task_id}")
        console.print(
            f"  Check status: [bold]plllm-mlx download-status {task_id}[/bold]"
        )
    except Exception as e:
        console.print(f"[red]✗ Failed to start download: {e}[/red]")
        sys.exit(1)


# ==================== Download-Status Command ====================


@app.command("download-status")
def download_status(task_id: str = typer.Argument(..., help="Download task ID")):
    """Check download status."""
    client = PlClient()

    try:
        status = client.get_download_status(task_id)
        console.print("[bold]Download Status:[/bold]")
        console.print(f"  Task ID: {status.get('task_id')}")
        console.print(f"  Model: {status.get('model_id')}")
        console.print(f"  Status: {status.get('status')}")

        progress = status.get("progress")
        if progress:
            percent = progress.get("percent", 0)
            files = f"{progress.get('downloaded_files', 0)}/{progress.get('total_files', 0)}"
            size_mb = progress.get("downloaded_mb", 0)
            current = progress.get("current_file", "")

            bar_width = 20
            filled = int(bar_width * percent / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            console.print(f"  Progress: [{bar}] {percent}%")
            console.print(f"  Files: {files}")
            console.print(f"  Downloaded: {size_mb}MB")
            if current:
                console.print(f"  Current: {current}")

        console.print(f"  Message: {status.get('message')}")

        if status.get("model_name"):
            console.print(f"  Model Name: {status.get('model_name')}")
    except Exception as e:
        console.print(f"[red]✗ Failed to get status: {e}[/red]")
        sys.exit(1)


# ==================== Delete Command ====================


@app.command()
def delete(model_name: str = typer.Argument(..., help="Model name to delete")):
    """Delete a local model."""
    client = PlClient()

    console.print(f"[bold]Deleting model: {model_name}[/bold]")

    try:
        result = client.delete_model(model_name)
        console.print(f"[green]✓[/green] Model {model_name} deleted")
    except Exception as e:
        console.print(f"[red]✗ Failed to delete model: {e}[/red]")
        sys.exit(1)


# ==================== Config Command ====================


@app.command()
def config(
    model_name: str = typer.Argument(..., help="Model name"),
    key_value: List[str] = typer.Argument(..., help="Key=value pairs"),
):
    """Configure model parameters."""
    client = PlClient()

    console.print(f"[bold]Configuring {model_name}...[/bold]")

    for kv in key_value:
        if "=" not in kv:
            console.print(f"[red]Invalid format: {kv}. Expected key=value[/red]")
            sys.exit(1)

        key, value_str = kv.split("=", 1)
        value = parse_value(value_str)

        console.print(f"  {key} = {value}")

        try:
            client.update_config(model_name, key, value)
        except Exception as e:
            console.print(f"[red]✗ Failed to set {key}: {e}[/red]")
            sys.exit(1)

    console.print("[green]✓[/green] Configuration updated")


# ==================== Status Command ====================


@app.command()
def status():
    """Check service status."""
    if not is_service_running():
        console.print("[red]● Service not running[/red]")
        console.print("Start with: [bold]plllm-mlx serve[/bold]")
        return

    client = PlClient()

    console.print("[green]● Service running[/green]")

    try:
        models = client.list_models(loaded_only=True)
        console.print(f"  Models loaded: {len(models)}")

        for m in models:
            console.print(f"    - {m.get('model_name')}")
    except Exception as e:
        console.print(f"[yellow]  Warning: Could not fetch model info: {e}[/yellow]")

    console.print(f"  Config: {DEFAULT_CONFIG}")
    console.print(f"  Log: {LOG_FILE}")


# ==================== Subprocess Commands ====================

subprocess_app = typer.Typer(
    name="subprocess",
    help="Manage model subprocess services",
)
app.add_typer(subprocess_app, name="subprocess")


@subprocess_app.command("serve")
def subprocess_serve(
    socket: str = typer.Option(
        ...,
        "--socket",
        "-s",
        help="Unix domain socket path for the subprocess server",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name to load on startup",
    ),
):
    """Start a model subprocess server."""
    from plllm_mlx.subprocess.python.server import run_server

    console.print("[bold]Starting subprocess server...[/bold]")
    console.print(f"  Socket: {socket}")
    if model:
        console.print(f"  Model: {model}")

    run_server(socket_path=socket, model_name=model)


@subprocess_app.command("status")
def subprocess_status(
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
):
    """Check subprocess status for a model."""
    import hashlib

    import httpx

    socket_dir = Path.home() / ".plllm-mlx" / "subprocess"
    model_hash = hashlib.md5(model.encode()).hexdigest()[:8]
    socket_path = socket_dir / f"{model_hash}.sock"

    if not socket_path.exists():
        console.print(f"[yellow]No subprocess found for model: {model}[/yellow]")
        console.print(f"  Expected socket: {socket_path}")
        return

    try:
        with httpx.Client(
            transport=httpx.HTTPTransport(uds=str(socket_path)),
            timeout=5.0,
        ) as client:
            resp = client.get("http://localhost/status")
            data = resp.json()
            console.print_json(data=data)
    except Exception as e:
        console.print(f"[red]Failed to connect to subprocess: {e}[/red]")
        console.print(f"  Socket may be stale: {socket_path}")


@subprocess_app.command("list")
def subprocess_list():
    """List all running subprocesses."""

    import httpx

    socket_dir = Path.home() / ".plllm-mlx" / "subprocess"

    if not socket_dir.exists():
        console.print("[yellow]No subprocess directory found[/yellow]")
        return

    sockets = list(socket_dir.glob("*.sock"))
    if not sockets:
        console.print("[yellow]No subprocess sockets found[/yellow]")
        return

    table = Table(title="Running Subprocesses")
    table.add_column("Socket")
    table.add_column("Status")
    table.add_column("Model")
    table.add_column("Loaded")

    for sock in sockets:
        try:
            with httpx.Client(
                transport=httpx.HTTPTransport(uds=str(sock)),
                timeout=2.0,
            ) as client:
                resp = client.get("http://localhost/status")
                data = resp.json()
                table.add_row(
                    sock.name,
                    "[green]running[/green]",
                    data.get("model_name", "-"),
                    str(data.get("is_loaded", False)),
                )
        except Exception:
            table.add_row(sock.name, "[red]dead[/red]", "-", "-")

    console.print(table)


@subprocess_app.command("stop")
def subprocess_stop(
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
):
    """Stop a model subprocess."""
    import hashlib

    import httpx

    socket_dir = Path.home() / ".plllm-mlx" / "subprocess"
    model_hash = hashlib.md5(model.encode()).hexdigest()[:8]
    socket_path = socket_dir / f"{model_hash}.sock"

    if not socket_path.exists():
        console.print(f"[yellow]No subprocess found for model: {model}[/yellow]")
        return

    try:
        with httpx.Client(
            transport=httpx.HTTPTransport(uds=str(socket_path)),
            timeout=5.0,
        ) as client:
            resp = client.post("http://localhost/unload")
            console.print(f"[green]Model unloaded: {model}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to stop subprocess: {e}[/red]")

    # Clean up socket file if still exists
    if socket_path.exists():
        socket_path.unlink()
        console.print(f"[dim]Cleaned up socket: {socket_path}[/dim]")


# ==================== Default: Start Server ====================


def main():
    """Main entry point."""
    import platform

    if platform.system() != "Darwin":
        console = Console()
        console.print(
            "[red]Error: plllm-mlx only supports macOS (Apple Silicon).[/red]"
        )
        console.print(
            "[yellow]This package requires MLX framework which is exclusive to macOS.[/yellow]"
        )
        sys.exit(1)

    if len(sys.argv) == 1 or (
        len(sys.argv) > 1
        and not sys.argv[1].startswith("-")
        and sys.argv[1]
        not in [
            "serve",
            "stop",
            "restart",
            "run-server",
            "ps",
            "ls",
            "search",
            "load",
            "unload",
            "reload",
            "download",
            "download-status",
            "delete",
            "config",
            "status",
            "chat",
            "subprocess",
        ]
    ):
        # Default to starting server with config
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", "-c", default=str(DEFAULT_CONFIG))
        parser.add_argument("--port", "-p", type=int, default=8000)
        parser.add_argument("--log-level", "-l", default="info")
        args = parser.parse_args()

        config_path = Path(args.config)
        serve(config=config_path, port=args.port, log_level=args.log_level)
    else:
        app()


if __name__ == "__main__":
    main()
