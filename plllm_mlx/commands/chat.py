"""
Chat command for interactive conversations with loaded models.

This module provides the CLI chat command for interacting with
LLM models through the plllm-mlx service.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import httpx
import typer
from rich.console import Console
from rich.prompt import Prompt

console = Console()

chat_app = typer.Typer(
    name="chat",
    help="Interactive chat with loaded models",
    add_completion=False,
)

DEFAULT_TIMEOUT = 300.0  # 5 minutes


@dataclass
class ChatUsage:
    """Usage statistics for a chat completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tps: float
    generation_tps: float
    prompt_process: float
    first_token: float


@dataclass
class ChatChunk:
    """A single chunk in the chat stream."""

    content: str
    is_reasoning: bool
    usage: Optional[ChatUsage] = None


class ChatClient:
    """Client for streaming chat completions."""

    def __init__(
        self,
        base_url: str,
        model: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.messages: List[Dict] = []

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat_stream(self, user_input: str) -> Iterator[ChatChunk]:
        """
        Stream chat completion.

        Yields:
            ChatChunk objects with content and usage info.
        """
        self.messages.append({"role": "user", "content": user_input})

        payload = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        full_content = ""
        full_reasoning = ""

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if not line:
                            continue

                        if not line.startswith("data: "):
                            continue

                        data = line[6:]

                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            usage_data = chunk.get("usage")
                            if usage_data:
                                yield ChatChunk(
                                    content="",
                                    is_reasoning=False,
                                    usage=self._parse_usage(usage_data),
                                )
                            continue

                        delta = choices[0].get("delta", {})

                        reasoning = delta.get("reasoning", "")
                        content = delta.get("content", "")

                        if reasoning:
                            full_reasoning += reasoning
                            yield ChatChunk(content=reasoning, is_reasoning=True)

                        if content:
                            full_content += content
                            yield ChatChunk(content=content, is_reasoning=False)

                        usage_data = chunk.get("usage")
                        if usage_data:
                            yield ChatChunk(
                                content="",
                                is_reasoning=False,
                                usage=self._parse_usage(usage_data),
                            )

            if full_content:
                self.messages.append({"role": "assistant", "content": full_content})
            elif full_reasoning:
                self.messages.append({"role": "assistant", "content": full_reasoning})

        except httpx.TimeoutException:
            console.print("\n[red]Error: Request timed out after 5 minutes[/red]")
            self.messages.pop()
            raise

    def _parse_usage(self, usage_data: Dict) -> ChatUsage:
        """Parse usage data from API response."""
        return ChatUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            prompt_tps=usage_data.get("prompt_tps", 0.0),
            generation_tps=usage_data.get("generation_tps", 0.0),
            prompt_process=usage_data.get("prompt_process", 0.0),
            first_token=usage_data.get("first_token", 0.0),
        )

    def clear_history(self) -> None:
        """Clear conversation history (keep system prompt)."""
        self.messages = [m for m in self.messages if m["role"] == "system"]


def _print_usage(usage: ChatUsage) -> None:
    """Print usage statistics."""
    console.print()
    console.print("[dim]" + "─" * 40 + "[/dim]")
    console.print(
        f"[dim]📊 prompt={usage.prompt_tokens}, "
        f"completion={usage.completion_tokens} "
        f"({usage.total_tokens} total)[/dim]"
    )
    console.print(
        f"[dim]⏱️  first: {usage.first_token:.2f}s | "
        f"speed: {usage.generation_tps:.1f} t/s[/dim]"
    )
    console.print("[dim]" + "─" * 40 + "[/dim]")


def _print_help() -> None:
    """Print help message."""
    console.print()
    console.print("[bold]Commands:[/bold]")
    console.print("  [cyan]/quit[/cyan]  - Exit the chat")
    console.print("  [cyan]/help[/cyan]  - Show this help message")
    console.print()


def _run_chat_round(client: ChatClient, user_input: str) -> bool:
    """
    Run one round of chat.

    Returns:
        True if should continue, False if should exit.
    """
    user_input = user_input.strip()

    if not user_input:
        return True

    if user_input.lower() == "/quit":
        console.print("[dim]Goodbye![/dim]")
        return False

    if user_input.lower() == "/help":
        _print_help()
        return True

    console.print()
    usage = None
    reasoning_started = False
    content_started = False
    try:
        for chunk in client.chat_stream(user_input):
            if chunk.content:
                if chunk.is_reasoning:
                    if not reasoning_started:
                        console.print()
                        console.print("[dim italic][Reasoning]: [/dim italic]", end="")
                        reasoning_started = True
                    console.print(chunk.content, style="dim", end="")
                else:
                    if not content_started:
                        if reasoning_started:
                            console.print()
                        console.print()
                        console.print("[bold green][Assistant]: [/bold green]", end="")
                        content_started = True
                    console.print(chunk.content, end="")
            if chunk.usage:
                usage = chunk.usage
    except httpx.TimeoutException:
        console.print()
        console.print("[red]Error: Request timed out[/red]")
        return True
    except Exception as e:
        console.print()
        console.print(f"[red]Error: {e}[/red]")
        return True

    console.print()

    if usage:
        _print_usage(usage)

    return True


def _select_model(models: List[Dict]) -> Optional[str]:
    """
    Let user select a model from the list.

    Returns:
        Selected model name or None if cancelled.
    """
    if not models:
        return None

    console.print("[bold]Select a model:[/bold]")
    for i, m in enumerate(models):
        model_name = m.get("model_name", "unknown")
        console.print(f"  [cyan]{i + 1}[/cyan]. {model_name}")

    console.print()

    while True:
        try:
            choice = Prompt.ask(
                "Enter number",
                default="1",
            )

            if choice.lower() == "q":
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx].get("model_name")

            console.print("[red]Invalid selection, please try again[/red]")
        except ValueError:
            console.print("[red]Please enter a number[/red]")


def get_base_url() -> str:
    """Get service base URL from status file."""
    from plllm_mlx.daemon import STATUS_FILE

    if STATUS_FILE.exists():
        try:
            status = json.loads(STATUS_FILE.read_text())
            port = status.get("port", 8000)
            return f"http://localhost:{port}"
        except Exception:
            pass

    return "http://localhost:8000"


@chat_app.callback(invoke_without_command=True)
def chat(
    model: str = typer.Option(None, "--model", "-m", help="Model name"),
    prompt: Optional[str] = typer.Option(
        None, "--prompt", help="Single prompt (non-interactive)"
    ),
    system: Optional[Path] = typer.Option(
        None, "--system", "-s", help="System prompt file"
    ),
    max_tokens: int = typer.Option(4096, "--max-tokens", help="Max generation tokens"),
):
    """
    Interactive chat with a loaded model.

    Examples:
        plllm-mlx chat                          # Select model interactively
        plllm-mlx chat -m Qwen/Qwen3-8B         # Use specific model
        plllm-mlx chat -m Qwen/Qwen3-8B -p "Hi" # Single prompt mode
        plllm-mlx chat -s system.txt            # With system prompt file
    """
    from plllm_mlx.client import PlClient

    try:
        client = PlClient(timeout=5.0)
        models = client.list_models(loaded_only=True)
    except Exception as e:
        console.print(f"[red]Error: Cannot connect to service: {e}[/red]")
        console.print("Start the service with: [cyan]plllm-mlx serve[/cyan]")
        raise typer.Exit(1)

    if not models:
        console.print("[red]Error: No models loaded[/red]")
        console.print()
        console.print("Load a model first:")
        console.print("  [cyan]plllm-mlx load <model_name>[/cyan]")
        console.print()
        console.print("Available models:")
        console.print("  [cyan]plllm-mlx ls[/cyan]")
        raise typer.Exit(1)

    if model is None:
        model = _select_model(models)
        if model is None:
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit(0)
        console.print()

    system_prompt = None
    if system:
        if not system.exists():
            console.print(f"[red]Error: System prompt file not found: {system}[/red]")
            raise typer.Exit(1)
        try:
            system_prompt = system.read_text().strip()
        except Exception as e:
            console.print(f"[red]Error reading system file: {e}[/red]")
            raise typer.Exit(1)

    chat_client = ChatClient(
        base_url=client.base_url,
        model=model,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
    )

    if prompt:
        _run_chat_round(chat_client, prompt)
        return

    console.print(f"[bold green]Chatting with {model}[/bold green]")
    console.print("[dim]Type /quit to exit, /help for commands[/dim]")
    console.print()

    while True:
        try:
            user_input = Prompt.ask("[bold cyan][You]: [/bold cyan]")
            if not _run_chat_round(chat_client, user_input):
                break
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break
