import typer
import os
import sys
from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from typing import List, Dict, Optional
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.utils.memory import print_gpu_info

# --- Rich Console Logger Adapter ---
class RichConsoleLogger:
    """A simple adapter to make rich.console.Console compatible with logging.Logger."""
    def __init__(self, console: Console):
        self._console = console

    def info(self, msg: str, *args, **kwargs):
        self._console.print(msg)
    
    # You can add other logging levels if needed (e.g., error, warning)
    def error(self, msg: str, *args, **kwargs):
        self._console.print(f"[bold red]ERROR:[/bold red] {msg}")

# --- Constants ---
HISTORY_SYSTEM_PROMPT = """You are a helpful AI assistant.
The user is having a conversation with you in a terminal.
Keep your answers concise and use markdown for formatting when appropriate."""

# --- Prompt Formatting (for local use) ---
tokenizer_for_format: Optional[AutoTokenizer] = None

def format_chat_prompt(messages: List[Dict[str, str]]) -> str:
    """Format chat messages into a single prompt string."""
    global tokenizer_for_format
    if tokenizer_for_format is None:
        return _format_chat_prompt_simple(messages)
    
    try:
        return tokenizer_for_format.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=True
        )
    except Exception:
        return _format_chat_prompt_simple(messages)

def _format_chat_prompt_simple(messages: List[Dict[str, str]]) -> str:
    """A simple fallback for formatting chat messages."""
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted += f"User: {content}\\n"
        elif role == "assistant":
            formatted += f"Assistant: {content}\\n"
        elif role == "system":
            formatted += f"System: {content}\\n"
    formatted += "Assistant: "
    return formatted

def main(
    model_path: str = typer.Option(
        None, 
        "--model-path", 
        "-m", 
        help="Path to the LLM model. If not provided, checks NANOVLLM_MODEL_PATH env var."
    ),
    tp: int = typer.Option(1, "--tp", help="Tensor parallel size for the model."),
    temperature: float = typer.Option(0.7, help="Sampling temperature."),
    top_p: float = typer.Option(1.0, help="Nucleus sampling parameter."),
    max_tokens: int = typer.Option(1024, help="Maximum number of tokens to generate."),
    visualize: bool = typer.Option(False, "--visualize", help="[Optional] Run a web server to visualize the BlockManager."),
):
    """
    A terminal-based chat client that runs the Nano-vLLM model locally.
    
    - Type your message and press Enter to send.
    - Type 'exit' or 'quit' to end the conversation.
    - Type 'clear' to reset the conversation history.
    """
    console = Console()
    global tokenizer_for_format

    if model_path is None:
        model_path = os.environ.get("NANOVLLM_MODEL_PATH", os.path.expanduser("~/llms/Qwen3-0.6B/"))

    if not os.path.isdir(model_path):
        console.print(f"[bold red]Error:[/bold red] Model path not found at [cyan]{model_path}[/cyan].")
        console.print("Please specify a valid path using --model-path or the NANOVLLM_MODEL_PATH environment variable.")
        raise typer.Exit(code=1)

    # Visualization check comes after argument parsing but before model loading
    if visualize:
        try:
            from nanovllm.cli.visualizer_server import run_visualizer_server
        except ImportError:
            console.print("[bold red]Error:[/bold red] To use the visualizer, please install the optional dependencies:")
            console.print("pip install fastapi \"uvicorn[standard]\"", style="yellow")
            raise typer.Exit(1)

    console.print(Panel(f"[bold green]Welcome to Nano-vLLM Local Chat![/bold green]\\nLoading model from [cyan]{model_path}[/cyan]...", title="Chat Client", border_style="dim"))
    
    # --- Model Loading ---
    print_gpu_info(logger=RichConsoleLogger(console))
    try:
        tokenizer_for_format = AutoTokenizer.from_pretrained(model_path)
        llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=tp)
        console.print("[bold green]Model loaded successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error loading model:[/bold red] {e}")
        raise typer.Exit(code=1)

    # Start visualizer server if requested
    if visualize:
        # The LLM object conveniently holds the engine instance
        run_visualizer_server(llm)

    messages: List[Dict[str, str]] = [{"role": "system", "content": HISTORY_SYSTEM_PROMPT}]

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    while True:
        try:
            console.print("[bold cyan]You: [/bold cyan]", end="")
            
            # More robust input handling to prevent crashes on decoding errors
            input_bytes = sys.stdin.buffer.readline()
            if not input_bytes: # Handle Ctrl+D (EOF)
                console.print("\n[bold yellow]Goodbye![/bold yellow]")
                break
            user_input = input_bytes.decode(sys.stdin.encoding or 'utf-8', errors='replace').rstrip()

            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break
            
            if user_input.lower() == "clear":
                messages = [{"role": "system", "content": HISTORY_SYSTEM_PROMPT}]
                console.print("[bold yellow]Conversation history cleared.[/bold yellow]")
                continue

            messages.append({"role": "user", "content": user_input})
            prompt = format_chat_prompt(messages)
            
            # Get the generator
            stream_iterator = llm.stream_generate([prompt], sampling_params)

            # --- Phase 1: Live-stream the Thinking Process ---
            thinking_content = ""
            remaining_buffer = ""
            has_started_thinking = False

            try:
                with Live(console=console, auto_refresh=False, vertical_overflow="visible") as live:
                    live.update(Panel("...", title="[bold yellow]Thinking Process[/bold yellow]", border_style="yellow", title_align="left"), refresh=True)

                    for _, token, token_id, is_finished in stream_iterator:
                        if token_id == llm.tokenizer.eos_token_id or token is None:
                            if is_finished: break
                            continue
                        
                        # Wait until we see the <think> tag
                        if not has_started_thinking:
                            if "<think>" in token:
                                has_started_thinking = True
                                # Discard the tag itself and anything before it in the chunk
                                token = token.split("<think>", 1)[1]
                            else:
                                continue # Discard all tokens before the thinking process starts

                        # Once thinking has started, stream it
                        if "</think>" in token:
                            last_think_part, remaining_buffer = token.split("</think>", 1)
                            thinking_content += last_think_part
                            live.update(Panel(Markdown(thinking_content.strip()), title="[bold yellow]Thinking Process[/bold yellow]", border_style="yellow", title_align="left"), refresh=True)
                            break # Exit the thinking live loop
                        
                        thinking_content += token
                        live.update(Panel(Markdown(thinking_content.strip()), title="[bold yellow]Thinking Process[/bold yellow]", border_style="yellow", title_align="left"), refresh=True)

                        if is_finished:
                            break
            except Exception as e:
                console.print(f"\\n[bold red]Error during thinking stream:[/bold red] {e}")
                messages.pop()
                continue
            
            # If no thinking was ever displayed, don't show an empty panel.
            # The remaining buffer will contain the whole response.
            if not has_started_thinking:
                remaining_buffer = thinking_content

            # --- Phase 2: Live-stream the Assistant Response ---
            ai_response_content = remaining_buffer
            try:
                # Only start this live session if there is something to say.
                if ai_response_content or next(stream_iterator, None) is not None:
                    with Live(console=console, auto_refresh=False, vertical_overflow="visible") as live:
                        live.update(Panel(Markdown(ai_response_content.strip() if ai_response_content else "..."), title="[bold magenta]Assistant[/bold magenta]", border_style="magenta", title_align="left"), refresh=True)
                        
                        for _, token, token_id, is_finished in stream_iterator:
                            if token_id == llm.tokenizer.eos_token_id or token is None:
                                if is_finished: break
                                continue
                            
                            ai_response_content += token
                            live.update(Panel(Markdown(ai_response_content.strip()), title="[bold magenta]Assistant[/bold magenta]", border_style="magenta", title_align="left"), refresh=True)
                            
                            if is_finished:
                                break
            except Exception as e:
                console.print(f"\\n[bold red]Error during assistant stream:[/bold red] {e}")
                messages.pop()
                continue

            if ai_response_content:
                messages.append({"role": "assistant", "content": ai_response_content.strip()})

        except KeyboardInterrupt:
            console.print("\\n[bold yellow]Interrupted. Type 'exit' to quit.[/bold yellow]")
            if messages and messages[-1]["role"] == "user":
                continue
        except Exception as e:
            console.print(f"\\n[bold red]An unexpected error occurred:[/bold red] {e}")


if __name__ == "__main__":
    typer.run(main) 