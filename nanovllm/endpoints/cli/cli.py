import typer
import subprocess
import sys
import os

app = typer.Typer(help="Nano-vLLM unified CLI")

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def api_server(ctx: typer.Context):
    """Run the full API server (OpenAI compatible)"""
    os.execvp(sys.executable, [sys.executable, "-m", "nanovllm.endpoints.api_server"] + [str(x) for x in ctx.args])

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def api_server_lite(ctx: typer.Context):
    """Run the lite API server (minimal, fast)"""
    os.execvp(sys.executable, [sys.executable, "-m", "nanovllm.endpoints.api_server_lite"] + [str(x) for x in ctx.args])

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def chat_cli(ctx: typer.Context):
    """Run the terminal chat CLI"""
    os.execvp(sys.executable, [sys.executable, "-m", "nanovllm.endpoints.chat_cli"] + [str(x) for x in ctx.args])

def main():
    app()

if __name__ == "__main__":
    main()
