# 🚀 Nano-vLLM OpenAI Gateway

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/sjy0727/nano-vllm/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-powered-green.svg)](https://fastapi.tiangolo.com/)

A lightweight, high-performance, OpenAI-compatible API server for the Nano-vLLM engine.

## ✨ Features

- 🔄 **OpenAI API Compatibility**: Full support for chat and text completions.
- 🌊 **Streaming Responses**: Real-time token streaming via Server-Sent Events (SSE).
- ⚡ **High-Performance Engine**: Powered by a custom PagedAttention-like CUDA engine.
- ⚙️ **Tensor Parallelism**: Run large models across multiple GPUs with `--tp` flag.
- 📝 **API Documentation**: Automatic interactive Swagger UI at `/docs`.
- 🔑 **API Key Security**: Optional bearer token authentication.
- 🛠️ **Function Calling / Tool Use**: Supports OpenAI-style function calling. You can pass `tools` and `tool_choice` in chat completions requests, and the server will parse and return tool calls (including streaming).
- 🧩 **JSON Mode**: Supports `response_format={"type": "json_object"}` to force the model to output valid JSON objects.
- 📦 **Dynamic Model List**: `/v1/models` always returns only the currently loaded model, matching the server's actual state.
- 🏗️ **Centralized App State**: All server state (model, tokenizer, config, keys) is managed via FastAPI's `app.state` for better maintainability and extensibility.

## 🚀 Installation

This server is run directly from the source repository.

```bash
# Clone the repository
git clone https://github.com/sjy0727/nano-vllm
cd nano-vllm

# Install dependencies
pip install -r requirements.txt 
# Or if using pyproject.toml
# pip install -e .
```
*Note: Please adapt the repository URL.*

## 🔧 Quick Start

### 1. Starting the Server

```bash
# Start with default settings
python -m nanovllm.cli.server --model-path /path/to/your/model

# Run on a specific host and port with 2 GPUs
python -m nanovllm.cli.server \
  --model-path /path/to/qwen3-0.6b \
  --host 0.0.0.0 \
  --port 8080 \
  --tp 2
```

### 2. API Examples

#### List Models

```bash
curl http://localhost:8000/v1/models
```

#### Chat Completions (Non-streaming)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

Example response:
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1719123456,
  "model": "qwen3-0.6b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 9,
    "total_tokens": 18
  }
}
```

### 3. API Usage with Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Required, but not checked if server has no keys
)

# Streaming request
for chunk in client.chat.completions.create(
    messages=[{"role": "user", "content": "Tell me a short story."}],
    model="qwen3-0.6b",
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="")
```

## ⚙️ Configuration

Configuration is managed via command-line arguments.

| Argument | Environment Variable | Description | Default |
|---|---|---|---|
| `--host` | - | Server host | `0.0.0.0` |
| `--port` | - | Server port | `8000` |
| `--model-path`| `NANOVLLM_MODEL_PATH` | Path to the LLM model | `~/llms/Qwen3-0.6B/` |
| `--api-key` | - | Secure the API with a key (can be used multiple times) | None |
| `--tp` | - | Tensor Parallelism size | `1` |

## 🔌 API Endpoints

The server provides the following main endpoints:

- `POST /v1/chat/completions`: Generate chat-based completions.
- `POST /v1/completions`: Generate standard text completions.
- `GET /v1/models`: List available models.
- `GET /health`: Health check for the server.

### Supported Parameters

Key parameters for `chat/completions` and `completions` requests include:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | **Required** | Model ID to use |
| `messages`/`prompt` | array/string | **Required** | Input for the model |
| `stream` | boolean | `false` | Whether to stream the response |
| `temperature` | float | `0.7` | Sampling temperature |
| `max_tokens` | integer | `2048` | Max tokens to generate |
| `top_p` | float | `1.0` | Nucleus sampling parameter |
| `n` | integer | `1` | Number of choices to generate |
| `presence_penalty`| float | `0.0` | Presence penalty |
| `frequency_penalty`| float | `0.0` | Frequency penalty |

**Note**: The `usage` field for token counts is only available in non-streaming responses.

## 🛠️ Development

```bash
# Clone the repository
git clone https://github.com/sjy0727/nano-vllm
cd nano-vllm

# Set up virtual environment
python -m venv venv
source venv/bin/activate

# Install in editable mode
pip install -e .
```

## 🚨 Troubleshooting

- **Connection refused**: Verify the `--host` and `--port` arguments and check firewall settings.
- **Model not found**: Ensure the `--model-path` is correct and you have permission to read the files.
- **CUDA out of memory**: Try reducing batch sizes or using a smaller model. If using tensor parallelism (`--tp`), ensure you have enough GPUs with sufficient VRAM.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit:
- Bug reports
- Feature requests
- Pull requests
- Documentation improvements

## ⚖️ License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.