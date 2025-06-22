# 🚀 Nano-vLLM OpenAI Gateway

[![许可证](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/sjy0727/nano-vllm/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-powered-green.svg)](https://fastapi.tiangolo.com/)

一个为 Nano-vLLM 引擎打造的轻量级、高性能、OpenAI 兼容的 API 服务器。

## ✨ 功能特性

- 🔄 **OpenAI API 兼容性**: 完全支持聊天和文本补全。
- 🌊 **流式响应**: 通过 Server-Sent Events (SSE) 实现实时 token 流。
- ⚡ **高性能引擎**: 由一个定制的类 PagedAttention 的 CUDA 引擎驱动。
- ⚙️ **张量并行**: 使用 `--tp` 参数在多个 GPU 上运行大模型。
- 📝 **API 文档**: 在 `/docs` 路径下自动生成交互式 Swagger UI。
- 🔑 **API 密钥安全**: 可选的 bearer token 身份验证。
- 🛠️ **函数调用/工具调用**：支持 OpenAI 风格的 Function Calling，可在 chat completions 请求中传递 `tools` 和 `tool_choice`，服务器会自动解析并返回工具调用（支持流式）。
- 🧩 **JSON 模式**：支持 `response_format={"type": "json_object"}`，强制模型输出合法 JSON 对象。
- 📦 **动态模型列表**：`/v1/models` 只返回当前实际加载的模型，始终与服务器状态一致。
- 🏗️ **全局状态集中管理**：所有服务端状态（模型、分词器、配置、密钥）均通过 FastAPI 的 `app.state` 统一管理，便于维护和扩展。

## 🚀 安装

本服务器直接从源代码仓库运行。

```bash
# 克隆仓库
git clone https://github.com/sjy0727/nano-vllm
cd nano-vllm

# 安装依赖
pip install -r requirements.txt 
# 或如果使用 pyproject.toml
# pip install -e .
```

## 🔧 快速开始

### 1. 启动服务器

```bash
# 使用默认设置启动
python -m nanovllm.cli.server --model-path /path/to/your/model

# 在指定的主机和端口上使用 2 个 GPU 运行
python -m nanovllm.cli.server \
  --model-path /path/to/qwen3-0.6b \
  --host 0.0.0.0 \
  --port 8080 \
  --tp 2
```

### 2. API 示例

#### 列出模型

```bash
curl http://localhost:8000/v1/models
```

#### 聊天补全 (非流式)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "你好!"}],
    "stream": false
  }'
```

示例响应:
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
      "content": "你好！我能为你做点什么？"
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

### 3. 使用 Python 客户端

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # 必需，但如果服务器未设置密钥则不会检查
)

# 流式请求
for chunk in client.chat.completions.create(
    messages=[{"role": "user", "content": "给我讲个短故事。"}],
    model="qwen3-0.6b",
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="")
```

## ⚙️ 配置

配置通过命令行参数进行管理。

| 参数 | 环境变量 | 描述 | 默认值 |
|---|---|---|---|
| `--host` | - | 服务器主机 | `0.0.0.0` |
| `--port` | - | 服务器端口 | `8000` |
| `--model-path`| `NANOVLLM_MODEL_PATH` | LLM 模型路径 | `~/llms/Qwen3-0.6B/` |
| `--api-key` | - | 用于保护 API 的密钥 (可多次使用) | 无 |
| `--tp` | - | 张量并行大小 | `1` |

## 🔌 API 端点

服务器提供以下主要端点：

- `POST /v1/chat/completions`: 生成基于聊天的补全。
- `POST /v1/completions`: 生成标准文本补全。
- `GET /v1/models`: 列出可用模型。
- `GET /health`: 服务器健康检查。

### 支持的参数

`chat/completions` 和 `completions` 请求的关键参数包括：

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `model` | string | **必需** | 要使用的模型ID |
| `messages`/`prompt` | array/string | **必需** | 模型的输入 |
| `stream` | boolean | `false` | 是否流式响应 |
| `temperature` | float | `0.7` | 采样温度 |
| `max_tokens` | integer | `2048` | 生成的最大 token 数 |
| `top_p` | float | `1.0` | 核采样参数 |
| `n` | integer | `1` | 要生成的选择数量 |
| `presence_penalty`| float | `0.0` | 存在惩罚 |
| `frequency_penalty`| float | `0.0` | 频率惩罚 |

**注意**: `usage` 字段仅在非流式响应中提供。

## 🛠️ 开发

```bash
# 克隆仓库
git clone https://github.com/sjy0727/nano-vllm
cd nano-vllm

# 设置虚拟环境
python -m venv venv
source venv/bin/activate

# 以可编辑模式安装
pip install -e .
```

## 🚨 问题排查

- **Connection refused (连接被拒绝)**: 检查 `--host` 和 `--port` 参数，并检查防火墙设置。
- **Model not found (模型未找到)**: 确保 `--model-path` 正确，并且您有权限读取文件。
- **CUDA out of memory (显存不足)**: 尝试减小批量大小或使用更小的模型。如果使用张量并行（`--tp`），请确保您有足够的 GPU 和显存。

## 🤝 贡献

欢迎各种贡献！请随时提交：
- Bug 报告
- 功能请求
- Pull requests
- 文档改进

## ⚖️ 许可证

本项目采用 Apache 2.0 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。 