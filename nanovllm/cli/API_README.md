# Nano-vLLM OpenAI API 兼容接口

这是一个基于 Nano-vLLM 的轻量级 OpenAI API 兼容服务器，支持聊天补全和文本补全功能。

## 功能特性

- ✅ **完整的 OpenAI API 兼容性** - 支持标准的 OpenAI API 格式
- ✅ **聊天补全接口** - `/v1/chat/completions`
- ✅ **文本补全接口** - `/v1/completions`
- ✅ **流式响应** - 支持 Server-Sent Events (SSE)
- ✅ **模型管理** - `/v1/models` 接口
- ✅ **错误处理** - 完整的错误处理和状态码
- ✅ **Token 统计** - 返回详细的 token 使用情况
- ✅ **CORS 支持** - 跨域请求支持
- ✅ **健康检查** - `/health` 接口
- ✅ **API 文档** - 自动生成的 Swagger 文档

## 快速开始

### 1. 启动服务器

```bash
# 启动服务器
uvicorn nanovllm.cli.server:app --host 0.0.0.0 --port 8000

# 或者使用 Python 模块方式
python -m uvicorn nanovllm.cli.server:app --host 0.0.0.0 --port 8000
```

### 2. 访问 API 文档

启动后访问 `http://localhost:8000/docs` 查看交互式 API 文档。

### 3. 健康检查

```bash
curl http://localhost:8000/health
```

## API 接口

### 聊天补全 (Chat Completions)

**端点:** `POST /v1/chat/completions`

**请求示例:**

```python
import requests

# 非流式请求
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "qwen3-0.6b",
        "messages": [
            {"role": "system", "content": "你是一个有用的AI助手。"},
            {"role": "user", "content": "你好，请介绍一下你自己。"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False
    }
)

print(response.json())
```

**流式请求示例:**

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "qwen3-0.6b",
        "messages": [
            {"role": "user", "content": "请用一句话解释什么是人工智能。"}
        ],
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                break
            try:
                data = json.loads(data_str)
                if 'choices' in data and data['choices']:
                    delta = data['choices'][0].get('delta', {})
                    if 'content' in delta:
                        print(delta['content'], end='', flush=True)
            except json.JSONDecodeError:
                continue
```

### 文本补全 (Completions)

**端点:** `POST /v1/completions`

**请求示例:**

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "qwen3-0.6b",
        "prompt": "人工智能是",
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": False
    }
)

print(response.json())
```

### 模型列表

**端点:** `GET /v1/models`

**请求示例:**

```python
import requests

response = requests.get("http://localhost:8000/v1/models")
models = response.json()

for model in models['data']:
    print(f"模型: {model['id']}, 所有者: {model['owned_by']}")
```

## 支持的参数

### 聊天补全参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `model` | string | 必需 | 模型ID |
| `messages` | array | 必需 | 消息数组 |
| `temperature` | float | 0.7 | 采样温度 (0.0-2.0) |
| `top_p` | float | 1.0 | 核采样参数 (0.0-1.0) |
| `n` | integer | 1 | 生成的选择数量 (1-128) |
| `stream` | boolean | false | 是否流式响应 |
| `max_tokens` | integer | 2048 | 最大生成token数 (1-8192) |
| `presence_penalty` | float | 0.0 | 存在惩罚 (-2.0-2.0) |
| `frequency_penalty` | float | 0.0 | 频率惩罚 (-2.0-2.0) |
| `user` | string | null | 用户标识符 |

### 文本补全参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `model` | string | 必需 | 模型ID |
| `prompt` | string | 必需 | 提示文本 |
| `temperature` | float | 0.7 | 采样温度 (0.0-2.0) |
| `top_p` | float | 1.0 | 核采样参数 (0.0-1.0) |
| `n` | integer | 1 | 生成的选择数量 (1-128) |
| `stream` | boolean | false | 是否流式响应 |
| `max_tokens` | integer | 2048 | 最大生成token数 (1-8192) |
| `presence_penalty` | float | 0.0 | 存在惩罚 (-2.0-2.0) |
| `frequency_penalty` | float | 0.0 | 频率惩罚 (-2.0-2.0) |
| `user` | string | null | 用户标识符 |

## 响应格式

### 非流式响应

```json
{
  "id": "req_1234567890abcdef",
  "object": "chat.completion",
  "created": 1703123456,
  "model": "qwen3-0.6b",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "你好！我是一个AI助手..."
      },
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 15,
    "total_tokens": 40
  }
}
```

### 流式响应

当设置 `stream: true` 时，服务器会通过 Server-Sent Events (SSE) 协议实时返回生成的 token。每个事件都以 `data: ` 开头，并以 `\n\n` 结尾。

流的最后会发送一个 `data: [DONE]\n\n` 事件来表示结束。

**注意**：在流式模式下，响应中不包含 `usage` 字段。如果需要获取 token 使用情况，请使用非流式请求。

#### 响应格式

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"qwen3-0.6b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"qwen3-0.6b","choices":[{"index":0,"delta":{"content":"当"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"qwen3-0.6b","choices":[{"index":0,"delta":{"content":"然"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"qwen3-0.6b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## 错误处理

API 使用标准的 HTTP 状态码：

- `200` - 成功
- `400` - 请求参数错误
- `404` - 模型不存在
- `500` - 服务器内部错误

错误响应格式：

```json
{
  "error": {
    "message": "错误描述",
    "type": "error_type",
    "code": "error_code"
  }
}
```

## 测试

### 使用 cURL 测试流式响应

```bash
# 使用 curl 测试流式响应
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

## 配置

### 模型路径

服务器通过 `--model-path` 命令行参数来指定加载的 LLM 模型路径。

如果未提供该参数，服务器会尝试从环境变量 `NANOVLLM_MODEL_PATH` 中获取路径。

如果两者都未设置，默认路径为 `~/llms/Qwen3-0.6B/`。

**启动示例:**
```bash
python -m nanovllm.cli.server --model-path /path/to/your/model
```

## 注意事项

1. **流式响应 (Streaming)**：当 `stream=true` 时，服务器返回标准的 Server-Sent Events 流。
2. **Usage 统计**：`usage` 字段仅在非流式响应中提供。
3. **模型路径**：确保通过 `--model-path` 参数或 `NANOVLLM_MODEL_PATH` 环境变量正确设置模型路径。
4. **内存使用**：大模型可能需要较多内存，请确保系统资源充足。