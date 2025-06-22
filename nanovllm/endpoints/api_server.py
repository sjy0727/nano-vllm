import os
import time
import asyncio
import uuid
import json
import re
import logging
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
import typer
import uvicorn
from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from nanovllm import LLM, SamplingParams
from nanovllm.utils.memory import print_gpu_info

# --- Security ---
security = HTTPBearer()

async def api_key_security(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_keys = getattr(request.app.state, "api_keys", [])
    if not api_keys:
        return
    if credentials.scheme != "Bearer" or credentials.credentials not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

# --- FastAPI App ---
app = FastAPI(
    title="Nano-vLLM OpenAI Compatible API",
    description="A lightweight vLLM implementation with OpenAI API compatibility",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class Function(BaseModel):
    name: str = Field(..., description="The name of the function to be called.")
    description: Optional[str] = Field(None, description="A description of what the function does.")
    parameters: Dict[str, Any] = Field(..., description="The parameters the functions accepts, described as a JSON Schema object.")

class Tool(BaseModel):
    type: str = Field("function", description="The type of the tool. Currently, only `function` is supported.")
    function: Function

class ResponseFormat(BaseModel):
    type: str = Field("text", description="The type of response format. Must be one of `text` or `json_object`.")

class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(..., description="A list of messages comprising the conversation")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    n: Optional[int] = Field(1, ge=1, le=128, description="Number of chat completion choices")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192, description="Maximum number of tokens to generate")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    user: Optional[str] = Field(None, description="A unique identifier for the end user")
    tools: Optional[List[Tool]] = Field(None, description="A list of tools the model may call.")
    tool_choice: Optional[Union[str, Dict]] = Field(None, description="Controls which tool the model should use.")
    response_format: Optional[ResponseFormat] = Field(None, description="An object specifying the format that the model must output.")

class CompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    prompt: str = Field(..., description="The prompt to complete")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    n: Optional[int] = Field(1, ge=1, le=128, description="Number of completion choices")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192, description="Maximum number of tokens to generate")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    user: Optional[str] = Field(None, description="A unique identifier for the end user")

# --- Lifespan: Model/Tokenizer/Config Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # These will be set by CLI entry
    model_path = getattr(app.state, "model_path", os.environ.get("NANOVLLM_MODEL_PATH", os.path.expanduser("~/llms/Qwen3-0.6B/")))
    tensor_parallel_size = getattr(app.state, "tensor_parallel_size", 1)
    api_keys = getattr(app.state, "api_keys", [])
    logger = logging.getLogger("uvicorn")
    print_gpu_info(logger=logger)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=tensor_parallel_size)
        if hasattr(tokenizer, 'name_or_path'):
            model_id = os.path.basename(tokenizer.name_or_path.rstrip('/'))
        else:
            model_id = os.path.basename(model_path.rstrip('/'))
        app.state.tokenizer = tokenizer
        app.state.llm = llm
        app.state.model_id = model_id
        app.state.model_path = model_path
        app.state.tensor_parallel_size = tensor_parallel_size
        app.state.api_keys = api_keys
        app.state.logger = logger
        logger.info(f"Model loaded successfully: {model_path} (model_id: {model_id})")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield

app.router.lifespan_context = lifespan

# --- Utility Functions ---
def get_sampling_params(body: Dict[str, Any], request: Request) -> SamplingParams:
    return SamplingParams(
        temperature=body.get("temperature", 0.7),
        top_p=body.get("top_p", 1.0),
        top_k=body.get("top_k"),
        max_tokens=body.get("max_tokens", 2048),
        min_tokens=body.get("min_tokens"),
        presence_penalty=body.get("presence_penalty", 0.0),
        frequency_penalty=body.get("frequency_penalty", 0.0),
        repetition_penalty=body.get("repetition_penalty", 1.0),
        stop=body.get("stop"),
        stop_token_ids=body.get("stop_token_ids"),
        seed=body.get("seed"),
        n=body.get("n", 1)
    )

def generate_request_id() -> str:
    return f"chatcmpl-{uuid.uuid4()}"

def count_tokens(text: str, request: Request) -> int:
    tokenizer = request.app.state.tokenizer
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text))

def parse_tool_calls(text: str) -> (Optional[List[Dict[str, Any]]], Optional[str]):
    pattern = r"<\|action_start\|>(.*?)<\|action_end\|>"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None, text
    tool_calls = []
    for match in matches:
        try:
            tool_data = json.loads(match)
            arguments_str = tool_data.get("arguments", "{}")
            if isinstance(arguments_str, dict):
                arguments_str = json.dumps(arguments_str)
            tool_call = {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": tool_data.get("name"),
                    "arguments": arguments_str
                }
            }
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue
    if not tool_calls:
        return None, text
    return tool_calls, None

def format_chat_prompt(messages: List[Dict[str, str]], request: Request, tools: Optional[List[Dict[str, Any]]] = None) -> str:
    tokenizer = request.app.state.tokenizer
    logger = request.app.state.logger
    if tokenizer is None:
        return _format_chat_prompt_simple(messages)
    try:
        return tokenizer.apply_chat_template(
            messages, 
            tools=tools,
            tokenize=False, 
            add_generation_prompt=True, 
            enable_thinking=True
        )
    except Exception as e:
        logger.warning(f"Failed to apply chat template with tools: {e}. Falling back to simple formatting.")
        return _format_chat_prompt_simple(messages)

def _format_chat_prompt_simple(messages: List[Dict[str, str]]) -> str:
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted += f"User: {content}\n"
        elif role == "assistant":
            formatted += f"Assistant: {content}\n"
        elif role == "system":
            formatted += f"System: {content}\n"
    formatted += "Assistant: "
    return formatted

def validate_model(model: str, request: Request):
    model_id = request.app.state.model_id
    if model.lower() != model_id.lower():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {model} not found (only loaded model: {model_id})"
        )

def create_model_data(request: Request):
    model_id = request.app.state.model_id
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1710000000,
                "owned_by": "nanovllm",
                "permission": [{
                    "id": f"modelperm-{hash(model_id) % 1000}",
                    "object": "model_permission",
                    "created": 1710000000,
                    "allow_create_engine": True,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False
                }],
                "root": model_id
            }
        ]
    }

# --- Streaming Helper Functions ---
async def create_streaming_chunk(request_id: str, created: int, model: str, token: str, is_final: bool = False, completion_type: str = "chat.completion.chunk"):
    if completion_type == "chat.completion.chunk":
        delta = {} if is_final else {"content": token}
    else:
        delta = {"text": "" if is_final else token}
    return {
        "id": request_id,
        "created": created,
        "object": completion_type,
        "choices": [{
            "delta" if completion_type == "chat.completion.chunk" else "text": delta if completion_type == "chat.completion.chunk" else (token if not is_final else ""),
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop" if is_final else None
        }],
        "model": model
    }

def create_streaming_tool_call_chunk(
    request_id: str, created: int, model: str, tool_call_id: str,
    function_name: Optional[str] = None, function_args_char: Optional[str] = None
):
    tool_call_delta = {
        "index": 0,
        "id": tool_call_id,
        "type": "function",
    }
    if function_name is not None:
        tool_call_delta["function"] = {"name": function_name, "arguments": ""}
    else:
        tool_call_delta["function"] = {"arguments": function_args_char}
    return {
        "id": request_id,
        "created": created,
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": None, "tool_calls": [tool_call_delta]},
            "finish_reason": None
        }],
        "model": model
    }

async def create_final_chunk(request_id: str, created: int, model: str, finish_reason: str):
    return {
        "id": request_id,
        "created": created,
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }],
        "model": model
    }

async def generate_stream(prompts, sampling_params, request_id, created, model, completion_type, request):
    ACTION_START_TAG = "<|action_start|>"
    ACTION_END_TAG = "<|action_end|>"
    state = "TEXT"
    buffer = ""
    action_json_buffer = ""
    finish_reason = "stop"
    llm = request.app.state.llm
    async def llm_stream_generator():
        for item in llm.stream_generate(prompts, sampling_params):
            yield item
            await asyncio.sleep(0)
    try:
        async for _, token, token_id, is_finished in llm_stream_generator():
            if token_id == llm.tokenizer.eos_token_id:
                continue  # skip eos token
            if token is None and not is_finished:
                continue
            if token:
                buffer += token
            while True:
                if state == "TEXT":
                    if ACTION_START_TAG in buffer:
                        parts = buffer.split(ACTION_START_TAG, 1)
                        text_to_yield = parts[0]
                        buffer = parts[1]
                        if text_to_yield:
                            chunk = await create_streaming_chunk(request_id, created, model, text_to_yield, False, completion_type)
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        state = "ACTION"
                        action_json_buffer = ""
                        continue
                    else:
                        if buffer:
                            chunk = await create_streaming_chunk(request_id, created, model, buffer, False, completion_type)
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        buffer = ""
                        break
                elif state == "ACTION":
                    if ACTION_END_TAG in buffer:
                        parts = buffer.split(ACTION_END_TAG, 1)
                        json_part = parts[0]
                        buffer = parts[1]
                        action_json_buffer += json_part
                        finish_reason = "tool_calls"
                        try:
                            tool_data = json.loads(action_json_buffer)
                            function_name = tool_data.get("name", "")
                            function_args = tool_data.get("arguments", {})
                            args_str = json.dumps(function_args) if isinstance(function_args, dict) else str(function_args)
                            tool_call_id = f"call_{uuid.uuid4().hex}"
                            initial_chunk = create_streaming_tool_call_chunk(
                                request_id, created, model, tool_call_id, function_name=function_name
                            )
                            yield f"data: {json.dumps(initial_chunk, ensure_ascii=False)}\n\n"
                            for char in args_str:
                                arg_chunk = create_streaming_tool_call_chunk(
                                    request_id, created, model, tool_call_id, function_args_char=char
                                )
                                yield f"data: {json.dumps(arg_chunk, ensure_ascii=False)}\n\n"
                                await asyncio.sleep(0.001)
                        except Exception as e:
                            fallback_content = f"{ACTION_START_TAG}{action_json_buffer}{ACTION_END_TAG}"
                            chunk = await create_streaming_chunk(request_id, created, model, fallback_content, False, completion_type)
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        state = "TEXT"
                        action_json_buffer = ""
                        continue
                    else:
                        action_json_buffer += buffer
                        buffer = ""
                        break
                break
            if is_finished:
                if buffer and state == "TEXT":
                    chunk = await create_streaming_chunk(request_id, created, model, buffer, False, completion_type)
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                break
    except Exception as e:
        error_chunk = await create_final_chunk(request_id, created, model, "error")
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
    final_chunk = await create_final_chunk(request_id, created, model, finish_reason)
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

# --- API Endpoints ---
@app.post("/v1/chat/completions", dependencies=[Depends(api_key_security)])
async def chat_completions(request_body: ChatCompletionRequest, http_request: Request):
    validate_model(request_body.model, http_request)
    if not request_body.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Messages cannot be empty"
        )
    if request_body.tools and request_body.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tool use is not supported in streaming mode."
        )
    messages = [{"role": msg.role, "content": msg.content} for msg in request_body.messages]
    if request_body.response_format and request_body.response_format.type == "json_object":
        messages.append({
            "role": "system",
            "content": "You must only output a valid JSON object."
        })
    formatted_tools = [tool.model_dump() for tool in request_body.tools] if request_body.tools else None
    prompt = format_chat_prompt(messages, http_request, tools=formatted_tools)
    prompts = [prompt]
    sampling_params = get_sampling_params(request_body.model_dump(), http_request)
    prompt_tokens = count_tokens(prompt, http_request)
    llm = http_request.app.state.llm
    if request_body.stream:
        created = int(time.time())
        request_id = generate_request_id()
        return StreamingResponse(
            generate_stream(prompts, sampling_params, request_id, created, 
                            request_body.model, "chat.completion.chunk", http_request),
            media_type="text/event-stream"
        )
    else:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        raw_content = outputs[0]["text"]
        tool_calls, clean_content = parse_tool_calls(raw_content)
        if tool_calls:
            response_message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        else:
            response_message = {"role": "assistant", "content": clean_content}
            finish_reason = "stop"
        completion_tokens = count_tokens(json.dumps(response_message), http_request)
        return JSONResponse(content={
            "id": generate_request_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_body.model,
            "choices": [{
                "message": response_message,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        })

@app.post("/v1/completions", dependencies=[Depends(api_key_security)])
async def completions(request_body: CompletionRequest, http_request: Request):
    validate_model(request_body.model, http_request)
    sampling_params = get_sampling_params(request_body.model_dump(), http_request)
    prompt_tokens = count_tokens(request_body.prompt, http_request)
    prompts = [request_body.prompt]
    llm = http_request.app.state.llm
    if request_body.stream:
        created = int(time.time())
        request_id = generate_request_id()
        return StreamingResponse(
            generate_stream(prompts, sampling_params, request_id, created, 
                            request_body.model, "text_completion.chunk", http_request),
            media_type="text/event-stream"
        )
    else:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        content = outputs[0]["text"]
        completion_tokens = count_tokens(content, http_request)
        return JSONResponse(content={
            "id": generate_request_id(),
            "object": "text_completion",
            "created": int(time.time()),
            "model": request_body.model,
            "choices": [{
                "text": content,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        })

@app.get("/v1/models", dependencies=[Depends(api_key_security)])
async def list_models(request: Request):
    return create_model_data(request)

@app.get("/v1/models/{model_id}", dependencies=[Depends(api_key_security)])
async def get_model(model_id: str, request: Request):
    model_id_loaded = request.app.state.model_id
    if model_id != model_id_loaded:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found (only loaded model: {model_id_loaded})"
        )
    models_data = create_model_data(request)
    return models_data["data"][0]

@app.get("/health")
async def health_check(request: Request):
    llm = request.app.state.llm
    model_path = request.app.state.model_path
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "model_path": model_path,
        "timestamp": int(time.time())
    }

@app.get("/")
async def root():
    return {
        "message": "Nano-vLLM OpenAI Compatible API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

cli = typer.Typer()

@cli.command()
def main(
    host: str = typer.Option("0.0.0.0", help="Host to run the server on."),
    port: int = typer.Option(8000, help="Port to run the server on."),
    model_path: str = typer.Option(None, "--model-path", help="Path to the LLM model. If not provided, it will check NANOVLLM_MODEL_PATH env var."),
    api_key: List[str] = typer.Option([], "--api-key", help="API key to protect the endpoints. Can be used multiple times."),
    tp: int = typer.Option(1, "--tp", help="Tensor parallel size for the model.")
):
    """Start the Nano-vLLM OpenAI compatible server."""
    # 通过 app.state 传递全局配置
    app.state.model_path = model_path or os.environ.get("NANOVLLM_MODEL_PATH", os.path.expanduser("~/llms/Qwen3-0.6B/"))
    app.state.api_keys = api_key
    app.state.tensor_parallel_size = tp
    log_level = "info"
    logger = logging.getLogger("uvicorn")
    if api_key:
        logger.info(f"API keys enabled. {len(api_key)} key(s) loaded.")
    else:
        logger.info("API key not provided. The server is open to all connections.")
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Using model from: {app.state.model_path}")
    logger.info(f"Using tensor parallel size: {tp}")
    uvicorn.run(app, host=host, port=port, log_level=log_level)

if __name__ == "__main__":
    cli()

# 启动命令: uvicorn nanovllm.endpoints.server:app --host 0.0.0.0 --port 8000
# python -m nanovllm.endpoints.server \
#   --model-path ~/llms/Qwen3-0.6B \
#   --api-key "token-abc123" \
#   --tp 2