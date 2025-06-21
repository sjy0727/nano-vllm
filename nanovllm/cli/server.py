import os
import time
import asyncio
import uuid
import json
import logging
from typing import List, Optional, Dict, Any
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

# --- Configuration Globals ---
API_KEYS: List[str] = []
MODEL_PATH: str = ""
TENSOR_PARALLEL_SIZE: int = 1

# --- Application Globals ---
tokenizer = None
llm = None
logger = logging.getLogger("uvicorn")

# --- Security ---
security = HTTPBearer()

# --- Constants ---
SUPPORTED_MODELS = ["qwen3-0.6b", "gpt-3.5-turbo", "gpt-4"]
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_N = 1

async def api_key_security(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to verify API key."""
    if not API_KEYS:
        return
    
    if credentials.scheme != "Bearer" or credentials.credentials not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    initialize_model()
    yield

app = FastAPI(
    title="Nano-vLLM OpenAI Compatible API",
    description="A lightweight vLLM implementation with OpenAI API compatibility",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(..., description="A list of messages comprising the conversation")
    temperature: Optional[float] = Field(DEFAULT_TEMPERATURE, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(DEFAULT_TOP_P, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    n: Optional[int] = Field(DEFAULT_N, ge=1, le=128, description="Number of chat completion choices")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    max_tokens: Optional[int] = Field(DEFAULT_MAX_TOKENS, ge=1, le=8192, description="Maximum number of tokens to generate")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    user: Optional[str] = Field(None, description="A unique identifier for the end user")

class CompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    prompt: str = Field(..., description="The prompt to complete")
    temperature: Optional[float] = Field(DEFAULT_TEMPERATURE, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(DEFAULT_TOP_P, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    n: Optional[int] = Field(DEFAULT_N, ge=1, le=128, description="Number of completion choices")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    max_tokens: Optional[int] = Field(DEFAULT_MAX_TOKENS, ge=1, le=8192, description="Maximum number of tokens to generate")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    user: Optional[str] = Field(None, description="A unique identifier for the end user")

def initialize_model():
    """Initialize model and tokenizer"""
    global tokenizer, llm

    print_gpu_info(logger=logger)

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        llm = LLM(MODEL_PATH, enforce_eager=True, tensor_parallel_size=TENSOR_PARALLEL_SIZE)
        logger.info(f"Model loaded successfully: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def get_sampling_params(body: Dict[str, Any]) -> SamplingParams:
    """Extract sampling parameters from request body"""
    return SamplingParams(
        temperature=body.get("temperature", DEFAULT_TEMPERATURE),
        top_p=body.get("top_p", DEFAULT_TOP_P),
        top_k=body.get("top_k"),
        max_tokens=body.get("max_tokens", DEFAULT_MAX_TOKENS),
        min_tokens=body.get("min_tokens"),
        presence_penalty=body.get("presence_penalty", 0.0),
        frequency_penalty=body.get("frequency_penalty", 0.0),
        repetition_penalty=body.get("repetition_penalty", 1.0),
        stop=body.get("stop"),
        stop_token_ids=body.get("stop_token_ids"),
        seed=body.get("seed"),
        n=body.get("n", DEFAULT_N)
    )

def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"req_{uuid.uuid4().hex[:16]}"

def count_tokens(text: str) -> int:
    """Count tokens in text"""
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text))

def format_chat_prompt(messages: List[Dict[str, str]]) -> str:
    """Format chat messages into prompt"""
    if tokenizer is None:
        return _format_chat_prompt_simple(messages)
    
    try:
        return tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
            enable_thinking=True
        )
    except Exception:
        return _format_chat_prompt_simple(messages)

def _format_chat_prompt_simple(messages: List[Dict[str, str]]) -> str:
    """Simple chat formatting fallback"""
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

def validate_model(model: str):
    """Validate if model is supported"""
    if model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {model} not found"
        )

def is_http2_request(request: Request) -> bool:
    """Check if request is HTTP/2"""
    return (hasattr(request, 'scope') and 
            request.scope.get('type') == 'http' and 
            request.scope.get('http_version') == '2')

async def create_streaming_chunk(request_id: str, created: int, model: str, 
                               token: str, is_final: bool = False, 
                               completion_type: str = "chat.completion.chunk"):
    """Create streaming response chunk"""
    if completion_type == "chat.completion.chunk":
        delta = {} if is_final else {"content": token}
    else:
        delta = {"text": "" if is_final else token}
    
    return {
        "id": request_id,
        "created": created,
        "object": completion_type,
        "choices": [{
            "delta" if completion_type == "chat.completion.chunk" else "text": 
                delta if completion_type == "chat.completion.chunk" else (token if not is_final else ""),
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop" if is_final else None
        }],
        "model": model
    }

async def generate_stream(prompts: List[str], sampling_params: SamplingParams, 
                         request_id: str, created: int, model: str, 
                         completion_type: str = "chat.completion.chunk"):
    """Generate streaming response"""
    content = ""
    completion_tokens = 0
    
    try:
        for idx, token, token_id, is_finished in llm.stream_generate(prompts, sampling_params):
            if token_id == llm.tokenizer.eos_token_id:
                continue
            
            if token is not None:
                content += token
                completion_tokens += 1
                
                chunk = await create_streaming_chunk(
                    request_id, created, model, token, False, completion_type
                )
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
            
            if is_finished:
                final_chunk = await create_streaming_chunk(
                    request_id, created, model, "", True, completion_type
                )
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                break
                
    except Exception as e:
        error_chunk = {
            "id": request_id,
            "created": created,
            "object": completion_type,
            "choices": [{
                "delta" if completion_type == "chat.completion.chunk" else "text": 
                    {} if completion_type == "chat.completion.chunk" else "",
                "index": 0,
                "logprobs": None,
                "finish_reason": "error"
            }],
            "model": model,
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
    
    yield "data: [DONE]\n\n"

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": str(exc),
                "type": "internal_error",
                "code": "internal_error"
            }
        }
    )

@app.post("/v1/chat/completions", dependencies=[Depends(api_key_security)])
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """Chat completion endpoint"""
    try:
        validate_model(request.model)
        
        if not request.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Messages cannot be empty"
            )
        
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        prompt = format_chat_prompt(messages)
        prompts = [prompt]
        
        sampling_params = get_sampling_params(request.model_dump())
        prompt_tokens = count_tokens(prompt)
        
        if request.stream:
            created = int(time.time())
            request_id = generate_request_id()
            
            return StreamingResponse(
                generate_stream(prompts, sampling_params, request_id, created, 
                              request.model, "chat.completion.chunk"),
                media_type="text/event-stream"
            )
        else:
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            content = outputs[0]["text"]
            completion_tokens = count_tokens(content)
            
            return JSONResponse(content={
                "id": generate_request_id(),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "message": {"role": "assistant", "content": content},
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
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/v1/completions", dependencies=[Depends(api_key_security)])
async def completions(request: CompletionRequest, http_request: Request):
    """Text completion endpoint"""
    try:
        validate_model(request.model)
        
        sampling_params = get_sampling_params(request.model_dump())
        prompt_tokens = count_tokens(request.prompt)
        prompts = [request.prompt]
        
        if request.stream:
            created = int(time.time())
            request_id = generate_request_id()
            
            return StreamingResponse(
                generate_stream(prompts, sampling_params, request_id, created, 
                              request.model, "text_completion.chunk"),
                media_type="text/event-stream"
            )
        else:
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            content = outputs[0]["text"]
            completion_tokens = count_tokens(content)
            
            return JSONResponse(content={
                "id": generate_request_id(),
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
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
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

def create_model_data():
    """Create model data structure"""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1710000000,
                "owned_by": "nanovllm" if model_id == "qwen3-0.6b" else "openai",
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
            for model_id in SUPPORTED_MODELS
        ]
    }

@app.get("/v1/models", dependencies=[Depends(api_key_security)])
async def list_models():
    """Get available models list"""
    return create_model_data()

@app.get("/v1/models/{model_id}", dependencies=[Depends(api_key_security)])
async def get_model(model_id: str):
    """Get specific model information"""
    if model_id not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    models_data = create_model_data()
    return next(model for model in models_data["data"] if model["id"] == model_id)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "model_path": MODEL_PATH,
        "timestamp": int(time.time())
    }

@app.get("/")
async def root():
    """Root endpoint"""
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
    global MODEL_PATH, API_KEYS, TENSOR_PARALLEL_SIZE
    
    if model_path is None:
        model_path = os.environ.get("NANOVLLM_MODEL_PATH", os.path.expanduser("~/llms/Qwen3-0.6B/"))
    
    MODEL_PATH = model_path
    API_KEYS = api_key
    TENSOR_PARALLEL_SIZE = tp

    log_level = "info"
    if API_KEYS:
        logger.info(f"API keys enabled. {len(API_KEYS)} key(s) loaded.")
    else:
        logger.info("API key not provided. The server is open to all connections.")
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Using model from: {MODEL_PATH}")
    logger.info(f"Using tensor parallel size: {TENSOR_PARALLEL_SIZE}")

    uvicorn.run(app, host=host, port=port, log_level=log_level)

if __name__ == "__main__":
    cli()
# 启动命令: uvicorn nanovllm.cli.server:app --host 0.0.0.0 --port 8000
# python -m nanovllm.cli.server \
#   --model-path ~/llms/Qwen3-0.6B \
#   --api-key "token-abc123" \
#   --tp 2