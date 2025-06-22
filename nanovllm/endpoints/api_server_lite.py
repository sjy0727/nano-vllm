import os
import json
import uuid
import time
import asyncio
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams
import typer

app = FastAPI()
security = HTTPBearer()

class ChatMessage(BaseModel):
    role: str
    content: str

class Tool(BaseModel):
    type: str = "function"
    function: dict

class ResponseFormat(BaseModel):
    type: str = "text"

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    max_tokens: int = 2048
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    tools: list[Tool] = None
    tool_choice: dict | str = None
    response_format: ResponseFormat = None

def api_key_security(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_keys = getattr(request.app.state, "api_keys", [])
    if not api_keys:
        return
    if credentials.scheme != "Bearer" or credentials.credentials not in api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

@app.on_event("startup")
def load_model():
    model_path = os.environ.get("NANOVLLM_MODEL_PATH", os.path.expanduser("~/llms/Qwen3-0.6B/"))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, enforce_eager=True)
    model_id = os.path.basename(tokenizer.name_or_path.rstrip('/')) if hasattr(tokenizer, 'name_or_path') else os.path.basename(model_path.rstrip('/'))
    app.state.llm = llm
    app.state.tokenizer = tokenizer
    app.state.model_id = model_id
    app.state.model_path = model_path

@app.post("/v1/chat/completions", dependencies=[Depends(api_key_security)])
async def chat_completions(body: ChatCompletionRequest, request: Request):
    if body.model.lower() != app.state.model_id.lower():
        raise HTTPException(400, f"Model {body.model} not found (only loaded model: {app.state.model_id})")
    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    if body.response_format and body.response_format.type == "json_object":
        messages.append({"role": "system", "content": "You must only output a valid JSON object."})
    prompt = app.state.tokenizer.apply_chat_template(messages, tools=[t.dict() for t in body.tools] if body.tools else None, tokenize=False, add_generation_prompt=True) if hasattr(app.state.tokenizer, 'apply_chat_template') else "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nAssistant: "
    sampling_params = SamplingParams(
        temperature=body.temperature, top_p=body.top_p, max_tokens=body.max_tokens, n=body.n,
        presence_penalty=body.presence_penalty, frequency_penalty=body.frequency_penalty
    )
    def parse_tool_calls(text):
        import re
        pattern = r"<\|action_start\|>(.*?)<\|action_end\|>"
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches: return None, text
        tool_calls = []
        for match in matches:
            try:
                tool_data = json.loads(match)
                args = tool_data.get("arguments", "{}")
                if isinstance(args, dict): args = json.dumps(args)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": tool_data.get("name"), "arguments": args}
                })
            except Exception: continue
        return tool_calls, None
    if body.stream:
        async def stream():
            ACTION_START_TAG = "<|action_start|>"
            ACTION_END_TAG = "<|action_end|>"
            buffer, action_json_buffer, state = "", "", "TEXT"
            request_id = f"chatcmpl-{uuid.uuid4()}"
            created = int(time.time())
            llm = app.state.llm
            tokenizer = app.state.tokenizer
            async def llm_stream():
                for item in llm.stream_generate([prompt], sampling_params):
                    yield item
                    await asyncio.sleep(0)
            async for _, token, token_id, is_finished in llm_stream():
                if token_id == tokenizer.eos_token_id:
                    continue
                if token: buffer += token
                while True:
                    if state == "TEXT":
                        if ACTION_START_TAG in buffer:
                            parts = buffer.split(ACTION_START_TAG, 1)
                            if parts[0]:
                                yield f"data: {json.dumps({'id': request_id, 'created': created, 'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': parts[0]}, 'index': 0, 'logprobs': None, 'finish_reason': None}], 'model': body.model})}\n\n"
                            buffer = parts[1]
                            state = "ACTION"; action_json_buffer = ""
                            continue
                        else:
                            if buffer:
                                yield f"data: {json.dumps({'id': request_id, 'created': created, 'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': buffer}, 'index': 0, 'logprobs': None, 'finish_reason': None}], 'model': body.model})}\n\n"
                            buffer = ""
                            break
                    elif state == "ACTION":
                        if ACTION_END_TAG in buffer:
                            parts = buffer.split(ACTION_END_TAG, 1)
                            action_json_buffer += parts[0]
                            buffer = parts[1]
                            try:
                                tool_data = json.loads(action_json_buffer)
                                function_name = tool_data.get("name", "")
                                function_args = tool_data.get("arguments", {})
                                args_str = json.dumps(function_args) if isinstance(function_args, dict) else str(function_args)
                                tool_call_id = f"call_{uuid.uuid4().hex}"
                                yield f"data: {json.dumps({'id': request_id, 'created': created, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': None, 'tool_calls': [{'index': 0, 'id': tool_call_id, 'type': 'function', 'function': {'name': function_name, 'arguments': ''}}]}, 'finish_reason': None}], 'model': body.model})}\n\n"
                                for char in args_str:
                                    yield f"data: {json.dumps({'id': request_id, 'created': created, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': None, 'tool_calls': [{'index': 0, 'id': tool_call_id, 'type': 'function', 'function': {'arguments': char}}]}, 'finish_reason': None}], 'model': body.model})}\n\n"
                                    await asyncio.sleep(0.001)
                            except Exception:
                                pass
                            state = "TEXT"; action_json_buffer = ""
                            continue
                        else:
                            action_json_buffer += buffer
                            buffer = ""
                            break
                    break
                if is_finished:
                    if buffer and state == "TEXT":
                        yield f"data: {json.dumps({'id': request_id, 'created': created, 'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': buffer}, 'index': 0, 'logprobs': None, 'finish_reason': None}], 'model': body.model})}\n\n"
                    break
            yield f"data: {json.dumps({'id': request_id, 'created': created, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'model': body.model})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream")
    else:
        outputs = app.state.llm.generate([prompt], sampling_params, use_tqdm=False)
        raw_content = outputs[0]["text"]
        tool_calls, clean_content = parse_tool_calls(raw_content)
        if tool_calls:
            response_message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        else:
            response_message = {"role": "assistant", "content": clean_content}
            finish_reason = "stop"
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [{
                "message": response_message,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason
            }],
            "usage": None
        })

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": app.state.model_id,
            "object": "model",
            "created": 1710000000,
            "owned_by": "nanovllm",
            "permission": [],
            "root": app.state.model_id
        }]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": app.state.llm is not None,
        "model_path": app.state.model_path,
        "timestamp": int(time.time())
    }

cli = typer.Typer()

@cli.command()
def main(model_path: str = typer.Option(None, "--model-path", help="Path to the LLM model."), api_key: list[str] = typer.Option([], "--api-key", help="API key to protect the endpoints. Can be used multiple times."), host: str = "0.0.0.0", port: int = 8000):
    if model_path:
        os.environ["NANOVLLM_MODEL_PATH"] = model_path
    app.state.api_keys = api_key
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    cli() 