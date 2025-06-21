import threading
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm.engine.llm_engine import LLMEngine

# Store engine in a global-like manner for the server thread to access
_engine_ref = {}

def get_engine() -> "LLMEngine":
    return _engine_ref.get("engine")

def run_visualizer_server(engine: "LLMEngine", host: str = "0.0.0.0", port: int = 8001):
    _engine_ref["engine"] = engine
    
    app = FastAPI()

    # Load HTML template
    template_path = Path(__file__).parent / "templates" / "visualizer.html"
    with open(template_path, "r") as f:
        html_template = f.read()

    @app.get("/", response_class=HTMLResponse)
    async def get_visualization_page():
        return HTMLResponse(content=html_template)

    @app.get("/api/status")
    async def get_status():
        engine_instance = get_engine()
        if not engine_instance:
            return {"error": "Engine not initialized"}

        scheduler = engine_instance.scheduler
        block_manager = scheduler.block_manager

        block_status = []
        for i, block in enumerate(block_manager.blocks):
            # To get seq_ids, we need to inspect the running sequences' block tables
            # This is the "non-invasive" part
            seq_ids = []
            if block.ref_count > 0:
                for seq in scheduler.running:
                    if block.block_id in seq.block_table:
                        seq_ids.append(seq.seq_id)
            
            block_status.append({
                "id": i,
                "ref_count": block.ref_count,
                "seq_ids": sorted(list(set(seq_ids))),
                "is_free": block.ref_count == 0,
            })
        
        status = {
            "num_total_blocks": len(block_manager.blocks),
            "num_free_blocks": len(block_manager.free_block_ids),
            "num_used_blocks": len(block_manager.used_block_ids),
            "num_waiting_seqs": len(scheduler.waiting),
            "num_running_seqs": len(scheduler.running),
            "blocks": block_status
        }
        return status

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    
    # Run in a thread so it doesn't block the main process
    thread = threading.Thread(target=server.run)
    thread.daemon = True
    thread.start()
    print(f"\n--- BlockManager Visualizer running on http://{host}:{port} ---") 