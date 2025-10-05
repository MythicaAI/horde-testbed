from typing import List, Dict
import uuid
import random
from pathlib import Path
from json import JSONEncoder

def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)

_default.default = JSONEncoder().default
JSONEncoder.default = _default

from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from pressure_math import generate_fields
from enemies import ENEMY_REGISTRY
from world_config import WORLD_SETUP
from models import FrameUpdate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["http://<your-ec2-ip>"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CORSMiddlewareForStaticFiles(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        print(f"Intercepted request for: {request.url.path}")
        response = await call_next(request)
        # Add CORS headers to static file responses
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

# Directory where static files are stored
STATIC_DIR = Path("/app/static")

@app.get("/assets/{file_path:path}")
async def serve_static(file_path: str):
    """
    Custom route to serve static files with CORS headers.
    """
    file_location = STATIC_DIR / file_path
    if not file_location.exists():
        return {"error": "File not found"}, 404

    response = FileResponse(file_location)
    # Add CORS headers
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

app.add_middleware(CORSMiddlewareForStaticFiles)


WORLD_SETUP = {
    "resolution": 1024,
    "latent_dim": 256,
    "dtype": "float16",
    "device": "cuda",
}


class GameMem:
    def __init__(self, setup_config: Dict):
        self.lock = asyncio.Lock()
        self.world_id = str(uuid.uuid4())
        world_size = (WORLD_SETUP["resolution"], WORLD_SETUP["resolution"], WORLD_SETUP["latent_dim"])
        self.world_latent = torch.randn(world_size, dtype=WORLD_SETUP["dtype"], device=WORLD_SETUP["device"])

GM = GameMem(WORLD_SETUP)

async def _save_world_async(world_id: str, world_cpu: torch.Tensor, meta: dict) -> None:
    """
    Save CPU copy to disk without blocking the event loop.
    Uses torch.save with a simple dict payload.
    """
    out_path = os.path.join(SAVE_DIR, f"{world_id}.pt")
    payload = {"meta": meta, "shape": tuple(world_cpu.shape), "world": world_cpu}
    await anyio.to_thread.run_sync(torch.save, payload, out_path, pickle_protocol=5)


# @app.get("/nika_world/start-game")
# async def start_game():
    
