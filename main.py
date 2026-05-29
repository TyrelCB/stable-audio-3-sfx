"""
Stable Audio 3 SFX — FastAPI service

Prerequisites (one-time):
  1. Accept the license at https://huggingface.co/stabilityai/stable-audio-3-small-sfx
  2. huggingface-cli login   (or export HF_TOKEN=<your_token>)

Install deps: see requirements.txt for uv commands.

Run:
  .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8765

Env vars:
  MODEL_NAME    model variant to load (default: small-sfx)
  IDLE_TIMEOUT  seconds of inactivity before unloading from GPU (default: 300, 0 = never)
"""

import asyncio
import io
import os
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from stable_audio_3 import StableAudioModel

MODEL_NAME = os.getenv("MODEL_NAME", "small-sfx")
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "300"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model() -> StableAudioModel:
    for attempt in range(5):
        try:
            torch.cuda.empty_cache()
            return StableAudioModel.from_pretrained(MODEL_NAME, device=DEVICE)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() and DEVICE == "cuda" and attempt < 4:
                wait = 10 * (attempt + 1)
                print(f"CUDA OOM on attempt {attempt + 1}, retrying in {wait}s...")
                torch.cuda.empty_cache()
                time.sleep(wait)
            else:
                raise


async def _idle_watcher(app: FastAPI) -> None:
    while True:
        await asyncio.sleep(30)
        with app.state.lock:
            if app.state.model is None:
                continue
            idle = time.monotonic() - app.state.last_used
            if idle >= IDLE_TIMEOUT:
                app.state.model = None
                torch.cuda.empty_cache()
                print(f"Model unloaded after {idle:.0f}s idle — GPU memory freed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.lock = threading.Lock()
    app.state.last_used = time.monotonic()
    app.state.model = _load_model()
    app.state.model_name = MODEL_NAME

    task = asyncio.create_task(_idle_watcher(app)) if IDLE_TIMEOUT > 0 else None
    yield
    if task:
        task.cancel()
    with app.state.lock:
        app.state.model = None
    torch.cuda.empty_cache()


app = FastAPI(title="Stable Audio 3 SFX", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    duration: float = Field(default=5.0, gt=0, le=60)
    steps: int = Field(default=8, ge=1, le=100)
    cfg_scale: float = Field(default=1.0, ge=0)
    seed: int = Field(default=-1)


@app.get("/health")
def health():
    loaded = app.state.model is not None
    idle = time.monotonic() - app.state.last_used
    return {
        "status": "ok",
        "model": app.state.model_name,
        "device": DEVICE,
        "loaded": loaded,
        "idle_seconds": round(idle),
        "idle_timeout": IDLE_TIMEOUT if IDLE_TIMEOUT > 0 else "disabled",
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    with app.state.lock:
        if app.state.model is None:
            print("Model unloaded — reloading...")
            try:
                app.state.model = _load_model()
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Model reload failed: {e}")

        app.state.last_used = time.monotonic()

        try:
            audio = app.state.model.generate(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt or None,
                duration=req.duration,
                steps=req.steps,
                cfg_scale=req.cfg_scale,
                seed=req.seed,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # generate() returns float32 (batch, channels, samples); squeeze batch dim
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().float().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 3:
        audio = audio[0]        # (channels, samples)
    if audio.ndim == 2:
        audio = audio.T         # soundfile expects (samples, channels)
    else:
        audio = audio[:, None]  # mono → (samples, 1)

    inner = getattr(app.state.model, "model", None)
    sample_rate = getattr(inner, "sample_rate", 44100)

    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)

    filename = req.prompt[:40].replace(" ", "_").replace("/", "-") + ".wav"
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
