"""
Stable Audio 3 SFX — FastAPI service

Prerequisites (one-time):
  1. Accept the license at https://huggingface.co/stabilityai/stable-audio-3-small-sfx
  2. huggingface-cli login   (or export HF_TOKEN=<your_token>)

Install deps: see requirements.txt for uv commands.

Run:
  .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
"""

import io
import os
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from stable_audio_3 import StableAudioModel

MODEL_NAME = os.getenv("MODEL_NAME", "small-sfx")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = DEVICE
    for attempt in range(5):
        try:
            torch.cuda.empty_cache()
            app.state.model = StableAudioModel.from_pretrained(MODEL_NAME, device=device)
            app.state.model_name = MODEL_NAME
            break
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() and device == "cuda" and attempt < 4:
                import time
                wait = 10 * (attempt + 1)
                print(f"CUDA OOM on attempt {attempt+1}, retrying in {wait}s...")
                torch.cuda.empty_cache()
                time.sleep(wait)
            else:
                raise
    yield


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
    return {
        "status": "ok",
        "model": app.state.model_name,
        "device": DEVICE,
    }


@app.post("/generate")
def generate(req: GenerateRequest):
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
        audio = audio[0]          # (channels, samples)
    if audio.ndim == 2:
        audio = audio.T           # soundfile expects (samples, channels)
    else:
        audio = audio[:, None]    # mono → (samples, 1)

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
