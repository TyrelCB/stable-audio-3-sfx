"""
Stable Audio 3 SFX — Gradio UI + MCP server

Prerequisites (one-time):
  1. Accept the license at https://huggingface.co/stabilityai/stable-audio-3-small-sfx
  2. huggingface-cli login   (or export HF_TOKEN=<your_token>)

Install deps: see requirements.txt for uv commands.

Run:
  .venv/bin/python main.py

Env vars:
  MODEL_NAME    model variant to load  (default: small-sfx)
  IDLE_TIMEOUT  seconds idle before GPU unload  (default: 300, 0 = never)
  PORT          server port  (default: 8765)

MCP endpoint:
  http://localhost:8765/gradio_api/mcp/sse
"""

import os
import threading
import time
from typing import Optional

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from stable_audio_3 import StableAudioModel

MODEL_NAME = os.getenv("MODEL_NAME", "small-sfx")
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "300"))
PORT = int(os.getenv("PORT", "8765"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model state (module-level for Gradio's functional style)
# ---------------------------------------------------------------------------

_model: Optional[StableAudioModel] = None
_lock = threading.Lock()
_last_used = time.monotonic()


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


def _idle_worker() -> None:
    global _model
    while True:
        time.sleep(30)
        with _lock:
            if _model is None or IDLE_TIMEOUT <= 0:
                continue
            idle = time.monotonic() - _last_used
            if idle >= IDLE_TIMEOUT:
                _model = None
                torch.cuda.empty_cache()
                print(f"Model unloaded after {idle:.0f}s idle — GPU memory freed.")


# ---------------------------------------------------------------------------
# Core generation function (exposed as Gradio endpoint + MCP tool)
# ---------------------------------------------------------------------------

def generate_sfx(
    prompt: str,
    duration: float = 5.0,
    steps: int = 8,
    cfg_scale: float = 1.0,
    seed: int = -1,
    negative_prompt: str = "",
) -> tuple[int, np.ndarray]:
    """Generate a sound effect from a text description.

    Args:
        prompt: Text description of the sound effect to generate, e.g. "thunder clap with distant rumble".
        duration: Length of the audio in seconds (1–60).
        steps: Number of diffusion steps. Higher values give better quality but are slower (1–50).
        cfg_scale: Classifier-free guidance scale. Higher values follow the prompt more strictly.
        seed: Random seed for reproducibility. Use -1 for a random result.
        negative_prompt: Description of what to avoid in the generated audio.

    Returns:
        Tuple of (sample_rate, stereo_audio_array) ready for playback or download.
    """
    global _model, _last_used

    with _lock:
        if _model is None:
            print("Model not loaded — loading now...")
            _model = _load_model()
        _last_used = time.monotonic()

        audio = _model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            duration=duration,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
        )

    # generate() → (batch, channels, samples) float32
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().float().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 3:
        audio = audio[0]        # (channels, samples)
    if audio.ndim == 2:
        audio = audio.T         # (samples, channels) for Gradio/soundfile

    inner = getattr(_model, "model", None)
    sample_rate = getattr(inner, "sample_rate", 44100)

    return (sample_rate, audio)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Stable Audio 3 SFX") as demo:
    gr.Markdown("# Stable Audio 3 SFX")
    gr.Markdown("Generate stereo sound effects from text prompts using the Stable Audio 3 diffusion model.")

    prompt = gr.Textbox(
        label="Prompt",
        placeholder='thunder clap with distant rumble',
        lines=2,
    )
    with gr.Row():
        duration = gr.Slider(1, 60, value=5, step=0.5, label="Duration (s)")
        steps    = gr.Slider(1, 50, value=8, step=1,   label="Steps")

    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            cfg_scale       = gr.Slider(0, 10, value=1.0, step=0.1, label="CFG Scale")
            seed            = gr.Number(value=-1, label="Seed  (−1 = random)", precision=0)
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="optional")

    btn       = gr.Button("Generate", variant="primary")
    audio_out = gr.Audio(label="Output", type="numpy")

    btn.click(
        fn=generate_sfx,
        inputs=[prompt, duration, steps, cfg_scale, seed, negative_prompt],
        outputs=audio_out,
    )


# ---------------------------------------------------------------------------
# FastAPI app — health endpoint + Gradio mounted at /
# ---------------------------------------------------------------------------

api = FastAPI()


@api.get("/health")
def health():
    loaded = _model is not None
    idle = time.monotonic() - _last_used
    return JSONResponse({
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "loaded": loaded,
        "idle_seconds": round(idle),
        "idle_timeout": IDLE_TIMEOUT if IDLE_TIMEOUT > 0 else "disabled",
    })


app = gr.mount_gradio_app(
    api,
    demo,
    path="/",
    mcp_server=True,
    theme=gr.themes.Soft(),
)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    threading.Thread(target=_idle_worker, daemon=True).start()
    _model = _load_model()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
