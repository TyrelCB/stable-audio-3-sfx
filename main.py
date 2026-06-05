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
  IDLE_TIMEOUT  seconds idle before the model worker is killed, freeing both
                GPU memory and host RAM  (default: 300, 0 = never)
  PORT          server port  (default: 8766)

The model runs in a separate subprocess (worker.py). On idle it is killed
outright so the OS reclaims the full torch/CUDA footprint — VRAM *and* the
~2.5 GB of host RAM the stack pins. It is respawned automatically on the next
request. The parent web server never imports torch, so it idles at a small,
constant footprint.

MCP endpoint:
  http://localhost:8766/gradio_api/mcp
"""

import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Optional

import numpy as np

import worker

MODEL_NAME = os.getenv("MODEL_NAME", "small-sfx")
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "300"))
PORT = int(os.getenv("PORT", "8766"))

# ---------------------------------------------------------------------------
# Model worker lifecycle (the model lives in a child process, not here)
# ---------------------------------------------------------------------------

# "spawn" gives the child a clean interpreter — it imports this module as
# __mp_main__, so nothing under `if __name__ == "__main__"` runs in the child.
_ctx = mp.get_context("spawn")

_proc: Optional[mp.context.SpawnProcess] = None
_req_q: Optional[mp.Queue] = None
_resp_q: Optional[mp.Queue] = None
_device: Optional[str] = None
_lock = threading.Lock()
_last_used = time.monotonic()


def _ensure_worker() -> None:
    """Spawn the model worker if it isn't running. Caller must hold _lock."""
    global _proc, _req_q, _resp_q, _device
    if _proc is not None and _proc.is_alive():
        return
    _req_q = _ctx.Queue()
    _resp_q = _ctx.Queue()
    _proc = _ctx.Process(
        target=worker.run_worker,
        args=(_req_q, _resp_q, MODEL_NAME),
        daemon=True,
    )
    _proc.start()
    print("Model worker starting — loading model...")
    status, payload = _resp_q.get()   # blocks until the model loads or fails
    if status == "ready":
        _device = payload
        print(f"Model worker ready on {_device}.")
    else:
        _stop_worker()
        raise RuntimeError(f"Model worker failed to load: {payload}")


def _stop_worker() -> None:
    """Kill the worker, reclaiming its GPU + host memory. Caller must hold _lock."""
    global _proc, _req_q, _resp_q
    if _proc is not None:
        _proc.terminate()
        _proc.join(timeout=10)
        if _proc.is_alive():
            _proc.kill()
            _proc.join()
    _proc = None
    _req_q = None
    _resp_q = None


def _idle_worker() -> None:
    while True:
        time.sleep(30)
        with _lock:
            if _proc is None or IDLE_TIMEOUT <= 0:
                continue
            idle = time.monotonic() - _last_used
            if idle >= IDLE_TIMEOUT:
                _stop_worker()
                print(f"Worker killed after {idle:.0f}s idle — GPU + RAM freed.")


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
    global _last_used

    with _lock:
        _ensure_worker()
        _last_used = time.monotonic()
        _req_q.put({
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "duration": duration,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
        })

        # Wait for the result, but don't hang forever if the worker dies.
        status, payload = "error", "Worker process died during generation."
        while True:
            try:
                status, payload = _resp_q.get(timeout=5)
                break
            except queue.Empty:
                if _proc is None or not _proc.is_alive():
                    _stop_worker()
                    break
        _last_used = time.monotonic()

    if status != "ok":
        raise RuntimeError(payload)
    return payload


# ---------------------------------------------------------------------------
# Entrypoint — build the Gradio UI + FastAPI app and serve.
# Kept under __main__ so the spawned worker never imports gradio/uvicorn.
# ---------------------------------------------------------------------------

def main() -> None:
    import gradio as gr
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

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

    api = FastAPI()

    @api.get("/health")
    def health():
        loaded = _proc is not None and _proc.is_alive()
        idle = time.monotonic() - _last_used
        return JSONResponse({
            "status": "ok",
            "model": MODEL_NAME,
            "device": _device or "unknown",
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

    threading.Thread(target=_idle_worker, daemon=True).start()
    with _lock:
        _ensure_worker()   # pre-warm so the first request is fast
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
