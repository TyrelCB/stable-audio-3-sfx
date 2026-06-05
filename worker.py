"""Model worker subprocess for Stable Audio 3 SFX.

Runs in its own process so the entire torch/CUDA stack — and all the host RAM
it pins — is reclaimed by the OS when the process is killed on idle. Imports of
torch / stable_audio_3 are kept *inside* the functions so the parent web server
can `import worker` without dragging in the heavy ML stack (and its ~2.5 GB of
resident memory).
"""

import time


def _load_model(model_name: str, device: str):
    import torch
    from stable_audio_3 import StableAudioModel

    for attempt in range(5):
        try:
            torch.cuda.empty_cache()
            return StableAudioModel.from_pretrained(model_name, device=device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() and device == "cuda" and attempt < 4:
                wait = 10 * (attempt + 1)
                print(f"CUDA OOM on attempt {attempt + 1}, retrying in {wait}s...")
                torch.cuda.empty_cache()
                time.sleep(wait)
            else:
                raise


def run_worker(req_q, resp_q, model_name: str) -> None:
    """Load the model, then serve generation requests until told to stop.

    Protocol over the queues:
      - on startup: puts ("ready", device) or ("error", message) on resp_q
      - per request: gets a kwargs dict (or None to exit) from req_q,
        replies with ("ok", (sample_rate, audio)) or ("error", message)
    """
    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = _load_model(model_name, device)
    except Exception as e:  # surface load failures to the parent
        resp_q.put(("error", f"{type(e).__name__}: {e}"))
        return
    resp_q.put(("ready", device))

    while True:
        req = req_q.get()
        if req is None:
            return
        try:
            audio = model.generate(
                prompt=req["prompt"],
                negative_prompt=req["negative_prompt"] or None,
                duration=req["duration"],
                steps=req["steps"],
                cfg_scale=req["cfg_scale"],
                seed=req["seed"],
            )

            # generate() → (batch, channels, samples) float32
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().float().numpy()
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim == 3:
                audio = audio[0]        # (channels, samples)
            if audio.ndim == 2:
                audio = audio.T         # (samples, channels) for Gradio/soundfile

            inner = getattr(model, "model", None)
            sample_rate = getattr(inner, "sample_rate", 44100)
            resp_q.put(("ok", (sample_rate, audio)))
        except Exception as e:
            resp_q.put(("error", f"{type(e).__name__}: {e}"))
