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
from fastapi.responses import HTMLResponse, StreamingResponse
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

UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stable Audio 3 SFX</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #22263a;
    --border: #2e3348;
    --accent: #7c6af7;
    --accent2: #56d8c8;
    --text: #e8eaf0;
    --muted: #7880a0;
    --danger: #f87171;
  }
  body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; min-height: 100vh; padding: 2rem 1rem; }
  h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }
  h1 span { color: var(--accent2); }
  header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 2rem; }
  #status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); display: inline-block; margin-right: 6px; transition: background 0.4s; }
  #status-dot.loaded { background: var(--accent2); box-shadow: 0 0 6px var(--accent2); }
  #status-dot.unloaded { background: var(--danger); }
  #status-label { font-size: 0.8rem; color: var(--muted); }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.25rem; }
  textarea { width: 100%; background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-size: 1rem; padding: 0.75rem 1rem; resize: vertical; min-height: 80px; outline: none; transition: border-color 0.2s; }
  textarea:focus { border-color: var(--accent); }
  textarea::placeholder { color: var(--muted); }
  .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
  @media (max-width: 500px) { .controls { grid-template-columns: 1fr; } }
  label { display: flex; flex-direction: column; gap: 0.35rem; font-size: 0.82rem; color: var(--muted); }
  .label-row { display: flex; justify-content: space-between; }
  .label-val { color: var(--text); font-weight: 600; }
  input[type=range] { width: 100%; accent-color: var(--accent); cursor: pointer; }
  input[type=number], input[type=text] { width: 100%; background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; color: var(--text); font-size: 0.9rem; padding: 0.45rem 0.7rem; outline: none; }
  input[type=number]:focus, input[type=text]:focus { border-color: var(--accent); }
  details { margin-top: 1rem; }
  summary { font-size: 0.82rem; color: var(--muted); cursor: pointer; user-select: none; }
  summary:hover { color: var(--text); }
  .advanced-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 0.75rem; }
  @media (max-width: 500px) { .advanced-grid { grid-template-columns: 1fr; } }
  button#generate {
    width: 100%; margin-top: 1.25rem; padding: 0.85rem;
    background: linear-gradient(135deg, var(--accent), #5e8aff);
    border: none; border-radius: 10px; color: #fff; font-size: 1rem;
    font-weight: 700; cursor: pointer; transition: opacity 0.2s, transform 0.1s;
    letter-spacing: 0.02em;
  }
  button#generate:hover:not(:disabled) { opacity: 0.9; }
  button#generate:active:not(:disabled) { transform: scale(0.98); }
  button#generate:disabled { opacity: 0.5; cursor: not-allowed; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.3); border-top-color: #fff; border-radius: 50%; animation: spin 0.7s linear infinite; margin-right: 8px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
  #result { margin-bottom: 1.25rem; }
  .result-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; }
  .result-prompt { font-size: 0.85rem; color: var(--muted); margin-bottom: 0.75rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .result-prompt strong { color: var(--text); }
  audio { width: 100%; margin-bottom: 0.75rem; }
  .result-meta { display: flex; justify-content: space-between; align-items: center; font-size: 0.78rem; color: var(--muted); }
  a.dl { color: var(--accent2); text-decoration: none; font-weight: 600; }
  a.dl:hover { text-decoration: underline; }
  .history-title { font-size: 0.85rem; font-weight: 600; color: var(--muted); margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
  .history-item { display: flex; align-items: center; gap: 0.75rem; padding: 0.6rem 0; border-bottom: 1px solid var(--border); }
  .history-item:last-child { border-bottom: none; }
  .history-item audio { flex: 1; height: 32px; }
  .history-prompt { flex: 2; font-size: 0.82rem; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .history-prompt strong { color: var(--text); display: block; }
  .history-dl { color: var(--accent2); font-size: 0.78rem; text-decoration: none; white-space: nowrap; }
  #error { color: var(--danger); font-size: 0.85rem; margin-top: 0.5rem; display: none; }
  .max-w { max-width: 640px; margin: 0 auto; }
</style>
</head>
<body>
<div class="max-w">
  <header>
    <h1>Stable Audio <span>3 SFX</span></h1>
    <div><span id="status-dot"></span><span id="status-label">checking…</span></div>
  </header>

  <div class="card">
    <textarea id="prompt" placeholder="Describe a sound effect… e.g. "thunder clap with distant rumble"" rows="3"></textarea>
    <div class="controls">
      <label>
        <span class="label-row">Duration <span class="label-val"><span id="dur-val">5</span>s</span></span>
        <input type="range" id="duration" min="1" max="60" value="5" oninput="document.getElementById('dur-val').textContent=this.value">
      </label>
      <label>
        <span class="label-row">Steps <span class="label-val" id="steps-val">8</span></span>
        <input type="range" id="steps" min="1" max="50" value="8" oninput="document.getElementById('steps-val').textContent=this.value">
      </label>
    </div>
    <details>
      <summary>Advanced</summary>
      <div class="advanced-grid">
        <label>CFG Scale<input type="number" id="cfg_scale" value="1.0" min="0" max="20" step="0.1"></label>
        <label>Seed (-1 = random)<input type="number" id="seed" value="-1"></label>
        <label>Negative prompt<input type="text" id="negative_prompt" placeholder="optional"></label>
      </div>
    </details>
    <button id="generate" onclick="generate()">Generate</button>
    <div id="error"></div>
  </div>

  <div id="result"></div>

  <div id="history-card" class="card" style="display:none">
    <div class="history-title">History</div>
    <div id="history"></div>
  </div>
</div>

<script>
const history = [];

async function pollStatus() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    const dot = document.getElementById('status-dot');
    const lbl = document.getElementById('status-label');
    if (d.loaded) {
      dot.className = 'loaded';
      lbl.textContent = 'model loaded · ' + d.device;
    } else {
      dot.className = 'unloaded';
      lbl.textContent = 'model unloaded · idle ' + d.idle_seconds + 's';
    }
  } catch {
    document.getElementById('status-dot').className = '';
    document.getElementById('status-label').textContent = 'offline';
  }
}

async function generate() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) { showError('Enter a prompt first.'); return; }
  hideError();

  const btn = document.getElementById('generate');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Generating…';

  const payload = {
    prompt,
    duration: parseFloat(document.getElementById('duration').value),
    steps: parseInt(document.getElementById('steps').value),
    cfg_scale: parseFloat(document.getElementById('cfg_scale').value),
    seed: parseInt(document.getElementById('seed').value),
    negative_prompt: document.getElementById('negative_prompt').value,
  };

  const t0 = Date.now();
  try {
    const resp = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({detail: resp.statusText}));
      throw new Error(err.detail || resp.statusText);
    }
    const blob = await resp.blob();
    const elapsed = ((Date.now() - t0) / 1000).toFixed(2);
    const url = URL.createObjectURL(blob);
    const filename = prompt.slice(0, 40).replace(/[^a-z0-9]+/gi, '_') + '.wav';

    showResult(prompt, url, filename, payload.duration, elapsed);
    addHistory(prompt, url, filename, payload.duration, elapsed);
    pollStatus();
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate';
  }
}

function showResult(prompt, url, filename, duration, elapsed) {
  document.getElementById('result').innerHTML = `
    <div class="result-card">
      <div class="result-prompt">Result for <strong>${esc(prompt)}</strong></div>
      <audio controls autoplay src="${url}"></audio>
      <div class="result-meta">
        <span>${duration}s · generated in ${elapsed}s</span>
        <a class="dl" href="${url}" download="${filename}">↓ download</a>
      </div>
    </div>`;
}

function addHistory(prompt, url, filename, duration, elapsed) {
  history.unshift({prompt, url, filename, duration, elapsed});
  if (history.length > 10) history.pop();
  const card = document.getElementById('history-card');
  const list = document.getElementById('history');
  if (history.length < 2) { card.style.display = 'none'; return; }
  card.style.display = '';
  list.innerHTML = history.slice(1).map(h => `
    <div class="history-item">
      <div class="history-prompt"><strong>${esc(h.prompt.slice(0, 60))}</strong>${h.duration}s · ${h.elapsed}s</div>
      <audio controls src="${h.url}"></audio>
      <a class="history-dl" href="${h.url}" download="${h.filename}">↓</a>
    </div>`).join('');
}

function showError(msg) {
  const el = document.getElementById('error');
  el.textContent = msg;
  el.style.display = '';
}
function hideError() { document.getElementById('error').style.display = 'none'; }
function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

document.getElementById('prompt').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) generate();
});

pollStatus();
setInterval(pollStatus, 10000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    return UI_HTML


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
