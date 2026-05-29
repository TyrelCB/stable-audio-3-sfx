# stable-audio-3-sfx

A minimal FastAPI service that generates sound effects using [Stable Audio 3 Small SFX](https://huggingface.co/stabilityai/stable-audio-3-small-sfx) — a 0.6B-parameter latent diffusion model from Stability AI. Generates stereo 44.1 kHz WAV files from text prompts in under a second on GPU.

## Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv)
- NVIDIA GPU with CUDA (CPU works but is ~20× slower)
- Hugging Face account with the [model license accepted](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)

## Installation

```bash
# 1. Create a virtual environment
uv venv --python 3.12 .venv

# 2. Install CUDA-enabled PyTorch (adjust cu130 to match your CUDA version)
uv pip install --torch-backend cu130 --python .venv/bin/python torch torchaudio

# 3. Install stable-audio-3 (skip its torch version pin to preserve the CUDA build)
uv pip install --python .venv/bin/python --no-deps git+https://github.com/Stability-AI/stable-audio-3.git

# 4. Install remaining dependencies
uv pip install --python .venv/bin/python \
  einops einops-exts soundfile safetensors \
  huggingface-hub transformers tqdm numpy packaging \
  fastapi uvicorn pydantic

# 5. Log in to Hugging Face (one-time)
huggingface-cli login
```

> **Note:** `stable-audio-3` pins `torch==2.7.1` but works fine with newer versions.
> `--no-deps` bypasses that constraint so the CUDA wheel is preserved.

## Running

```bash
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8765
```

The model downloads and loads into GPU memory on first startup (~3.4 GB, cached to `~/.cache/huggingface` afterward).

## API

### `GET /health`

```bash
curl http://localhost:8765/health
```

```json
{"status": "ok", "model": "small-sfx", "device": "cuda"}
```

### `POST /generate`

Returns a WAV file.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | **required** | Text description of the sound |
| `duration` | float | `5.0` | Length in seconds (max 60) |
| `steps` | int | `8` | Diffusion steps — more = higher quality, slower |
| `cfg_scale` | float | `1.0` | Classifier-free guidance scale |
| `seed` | int | `-1` | Random seed (-1 = random) |
| `negative_prompt` | string | `""` | What to avoid in the output |

```bash
curl -X POST http://localhost:8765/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "thunder clap with distant rumble", "duration": 5}' \
  --output thunder.wav
```

## Examples

```bash
# Thunder
curl -X POST http://localhost:8765/generate \
  -d '{"prompt": "thunder clap with distant rumble", "duration": 5}' \
  -H "Content-Type: application/json" --output thunder.wav

# Gunshot
curl -X POST http://localhost:8765/generate \
  -d '{"prompt": "single gunshot with echo", "duration": 3}' \
  -H "Content-Type: application/json" --output gunshot.wav

# Explosion
curl -X POST http://localhost:8765/generate \
  -d '{"prompt": "large explosion with deep rumbling shockwave", "duration": 5}' \
  -H "Content-Type: application/json" --output explosion.wav

# Campfire
curl -X POST http://localhost:8765/generate \
  -d '{"prompt": "crackling campfire with wood popping", "duration": 6}' \
  -H "Content-Type: application/json" --output campfire.wav

# Rain
curl -X POST http://localhost:8765/generate \
  -d '{"prompt": "heavy rain on a rooftop", "duration": 7}' \
  -H "Content-Type: application/json" --output rain.wav

# Reproducible result with a seed
curl -X POST http://localhost:8765/generate \
  -d '{"prompt": "dog barking loudly", "duration": 4, "seed": 42}' \
  -H "Content-Type: application/json" --output dog.wav
```

## systemd Service (auto-start on boot)

A service unit is included. Install it as a user service:

```bash
cp stable-audio-sfx.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now stable-audio-sfx.service
```

Check status and logs:

```bash
systemctl --user status stable-audio-sfx.service
journalctl --user -u stable-audio-sfx.service -f
```

### Note on unified memory systems (NVIDIA Grace Blackwell / Jetson)

On systems where CPU and GPU share a unified memory pool, other large models (e.g. a llama.cpp server) may over-commit virtual address space. The service handles this with a built-in retry loop — it will back off and retry up to 5 times on CUDA OOM before failing. This means it may take a minute or two to come online after boot while other services settle.

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `small-sfx` | Model variant to load |

## Performance

Tested on NVIDIA GB10 (Grace Blackwell, CUDA 13.0):

| | Time |
|-|------|
| Cold start (model load) | ~8s |
| Generation (3s clip) | ~0.45s |
| Generation (7s clip) | ~0.55s |

## License

Model weights: [Stability AI Community License](https://stability.ai/license). The T5Gemma text encoder is redistributed under the [Gemma Terms of Use](https://huggingface.co/stabilityai/stable-audio-3-small-sfx/blob/main/LICENSE_GEMMA.md).
