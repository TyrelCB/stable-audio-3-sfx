# stable-audio-3-sfx

A Gradio service that generates sound effects using [Stable Audio 3 Small SFX](https://huggingface.co/stabilityai/stable-audio-3-small-sfx) — a 0.6B-parameter latent diffusion model from Stability AI. Includes a web UI, a REST health endpoint, and an MCP server so AI agents can generate sound effects as a tool call.

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
  gradio

# 5. Log in to Hugging Face (one-time)
huggingface-cli login
```

> **Note:** `stable-audio-3` pins `torch==2.7.1` but works fine with newer versions.
> `--no-deps` bypasses that constraint so the CUDA wheel is preserved.

## Running

```bash
.venv/bin/python main.py
```

The model downloads and loads into GPU memory on first startup (~3.4 GB, cached to `~/.cache/huggingface` afterward). Open **http://localhost:8765** for the UI.

## Web UI

The Gradio interface at `http://localhost:8765` provides:
- Prompt text input
- Duration and Steps sliders
- Advanced controls (CFG scale, seed, negative prompt)
- Inline audio player with download

## MCP Server

The service exposes a `generate_sfx` MCP tool at:

```
http://localhost:8765/gradio_api/mcp/sse
```

### Connecting from Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "stable-audio-sfx": {
      "url": "http://localhost:8765/gradio_api/mcp/sse"
    }
  }
}
```

### Connecting from any MCP client

The tool schema is available at:
```bash
curl http://localhost:8765/gradio_api/mcp/schema | python3 -m json.tool
```

The `generate_sfx` tool accepts: `prompt`, `duration`, `steps`, `cfg_scale`, `seed`, `negative_prompt`.

## Health endpoint

```bash
curl http://localhost:8765/health
```

```json
{"status": "ok", "model": "small-sfx", "device": "cuda", "loaded": true, "idle_seconds": 12, "idle_timeout": 300}
```

## systemd Service (auto-start on boot)

A service unit is included. Install it as a user service:

```bash
cp stable-audio-sfx-mcp.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now stable-audio-sfx-mcp.service
```

Check status and logs:

```bash
systemctl --user status stable-audio-sfx-mcp.service
journalctl --user -u stable-audio-sfx-mcp.service -f
```

### Note on unified memory systems (NVIDIA Grace Blackwell / Jetson)

On systems where CPU and GPU share a unified memory pool, other large models (e.g. a llama.cpp server) may over-commit virtual address space. The service handles this with a built-in retry loop — it will back off and retry up to 5 times on CUDA OOM before failing. This means it may take a minute or two to come online after boot while other services settle.

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `small-sfx` | Model variant to load |
| `PORT` | `8765` | Server port |
| `IDLE_TIMEOUT` | `300` | Seconds of inactivity before unloading the model from GPU. Set to `0` to disable. |

### Idle sleep

When `IDLE_TIMEOUT` is set (default 5 minutes), a background thread monitors inactivity and unloads the model weights from GPU memory after the timeout, freeing VRAM for other processes. The model reloads automatically on the next request (~8s cold start).

To override at runtime:

```bash
IDLE_TIMEOUT=600 .venv/bin/python main.py  # 10 min
IDLE_TIMEOUT=0   .venv/bin/python main.py  # never unload
```

Or in the systemd unit:

```ini
Environment=IDLE_TIMEOUT=600
```

## Performance

Tested on NVIDIA GB10 (Grace Blackwell, CUDA 13.0):

| | Time |
|-|------|
| Cold start (model load) | ~8s |
| Generation (3s clip) | ~0.45s |
| Generation (7s clip) | ~0.55s |

## License

Model weights: [Stability AI Community License](https://stability.ai/license). The T5Gemma text encoder is redistributed under the [Gemma Terms of Use](https://huggingface.co/stabilityai/stable-audio-3-small-sfx/blob/main/LICENSE_GEMMA.md).
