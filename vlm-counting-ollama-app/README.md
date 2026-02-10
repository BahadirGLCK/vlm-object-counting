# vlm-counting-ollama-app

Flask web app for running your served VLM endpoint (vLLM OpenAI-compatible or Ollama native) with:
- video upload
- SFT prompt upload/edit
- frame preprocessing (uniform sampling aligned with finetuning + resize + JPEG encode)
- model response display

No Gradio or Hugging Face dependencies are used.

## Run

```bash
cd vlm-counting-ollama-app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open: `http://localhost:7860`

## Endpoint Input

You can provide:
- base host for vLLM: `http://127.0.0.1:18080` (app uses `/v1/chat/completions`)
- base host for Ollama: `https://your-host` (app uses `/api/chat`)
- native endpoint: `https://your-host/api/chat`
- OpenAI-compatible endpoint: `https://your-host/v1/chat/completions`
- SSH tunnel/local Ollama: `http://127.0.0.1:11434` (recommended when server is SSH-accessible)

Auth options in UI:
- `Auto (Bearer + X-API-Key)` (recommended)
- `Bearer`
- `X-API-Key`
- `Basic (username/password)`
- `None`

API Style options in UI:
- `OpenAI-Compatible (vLLM)` (recommended for vLLM)
- `Ollama Native`
- `Auto Detect`

## Optional Environment Variables

```bash
export OLLAMA_ENDPOINT="http://127.0.0.1:18080"
export OLLAMA_MODEL="qwen3-vl-object-counting-4bit"
export OLLAMA_API_KEY="..."
export OLLAMA_BASIC_USERNAME="..."
export OLLAMA_BASIC_PASSWORD="..."
export OLLAMA_AUTH_MODE="auto"   # auto | bearer | x-api-key | basic | none
export MODEL_API_STYLE="openai"  # openai | native | auto
export RESPONSE_JSON_ONLY="1"    # 1 => response_format json_object for OpenAI APIs
export HOST="0.0.0.0"
export PORT="7860"
```

## Prompt Behavior

- If you upload a `.txt` prompt file, its content fills the prompt editor.
- If prompt editor is empty, app falls back to `../data/inputs/sft_prompt.txt` if available.

## 401 Troubleshooting

If you get `HTTP 401`:
- enter your API key in the `API Key` field
- set `Auth Mode` to `Auto (Bearer + X-API-Key)` first
- if still unauthorized, switch to `Bearer` or `X-API-Key` based on your gateway config
- check whether your server protects `/api/chat` or expects a different endpoint path

If response includes `WWW-Authenticate: Basic ...`, your URL is behind HTTP Basic Auth (for example Caddy). SSH keys do not satisfy HTTP Basic Auth.

In that case, use:
- `Auth Mode = Basic (username/password)`
- `Basic Auth Username` and `Basic Auth Password` fields with your gateway credentials

### Use SSH Tunnel Instead Of Gateway Auth

```bash
ssh -N -L 11434:127.0.0.1:11434 <ssh_user>@207.81.166.196
```

Then set endpoint in the app to:

```text
http://127.0.0.1:11434
```

### Your Current Host (`128.24.60.121:9211`)

For this host, Ollama is listening on server port `11434` (not `8080`).

Use:

```bash
ssh -F /dev/null -p 9211 -N -L 11434:127.0.0.1:11434 root@128.24.60.121
```

If local `11434` is busy:

```bash
ssh -F /dev/null -p 9211 -N -L 18134:127.0.0.1:11434 root@128.24.60.121
```

Then set app endpoint to `http://127.0.0.1:11434` (or `http://127.0.0.1:18134` for the second command).
For your current setup, `18134` is the safer default because local `11434` may be occupied by IDE forwarding.

If `11434` on your local machine is occupied (for example by VS Code forwarding), check:

```bash
lsof -nP -iTCP:11434 -sTCP:LISTEN
```

Then use the `18134` tunnel variant and endpoint `http://127.0.0.1:18134`.

If your request reaches Ollama but returns:
- `HTTP 500 ... model runner has unexpectedly stopped`

that is a server-side model runner crash (not app connectivity). Check server logs:

```bash
tail -n 200 /var/log/portal/ollama.log
```

### vLLM Host (`207.180.148.74:46219`)

Tunnel local `18080` to remote `8000` (recommended to avoid local `8080` collisions):

```bash
ssh -F /dev/null -N -p 46219 root@207.180.148.74 -L 18080:localhost:8000
```

Then use in app:
- `Endpoint URL`: `http://127.0.0.1:18080`
- `API Style`: `OpenAI-Compatible (vLLM)`
- `Model`: `qwen3-vl-object-counting-4bit`
- `Response Format`: `JSON Object`

Quick checks:

```bash
curl -i http://127.0.0.1:18080/health
curl -s http://127.0.0.1:18080/v1/models
```

If you see `RemoteDisconnected('Remote end closed connection without response')`, verify tunnel and listeners:

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
lsof -nP -iTCP:18080 -sTCP:LISTEN
```

If `/health` and `/v1/models` pass but inference still disconnects, reduce request size:
- sampled frames: `8-16`
- max frame side: `640-768`
- max output tokens: `128-256`

## Frame Sampling Alignment

- Uniform frame selection now follows the same rule as finetuning (`src/utils.py`):
  - `step = frame_count / k`
  - frame indices: `int(i * step)` for `i in range(k)`
- Default `k` is `25` to match `NUM_FRAMES_PER_VIDEO = 25`.

## Timeout and Generation Controls

- `Timeout (seconds)` default is `600` (increase for first-run model warmup).
- `Max Output Tokens` default is `256` to prevent overlong generations.
- App performs a quick endpoint/model preflight before video processing to fail fast on tunnel/auth/model mismatches.
