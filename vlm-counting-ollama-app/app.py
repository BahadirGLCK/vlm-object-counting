from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_PATH = REPO_ROOT / "data" / "inputs" / "sft_prompt.txt"
DEFAULT_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:18080")
DEFAULT_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3-vl-object-counting-4bit",
)
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY", "")
DEFAULT_BASIC_USERNAME = os.getenv("OLLAMA_BASIC_USERNAME", "")
DEFAULT_BASIC_PASSWORD = os.getenv("OLLAMA_BASIC_PASSWORD", "")
DEFAULT_AUTH_MODE = os.getenv("OLLAMA_AUTH_MODE", "auto").strip().lower() or "auto"
DEFAULT_API_STYLE = os.getenv("MODEL_API_STYLE", "openai").strip().lower() or "openai"
DEFAULT_RESPONSE_JSON_ONLY = os.getenv("RESPONSE_JSON_ONLY", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}

app = Flask(__name__, template_folder="templates", static_folder="static")


def load_default_prompt() -> str:
    if DEFAULT_PROMPT_PATH.is_file():
        return DEFAULT_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return ""


def normalize_base_endpoint(raw_endpoint: str) -> str:
    endpoint = (raw_endpoint or "").strip()
    if not endpoint:
        raise ValueError("Endpoint URL is required.")
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"http://{endpoint}"
    return endpoint.rstrip("/")


def build_mode_url(base_endpoint: str, mode: str) -> str:
    if mode == "openai":
        if base_endpoint.endswith("/v1/chat/completions"):
            return base_endpoint
        if base_endpoint.endswith("/v1"):
            return f"{base_endpoint}/chat/completions"
        return f"{base_endpoint}/v1/chat/completions"

    if base_endpoint.endswith("/api/chat"):
        return base_endpoint
    if base_endpoint.endswith("/api"):
        return f"{base_endpoint}/chat"
    return f"{base_endpoint}/api/chat"


def normalize_endpoint_url(raw_endpoint: str, api_style: str = "auto") -> tuple[str, str]:
    base_endpoint = normalize_base_endpoint(raw_endpoint)
    style = (api_style or "auto").strip().lower()
    if style not in {"auto", "native", "openai"}:
        style = "auto"

    # Explicit endpoint suffix wins.
    if base_endpoint.endswith("/api/chat"):
        return "native", base_endpoint
    if base_endpoint.endswith("/v1/chat/completions"):
        return "openai", base_endpoint
    if base_endpoint.endswith("/v1/models"):
        return "openai", base_endpoint.rsplit("/v1/models", 1)[0] + "/v1/chat/completions"
    if base_endpoint.endswith("/health"):
        return "openai", base_endpoint.rsplit("/health", 1)[0] + "/v1/chat/completions"
    if base_endpoint.endswith("/v1"):
        return "openai", f"{base_endpoint}/chat/completions"
    if base_endpoint.endswith("/api"):
        return "native", f"{base_endpoint}/chat"

    if style == "native":
        return "native", build_mode_url(base_endpoint, "native")
    if style == "openai":
        return "openai", build_mode_url(base_endpoint, "openai")

    # Auto: prefer OpenAI-compatible for common vLLM ports.
    parsed = urlparse(base_endpoint)
    parsed_port = str(parsed.port) if parsed.port else ""
    if parsed_port in {"8000", "8080"}:
        return "openai", build_mode_url(base_endpoint, "openai")
    return "native", build_mode_url(base_endpoint, "native")


def sample_frame_indices(total_frames: int, requested_frames: int) -> np.ndarray:
    """
    Mirrors training preprocessing in src/utils.py -> sample_frames_from_video(..., strategy="uniform"):
    - clamp k to available frame_count
    - step = frame_count / k
    - indices = [int(i * step) for i in range(k)]
    """
    if total_frames <= 0:
        raise ValueError("Video has no readable frames.")

    k = int(requested_frames)
    if k <= 0:
        raise ValueError(f"Number of frames to sample must be positive, got {k}")
    if k > total_frames:
        k = total_frames

    step = total_frames / k
    indices = [int(i * step) for i in range(k)]
    return np.array(indices, dtype=int)


def resize_if_needed(frame_bgr: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame_bgr
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_video_frames_base64(
    video_path: Path,
    frame_limit: int,
    max_side: int,
    jpeg_quality: int,
) -> list[str]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_frame_indices(total_frames=total_frames, requested_frames=frame_limit)
        encoded_frames: list[str] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue

            frame_bgr = resize_if_needed(frame_bgr, max_side=max_side)
            ok, encoded = cv2.imencode(
                ".jpg",
                frame_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            if not ok:
                continue

            encoded_frames.append(base64.b64encode(encoded.tobytes()).decode("ascii"))

        if not encoded_frames:
            raise ValueError("No frames could be extracted from the uploaded video.")

        return encoded_frames
    finally:
        cap.release()


def parse_openai_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_value = item.get("text")
                if isinstance(text_value, str):
                    texts.append(text_value)
        return "\n".join(texts).strip()
    return ""


def extract_model_response(response_json: dict[str, Any]) -> str:
    message = response_json.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        parsed = parse_openai_content(content)
        if parsed:
            return parsed.strip()

    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            choice_message = first_choice.get("message")
            if isinstance(choice_message, dict):
                parsed = parse_openai_content(choice_message.get("content"))
                if parsed:
                    return parsed.strip()

    direct_response = response_json.get("response")
    if isinstance(direct_response, str) and direct_response.strip():
        return direct_response.strip()

    raw_content = response_json.get("content")
    if isinstance(raw_content, str) and raw_content.strip():
        return raw_content.strip()

    return json.dumps(response_json, indent=2, ensure_ascii=False)


def build_native_payload(
    model_name: str,
    system_prompt: str,
    frame_images_b64: list[str],
    temperature: float,
) -> dict[str, Any]:
    user_instruction = (
        "These images are sampled frames from one video in temporal order. "
        "Follow the system prompt exactly and return only your final response."
    )
    return {
        "model": model_name,
        "stream": False,
        "options": {
            "temperature": float(temperature),
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_instruction,
                "images": frame_images_b64,
            },
        ],
    }


def build_openai_payload(
    model_name: str,
    system_prompt: str,
    frame_images_b64: list[str],
    temperature: float,
    response_json_only: bool,
) -> dict[str, Any]:
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "These images are sampled frames from one video in temporal order. "
                "Follow the system prompt exactly and return only your final response."
            ),
        }
    ]
    for frame_b64 in frame_images_b64:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}",
                },
            }
        )

    payload: dict[str, Any] = {
        "model": model_name,
        "temperature": float(temperature),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if response_json_only:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _build_basic_auth_value(username: str, password: str) -> str:
    user = username or ""
    pwd = password or ""
    token = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def build_auth_headers(
    api_key: str,
    auth_mode: str,
    basic_username: str,
    basic_password: str,
) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = (api_key or "").strip()
    mode = (auth_mode or "auto").strip().lower()
    basic_user = (basic_username or "").strip()
    basic_pass = basic_password or ""
    has_basic_creds = bool(basic_user or basic_pass)

    if mode == "none":
        return headers

    if mode == "basic":
        headers["Authorization"] = _build_basic_auth_value(basic_user, basic_pass)
        return headers

    if mode == "bearer":
        if not key:
            return headers
        headers["Authorization"] = f"Bearer {key}"
        return headers

    if mode == "x-api-key":
        if not key:
            return headers
        headers["X-API-Key"] = key
        return headers

    # Auto mode:
    # - If Basic credentials exist, prefer Basic auth.
    # - Else try common API key header variants.
    if has_basic_creds:
        headers["Authorization"] = _build_basic_auth_value(basic_user, basic_pass)
        return headers

    if not key:
        return headers

    headers["Authorization"] = f"Bearer {key}"
    headers["X-API-Key"] = key
    headers["api-key"] = key
    return headers


def call_ollama(
    endpoint_url: str,
    api_style: str,
    api_key: str,
    auth_mode: str,
    basic_username: str,
    basic_password: str,
    model_name: str,
    prompt_text: str,
    frame_images_b64: list[str],
    temperature: float,
    response_json_only: bool,
    max_output_tokens: int,
    timeout_seconds: int,
    headers: dict[str, str] | None = None,
) -> str:
    mode, normalized_url = normalize_endpoint_url(endpoint_url, api_style=api_style)
    if headers is None:
        headers = build_auth_headers(
            api_key=api_key,
            auth_mode=auth_mode,
            basic_username=basic_username,
            basic_password=basic_password,
        )

    if mode == "openai":
        payload = build_openai_payload(
            model_name=model_name,
            system_prompt=prompt_text,
            frame_images_b64=frame_images_b64,
            temperature=temperature,
            response_json_only=response_json_only,
        )
        payload["max_tokens"] = int(max_output_tokens)
    else:
        payload = build_native_payload(
            model_name=model_name,
            system_prompt=prompt_text,
            frame_images_b64=frame_images_b64,
            temperature=temperature,
        )
        payload.setdefault("options", {})["num_predict"] = int(max_output_tokens)
        payload["keep_alive"] = "30m"

    response = post_with_retry(
        normalized_url=normalized_url,
        headers=headers,
        payload=payload,
        timeout_seconds=timeout_seconds,
    )
    response.raise_for_status()

    try:
        body = response.json()
    except ValueError:
        text = response.text.strip()
        if text:
            return text
        raise ValueError("Endpoint returned an empty non-JSON response.")

    if not isinstance(body, dict):
        return json.dumps(body, indent=2, ensure_ascii=False)

    return extract_model_response(body)


def is_retryable_connection_error(exc: requests.ConnectionError) -> bool:
    lowered = str(exc).lower()
    retry_markers = (
        "remotedisconnected",
        "connection reset by peer",
        "broken pipe",
        "badstatusline",
        "protocolerror",
        "unexpected eof",
    )
    return any(marker in lowered for marker in retry_markers)


def post_with_retry(
    normalized_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: int,
) -> requests.Response:
    """
    Retry once on abrupt disconnects (common with stale keep-alive sockets or flaky tunnels).
    """
    request_headers = dict(headers)
    for attempt in range(2):
        try:
            return requests.post(
                normalized_url,
                headers=request_headers,
                json=payload,
                timeout=(15, int(timeout_seconds)),
            )
        except requests.ConnectionError as exc:
            if attempt == 0 and is_retryable_connection_error(exc):
                # Force a fresh connection for one retry attempt.
                request_headers = dict(headers)
                request_headers["Connection"] = "close"
                continue
            raise

    raise RuntimeError("Unexpected retry loop exit")


def clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(parsed, max_value))


def clamp_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(parsed, max_value))


def model_name_matches(requested: str, available: str) -> bool:
    req = (requested or "").strip()
    avail = (available or "").strip()
    if not req or not avail:
        return False
    if req == avail:
        return True
    return req.split(":")[0] == avail.split(":")[0]


def build_probe_url(mode: str, normalized_url: str) -> str:
    if mode == "native":
        return normalized_url.rsplit("/api/chat", 1)[0] + "/api/tags"
    return normalized_url.rsplit("/v1/chat/completions", 1)[0] + "/v1/models"


def build_health_url(mode: str, normalized_url: str) -> str:
    if mode == "openai":
        return normalized_url.rsplit("/v1/chat/completions", 1)[0] + "/health"
    return normalized_url.rsplit("/api/chat", 1)[0] + "/"


def probe_endpoint(
    endpoint_url: str,
    api_style: str,
    headers: dict[str, str],
    model_name: str,
    timeout_seconds: int = 8,
) -> None:
    mode, normalized_url = normalize_endpoint_url(endpoint_url, api_style=api_style)
    health_url = build_health_url(mode, normalized_url)
    health_resp = requests.get(
        health_url,
        headers=headers,
        timeout=(5, int(timeout_seconds)),
    )
    health_resp.raise_for_status()

    probe_url = build_probe_url(mode, normalized_url)
    response = requests.get(
        probe_url,
        headers=headers,
        timeout=(5, int(timeout_seconds)),
    )
    response.raise_for_status()

    try:
        body = response.json()
    except ValueError:
        # Endpoint is reachable and authorized; keep going.
        return

    available_models: list[str] = []
    if mode == "native":
        for item in body.get("models", []) if isinstance(body, dict) else []:
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str):
                    available_models.append(name)
    else:
        for item in body.get("data", []) if isinstance(body, dict) else []:
            if isinstance(item, dict):
                name = item.get("id")
                if isinstance(name, str):
                    available_models.append(name)

    if available_models and not any(model_name_matches(model_name, m) for m in available_models):
        preview = ", ".join(available_models[:8])
        raise ValueError(
            f"Model '{model_name}' is not visible on endpoint. Available models: {preview}"
        )


def summarize_http_error(
    exc: requests.HTTPError,
    endpoint_url: str,
    auth_mode: str,
    has_api_key: bool,
    has_basic_credentials: bool,
) -> str:
    response = exc.response
    if response is None:
        return str(exc)

    status_code = response.status_code
    body = (response.text or "").strip()
    www_authenticate = response.headers.get("WWW-Authenticate", "").strip()

    parts = [f"HTTP {status_code}"]

    if status_code == 401:
        is_basic_auth = "basic" in www_authenticate.lower()
        if is_basic_auth:
            parts.append(
                "Unauthorized: endpoint is protected by HTTP Basic Auth (likely Caddy). "
                "SSH key authentication does not apply to HTTP requests."
            )
            if has_basic_credentials:
                parts.append("Provided Basic credentials were rejected.")
            else:
                parts.append("No Basic credentials were provided.")
            parts.append(
                "Use one of: 1) set Auth mode to Basic and enter gateway username/password, "
                "2) SSH tunnel to remote Ollama and endpoint http://127.0.0.1:11434, "
                "3) remove gateway Basic Auth."
            )
        elif has_api_key:
            parts.append("Unauthorized: provided API key/auth headers were rejected.")
        else:
            parts.append("Unauthorized: this endpoint likely requires an API key.")
        if not is_basic_auth:
            parts.append(
                "Try auth mode 'Auto (Bearer + X-API-Key)' or the one your gateway expects."
            )
        if www_authenticate:
            parts.append(f"WWW-Authenticate: {www_authenticate}")

    parts.append(f"Request URL: {endpoint_url}")
    parts.append(f"Auth mode: {auth_mode}")

    if body:
        parts.append(f"Response body: {body}")

    return " ".join(parts).strip()


def summarize_read_timeout(endpoint_url: str, timeout_seconds: int) -> str:
    message = (
        f"Read timeout after {timeout_seconds}s while waiting for endpoint response. "
        f"Request URL: {endpoint_url}. "
    )
    local_ports = ("http://127.0.0.1:11434", "http://localhost:11434", "http://127.0.0.1:8080")
    if endpoint_url.startswith(local_ports):
        message += (
            "If you use SSH tunneling, your local forwarded port may be occupied by another process "
            "(for example IDE port forwarding). Check with `lsof -nP -iTCP:<port> -sTCP:LISTEN` "
            "and use a clean local port if needed. "
        )
    message += (
        "Also try increasing timeout, reducing sampled frames or frame size, and set max output tokens lower."
    )
    return message


def summarize_connection_error(endpoint_url: str, exc: Exception) -> str:
    raw = str(exc)
    message = f"Connection failed while contacting endpoint. Request URL: {endpoint_url}. Error: {raw}. "
    lowered = raw.lower()
    if "remotedisconnected" in lowered or "empty reply from server" in lowered:
        message += (
            "Remote closed the connection without response. This is commonly a stale/wrong SSH forward "
            "or local port collision. "
        )
        message += (
            "If health/model checks pass but this fails only on inference, backend may be hitting memory/body "
            "limits; try fewer frames (8-16), lower max frame side (640-768), and lower max output tokens. "
        )

    if endpoint_url.startswith("http://127.0.0.1:8080") or endpoint_url.startswith("http://localhost:8080"):
        message += (
            "For vLLM tunnel, use a clean local port and bypass SSH config forwards: "
            "`ssh -F /dev/null -N -p 46219 root@207.180.148.74 -L 18080:localhost:8000`, "
            "then set endpoint to `http://127.0.0.1:18080`. "
            "Check listeners with `lsof -nP -iTCP:8080 -sTCP:LISTEN` and "
            "`lsof -nP -iTCP:18080 -sTCP:LISTEN`."
        )
        return message

    if endpoint_url.startswith("http://127.0.0.1:11434") or endpoint_url.startswith("http://localhost:11434"):
        message += (
            "For Ollama tunnel, local 11434 may be occupied. Use "
            "`ssh -F /dev/null -N -p 9211 root@128.24.60.121 -L 18134:127.0.0.1:11434` "
            "and endpoint `http://127.0.0.1:18134`."
        )
        return message

    message += "Check endpoint health manually and confirm tunnel/process is running."
    return message


@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        default_prompt=load_default_prompt(),
        default_endpoint=DEFAULT_ENDPOINT,
        default_model=DEFAULT_MODEL,
        default_api_key=DEFAULT_API_KEY,
        default_basic_username=DEFAULT_BASIC_USERNAME,
        default_basic_password=DEFAULT_BASIC_PASSWORD,
        default_auth_mode=DEFAULT_AUTH_MODE,
        default_api_style=DEFAULT_API_STYLE,
        default_response_json_only=DEFAULT_RESPONSE_JSON_ONLY,
    )


@app.post("/api/infer")
def api_infer():
    video_file = request.files.get("video")
    if video_file is None or not video_file.filename:
        return jsonify({"error": "Please upload a video file."}), 400

    prompt_text = (request.form.get("prompt_text") or "").strip()
    if not prompt_text:
        prompt_file = request.files.get("prompt_file")
        if prompt_file is not None and prompt_file.filename:
            prompt_text = prompt_file.read().decode("utf-8", errors="ignore").strip()
    if not prompt_text:
        prompt_text = load_default_prompt().strip()
    if not prompt_text:
        return jsonify({"error": "Prompt is empty. Upload a prompt file or paste prompt text."}), 400

    endpoint_url = (request.form.get("endpoint_url") or "").strip()
    model_name = (request.form.get("model_name") or "").strip()
    api_key = (request.form.get("api_key") or "").strip()
    basic_username = (request.form.get("basic_username") or "").strip()
    basic_password = request.form.get("basic_password") or ""
    auth_mode = (request.form.get("auth_mode") or DEFAULT_AUTH_MODE).strip().lower()
    api_style = (request.form.get("api_style") or DEFAULT_API_STYLE).strip().lower()
    default_response_json_value = "1" if DEFAULT_RESPONSE_JSON_ONLY else "0"
    response_json_only = (
        request.form.get("response_json_only", default_response_json_value).strip().lower()
        in {"1", "true", "on", "yes"}
    )
    if auth_mode not in {"auto", "bearer", "x-api-key", "basic", "none"}:
        auth_mode = "auto"
    if api_style not in {"auto", "native", "openai"}:
        api_style = "auto"

    if not endpoint_url:
        return jsonify({"error": "Endpoint URL is required."}), 400
    if not model_name:
        return jsonify({"error": "Model name is required."}), 400

    frame_limit = clamp_int(request.form.get("frame_limit"), default=25, min_value=4, max_value=40)
    max_side = clamp_int(request.form.get("max_side"), default=1024, min_value=480, max_value=1920)
    jpeg_quality = clamp_int(request.form.get("jpeg_quality"), default=85, min_value=55, max_value=100)
    temperature = clamp_float(request.form.get("temperature"), default=0.2, min_value=0.0, max_value=1.0)
    max_output_tokens = clamp_int(
        request.form.get("max_output_tokens"), default=256, min_value=32, max_value=2048
    )
    timeout_seconds = clamp_int(
        request.form.get("timeout_seconds"), default=600, min_value=30, max_value=1800
    )

    suffix = Path(video_file.filename).suffix or ".mp4"
    temp_path: Path | None = None

    try:
        headers = build_auth_headers(
            api_key=api_key,
            auth_mode=auth_mode,
            basic_username=basic_username,
            basic_password=basic_password,
        )
        # Fail fast before expensive video frame processing if endpoint/auth/model is wrong.
        probe_endpoint(
            endpoint_url=endpoint_url,
            api_style=api_style,
            headers=headers,
            model_name=model_name,
            timeout_seconds=min(timeout_seconds, 12),
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            video_file.save(temp_file.name)
            temp_path = Path(temp_file.name)

        frame_images_b64 = extract_video_frames_base64(
            video_path=temp_path,
            frame_limit=frame_limit,
            max_side=max_side,
            jpeg_quality=jpeg_quality,
        )

        model_response = call_ollama(
            endpoint_url=endpoint_url,
            api_style=api_style,
            api_key=api_key,
            auth_mode=auth_mode,
            basic_username=basic_username,
            basic_password=basic_password,
            model_name=model_name,
            prompt_text=prompt_text,
            frame_images_b64=frame_images_b64,
            temperature=temperature,
            response_json_only=response_json_only,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            headers=headers,
        )

        return jsonify({"response": model_response.strip()})
    except requests.HTTPError as exc:
        try:
            _, request_url = normalize_endpoint_url(endpoint_url, api_style=api_style)
        except Exception:
            request_url = endpoint_url
        details = summarize_http_error(
            exc=exc,
            endpoint_url=request_url,
            auth_mode=auth_mode,
            has_api_key=bool(api_key),
            has_basic_credentials=bool(basic_username or basic_password),
        )
        return jsonify({"error": details}), 502
    except requests.ReadTimeout:
        try:
            _, request_url = normalize_endpoint_url(endpoint_url, api_style=api_style)
        except Exception:
            request_url = endpoint_url
        return jsonify({"error": summarize_read_timeout(request_url, timeout_seconds)}), 504
    except requests.ConnectionError as exc:
        try:
            _, request_url = normalize_endpoint_url(endpoint_url, api_style=api_style)
        except Exception:
            request_url = endpoint_url
        return jsonify({"error": summarize_connection_error(request_url, exc)}), 502
    except requests.RequestException as exc:
        try:
            _, request_url = normalize_endpoint_url(endpoint_url, api_style=api_style)
        except Exception:
            request_url = endpoint_url
        return jsonify({"error": f"Request failed for {request_url}: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
        debug=os.getenv("DEBUG", "0") == "1",
    )
