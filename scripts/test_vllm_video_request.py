#!/usr/bin/env python3
import argparse
import base64
import json
import pathlib
import random
import re
import sys
import urllib.error
import urllib.request

import cv2


def _sample_indices(frame_count: int, k: int, strategy: str, seed: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("Video has no frames.")
    if k <= 0:
        raise ValueError(f"num_frames must be positive, got {k}")
    if k > frame_count:
        k = frame_count

    if strategy == "uniform":
        step = frame_count / k
        return [int(i * step) for i in range(k)]
    if strategy == "index":
        if k == 1:
            return [frame_count // 2]
        if k == 2:
            return [0, frame_count - 1]
        if k == 3:
            return [0, frame_count // 2, frame_count - 1]
        return [int(frame_count * (i + 1) / (k + 1)) for i in range(k)]
    if strategy == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(range(frame_count), k))

    raise ValueError(f"Unknown sampling strategy: {strategy}")


def sample_video_frames_as_data_urls(
    video_path: pathlib.Path,
    num_frames: int,
    strategy: str,
    seed: int,
    jpeg_quality: int,
) -> list[str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _sample_indices(frame_count, num_frames, strategy, seed)

    data_urls: list[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        ok, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            continue
        data = base64.b64encode(encoded.tobytes()).decode("ascii")
        data_urls.append(f"data:image/jpeg;base64,{data}")

    cap.release()
    if not data_urls:
        raise RuntimeError(f"No frames were extracted from {video_path}")
    return data_urls


def _extract_json_object(text: str) -> dict | None:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _post_json(url: str, payload: dict, timeout: int) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            response_body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {error_body}") from e

    try:
        return status, json.loads(response_body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Non-JSON response (status {status}): {response_body[:500]}") from e


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send 25-frame video + text prompt request to a vLLM OpenAI-compatible chat endpoint."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="vLLM server base URL",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl-object-counting-base-q4",
        help="Model name for /v1/chat/completions (use qwen3-vl-object-counting-ft-q4 when serving in lora_4bit mode)",
    )
    parser.add_argument(
        "--prompt-file",
        default="/workspace/data/inputs/PROMPT.txt",
        help="Path to prompt text file",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to mp4 video file",
    )
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument(
        "--sampling-strategy",
        choices=["uniform", "index", "random"],
        default="uniform",
    )
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save full request/response JSON",
    )
    args = parser.parse_args()

    prompt_path = pathlib.Path(args.prompt_file)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Prompt file is empty: {prompt_path}")

    video_path = pathlib.Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    image_urls = sample_video_frames_as_data_urls(
        video_path=video_path,
        num_frames=args.num_frames,
        strategy=args.sampling_strategy,
        seed=args.sampling_seed,
        jpeg_quality=args.jpeg_quality,
    )

    content = [{"type": "text", "text": prompt_text}]
    content += [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    status_code, response_obj = _post_json(endpoint, payload, timeout=args.timeout)

    choice = response_obj.get("choices", [{}])[0]
    message = choice.get("message", {})
    raw_content = message.get("content", "")
    if isinstance(raw_content, list):
        text_parts = [part.get("text", "") for part in raw_content if isinstance(part, dict)]
        raw_text = "\n".join(text_parts)
    else:
        raw_text = str(raw_content)

    parsed_json = _extract_json_object(raw_text)

    print(f"status_code={status_code}")
    print(f"video={video_path}")
    print(f"frames_sent={len(image_urls)}")
    print(f"model={args.model}")
    usage = response_obj.get("usage")
    if usage:
        print(f"usage={json.dumps(usage)}")

    print("raw_response_text_start")
    print(raw_text)
    print("raw_response_text_end")

    if parsed_json is not None:
        print("parsed_json_start")
        print(json.dumps(parsed_json, indent=2))
        print("parsed_json_end")
    else:
        print("parsed_json_start")
        print("null")
        print("parsed_json_end")

    if args.output_json:
        output_path = pathlib.Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "request": payload,
                    "response": response_obj,
                    "parsed_json": parsed_json,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved_output={output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
