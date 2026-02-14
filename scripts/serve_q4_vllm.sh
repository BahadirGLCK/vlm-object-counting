#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_ROOT/experiments/2026-02-08_19-57-23_finetune}"
CONFIG_PATH="${CONFIG_PATH:-$EXPERIMENT_DIR/experiment_config.json}"
LORA_DIR="${LORA_DIR:-$EXPERIMENT_DIR/finetuned_model/final}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-$EXPERIMENT_DIR/finetuned_model/merged_16bit}"

# Serve mode:
# - merged_bnb : serve merged_16bit weights with bitsandbytes quantization (most faithful to finetuned model)
# - lora_4bit  : serve quantized base model + LoRA adapter (lower memory, but see note below)
#
# Notes:
# 1) vLLM may fail with shape-mismatch assertions when lora_4bit is used with
#    Unsloth bnb 4-bit base checkpoints.
# 2) For Qwen3-VL, vLLM currently applies LoRA to language layers only; visual
#    LoRA layers are ignored.
SERVE_MODE="${VLLM_SERVE_MODE:-merged_bnb}"

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-qwen3-vl-object-counting-base-q4}"
LORA_ALIAS="${VLLM_LORA_ALIAS:-qwen3-vl-object-counting-ft-q4}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
LORA_BASE_MODEL_FALLBACK="${VLLM_LORA_BASE_MODEL_FALLBACK:-Qwen/Qwen3-VL-32B-Instruct}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed in the active environment." >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

# Pull serving + evaluation settings from experiment_config.json.
readarray -t EVAL_VALUES < <(
  python - "$CONFIG_PATH" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
eval_cfg = cfg.get("EVAL_GENERATION_CONFIG", {})
print(cfg.get("BASE_MODEL_ID", "unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit"))
print(cfg.get("MAX_SEQ_LEN_EVAL", 2000))
print(eval_cfg.get("temperature", 0.2))
print(eval_cfg.get("top_p", 0.95))
print(eval_cfg.get("max_new_tokens", 2000))
print(cfg.get("MAX_SEQ_LEN_FINETUNE", 10000))
print(cfg.get("NUM_FRAMES_PER_VIDEO", 25))
PY
)

BASE_MODEL_ID="${VLLM_BASE_MODEL_ID:-${EVAL_VALUES[0]}}"
MAX_SEQ_LEN_EVAL="${EVAL_VALUES[1]}"
TEMPERATURE="${VLLM_TEMPERATURE:-${EVAL_VALUES[2]}}"
TOP_P="${VLLM_TOP_P:-${EVAL_VALUES[3]}}"
MAX_NEW_TOKENS="${VLLM_MAX_NEW_TOKENS:-${EVAL_VALUES[4]}}"
MAX_SEQ_LEN_FINETUNE="${EVAL_VALUES[5]}"
NUM_FRAMES_PER_VIDEO="${NUM_FRAMES_PER_VIDEO:-${EVAL_VALUES[6]}}"

DEFAULT_MAX_MODEL_LEN="${MAX_SEQ_LEN_EVAL}"
if [[ "${NUM_FRAMES_PER_VIDEO}" -ge 25 && "${MAX_SEQ_LEN_EVAL}" -lt 6000 ]]; then
  # 25-frame multimodal prompts typically exceed a 2k context budget in vLLM.
  # Prefer the finetuning context length for serving while preserving eval generation settings.
  DEFAULT_MAX_MODEL_LEN="${MAX_SEQ_LEN_FINETUNE}"
fi
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${DEFAULT_MAX_MODEL_LEN}}"

if [[ -n "${VLLM_LIMIT_MM_PER_PROMPT:-}" ]]; then
  LIMIT_MM_PER_PROMPT="${VLLM_LIMIT_MM_PER_PROMPT}"
else
  LIMIT_MM_PER_PROMPT="{\"image\":${NUM_FRAMES_PER_VIDEO},\"video\":1}"
fi

OVERRIDE_GENERATION_CONFIG="{\"temperature\":${TEMPERATURE},\"top_p\":${TOP_P},\"max_new_tokens\":${MAX_NEW_TOKENS}}"
MAX_LORAS="${VLLM_MAX_LORAS:-1}"

if [[ -f "$LORA_DIR/adapter_config.json" ]]; then
  LORA_RANK_DEFAULT="$(python - "$LORA_DIR/adapter_config.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get("r", 16))
PY
)"
else
  LORA_RANK_DEFAULT="16"
fi
MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-$LORA_RANK_DEFAULT}"

if [[ "$SERVE_MODE" != "lora_4bit" && "$SERVE_MODE" != "merged_bnb" ]]; then
  echo "Invalid VLLM_SERVE_MODE=$SERVE_MODE. Use one of: lora_4bit, merged_bnb" >&2
  exit 1
fi

if [[ "$SERVE_MODE" == "lora_4bit" ]]; then
  if [[ ! -f "$LORA_DIR/adapter_config.json" ]]; then
    echo "LoRA adapter not found at: $LORA_DIR" >&2
    exit 1
  fi

  # Unsloth bnb 4-bit bases can fail in vLLM with shape mismatch assertions.
  # Default to a compatible upstream base unless user explicitly forces it.
  if [[ "$BASE_MODEL_ID" == unsloth/*-bnb-4bit* ]]; then
    if [[ "${VLLM_ALLOW_UNSLOTH_BNB_LORA:-0}" == "1" ]]; then
      echo "WARNING: Forcing Unsloth bnb 4-bit base in lora_4bit mode."
      echo "         If vLLM fails with shape mismatch, unset VLLM_ALLOW_UNSLOTH_BNB_LORA."
    else
      echo "Detected Unsloth bnb 4-bit base in lora_4bit mode: $BASE_MODEL_ID"
      echo "Switching LoRA base to compatible model: $LORA_BASE_MODEL_FALLBACK"
      echo "Set VLLM_ALLOW_UNSLOTH_BNB_LORA=1 to force the original base."
      BASE_MODEL_ID="$LORA_BASE_MODEL_FALLBACK"
    fi
  fi

  TARGET_MODEL="$BASE_MODEL_ID"
  REQUEST_MODEL_NAME="$LORA_ALIAS"
else
  if [[ ! -f "$MERGED_MODEL_DIR/config.json" ]]; then
    echo "Merged model not found at: $MERGED_MODEL_DIR" >&2
    exit 1
  fi
  TARGET_MODEL="$MERGED_MODEL_DIR"
  REQUEST_MODEL_NAME="$SERVED_MODEL_NAME"
fi

echo "Serve mode: $SERVE_MODE"
echo "Serving quantized model target: $TARGET_MODEL"
if [[ "$SERVE_MODE" == "lora_4bit" ]]; then
  echo "Attached LoRA adapter: $LORA_DIR (alias: $LORA_ALIAS, rank: $MAX_LORA_RANK)"
  echo "NOTE: vLLM applies Qwen3-VL LoRA only to language layers; vision LoRA layers are ignored."
fi
echo "Endpoint: http://$HOST:$PORT"
echo "Served model name: $SERVED_MODEL_NAME"
echo "Use this model name in requests: $REQUEST_MODEL_NAME"
echo "Eval defaults: eval_max_seq_len=$MAX_SEQ_LEN_EVAL, max_new_tokens=$MAX_NEW_TOKENS, temperature=$TEMPERATURE, top_p=$TOP_P, num_frames_per_video=$NUM_FRAMES_PER_VIDEO"
echo "Serving max_model_len: $MAX_MODEL_LEN"
echo "MM limit per prompt: $LIMIT_MM_PER_PROMPT"

if [[ "$SERVE_MODE" == "lora_4bit" ]]; then
  exec vllm serve "$TARGET_MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --load-format bitsandbytes \
    --quantization bitsandbytes \
    --enable-lora \
    --lora-modules "${LORA_ALIAS}=${LORA_DIR}" \
    --max-loras "$MAX_LORAS" \
    --max-lora-rank "$MAX_LORA_RANK" \
    --dtype bfloat16 \
    --chat-template-content-format openai \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
    --generation-config vllm \
    --override-generation-config "$OVERRIDE_GENERATION_CONFIG" \
    --disable-log-requests
else
  exec vllm serve "$TARGET_MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --load-format bitsandbytes \
    --quantization bitsandbytes \
    --dtype bfloat16 \
    --chat-template-content-format openai \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
    --generation-config vllm \
    --override-generation-config "$OVERRIDE_GENERATION_CONFIG" \
    --disable-log-requests
fi
