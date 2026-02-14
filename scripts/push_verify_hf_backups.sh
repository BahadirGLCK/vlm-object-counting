#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

retry_cmd() {
  local tries="${RETRIES:-3}"
  local sleep_s="${RETRY_SLEEP_SECONDS:-15}"
  local attempt=1

  while true; do
    if "$@"; then
      return 0
    fi
    local rc=$?
    if [[ "$attempt" -ge "$tries" ]]; then
      echo "Command failed after ${tries} attempts: $*" >&2
      return "$rc"
    fi
    log "Retry ${attempt}/${tries} failed (exit=${rc}) for: $*"
    log "Sleeping ${sleep_s}s before retry..."
    sleep "$sleep_s"
    attempt=$((attempt + 1))
  done
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_ROOT/experiments/2026-02-08_19-57-23_finetune}"
LORA_DIR="${LORA_DIR:-$EXPERIMENT_DIR/finetuned_model/final}"
MERGED_DIR="${MERGED_DIR:-$EXPERIMENT_DIR/finetuned_model/merged_16bit}"
PROMPT_PATH="${PROMPT_PATH:-$PROJECT_ROOT/data/inputs/PROMPT.txt}"
CONFIG_PATH="${CONFIG_PATH:-$EXPERIMENT_DIR/experiment_config.json}"
META_DIR="${META_DIR:-$EXPERIMENT_DIR/hf_export_meta}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TAG="${TAG:-backup-$(date +%Y%m%d-%H%M%S)}"
VERIFY_ROOT="${VERIFY_ROOT:-/tmp/hf-verify-$(basename "$EXPERIMENT_DIR")-$TAG}"
KEEP_VERIFY_DOWNLOAD="${KEEP_VERIFY_DOWNLOAD:-1}"
HF_TOKEN="${HF_TOKEN:-}"
REQUIRE_PRIVATE_REPO="${REQUIRE_PRIVATE_REPO:-1}"

require_cmd python
require_cmd sha256sum
require_cmd find
require_cmd sort
require_cmd xargs

if ! python - <<'PY' >/dev/null 2>&1
import huggingface_hub  # noqa: F401
PY
then
  die "Python package 'huggingface_hub' is required. Install it and retry."
fi

[[ -d "$EXPERIMENT_DIR" ]] || die "Experiment directory not found: $EXPERIMENT_DIR"
[[ -d "$LORA_DIR" ]] || die "LoRA directory not found: $LORA_DIR"
[[ -d "$MERGED_DIR" ]] || die "Merged model directory not found: $MERGED_DIR"
[[ -f "$CONFIG_PATH" ]] || die "Config file not found: $CONFIG_PATH"
[[ -f "$PROMPT_PATH" ]] || die "Prompt file not found: $PROMPT_PATH"

HF_USER_DEFAULT="$(
HF_TOKEN="$HF_TOKEN" python - <<'PY'
from huggingface_hub import HfApi
import os
token = os.getenv("HF_TOKEN") or None
try:
    print(HfApi(token=token).whoami()["name"])
except Exception:
    print("")
PY
)"
HF_USER="${HF_USER:-$HF_USER_DEFAULT}"
[[ -n "$HF_USER" ]] || die "Cannot determine Hugging Face username. Run 'hf auth login' first or set HF_USER."

LORA_REPO="${LORA_REPO:-$HF_USER/qwen3-vl-object-counting-lora}"
MERGED_REPO="${MERGED_REPO:-$HF_USER/qwen3-vl-object-counting-merged16}"

hfi_repo_create() {
  local repo_id="$1"
  HF_REPO_ID="$repo_id" HF_TOKEN="$HF_TOKEN" python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
token = os.getenv("HF_TOKEN") or None
api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
print(f"repo_ready={repo_id}")
PY
}

hfi_upload_large_folder() {
  local repo_id="$1"
  local folder_path="$2"
  local ignore_patterns="${3:-}"

  HF_REPO_ID="$repo_id" \
  HF_FOLDER_PATH="$folder_path" \
  HF_IGNORE_PATTERNS="$ignore_patterns" \
  HF_NUM_WORKERS="$NUM_WORKERS" \
  HF_TOKEN="$HF_TOKEN" \
  python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
folder_path = os.environ["HF_FOLDER_PATH"]
ignore_patterns = os.getenv("HF_IGNORE_PATTERNS", "").strip()
num_workers = int(os.getenv("HF_NUM_WORKERS", "8"))
token = os.getenv("HF_TOKEN") or None

kwargs = {}
if ignore_patterns:
    kwargs["ignore_patterns"] = [p.strip() for p in ignore_patterns.split(",") if p.strip()]

api = HfApi(token=token)
api.upload_large_folder(
    repo_id=repo_id,
    folder_path=folder_path,
    repo_type="model",
    num_workers=num_workers,
    print_report=True,
    print_report_every=60,
    **kwargs,
)
print(f"uploaded_folder={repo_id}:{folder_path}")
PY
}

hfi_upload_file() {
  local repo_id="$1"
  local local_path="$2"
  local path_in_repo="$3"

  HF_REPO_ID="$repo_id" \
  HF_LOCAL_PATH="$local_path" \
  HF_PATH_IN_REPO="$path_in_repo" \
  HF_TOKEN="$HF_TOKEN" \
  python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
local_path = os.environ["HF_LOCAL_PATH"]
path_in_repo = os.environ["HF_PATH_IN_REPO"]
token = os.getenv("HF_TOKEN") or None

api = HfApi(token=token)
api.upload_file(
    path_or_fileobj=local_path,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="model",
)
print(f"uploaded_file={repo_id}:{path_in_repo}")
PY
}

hfi_assert_private_repo() {
  local repo_id="$1"
  local require_private="$2"

  HF_REPO_ID="$repo_id" \
  HF_REQUIRE_PRIVATE="$require_private" \
  HF_TOKEN="$HF_TOKEN" \
  python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
require_private = os.getenv("HF_REQUIRE_PRIVATE", "1") == "1"
token = os.getenv("HF_TOKEN") or None

api = HfApi(token=token)
info = api.repo_info(repo_id=repo_id, repo_type="model")
is_private = bool(getattr(info, "private", False))

if require_private and not is_private:
    raise SystemExit(
        f"Repository is public but REQUIRE_PRIVATE_REPO=1: {repo_id}. "
        "Make it private on HF or set REQUIRE_PRIVATE_REPO=0."
    )

print(f"repo_visibility_ok={repo_id}:private={is_private}")
PY
}

hfi_create_tag() {
  local repo_id="$1"
  local tag="$2"
  local tag_msg="$3"

  HF_REPO_ID="$repo_id" \
  HF_TAG="$tag" \
  HF_TAG_MESSAGE="$tag_msg" \
  HF_TOKEN="$HF_TOKEN" \
  python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
tag = os.environ["HF_TAG"]
tag_message = os.environ["HF_TAG_MESSAGE"]
token = os.getenv("HF_TOKEN") or None

api = HfApi(token=token)
api.create_tag(
    repo_id=repo_id,
    repo_type="model",
    tag=tag,
    tag_message=tag_message,
    exist_ok=True,
)
print(f"tag_ready={repo_id}:{tag}")
PY
}

hfi_download_snapshot() {
  local repo_id="$1"
  local revision="$2"
  local local_dir="$3"

  HF_REPO_ID="$repo_id" \
  HF_REVISION="$revision" \
  HF_LOCAL_DIR="$local_dir" \
  HF_NUM_WORKERS="$NUM_WORKERS" \
  HF_TOKEN="$HF_TOKEN" \
  python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
revision = os.environ["HF_REVISION"]
local_dir = os.environ["HF_LOCAL_DIR"]
num_workers = int(os.getenv("HF_NUM_WORKERS", "8"))
token = os.getenv("HF_TOKEN") or None

api = HfApi(token=token)
api.snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    revision=revision,
    local_dir=local_dir,
    max_workers=num_workers,
)
print(f"downloaded_snapshot={repo_id}:{revision}")
PY
}

compute_checksums() {
  local src_dir="$1"
  local out_file="$2"
  local exclude_cache="${3:-0}"

  if [[ "$exclude_cache" == "1" ]]; then
    (
      cd "$src_dir"
      find . -type f ! -path './.cache/*' -print0 | sort -z | xargs -0 -r sha256sum
    ) >"$out_file"
  else
    (
      cd "$src_dir"
      find . -type f -print0 | sort -z | xargs -0 -r sha256sum
    ) >"$out_file"
  fi

  [[ -s "$out_file" ]] || die "Checksum file is empty: $out_file"
}

upload_shared_metadata() {
  local repo_id="$1"
  local checksum_file="$2"

  retry_cmd hfi_upload_file "$repo_id" "$checksum_file" "SHA256SUMS.txt"
  retry_cmd hfi_upload_file "$repo_id" "$CONFIG_PATH" "experiment_config.json"
  retry_cmd hfi_upload_file "$repo_id" "$PROMPT_PATH" "PROMPT.txt"
}

log "Preparing backup for experiment: $EXPERIMENT_DIR"
log "HF account: $HF_USER"
log "LoRA repo: $LORA_REPO"
log "Merged repo: $MERGED_REPO"
log "Tag: $TAG"

mkdir -p "$META_DIR"
LORA_SUMS="$META_DIR/final.SHA256SUMS"
MERGED_SUMS="$META_DIR/merged_16bit.SHA256SUMS"

log "Computing local checksums (LoRA)..."
compute_checksums "$LORA_DIR" "$LORA_SUMS" 0

log "Computing local checksums (merged_16bit)..."
compute_checksums "$MERGED_DIR" "$MERGED_SUMS" 1

log "Ensuring private model repos exist..."
retry_cmd hfi_repo_create "$LORA_REPO"
retry_cmd hfi_repo_create "$MERGED_REPO"
retry_cmd hfi_assert_private_repo "$LORA_REPO" "$REQUIRE_PRIVATE_REPO"
retry_cmd hfi_assert_private_repo "$MERGED_REPO" "$REQUIRE_PRIVATE_REPO"

log "Uploading LoRA folder (resumable)..."
retry_cmd hfi_upload_large_folder "$LORA_REPO" "$LORA_DIR" ""

log "Uploading merged_16bit folder (resumable)..."
retry_cmd hfi_upload_large_folder "$MERGED_REPO" "$MERGED_DIR" ".cache/*"

log "Uploading metadata and checksum manifests..."
upload_shared_metadata "$LORA_REPO" "$LORA_SUMS"
upload_shared_metadata "$MERGED_REPO" "$MERGED_SUMS"

log "Creating immutable backup tags..."
retry_cmd hfi_create_tag "$LORA_REPO" "$TAG" "Backup of $(basename "$EXPERIMENT_DIR")"
retry_cmd hfi_create_tag "$MERGED_REPO" "$TAG" "Backup of $(basename "$EXPERIMENT_DIR")"

log "Downloading tagged snapshots for verification..."
mkdir -p "$VERIFY_ROOT/lora" "$VERIFY_ROOT/merged"
retry_cmd hfi_download_snapshot "$LORA_REPO" "$TAG" "$VERIFY_ROOT/lora"
retry_cmd hfi_download_snapshot "$MERGED_REPO" "$TAG" "$VERIFY_ROOT/merged"

log "Verifying downloaded LoRA snapshot checksums..."
(
  cd "$VERIFY_ROOT/lora"
  sha256sum -c SHA256SUMS.txt
)

log "Verifying downloaded merged_16bit snapshot checksums..."
(
  cd "$VERIFY_ROOT/merged"
  sha256sum -c SHA256SUMS.txt
)

if [[ "$KEEP_VERIFY_DOWNLOAD" == "0" ]]; then
  log "Removing verification download directory: $VERIFY_ROOT"
  rm -rf "$VERIFY_ROOT"
fi

echo
echo "Backup + verification completed successfully."
echo "LoRA repo   : https://huggingface.co/$LORA_REPO/tree/$TAG"
echo "Merged repo : https://huggingface.co/$MERGED_REPO/tree/$TAG"
echo "Tag         : $TAG"
if [[ "$KEEP_VERIFY_DOWNLOAD" == "1" ]]; then
  echo "Verified local snapshot directory: $VERIFY_ROOT"
fi
