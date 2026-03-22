#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_REPO="${HF_REPO:-SmilingWolf/wd-swinv2-tagger-v3}"

MODEL_URL="${MODEL_URL:-${HF_ENDPOINT%/}/${HF_REPO}/resolve/main/model.onnx?download=1}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/model/swinv2-v3.onnx}"

TAGS_URL="${TAGS_URL:-${HF_ENDPOINT%/}/${HF_REPO}/resolve/main/selected_tags.csv?download=1}"
TAGS_PATH="${TAGS_PATH:-${ROOT_DIR}/csv/selected_tags.csv}"

IMAGE_ROOT="${IMAGE_ROOT:-/srv/shared-images}"
DOWNLOAD_TAGS="${DOWNLOAD_TAGS:-0}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

usage() {
  cat <<'EOF'
Usage: ./init.sh [options]

Initialize local folders and download model assets from Hugging Face.

Options:
  --model-url <url>      Override model download URL
  --model-path <path>    Override output path for model.onnx
  --download-tags        Also download selected_tags.csv
  --tags-url <url>       Override tag CSV download URL
  --tags-path <path>     Override output path for tag CSV
  --image-root <path>    Directory for shared images (default: /srv/shared-images)
  --force                Force re-download even if file exists
  --help                 Show this help message

Environment overrides:
  HF_ENDPOINT, HF_REPO, MODEL_URL, MODEL_PATH, TAGS_URL, TAGS_PATH,
  IMAGE_ROOT, DOWNLOAD_TAGS=1, FORCE_DOWNLOAD=1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-url)
      MODEL_URL="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --download-tags)
      DOWNLOAD_TAGS=1
      shift
      ;;
    --tags-url)
      TAGS_URL="$2"
      shift 2
      ;;
    --tags-path)
      TAGS_PATH="$2"
      DOWNLOAD_TAGS=1
      shift 2
      ;;
    --image-root)
      IMAGE_ROOT="$2"
      shift 2
      ;;
    --force)
      FORCE_DOWNLOAD=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

download_file() {
  local url="$1"
  local output="$2"

  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" && "$FORCE_DOWNLOAD" != "1" ]]; then
    echo "Skip existing file: $output"
    return 0
  fi

  echo "Downloading -> $output"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 --connect-timeout 15 -o "$output" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$output" "$url"
  else
    echo "Either curl or wget is required for download." >&2
    return 1
  fi
}

mkdir -p "${ROOT_DIR}/model" "${ROOT_DIR}/csv"
download_file "$MODEL_URL" "$MODEL_PATH"

if [[ "$DOWNLOAD_TAGS" == "1" ]]; then
  download_file "$TAGS_URL" "$TAGS_PATH"
fi

if mkdir -p "$IMAGE_ROOT" 2>/dev/null; then
  :
else
  IMAGE_ROOT="${ROOT_DIR}/data/images"
  mkdir -p "$IMAGE_ROOT"
  echo "No permission to create /srv path, fallback IMAGE_ROOT: $IMAGE_ROOT"
  echo "Remember to update docker-compose.yaml volume mapping if needed."
fi

cat <<EOF
Init complete.

Model:      $MODEL_PATH
Tags CSV:   ${TAGS_PATH}
Image root: ${IMAGE_ROOT}

Next steps:
  1) ./build.sh
  2) docker compose up -d
EOF
