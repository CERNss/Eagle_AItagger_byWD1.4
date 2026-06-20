#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_REPO="${HF_REPO:-SmilingWolf/wd-swinv2-tagger-v3}"

MODEL_URL="${MODEL_URL:-${HF_ENDPOINT%/}/${HF_REPO}/resolve/main/model.onnx?download=1}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/model/swinv2-v3.onnx}"

TAGS_URL="${TAGS_URL:-https://raw.githubusercontent.com/CERNss/Eagle_AItagger_byWD1.4/main/csv/Tags-cn_2024_ver-1.0.csv}"
TAGS_PATH="${TAGS_PATH:-${ROOT_DIR}/csv/Tags-cn_2024_ver-1.0.csv}"

IMAGE_ROOT="${IMAGE_ROOT:-/srv/shared-images}"
DOWNLOAD_TAGS="${DOWNLOAD_TAGS:-0}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
REGISTRY="${REGISTRY:-${REMOTE_REGISTRY:-192.168.10.142:5000}}"
IMAGE_NAME="${IMAGE_NAME:-eagle-ai-tagger}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REMOTE_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
LOCAL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
SKIP_IMAGE_PULL="${SKIP_IMAGE_PULL:-0}"

usage() {
  cat <<'EOF'
Usage: ./init.sh [options]

Initialize local folders, download model assets from Hugging Face, and
pull container image from private registry for local retagging.

Options:
  --model-url <url>      Override model download URL
  --model-path <path>    Override output path for model.onnx
  --download-tags        Also download the repo-compatible tag CSV
  --tags-url <url>       Override tag CSV download URL
  --tags-path <path>     Override output path for tag CSV
  --image-root <path>    Directory for shared images (default: /srv/shared-images)
  --remote-registry <h>  Private registry host:port (default: 192.168.10.142:5000)
  --image-name <name>    Image name to pull/retag (default: eagle-ai-tagger)
  --image-tag <tag>      Image tag to pull/retag (default: latest)
  --skip-image-pull      Skip docker pull + retag step
  --force                Force re-download even if file exists
  --help                 Show this help message

Environment overrides:
  HF_ENDPOINT, HF_REPO, MODEL_URL, MODEL_PATH, TAGS_URL, TAGS_PATH,
  IMAGE_ROOT, DOWNLOAD_TAGS=1, FORCE_DOWNLOAD=1,
  REGISTRY, REMOTE_REGISTRY, IMAGE_NAME, IMAGE_TAG, SKIP_IMAGE_PULL=1
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
    --remote-registry)
      REGISTRY="$2"
      REMOTE_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
      shift 2
      ;;
    --image-name)
      IMAGE_NAME="$2"
      LOCAL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
      REMOTE_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
      shift 2
      ;;
    --image-tag)
      IMAGE_TAG="$2"
      LOCAL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
      REMOTE_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
      shift 2
      ;;
    --skip-image-pull)
      SKIP_IMAGE_PULL=1
      shift
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
  local tmp_output="${output}.tmp.$$"
  rm -f "$tmp_output"
  if command -v curl >/dev/null 2>&1; then
    if ! curl -fL --retry 3 --retry-delay 2 --connect-timeout 15 -o "$tmp_output" "$url"; then
      rm -f "$tmp_output"
      return 1
    fi
  elif command -v wget >/dev/null 2>&1; then
    if ! wget -O "$tmp_output" "$url"; then
      rm -f "$tmp_output"
      return 1
    fi
  else
    echo "Either curl or wget is required for download." >&2
    rm -f "$tmp_output"
    return 1
  fi
  if [[ ! -s "$tmp_output" ]]; then
    echo "Downloaded file is empty: $url" >&2
    rm -f "$tmp_output"
    return 1
  fi
  mv "$tmp_output" "$output"
}

sync_image() {
  if [[ "$SKIP_IMAGE_PULL" == "1" ]]; then
    echo "Skip image pull/retag step."
    return 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required for image pull/retag step." >&2
    return 1
  fi

  echo "Pulling image: ${REMOTE_IMAGE}"
  docker pull "${REMOTE_IMAGE}"

  if [[ "${REMOTE_IMAGE}" != "${LOCAL_IMAGE}" ]]; then
    echo "Retagging image locally: ${LOCAL_IMAGE}"
    docker tag "${REMOTE_IMAGE}" "${LOCAL_IMAGE}"
  fi
}

mkdir -p "${ROOT_DIR}/model" "${ROOT_DIR}/csv"

# Scaffold local config from tracked templates (real files are git-ignored).
if [[ ! -f "${ROOT_DIR}/config.yaml" && -f "${ROOT_DIR}/config.example.yaml" ]]; then
  cp "${ROOT_DIR}/config.example.yaml" "${ROOT_DIR}/config.yaml"
  echo "Created config.yaml from config.example.yaml"
fi
if [[ ! -f "${ROOT_DIR}/.env" && -f "${ROOT_DIR}/.env.example" ]]; then
  cp "${ROOT_DIR}/.env.example" "${ROOT_DIR}/.env"
  echo "Created .env from .env.example"
fi

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
  echo "Use IMAGE_ROOT=\"$IMAGE_ROOT\" docker compose up -d for this fallback path."
fi

sync_image

cat <<EOF
Init complete.

Model:      $MODEL_PATH
Tags CSV:   ${TAGS_PATH}
Image root: ${IMAGE_ROOT}
Remote img: ${REMOTE_IMAGE}
Local img:  ${LOCAL_IMAGE}

Next steps:
  1) ./build.sh
  2) IMAGE_ROOT="$IMAGE_ROOT" docker compose up -d
EOF
