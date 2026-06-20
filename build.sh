#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

IMAGE_NAME="${IMAGE_NAME:-cernss/eagle-ai-tagger}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
LOCAL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
REGISTRY="${REGISTRY:-${REMOTE_REGISTRY:-}}"
if [[ -n "${REGISTRY}" ]]; then
  REMOTE_IMAGE="${REGISTRY%/}/${IMAGE_NAME}:${IMAGE_TAG}"
else
  REMOTE_IMAGE="${LOCAL_IMAGE}"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but was not found in PATH" >&2
  exit 1
fi

if ! docker buildx version >/dev/null 2>&1; then
  echo "docker buildx is required but was not found" >&2
  exit 1
fi

echo "Building linux/amd64 image: ${LOCAL_IMAGE}"
docker buildx build --platform linux/amd64 -t "${LOCAL_IMAGE}" --load .

if [[ "${REMOTE_IMAGE}" != "${LOCAL_IMAGE}" ]]; then
  echo "Retagging image to registry target: ${REMOTE_IMAGE}"
  docker tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"
fi

echo "Pushing image: ${REMOTE_IMAGE}"
docker push "${REMOTE_IMAGE}"

echo "Done: ${REMOTE_IMAGE}"
