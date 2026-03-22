#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but was not found in PATH" >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  echo "Building image with docker compose..."
  docker compose build "$@"
else
  echo "docker compose not found, falling back to docker build..."
  docker build "$@" -t eagle-ai-tagger:latest -f Dockerfile .
fi
