#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
mkdir -p "$DATA_DIR"
exec "${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}" -m uvicorn app.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --workers 1
