#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required. On Ubuntu/WSL install it with: sudo apt update && sudo apt install -y ffmpeg" >&2
  exit 1
fi

if ! command -v gcc >/dev/null 2>&1; then
  echo "A C compiler is required by Triton. On Ubuntu/WSL install it with: sudo apt update && sudo apt install -y build-essential" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN before running this script; Pyannote Community-1 requires Hugging Face access." >&2
  exit 1
fi

echo "Installing project dependencies with uv ..."
uv sync --python 3.12

echo "Downloading/caching all pipeline models ..."
uv run --no-sync python scripts/download_models.py

echo "Setup complete. Start the dashboard with: uv run --no-sync python dashboard.py"
