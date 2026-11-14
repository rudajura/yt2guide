#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
for t in uv ffmpeg node ollama; do
  command -v "$t" >/dev/null || { echo "missing: $t (install it first)"; exit 1; }
done
uv venv --python 3.12 "$D/.venv"
uv pip install --python "$D/.venv/bin/python" -r "$D/requirements.txt"
mkdir -p "$HOME/bin"
ln -sf "$D/yt2guide" "$HOME/bin/yt2guide"
echo "installed: ~/bin/yt2guide -> $D/yt2guide (make sure ~/bin is on your PATH)"
