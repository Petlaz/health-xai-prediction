#!/usr/bin/env bash
set -Eeuo pipefail

FRPC_PATH="/usr/local/lib/python3.11/site-packages/gradio/frpc_linux_aarch64_v0.2"
FRPC_URL="https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_aarch64"

if [ ! -f "$FRPC_PATH" ]; then
  echo "[INFO] Downloading Gradio frpc binary..."
  curl -L -o "$FRPC_PATH" "$FRPC_URL"
  chmod +x "$FRPC_PATH"
fi

exec python -m app.app_gradio
