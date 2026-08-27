#!/usr/bin/env bash

set -euo pipefail

MODEL_URL="https://huggingface.co/xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
TOKENIZER_URL="https://huggingface.co/xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json"

echo "Downloading all-MiniLM-L6-v2.onnx..."
curl -L --progress-bar -o "all-MiniLM-L6-v2.onnx" "$MODEL_URL"

echo "Downloading tokenizer.json..."
curl -L --progress-bar -o "tokenizer.json" "$TOKENIZER_URL"

echo "Download completed successfully."
