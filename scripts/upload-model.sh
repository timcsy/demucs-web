#!/bin/bash
# Upload ONNX model to Hugging Face Hub
#
# Prerequisites:
#   uv tool install huggingface_hub
#   hf auth login
#
# Usage:
#   ./scripts/upload-model.sh <hf-username>

set -e

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -z "$1" ]; then
    echo "Usage: $0 <huggingface-username>"
    echo "Example: $0 myusername"
    exit 1
fi

HF_USER="$1"
REPO_NAME="demucs-web-onnx"
MODEL_FILE="$PROJECT_ROOT/models/htdemucs_embedded.onnx"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found at $MODEL_FILE"
    exit 1
fi

REPO_ID="$HF_USER/$REPO_NAME"

echo "Creating/updating Hugging Face repo: $REPO_ID"

# Create repo if not exists
hf repo create "$REPO_ID" --type model 2>/dev/null || echo "Repo already exists or created"

# Upload model
echo "Uploading model (this may take a while)..."
hf upload "$REPO_ID" "$MODEL_FILE" htdemucs_embedded.onnx --repo-type model

echo ""
echo "Done! Model URL:"
echo "https://huggingface.co/$REPO_ID/resolve/main/htdemucs_embedded.onnx"
echo ""
echo "Update your code to use this URL for model loading."
