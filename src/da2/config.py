from __future__ import annotations

from pathlib import Path

# Default Hugging Face repo and filenames for Depth Anything v2 Small (ViT-S) ONNX
# Note: If these defaults ever change upstream, override via CLI options.
DEFAULT_HF_REPO_ID = "onnx-community/depth-anything-v2-small"
DEFAULT_MODEL_FILENAME = "onnx/model_fp16.onnx"

# Cache directory (inside user cache dir)
CACHE_DIR = Path.home() / ".cache" / "da2"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
