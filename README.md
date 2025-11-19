# da2 — Depth Anything v2 Small + ONNX Runtime (CUDA) Toy Project

Tiny, CUDA‑accelerated depth estimation pipeline using Depth Anything v2 Small (ViT‑S) ONNX and ONNX Runtime. Includes a simple CLI for running inference on images and saving both 16‑bit depth and a colorized preview.

Tested targets: NVIDIA RTX 30xx ~ 50xx with CUDA. Falls back to CPU automatically if CUDA provider is unavailable or you pass `--cpu`.

## Quickstart (with uv)

Prereqs:
- Python 3.10–3.12
- NVIDIA GPU + CUDA (for GPU). CPU also works but is slower.
- OS: Linux/Windows recommended for CUDA. macOS will use CPU.

Setup with uv:

```bash
# Create and activate a virtual env (optional but recommended)
uv venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .

# Verify the CLI is available
da2 --help
```

If you prefer plain pip:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Note: `onnxruntime-gpu` wheels target specific CUDA versions. If your CUDA is newer, it still typically works because ORT wheels bundle their own CUDA runtime parts; otherwise pin `onnxruntime-gpu` to a version that matches your environment.

## Download an example image
```bash
da2 example --out assets/example.jpg
```

## Run inference
```bash
# Basic: saves grayscale 16-bit PNG and a colorized PNG next to input
# Model and weights are auto-downloaded from Hugging Face cache on first run

da2 infer assets/example.jpg

# Specify output paths and color map
 da2 infer assets/example.jpg \
   --out16 out/depth16.png \
   --outcolor out/depth_color.png \
   --cmap turbo

# Resize output to a specific long side (keeping aspect)
 da2 infer assets/example.jpg --long-side 1024

# Force CPU (fallback if CUDA provider isn’t available)
 da2 infer assets/example.jpg --cpu

# Use a local ONNX model file
 da2 infer assets/example.jpg --model-path /path/to/model.onnx

# Or override Hugging Face repo/filename (if upstream layout changes)
 da2 infer assets/example.jpg \
   --repo depth-anything/Depth-Anything-V2-onnx \
   --filename depth_anything_v2_small.onnx
```

## Providers (CUDA vs CPU)
ONNX Runtime provider selection prefers CUDA when available, and always includes CPU fallback.

Check providers:
```bash
da2 providers
```
Example output:
```
Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

## Project layout
```
src/da2/
  cli.py         # Click-based CLI: infer, providers, example
  model.py       # ORT session setup, HF download, inference
  preprocess.py  # Resize, pad/crop, normalize to NCHW float32
  postprocess.py # Remove pad, resize to original, uint16/colorize
  io.py          # Image loading/saving helpers
  config.py      # Defaults (HF repo, filenames, cache dir)
  providers.py   # Provider selection (CUDA preferred)
```

## Notes
- Default model target size is 518 (square). You can change via `--size`.
- Output 16-bit PNG is min-max normalized per image (relative depth). For absolute metric depth you’d need scale calibration (beyond this toy scope).
- Colorized output is normalized to 8-bit and color-mapped (magma/inferno/turbo/plasma/jet).

## Troubleshooting
- If `CUDAExecutionProvider` is missing:
  - Ensure you installed `onnxruntime-gpu` (not just `onnxruntime`).
  - Windows: install NVIDIA drivers; Linux: drivers + CUDA libraries. Many ORT GPU wheels ship their own CUDA but still require a compatible driver.
  - Try CPU mode: `--cpu`.
- Hugging Face download issues: set `HF_HUB_ENABLE_HF_TRANSFER=1` for faster downloads (optional), or pass `--model-path` to use a local file.

## License
This repository is a minimal example. The model and weights are owned by their respective authors and distributed under their original licenses. See the corresponding Hugging Face model card for details.
