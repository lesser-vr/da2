#!/usr/bin/env python3
"""
PyTorch benchmark for Depth Anything v2 Small under fixed settings.
- Device: cuda/cpu (default: cuda)
- Batch: 1 (fixed)
- Input size: default 518x518 (fixed-size inputs recommended)
- AMP: enabled via --amp (default: off)
- torch.compile: not used
- channels_last: optional flag (recommended on for recent NVIDIA GPUs)

Model loading:
- Downloads weights from HF: repo "depth-anything/Depth-Anything-V2-Small",
  filename "depth_anything_v2_vits.pth".
- Uses a built-in placeholder class `da2.torch_model.TorchDepthAnythingV2Small`
  so you can run without any environment variables. If you later implement the
  real architecture in that class, the same benchmark will load weights
  automatically (strict=False) and proceed.

Results are appended to a CSV with timing statistics.
"""
import argparse
import csv
import os
import time
from statistics import mean, pstdev

import torch
from huggingface_hub import hf_hub_download
from da2.torch_model import TorchDepthAnythingV2Small

HF_REPO_ID = "depth-anything/Depth-Anything-V2-Small"
HF_FILENAME = "depth_anything_v2_vits.pth"


def maybe_load_state_dict(model: torch.nn.Module, ckpt_path: str) -> None:
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # Try strict=True first (preferred for real model)
        try:
            model.load_state_dict(sd, strict=True)
            print("[da2:bench] state_dict loaded with strict=True")
            return
        except Exception as e_strict:
            print(f"[da2:bench] strict=True load failed: {e_strict}. Retrying with strict=False ...")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(
                f"[da2:bench] strict=False loaded. missing={len(missing)}, unexpected={len(unexpected)}"
            )
    except Exception as e:
        print(f"[da2:bench] Warning: failed to load state_dict from {ckpt_path}: {e}")


def bench_once(model, inp, use_amp: bool, device: torch.device):
    with torch.inference_mode():
        if device.type == "cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return model(inp)
        else:
            return model(inp)


def run(device: str, h: int, w: int, warmup: int, iters: int, use_amp: bool, channels_last: bool):
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Download weights from HF
    ckpt = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)

    # Build model
    model = TorchDepthAnythingV2Small()
    maybe_load_state_dict(model, ckpt)
    model.to(dev)

    if channels_last and dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    x = torch.randn(1, 3, h, w, device=dev)
    if channels_last and dev.type == "cuda":
        x = x.to(memory_format=torch.channels_last)

    # Warmup
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed-size input -> allow autotune
        torch.cuda.synchronize()
    for _ in range(warmup):
        _ = bench_once(model, x, use_amp, dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    lats = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = bench_once(model, x, use_amp, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        lats.append((t1 - t0) * 1000.0)

    return mean(lats), pstdev(lats)


def main():
    ap = argparse.ArgumentParser(description="Depth Anything v2 Small (PyTorch) benchmark")
    ap.add_argument("--height", type=int, default=518)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--amp", action="store_true", help="Enable autocast(fp16) on CUDA")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--csv", type=str, default="torch_results.csv")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    if args.amp:
        # often beneficial on newer GPUs
        torch.set_float32_matmul_precision("high")

    avg, std = run(
        device=args.device,
        h=args.height,
        w=args.width,
        warmup=args.warmup,
        iters=args.iters,
        use_amp=args.amp,
        channels_last=args.channels_last,
    )

    ips = 1000.0 / avg  # batch=1
    row = {
        "tag": args.tag,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": args.device,
        "h": args.height,
        "w": args.width,
        "amp": args.amp,
        "channels_last": args.channels_last,
        "warmup": args.warmup,
        "iters": args.iters,
        "avg_ms": round(avg, 3),
        "std_ms": round(std, 3),
        "throughput_ips": round(ips, 2),
    }

    print(row)

    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
