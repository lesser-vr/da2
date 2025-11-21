#!/usr/bin/env python3
"""
ONNX Runtime benchmark using the existing da2 model wrapper.
- Measures either pure session.run (forward-only) or end-to-end with postprocess.
- Fixed input size recommended (default 518), batch=1.
- Uses the project's default FP16 ONNX model from HF.

Outputs timing stats and optionally appends to a CSV.
"""
import argparse
import csv
import os
import time
from statistics import mean, pstdev
from pathlib import Path

from PIL import Image
import numpy as np

from da2.model import DepthAnythingV2
from da2.preprocess import preprocess_image
from da2.postprocess import postprocess_depth


def bench_forward_only(model: DepthAnythingV2, img: Image.Image, size: int, warmup: int, iters: int):
    # Preprocess once (fixed input)
    prep = preprocess_image(img, target_size=size)
    input_feed = {model.info.input_name: prep.tensor}
    out_name = model.info.output_name

    # Warmup
    for _ in range(warmup):
        _ = model.session.run([out_name], input_feed)[0]

    # Measure: session.run only (exclude postprocess)
    lats = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model.session.run([out_name], input_feed)[0]
        t1 = time.perf_counter()
        lats.append((t1 - t0) * 1000.0)
    return mean(lats), pstdev(lats)


def bench_end_to_end(model: DepthAnythingV2, img: Image.Image, size: int, warmup: int, iters: int):
    # Warmup
    for _ in range(warmup):
        _ = model.infer_and_postprocess(img, target_size=size)

    # Measure: includes preprocess+session.run+postprocess
    lats = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model.infer_and_postprocess(img, target_size=size)
        t1 = time.perf_counter()
        lats.append((t1 - t0) * 1000.0)
    return mean(lats), pstdev(lats)


def main():
    ap = argparse.ArgumentParser(description="da2 ONNX Runtime benchmark")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--cpu", action="store_true", help="Force CPUExecutionProvider")
    ap.add_argument("--forward-only", action="store_true", help="Measure session.run only (no postprocess)")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")

    m = DepthAnythingV2(prefer_cpu=args.cpu)

    if args.forward_only:
        avg, std = bench_forward_only(m, img, args.size, args.warmup, args.iters)
    else:
        avg, std = bench_end_to_end(m, img, args.size, args.warmup, args.iters)

    row = {
        "tag": args.tag,
        "providers": ",".join(m.providers()),
        "cpu": args.cpu,
        "forward_only": args.forward_only,
        "size": args.size,
        "warmup": args.warmup,
        "iters": args.iters,
        "avg_ms": round(avg, 3),
        "std_ms": round(std, 3),
    }

    print(row)

    if args.csv is not None:
        write_header = not args.csv.exists()
        with open(args.csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    main()
