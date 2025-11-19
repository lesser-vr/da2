from __future__ import annotations

from pathlib import Path
from typing import Optional

import sys
import click
from PIL import Image
import numpy as np
import requests
from tqdm import tqdm

from .model import DepthAnythingV2
from .io import load_image, save_depth_uint16, save_image_rgb
from .postprocess import colorize_depth
from .config import DEFAULT_HF_REPO_ID, DEFAULT_MODEL_FILENAME, CACHE_DIR


@click.group(help="Depth Anything v2 Small (ONNX Runtime + CUDA) toy CLI")
def app() -> None:
    pass


@app.command("providers")
def cmd_providers() -> None:
    """Show available ONNX Runtime providers and which will be used."""
    m = DepthAnythingV2(prefer_cpu=False)
    click.echo("Providers (available/selected order):")
    for p in m.providers():
        click.echo(f"- {p}")


@app.command("infer")
@click.argument("input_image", type=click.Path(exists=True, path_type=Path))
@click.option("--out16", type=click.Path(path_type=Path), default=None, help="Save 16-bit depth PNG path (grayscale)")
@click.option("--outcolor", type=click.Path(path_type=Path), default=None, help="Save colorized depth (PNG/JPG)")
@click.option("--cmap", type=click.Choice(["magma","inferno","turbo","plasma","jet"]), default="magma")
@click.option("--long-side", type=int, default=None, help="Resize output so long side equals this value")
@click.option("--size", type=int, default=518, help="Model input square size (e.g., 518)")
@click.option("--cpu", is_flag=True, help="Force CPUExecutionProvider")
@click.option("--repo", type=str, default=DEFAULT_HF_REPO_ID, help="HF repo id for ONNX model")
@click.option("--filename", type=str, default=DEFAULT_MODEL_FILENAME, help="Model filename in repo")
@click.option("--model-path", type=click.Path(exists=True, path_type=Path), default=None, help="Use local ONNX model path")
def cmd_infer(
    input_image: Path,
    out16: Optional[Path],
    outcolor: Optional[Path],
    cmap: str,
    long_side: Optional[int],
    size: int,
    cpu: bool,
    repo: str,
    filename: str,
    model_path: Optional[Path],
) -> None:
    if out16 is None and outcolor is None:
        # default outputs next to input
        stem = input_image.stem
        outdir = input_image.parent
        out16 = outdir / f"{stem}_depth16.png"
        outcolor = outdir / f"{stem}_depth_{cmap}.png"

    click.echo("Loading model...")
    m = DepthAnythingV2(
        model_path=model_path,
        hf_repo_id=repo,
        filename=filename,
        prefer_cpu=cpu,
    )
    click.echo(f"Providers: {m.providers()}")

    img = load_image(input_image)
    click.echo("Running inference...")
    depth16 = m.infer_and_postprocess(img, target_size=size, output_long_side=long_side)

    if out16 is not None:
        save_depth_uint16(depth16, out16)
        click.echo(f"Saved uint16 depth: {out16}")
    if outcolor is not None:
        colored = colorize_depth(depth16)
        save_image_rgb(colored, outcolor)
        click.echo(f"Saved colorized depth: {outcolor}")


@app.command("example")
@click.option("--out", type=click.Path(path_type=Path), default=Path("assets/example.jpg"), help="Where to save the example image")
def cmd_example(out: Path) -> None:
    """Download a small public-domain example image for testing."""
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_winter_panorama.jpg/1280px-Fronalpstock_winter_panorama.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"Downloading example image to {out} ...")
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    click.echo("Done. Try: da2 infer assets/example.jpg")
