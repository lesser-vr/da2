from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return Image.open(p)


def save_depth_uint16(depth16: np.ndarray, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Pillow expects mode "I;16" for 16-bit unsigned
    img = Image.fromarray(depth16, mode="I;16")
    img.save(p)


def save_image_rgb(arr: np.ndarray, path: str | Path, quality: Optional[int] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    if quality is not None and p.suffix.lower() in {".jpg", ".jpeg"}:
        img.save(p, quality=int(quality))
    else:
        img.save(p)
