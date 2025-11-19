from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import cv2

from .preprocess import PreprocessResult


def remove_padding(depth_sq: np.ndarray, pad: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Remove padding from square depth map using (top,bottom,left,right).
    """
    t, b, l, r = pad
    h, w = depth_sq.shape
    y0 = t
    y1 = h - b if b > 0 else h
    x0 = l
    x1 = w - r if r > 0 else w
    return depth_sq[y0:y1, x0:x1]


def normalize_to_uint16(depth: np.ndarray) -> np.ndarray:
    """
    Min-max normalize float depth to uint16 range [0, 65535].
    """
    d = depth.astype(np.float32)
    d = d - d.min()
    if d.max() > 1e-8:
        d = d / d.max()
    d16 = np.clip(d * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    return d16


def colorize_depth(depth: np.ndarray, cmap: str = "magma") -> np.ndarray:
    """
    Colorize depth (float32 or uint16). Uses OpenCV colormaps.
    """
    if depth.dtype != np.uint8:
        d = depth.astype(np.float32)
        d = d - d.min()
        if d.max() > 1e-8:
            d = d / d.max()
        depth_u8 = np.clip(d * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        depth_u8 = depth

    # Map common names to OpenCV
    cmap_map = {
        "magma": cv2.COLORMAP_MAGMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "turbo": cv2.COLORMAP_TURBO,
        "plasma": cv2.COLORMAP_PLASMA,
        "jet": cv2.COLORMAP_JET,
    }
    code = cmap_map.get(cmap, cv2.COLORMAP_MAGMA)
    colored_bgr = cv2.applyColorMap(depth_u8, code)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return colored_rgb


def postprocess_depth(
    pred_sq: np.ndarray,
    prep: PreprocessResult,
    output_long_side: Optional[int] = None,
    return_uint16: bool = True,
) -> np.ndarray:
    """
    Convert square prediction back to original aspect. Optionally resize so that
    the long side equals `output_long_side`. Returns uint16 (grayscale) if
    `return_uint16` else float32 in [0,1].
    """
    # Remove padding if any
    d = remove_padding(pred_sq, prep.pad)
    # Resize back to original size
    oh, ow = prep.orig_size
    if d.shape[0] != oh or d.shape[1] != ow:
        d = cv2.resize(d, (ow, oh), interpolation=cv2.INTER_CUBIC)

    # Optionally resize to a specific long side
    if output_long_side is not None and output_long_side > 0:
        h, w = d.shape
        if h >= w:
            new_h = output_long_side
            new_w = int(round(w * output_long_side / h))
        else:
            new_w = output_long_side
            new_h = int(round(h * output_long_side / w))
        d = cv2.resize(d, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if return_uint16:
        return normalize_to_uint16(d)
    else:
        # return float32 normalized 0-1
        d = d.astype(np.float32)
        d = d - d.min()
        if d.max() > 1e-8:
            d = d / d.max()
        return d
