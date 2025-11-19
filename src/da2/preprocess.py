from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image
import cv2


@dataclass
class PreprocessResult:
    tensor: np.ndarray  # NCHW float32
    resize_size: int
    orig_size: Tuple[int, int]  # (H, W)
    pad: Tuple[int, int, int, int]  # (top, bottom, left, right)


def resize_keep_aspect_pad(img: Image.Image, target: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Resize shortest side to target, keep aspect, then pad to square (target x target).
    Returns (resized_rgb_uint8, (top, bottom, left, right)).
    """
    img = img.convert("RGB")
    w, h = img.size
    if w <= h:
        new_w = target
        new_h = int(round(h * target / w))
    else:
        new_h = target
        new_w = int(round(w * target / h))
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)
    arr = np.array(img_resized, dtype=np.uint8)

    # compute center crop/pad to target x target
    if new_h >= target and new_w >= target:
        # center crop
        y0 = (new_h - target) // 2
        x0 = (new_w - target) // 2
        crop = arr[y0:y0 + target, x0:x0 + target]
        pad = (0, 0, 0, 0)
        return crop, pad
    # need to pad
    top = (target - new_h) // 2 if new_h < target else 0
    bottom = target - new_h - top if new_h < target else 0
    left = (target - new_w) // 2 if new_w < target else 0
    right = target - new_w - left if new_w < target else 0
    padded = cv2.copyMakeBorder(arr, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, (top, bottom, left, right)


def preprocess_image(image: Image.Image, target_size: int = 518) -> PreprocessResult:
    """
    Convert PIL image to model input tensor (1,3,H,W) float32 normalized.
    Normalization: [0,1] then ImageNet mean/std.
    """
    orig_h, orig_w = image.height, image.width
    img_resized, pad = resize_keep_aspect_pad(image, target_size)
    img_f = img_resized.astype(np.float32) / 255.0
    # ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_f = (img_f - mean) / std
    # HWC -> CHW
    chw = np.transpose(img_f, (2, 0, 1))
    tensor = np.expand_dims(chw, 0)  # (1,3,H,W)
    return PreprocessResult(tensor=tensor, resize_size=target_size, orig_size=(orig_h, orig_w), pad=pad)
