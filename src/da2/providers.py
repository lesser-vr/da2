from __future__ import annotations

from typing import List, Tuple

import onnxruntime as ort


def get_providers(prefer_cpu: bool = False) -> Tuple[List[str], list]:
    """
    Build provider list for ONNX Runtime, preferring CUDA when available.
    Returns (providers, provider_options)
    """
    if prefer_cpu:
        return ["CPUExecutionProvider"], []

    available = ort.get_available_providers()
    providers: List[str] = []
    provider_options: list = []

    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
        # Reasonable defaults; users with advanced needs can set env vars.
        provider_options.append({
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_use_max_workspace": "1",
            "do_copy_in_default_stream": "1",
        })

    # Always put CPU as a fallback
    providers.append("CPUExecutionProvider")
    provider_options.append({})

    return providers, provider_options
