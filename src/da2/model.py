from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path

import onnxruntime as ort
from huggingface_hub import hf_hub_download
import numpy as np

from .config import DEFAULT_HF_REPO_ID, DEFAULT_MODEL_FILENAME, CACHE_DIR
from .providers import get_providers
from .preprocess import preprocess_image, PreprocessResult
from .postprocess import postprocess_depth


ort.preload_dlls()


@dataclass
class ModelInfo:
    path: Path
    input_name: str
    output_name: str
    input_shape: Tuple[int, int, int, int]  # NCHW


class DepthAnythingV2:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        hf_repo_id: str = DEFAULT_HF_REPO_ID,
        filename: str = DEFAULT_MODEL_FILENAME,
        prefer_cpu: bool = False,
    ) -> None:
        self.model_path = self._resolve_model_path(model_path, hf_repo_id, filename)
        providers, provider_options = get_providers(prefer_cpu=prefer_cpu)
        sess_options = ort.SessionOptions()
        if prefer_cpu:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        else:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )
        self.info = self._introspect_model()

    def _resolve_model_path(self, model_path: Optional[Path], repo_id: str, filename: str) -> Path:
        if model_path is not None:
            p = Path(model_path)
            if not p.exists():
                raise FileNotFoundError(f"Model path not found: {p}")
            return p
        # download via HF
        local = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(CACHE_DIR))
        return Path(local)

    def _introspect_model(self) -> ModelInfo:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if not inputs:
            raise RuntimeError("Model has no inputs")
        if not outputs:
            raise RuntimeError("Model has no outputs")
        inp = inputs[0]
        out = outputs[0]
        shape = tuple(
            int(dim) if isinstance(dim, int) or (isinstance(dim, str) and dim.isdigit()) else -1
            for dim in inp.shape
        )

        return ModelInfo(path=self.model_path, input_name=inp.name, output_name=out.name, input_shape=shape)  # type: ignore

    def infer(
        self,
        image: "PIL.Image.Image",
        target_size: int = 518,
    ) -> Tuple[np.ndarray, PreprocessResult]:
        prep = preprocess_image(image, target_size=target_size)
        input_feed = {self.info.input_name: prep.tensor}
        preds = self.session.run([self.info.output_name], input_feed)[0]
        # Expect (1,1,H,W) or (1,H,W)
        if preds.ndim == 4:
            preds = preds[:, 0]
        elif preds.ndim == 3:
            # already (1,H,W)
            pass
        else:
            raise RuntimeError(f"Unexpected output shape: {preds.shape}")
        return preds[0], prep

    def infer_and_postprocess(
        self,
        image: "PIL.Image.Image",
        target_size: int = 518,
        output_long_side: Optional[int] = None,
    ) -> np.ndarray:
        t0 = time.perf_counter()
        pred, prep = self.infer(image, target_size=target_size)
        t1 = time.perf_counter()
        print(f"Inference time: {(t1 - t0) * 1000:.2f} ms")
        depth = postprocess_depth(pred, prep, output_long_side=output_long_side)
        return depth

    def providers(self) -> List[str]:
        return self.session.get_providers()
