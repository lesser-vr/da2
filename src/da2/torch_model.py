from __future__ import annotations

"""
Depth Anything v2 Small (PyTorch) wrapper used by the bench harness.

사용자가 선택한 통합 방식(A: 외부 의존성 사용)에 맞춰, 이 클래스는
"공식/레퍼런스 PyTorch 구현"을 외부 패키지로부터 import 하여
인스턴스화합니다. 즉, 이 파일은 실제 아키텍처 코드를 벤더링하지 않고
외부 패키지에 위임합니다.

예상 사용법(하나를 설치):
- pip install depth-anything-v2  (예시 패키지명; 실제 배포명에 맞춰 설치)
- 또는 GitHub에서 직접 설치: pip install "git+https://github.com/DepthAnything/Depth-Anything-V2.git"

지원하는 임포트 후보 경로(순차 시도):
- depth_anything_v2 (ex: from depth_anything_v2 import DepthAnythingV2)
- depth_anything (ex: from depth_anything import DepthAnythingV2)
- depth_anything_v2_small (ex: from depth_anything_v2_small import DepthAnythingV2Small)

주의: 실제 공개 패키지/모듈 경로는 배포 시점에 따라 다를 수 있습니다. 아래
코드는 여러 후보 경로를 순차적으로 시도하며, 실패 시 설치 안내 에러를
명확하게 발생시킵니다.
"""

from typing import Optional, Callable

import torch


class TorchDepthAnythingV2Small(torch.nn.Module):
    """Depth Anything v2 Small 실제 모델 래퍼.

    외부 패키지로부터 실제 모델 클래스를 동적으로 임포트하고, Small(ViT-S)
    변형을 인스턴스화합니다. 이 래퍼의 목적은 벤치마크 스크립트가
    환경변수 없이도 일관된 import 경로(`da2.torch_model.TorchDepthAnythingV2Small`)
    를 통해 실제 PyTorch 모델을 사용할 수 있게 하는 것입니다.

    참고: 입력/출력 정규화 및 후처리는 "공식 구현"의 forward 로직을 따릅니다.
    벤치마크는 순수 forward latency를 측정하므로, 사전/후처리는 이 래퍼에서
    수행하지 않습니다.
    """

    def __init__(self) -> None:
        super().__init__()
        self._impl = self._instantiate_upstream_impl()

    def _instantiate_upstream_impl(self) -> torch.nn.Module:
        """여러 가능한 import 경로를 순차 시도하여 실제 구현을 생성합니다.

        반환되는 모듈은 nn.Module이며, Small(ViT-S) 백본을 사용하도록
        생성자 인자를 시도합니다. 각 구현의 생성자 시그니처가 다를 수
        있으므로, 몇 가지 흔한 패턴을 순차적으로 시도합니다.
        """
        import importlib

        candidates: list[tuple[str, str]] = [
            ("depth_anything_v2", "DepthAnythingV2"),
            ("depth_anything", "DepthAnythingV2"),
            ("depth_anything_v2_small", "DepthAnythingV2Small"),
        ]

        last_err: Optional[Exception] = None
        for mod_name, cls_name in candidates:
            try:
                mod = importlib.import_module(mod_name)
                cls = getattr(mod, cls_name)
                # 시그니처 후보들을 순차 시도
                ctor_candidates: list[tuple[tuple, dict]] = [
                    ((), {"variant": "vits"}),
                    ((), {"backbone": "vits"}),
                    ((), {"model_name": "vits"}),
                    ((), {}),  # 기본 생성자
                ]
                for args, kwargs in ctor_candidates:
                    try:
                        m = cls(*args, **kwargs)
                        if isinstance(m, torch.nn.Module):
                            m.eval()
                            return m
                    except Exception as e:  # 시그니처 불일치면 다음 후보 시도
                        last_err = e
                        continue
            except Exception as e:
                last_err = e
                continue

        help_msg = (
            "Depth Anything v2 PyTorch 구현을 찾을 수 없습니다.\n"
            "다음 중 하나를 설치한 뒤 다시 시도하세요:\n"
            "- pip install depth-anything-v2   (공식 배포명이 다를 수 있습니다)\n"
            "- pip install \"git+https://github.com/DepthAnything/Depth-Anything-V2.git\"\n"
            "그리고 모듈 경로가 위 후보(candidates)에 포함되지 않는 경우,\n"
            "src/da2/torch_model.py 의 candidates 리스트에 해당 import 경로를 추가하세요.\n"
        )
        if last_err is not None:
            raise ImportError(help_msg) from last_err
        raise ImportError(help_msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._impl(x)


__all__ = ["TorchDepthAnythingV2Small"]
