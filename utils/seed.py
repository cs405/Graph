"""随机种子与设备：不加这个，异配图上的 0.5 个点差异说不清是模型还是噪声。"""

from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ['set_seed', 'resolve_device', 'count_parameters']


def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    r"""统一随机源。

    注意 ``index_add_`` 在 CUDA 上是**非确定性**的（浮点加法不满足结合律），
    所以逐元素比对的测试必须跑在 CPU + float64 下，``deterministic`` 只保证
    同种子同设备可复现，不保证 GPU/CPU 位级一致。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(spec: str = 'auto') -> torch.device:
    if spec == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(spec)


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    ps = [p for p in module.parameters() if p.requires_grad or not trainable_only]
    return sum(p.numel() for p in ps)
