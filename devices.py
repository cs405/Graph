"""设备解析：所有入口共用一份，免得「auto」在三个文件里是三种意思。

与 yolov8 的 ``devices.py`` 同位。注意本项目里 GPU 有两个真实陷阱，都在这里处理：

* ``cuda`` 不可用时要**降级并说清楚**，不能抛一句看不懂的 RuntimeError；
* 逐元素位级比对的测试**必须跑 CPU + float64**：``index_add_`` 在 CUDA 上不满足
  结合律，同一份代码两次运行的 1e-16 量级结果都会不同（详见 test/support.py）。
"""

from __future__ import annotations

import logging

import torch

__all__ = ['resolve_device', 'describe_device', 'is_cuda']

_log = logging.getLogger('oca')


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except (AssertionError, RuntimeError):        # 驱动/CUDA 运行时坏掉时不让导入炸
        return False


def is_cuda(device) -> bool:
    return str(getattr(device, 'type', device)).startswith('cuda')


def resolve_device(spec: str = 'auto') -> torch.device:
    """``'auto' | 'cpu' | 'cuda' | 'cuda:1' | 0`` -> ``torch.device``。"""
    name = str(spec).strip().lower()
    if name in ('', 'auto'):
        return torch.device('cuda' if _cuda_available() else 'cpu')
    if name.startswith('cuda') and not _cuda_available():
        _log.warning('请求 %s 但 CUDA 不可用，回退 CPU', spec)
        return torch.device('cpu')
    return torch.device(name)


def describe_device(device) -> str:
    """一行设备签名，写进日志与 config_used 旁边，方便复现时对口径。"""
    d = resolve_device(device) if isinstance(device, str) else torch.device(device)
    if d.type == 'cpu':
        return f'cpu (torch {torch.__version__})'
    idx = d.index if d.index is not None else torch.cuda.current_device()
    try:
        props = torch.cuda.get_device_properties(idx)
        mem = props.total_memory // (1024 ** 2)
        return f'cuda:{idx} ({props.name}, {mem}MiB, torch {torch.__version__})'
    except Exception as e:                                   # noqa: BLE001
        return f'cuda:{idx} (信息不可用: {type(e).__name__})'
