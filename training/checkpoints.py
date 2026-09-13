"""checkpoint：命名、原子写、滚动保留、RNG 快照（对齐 yolov8 的 ``training/checkpoints.py``）。

命名规则（一个保存点一个文件，永不合并）：

    checkpoint_e0042c0000.pth

``e0042`` 是 1-indexed 的 epoch，``c0000`` 是该 epoch 内已完成的优化步数。
本项目是 **transductive 全图**，一个 epoch 只有一次 forward/backward，所以
``c`` 恒为 0 —— 保留这个字段是为了与 yolov8 的命名格式兼容（以及将来若要按
mini-batch 切子图时不必改文件名）。
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

__all__ = ['checkpoint_name', 'parse_checkpoint_name', 'atomic_save',
           'CheckpointManager', 'capture_rng_state', 'restore_rng_state',
           'save_model', 'load_checkpoint']

_log = logging.getLogger('oca')

CKPT_PATTERN = re.compile(r'^checkpoint_e(\d+)c(\d+)\.pth$')


def checkpoint_name(epoch: int, step: int = 0) -> str:
    """一个 checkpoint 的文件名（epoch 从 1 开始）。"""
    return f'checkpoint_e{int(epoch):04d}c{int(step):04d}.pth'


def parse_checkpoint_name(path) -> Optional[tuple]:
    """从文件名解出 ``(epoch, step)``；不匹配返回 ``None``。"""
    m = CKPT_PATTERN.match(Path(path).name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def atomic_save(state: Any, path) -> Path:
    """先写临时文件再改名：中途断电/崩掉也不会留下半个「看起来完整」的权重。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, tmp)
    tmp.replace(path)
    return path


class CheckpointManager:
    """一个目录内的 checkpoint 保存 / 列出 / 滚动清理。"""

    def __init__(self, directory, max_keep: int = 5):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_keep = int(max_keep)

    def list(self) -> List[Path]:
        """按 ``(epoch, step)`` 排序，旧的在前。"""
        found = []
        for p in self.directory.glob('checkpoint_e*.pth'):
            parsed = parse_checkpoint_name(p)
            if parsed is not None:
                found.append((parsed, p))
        return [p for _, p in sorted(found)]

    def latest(self) -> Optional[Path]:
        ckpts = self.list()
        return ckpts[-1] if ckpts else None

    def save(self, state: Any, epoch: int, step: int = 0) -> Path:
        path = self.directory / checkpoint_name(epoch, step)
        atomic_save(state, path)
        _log.info('checkpoint 已保存：%s', path.name)
        self.rotate()
        return path

    def rotate(self) -> None:
        """超出 ``max_keep`` 就删最老的（全图模型不大，但 400 epoch 存满会淹掉磁盘）。"""
        ckpts = self.list()
        if len(ckpts) <= self.max_keep:
            return
        for p in ckpts[:len(ckpts) - self.max_keep]:
            try:
                p.unlink()
                _log.info('checkpoint 滚动清理：%s', p.name)
            except OSError as e:
                _log.warning('删不掉 %s：%s', p, e)


def capture_rng_state() -> Dict[str, Any]:
    """把三套随机源快照下来，供 ``resume`` 时接回同一条轨迹。"""
    np_state = np.random.get_state()
    state: Dict[str, Any] = {
        'python': random.getstate(),
        'numpy': [np_state[0], np.asarray(np_state[1]).tolist(),
                  int(np_state[2]), int(np_state[3]), float(np_state[4])],
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    """恢复 :func:`capture_rng_state` 存的随机源；失败只告警，不中断训练。"""
    if not state:
        return
    try:
        py_state = state.get('python')
        if py_state is not None:
            random.setstate(_to_tuple(py_state))
        np_state = state.get('numpy')
        if np_state is not None:
            np.random.set_state((np_state[0],
                                 np.asarray(np_state[1], dtype=np.uint32),
                                 int(np_state[2]), int(np_state[3]),
                                 float(np_state[4])))
        if state.get('torch') is not None:
            torch.set_rng_state(state['torch'].to(torch.uint8).cpu())
        if state.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [t.to(torch.uint8).cpu() for t in state['cuda']])
    except Exception as e:                            # noqa: BLE001
        _log.warning('RNG 状态恢复失败（%s）：继续训练，但随机性是新起的', e)


def save_model(net: torch.nn.Module, path, meta: Optional[Dict[str, Any]] = None
               ) -> Path:
    """写 ``best.pt``：权重 + 重建所需的元信息（结构表/scale/nc/in_dim/配置）。

    只存 state_dict 的 checkpoint 在半年后基本等于废纸 —— 没有 ``spec/scale/nc``
    就不知道要搭哪个网络，所以这里把重建信息一并落盘。
    """
    payload: Dict[str, Any] = {'model': net.state_dict()}
    payload.update(meta or {})
    return atomic_save(payload, path)


def load_checkpoint(path, net: Optional[torch.nn.Module] = None,
                    device: str = 'auto') -> Dict[str, Any]:
    """读回 ``best.pt``；给了 ``net`` 就把权重装进去。

    ``weights_only=False`` 是有意的：文件里除了张量还有配置 dict。这些文件全部由
    本项目自己产出，不存在「加载别人发来的 pickle」这个攻击面 —— 真要加载外部
    权重，请只留 ``model`` 那一项。
    """
    from devices import resolve_device

    payload = torch.load(Path(path), map_location=resolve_device(device),
                         weights_only=False)
    if not isinstance(payload, dict) or 'model' not in payload:
        raise ValueError(f'{path} 不是本项目的 checkpoint（缺 "model" 键）')
    if net is not None:
        missing = net.load_state_dict(payload['model'], strict=False)
        if missing.missing_keys or missing.unexpected_keys:
            _log.warning('权重与结构不完全对应：缺 %s / 多 %s',
                         missing.missing_keys, missing.unexpected_keys)
        net.to(resolve_device(device))
    return payload
