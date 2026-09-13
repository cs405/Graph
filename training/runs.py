"""运行目录：``runs/<run_name>/train``、``train2``、``train3`` …（对齐 yolov8 的 ``training/runs.py``）。

规则（与 yolov8 同）：

* 第一次不带序号：``train``，不是 ``train1``；
* 之后依次 ``train2``、``train3``；
* ``resume=True`` 时复用**编号最大且含 checkpoint** 的那个目录，而不是新建。

为什么不直接用时间戳当目录名：消融表要反复比对同一份配置的历史运行，``train3``
这种可复现的序号比 ``20260912_153012`` 好写在论文里，也方便 ``--resume``。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

__all__ = ['SUBFOLDERS', 'prepare_run_dir', 'list_run_dirs', 'has_checkpoint',
           'save_config_used', 'save_json']

_log = logging.getLogger('oca')

SUBFOLDERS = ('weights', 'checkpoints', 'logs')


def _dir_number(name: str, kind: str) -> Optional[int]:
    """``train`` -> 1，``train7`` -> 7，``train1`` -> None（序号 1 留给裸名）。"""
    if name == kind:
        return 1
    m = re.fullmatch(rf'{re.escape(kind)}(\d+)', name)
    if m and int(m.group(1)) >= 2:
        return int(m.group(1))
    return None


def list_run_dirs(base, kind: str) -> List[Tuple[int, Path]]:
    """``base`` 下属于 ``kind`` 的运行目录，按编号升序。"""
    base = Path(base)
    if not base.is_dir():
        return []
    found = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        num = _dir_number(p.name, kind)
        if num is not None:
            found.append((num, p))
    return sorted(found)


def has_checkpoint(run_dir) -> bool:
    """目录里是否已有可用的 checkpoint（``resume`` 判断复用哪个目录）。"""
    ckpt_dir = Path(run_dir) / 'checkpoints'
    if not ckpt_dir.is_dir():
        return False
    return any(ckpt_dir.glob('checkpoint_e*.pth'))


def _make_subfolders(run_dir) -> Path:
    run_dir = Path(run_dir)
    for sub in SUBFOLDERS:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def prepare_run_dir(project, run_name: str, kind: str = 'train',
                    resume: bool = False) -> Tuple[Path, bool]:
    """创建（或复用）运行目录，返回 ``(run_dir, resumed)``。"""
    base = Path(project) / run_name
    existing = list_run_dirs(base, kind)

    if resume:
        for _, path in reversed(existing):
            if has_checkpoint(path):
                _log.info('恢复训练：复用运行目录 %s', path)
                return _make_subfolders(path), True
        _log.info('要求 resume，但没找到任何 checkpoint；改为新建运行目录')

    if existing:
        run_dir = base / f'{kind}{existing[-1][0] + 1}'
    else:
        run_dir = base / kind
    _make_subfolders(run_dir)
    _log.info('运行目录：%s', run_dir)
    return run_dir, False


def save_config_used(run_dir, config: Dict[str, Any]) -> Path:
    """把**实际生效**的配置快照写进 ``config_used.yaml``（追溯以它为准）。"""
    path = Path(run_dir) / 'config_used.yaml'
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    _log.info('配置快照：%s', path)
    return path


def save_json(path, payload: Any) -> Path:
    """结果落盘（``results.json``）。numpy 标量不是 JSON 类型，故先 ``default=str`` 兜。"""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path
