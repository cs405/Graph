"""日志：控制台 + ``runs/<name>/train/logs/train.log``。

不用 loguru（环境里没有），标准 logging 足够。Windows 下必须做两件事，否则
第一行中文/希腊字母就 UnicodeEncodeError：

* ``sys.stdout.reconfigure(encoding='utf-8')``；
* 文件 handler 显式 ``encoding='utf-8'``。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

__all__ = ['setup_logging', 'add_file_logging', 'close_file_logging',
           'log_model_summary', 'get_logger']

_FMT = '%(asctime)s | %(levelname)-7s | %(message)s'
_DATEFMT = '%H:%M:%S'
ROOT = 'oca'


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(ROOT if name is None else f'{ROOT}.{name}')


def setup_logging(level: int = logging.INFO, stream: bool = True) -> logging.Logger:
    """幂等：重复调用不会叠出多份 console handler。"""
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except (ValueError, OSError):               # 输出被重定向成已关闭的流
            pass
    log = logging.getLogger(ROOT)
    log.setLevel(level)
    for h in list(log.handlers):                    # 清掉上一次留下的，避免重复输出
        log.removeHandler(h)
        h.close()
    if stream:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        log.addHandler(h)
    return log


def add_file_logging(run_dir, name: str = 'train.log') -> Path:
    """把日志同时写进 ``run_dir/logs/<name>``，返回文件路径。"""
    log_dir = Path(run_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    h = logging.FileHandler(path, mode='a', encoding='utf-8')
    h.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s'))
    logging.getLogger(ROOT).addHandler(h)
    return path


def close_file_logging() -> None:
    """关掉文件 handler —— Windows 上不关就删不掉临时目录（PermissionError）。"""
    log = logging.getLogger(ROOT)
    for h in list(log.handlers):
        if isinstance(h, logging.FileHandler):
            log.removeHandler(h)
            h.close()


def log_model_summary(model, logger=None) -> int:
    """一行一个子模块的参数量 + 总数（yolov8 的 ``model.info()`` 的朴素版）。"""
    log = logger or get_logger()
    total = 0
    for name, p in model.named_parameters():
        total += p.numel() if p.requires_grad else 0
    log.info('模型 %s：可训练参数 %d', type(model).__name__, total)
    return total
