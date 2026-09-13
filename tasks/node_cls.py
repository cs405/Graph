r"""节点分类任务（本项目原有任务；对应 ultralytics ``models/yolo/detect/train.py``）。

这一层是**允许认识具体算子**的地方 —— 框架层（``model_builder``/``training.trainer``）
不许，任务层必须：诊断报什么、往 ``run_dir`` 落什么，本来就是算法特定的。
分界线是：本文件只 import ``family='oca'`` 这一族需要的东西，绝不 import
:mod:`modules.dia`；边级任务的对称约束见 :mod:`tasks.edge_cls`。

``build_hooks`` 的返回值就是 :class:`training.base.TrainHooks`，trainer 拿到它以后
不再问「这是什么模型」。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from torch import nn

from config import TrainConfig
from dataset.base import GraphBundle
from modules.base import explainable_layers
from training.base import DiagFn, TrainHooks
from training.diagnostics import dump_gates, gate_report

__all__ = ['build_hooks', 'diagnostics_for']


def _has_gates(model: nn.Module) -> bool:
    """模型里有没有 OCA 那一族的可解释层。

    用 ``family`` 字符串筛而不是 ``isinstance(m, OCALayer)``：后者会让本文件
    import 算子，而基线（gat/gcn/mlp 结构表）上根本没有 λ 可报。
    """
    return bool(explainable_layers(model, family='oca'))


def diagnostics_for(model: nn.Module) -> Optional[DiagFn]:
    """有门控才返回 :func:`training.diagnostics.gate_report`，否则 ``None``。"""
    return gate_report if _has_gates(model) else None


def build_hooks(cfg: TrainConfig, ds: GraphBundle, model: nn.Module,
                device: Optional[Any] = None) -> TrainHooks:
    dumps: Dict[str, Callable] = {}
    if cfg.run.dump_gates and _has_gates(model):
        # 放 run_dir 根而不是 weights/：它不是权重，是画图用的导出数据
        dumps['gates'] = dump_gates
    return TrainHooks.default(cfg, ds, device,
                              diagnostics=diagnostics_for(model), dumps=dumps)
