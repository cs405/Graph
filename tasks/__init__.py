r"""任务表（对应 ultralytics ``nn/tasks.py`` 里的 ``YOLO.task_map``）。

``task_map`` 在 ultralytics 里长这样：

.. code-block:: python

    task_map = {'detect': {'model': DetectionModel, 'trainer': yolo.detect.train,
                           'validator': ..., 'predictor': ...}, ...}

本项目照搬这个分派思路，但每个任务只需要提供**一样东西**：一份
:class:`training.base.TrainHooks`。因为训练循环本身（全图前向 → 损失 → 反传 →
选 val 最优 → 落盘）在节点分类与边分类之间是完全一样的，不一样的只有
「损失怎么算、诊断报什么、落什么盘」—— 那三件事正是 hooks 的三个字段。

于是分工是：

* :mod:`training.trainer` 只认 ``TrainHooks``，不认识任务；
* :mod:`tasks.node_cls` / :mod:`tasks.edge_cls` 各自认识自己那一族的算子；
* 本文件只做「名字 -> 任务」的查表与两处校验（数据集的监督层级、结构表的输出层级）。

新加一个任务（比如整个子图的分类）＝ 新写一个 ``tasks/xxx.py`` + 在这里登记一行，
trainer 与 builder 都不用动。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from torch import nn

from config import TrainConfig
from dataset.base import GraphBundle
from tasks import edge_cls, node_cls
from training.base import TrainHooks

__all__ = ['Task', 'TASK_MAP', 'task_names', 'resolve_task', 'build_hooks',
           'assert_level']

HooksFn = Callable[[TrainConfig, GraphBundle, nn.Module, Optional[Any]], TrainHooks]


@dataclass(frozen=True)
class Task:
    """一个任务 = 名字 + 它监督的层级 + 怎么造 hooks。"""

    name: str
    supervision: str                 # 'node' | 'edge'，必须与 ds.supervision 对上
    hooks: HooksFn

    def check_dataset(self, ds: GraphBundle) -> None:
        if getattr(ds, 'supervision', 'node') != self.supervision:
            raise ValueError(
                f'任务 {self.name!r} 要的是 {self.supervision} 级监督，'
                f'数据集 {ds.name!r} 给的是 {ds.supervision} 级'
                f'（y 的长度 {ds.y.numel()}，节点数 {ds.num_nodes}，'
                f'边数 {ds.num_edges}）—— 换一个数据集，或把 task_type 改成 auto')

    def check_model(self, net: nn.Module, ds: GraphBundle) -> None:
        """结构表的输出层级必须与数据集的监督层级一致。

        两边都是二维张量，接错了**不会报错**：``logits[train_mask]`` 在
        ``[N,nc]`` 与 ``[E,nc]`` 上都跑得通（只要 mask 长度碰巧对得上），
        指标也照样出数，只是每一个数都是错的。所以在这里钉住。
        """
        level = getattr(net, 'level', 'node')
        if level != self.supervision:
            raise ValueError(
                f'结构表最后一层输出 {level} 级（{level}_logits），'
                f'但任务 {self.name!r} 监督的是 {self.supervision} 级；'
                f'换一张结构表（cfg/models/*.yaml），或改 task.task_type')


TASK_MAP: Dict[str, Task] = {
    'node_classification': Task('node_classification', 'node',
                                node_cls.build_hooks),
    'edge_classification': Task('edge_classification', 'edge',
                                edge_cls.build_hooks),
}

#: ``ds.supervision`` -> 任务名（``task_type='auto'`` 时走这张表）
_BY_SUPERVISION: Dict[str, str] = {t.supervision: n for n, t in TASK_MAP.items()}


def task_names() -> list:
    return sorted(TASK_MAP)


def resolve_task(cfg: TrainConfig, ds: GraphBundle) -> Task:
    """按 ``cfg.task.task_type`` 取任务；``'auto'`` 时按数据集的监督层级推。"""
    name = str(cfg.task.task_type)
    if name == 'auto':
        sup = getattr(ds, 'supervision', 'node')
        if sup not in _BY_SUPERVISION:
            raise KeyError(f'数据集 {ds.name!r} 声明了未知的监督层级 {sup!r}，'
                           f'可选：{sorted(_BY_SUPERVISION)}')
        name = _BY_SUPERVISION[sup]
    if name not in TASK_MAP:
        raise KeyError(f'未知任务 {name!r}，可选：{task_names()} 或 auto')
    task = TASK_MAP[name]
    task.check_dataset(ds)
    return task


def assert_level(net: nn.Module, ds: GraphBundle,
                 cfg: Optional[TrainConfig] = None) -> str:
    """校验并返回任务名（``model.GraphModel.build`` 与 ``trainer.build`` 都用它）。"""
    task = resolve_task(cfg, ds) if cfg is not None else \
        TASK_MAP[_BY_SUPERVISION[getattr(ds, 'supervision', 'node')]]
    task.check_model(net, ds)
    return task.name


def build_hooks(cfg: TrainConfig, ds: GraphBundle, model: nn.Module,
                device: Optional[Any] = None) -> TrainHooks:
    """trainer 的唯一入口：解析任务 → 校验层级 → 造 hooks。"""
    task = resolve_task(cfg, ds)
    task.check_model(model, ds)
    return task.hooks(cfg, ds, model, device).with_task(task.name)
