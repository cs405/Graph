"""融合层（对应 yolov8 neck 里的 ``Concat`` + ``Conv`` 组合）。

图侧与图像侧的关键差别：**所有尺度共享同一批节点**（transductive 全图，没有
FPN 那种空间下采样）。所以「跨尺度融合」只有两种合法写法：

* :class:`Merge` —— 等宽相加（v1 的 ``GraphNeck`` 写法，骨架代码已删，拓扑如今在
  ``cfg/models/oca.yaml``）；
* :class:`Concat` —— 沿特征维拼接（老 ``GraphHead`` 读出前的写法）。

宽度不等时**不能**用 ``Merge``：torch 会把 ``[N,64] + [N,128]`` 直接广播失败抛
shape 错，这是好事（比静默加错强），但结构表里更该改用 ``Concat``。
"""

from __future__ import annotations

from typing import List, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from modules.base import GraphOp, register_module
from modules.convs import ACTS, _tag

__all__ = ['Merge', 'Concat']

Feats = Union[Tensor, Sequence[Tensor]]


def _as_list(xs: Feats) -> List[Tensor]:
    return list(xs) if isinstance(xs, (list, tuple)) else [xs]


@register_module('Merge')
class Merge(GraphOp):
    r"""多输入融合：等宽相加 -> Linear -> act -> LayerNorm。

    与 v1 ``GraphNeck`` 的两处有意偏离，都写在 docs/OCA.md §六：

    1. **只留一次 LayerNorm**。老实现每个尺度先过 ``blk`` 尾部的 LN、再被
       ``norms`` 的 LN 过一次，同一个尺度连归一化两遍，第二次只是重参数化，
       白占参数；这里改成一次。
    2. **没有 dropout**。层间正则会让 §八 的消融表解释不清（融合路径加了 dropout
       之后，「算子开关的差值」和「正则强度的差值」混在一起）。

    ``norm`` 放在 act 之后（而不是 Linear 之前）也是同样的理由：与老实现的
    ``Linear -> ReLU -> LayerNorm`` 保持同一顺序，换骨架时不必重调 lr。
    """

    def __init__(self, c1: Union[int, List[int]], c2: int, act: str = 'relu',
                 norm: bool = True):
        super().__init__()
        dims = [c1] if isinstance(c1, int) else list(c1)
        assert len(dims) >= 2, f'Merge 至少要有 2 个输入（收到 {dims}）'
        assert len(set(dims)) == 1, (
            f'Merge 要求各输入同宽，收到 {dims}：不同宽请在结构表里改用 Concat')
        self.sum = sum(dims)                       # 只做展示/诊断用
        self.lin = nn.Linear(dims[0], int(c2))
        self.norm = nn.LayerNorm(int(c2)) if norm else None
        self.act = ACTS[act]
        _tag(self, c2)

    def forward(self, xs: Feats, edge_index: Tensor = None) -> Tensor:
        h = torch.stack(_as_list(xs), 0).sum(0) if isinstance(
            xs, (list, tuple)) else xs
        h = self.act(self.lin(h))
        return h if self.norm is None else self.norm(h)


@register_module('Concat')
class Concat(GraphOp):
    """多输入沿特征维拼接：``out_dim = sum(c1)``。

    节点集恒同，所以直接 ``cat(dim=-1)``，不需要图像侧那套「对齐空间尺寸」的逻辑，
    也不需要按中心重排（本项目是单张全图，不存在 batch 内 topk 不一致的情况）。
    """

    derived_out = True       # 输出宽度由输入推导，args 里没有通道数（builder 据此跳过缩放）
    scale_out = False

    def __init__(self, c1: Union[int, List[int]]):
        super().__init__()
        dims = [c1] if isinstance(c1, int) else list(c1)
        assert len(dims) >= 2, f'Concat 至少要有 2 个输入（收到 {dims}）'
        self.dims, self.sum = dims, sum(dims)
        _tag(self, sum(dims))

    def forward(self, xs: Feats, edge_index: Tensor = None) -> Tensor:
        ts = _as_list(xs)
        assert len(ts) == len(self.dims), \
            f'Concat 声明 {len(self.dims)} 个输入，实际收到 {len(ts)} 个'
        for d, t in zip(self.dims, ts):
            assert t.size(-1) == d, f'拼接宽度对不上：声明 {d}，实际 {t.size(-1)}'
        return torch.cat(ts, -1)
