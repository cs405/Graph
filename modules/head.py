"""读出层（对应 yolov8 的 ``Classify``）：多尺度特征 -> 节点类别 logits。

**多输入自己拼**：``Classify`` 收到 list 就沿特征维 cat，收到单个 tensor 就退化成
普通 MLP 头（基线的 ``- [-1, 1, Classify, [nc]]`` 走的就是这条路）。这样「多尺度读出」
在结构表里是一行而不是两行，代价是 ``Concat`` 与 ``Classify`` 的职责有一点重叠 ——
想显式写就 ``Concat`` 后接单输入 ``Classify``，两者等价。

与 v1 的 ``GraphHead``（骨架代码已删）的偏离：老实现是 ``cat -> Linear(4h->h) -> ReLU
-> Linear(h->h) -> ReLU -> Dropout -> Linear(h->nc)``（两个隐藏层），这里只留一个
隐藏层。多出来的那层对合成数据上的对照几乎没有影响，但会让「换 scale」时参数量
平方级膨胀，所以按 yolov8 的 ``Classify`` 口径收敛成一层。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from modules.base import GraphOp, register_module
from modules.convs import ACT_MODULES, _tag

__all__ = ['Classify']

Feats = Union[Tensor, Sequence[Tensor]]


@register_module('Classify')
class Classify(GraphOp):
    """``Linear(sum(dims) -> hidden) -> act -> Dropout -> Linear(hidden -> nc)``。

    ``hidden=None`` 时取 **最宽尺度的宽度**（老实现的 ``fusion`` 输出正是 ``h``，
    多尺度拼接后就是 ``max(dims)``），故 scale 变宽时头也同步变宽。
    """

    scale_out = False                  # args[0] 是类别数，不是通道数，不参与缩放
    inject = {'dropout': 'dropout'}

    def __init__(self, c1: Union[int, List[int]], nc: int,
                 hidden: Optional[int] = None, dropout: float = 0.5,
                 act: str = 'relu'):
        super().__init__()
        dims = [c1] if isinstance(c1, int) else list(c1)
        assert nc and int(nc) > 0, \
            f'Classify 需要正整数 nc，收到 {nc!r}（结构 YAML 里的 nc 需由数据集注入）'
        self.dims = dims
        self.num_classes = int(nc)
        dim_in, h = sum(dims), int(hidden) if hidden else max(dims)
        self.net = nn.Sequential(
            nn.Linear(dim_in, h), ACT_MODULES[act](), nn.Dropout(float(dropout)),
            nn.Linear(h, self.num_classes))
        _tag(self, self.num_classes)

    def forward(self, x: Feats, edge_index: Tensor = None) -> Tensor:
        if isinstance(x, (list, tuple)):
            x = torch.cat(list(x), -1)
        return self.net(x)
