"""结构层（对应 yolov8 的 ``modules/conv.py`` 一类）：把算子/基线包装成
「有输入输出宽度、能被 cfg/models/*.yaml 引用」的节点。

三条硬约定，model_builder 依赖它们（全部写在 :class:`modules.base.GraphOp`
的类属性上，builder 不再维护「认识哪个类」的表）：

1. 第一个构造参数恒为 ``c1``（输入宽度），第二个恒为 ``c2``（输出宽度），
   builder 负责按 scale 的 width_multiple 缩放后填进来；
2. ``forward(x, edge_index)`` —— 所有结构层同签名，``LinearBlock`` 只是不用图；
   需要边级侧信息的层声明 ``takes_batch=True``，builder 会把 ``batch`` 一并递下去；
3. 暴露 ``.out_dim``，让 builder 能逐层推算下一层的 ``c1``（重复堆叠的
   ``Repeat`` 也能取到，因为它本身带 ``out_dim`` 属性）。

不写 ``n_parameters`` 属性就没法在结构表里报每层参数量，所以三个包装类都带。

注册名与类名可以不同（``GATConv`` ↔ :class:`GATBlock`）：YAML 用 PyG 风味的
名字，是为了让基线行读起来就是「这里是一层 GAT」；``m.type`` 存类名，日志与
``test_model_builder`` 对的是类名。两边不一致是故意的，故 ``@register_module``
里显式写了名字。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, GCNConv

from modules.base import GraphOp, register_module
from modules.oca import OCAConfig, OCALayer

__all__ = ['OCAConv', 'GATBlock', 'GCNBlock', 'LinearBlock', 'ACTS',
           'ACT_MODULES']

ACTS = {'relu': F.relu, 'elu': F.elu, 'gelu': F.gelu, 'none': lambda x: x}
# 同上，但给想要 ``nn.Sequential`` 的模块用（函数不是 Module，装不进去）
ACT_MODULES = {'relu': nn.ReLU, 'elu': nn.ELU, 'gelu': nn.GELU,
               'none': nn.Identity}


def _tag(m: nn.Module, out_dim: int) -> nn.Module:
    m.out_dim = int(out_dim)                       # type: ignore[attr-defined]
    m.n_parameters = sum(p.numel() for p in m.parameters())  # type: ignore[attr-defined]
    return m


@register_module('OCAConv')
class OCAConv(GraphOp):
    """:class:`OCALayer` 的 cfg 化包装：``oca`` 字典 = 该层的算子超参。

    全局默认来自结构 YAML 的 ``oca:`` 块，``build_model`` 会把 train cfg / CLI 的
    覆盖项合并进来再逐层传入 —— 因此「同一网络里每层用不同 T」这种消融是可以直接
    在 YAML 里按行写的（args 里放一个 dict 即可）。
    """

    inject = {'oca': 'oca'}

    def __init__(self, c1: int, c2: int, oca: Optional[Dict[str, Any]] = None):
        super().__init__()
        kwargs: Dict[str, Any] = {k: v for k, v in (oca or {}).items()
                                  if k != 'out_dim'}
        # out_dim 由结构表决定（要被 width 缩放），不允许 oca 块偷偷覆盖
        self.cfg = OCAConfig(out_dim=int(c2), **kwargs)
        self.layer = OCALayer(int(c1), self.cfg)
        _tag(self, c2)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.layer(x, edge_index)


@register_module('GATConv')
class GATBlock(GraphOp):
    """标准 GAT（PyG GATConv）。

    ``add_self_loops=False`` 是写死的：OCA 的中心槽位不进聚合，GAT 若开自环，
    两边的邻域口径就不一致，差值会被误读成算子差异。
    """

    inject = {'dropout': 'dropout'}

    def __init__(self, c1: int, c2: int, heads: int = 8, dropout: float = 0.5,
                 act: str = 'elu'):
        super().__init__()
        assert c2 % heads == 0, \
            f'GAT 的 out_dim({c2}) 需能被 heads({heads}) 整除（换 scale 或 heads）'
        self.conv = GATConv(int(c1), int(c2) // heads, heads=heads,
                            add_self_loops=False)
        self.dropout, self.act = dropout, ACTS[act]
        _tag(self, c2)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = self.act(self.conv(x, edge_index))
        return F.dropout(h, p=self.dropout, training=self.training)


@register_module('GCNConv')
class GCNBlock(GraphOp):

    inject = {'dropout': 'dropout'}

    def __init__(self, c1: int, c2: int, dropout: float = 0.5, act: str = 'relu'):
        super().__init__()
        self.conv = GCNConv(int(c1), int(c2))
        self.dropout, self.act = dropout, ACTS[act]
        _tag(self, c2)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = self.act(self.conv(x, edge_index))
        return F.dropout(h, p=self.dropout, training=self.training)


@register_module('Linear')
class LinearBlock(GraphOp):
    """无图结构基线里的一层 MLP；照样吃 edge_index（忽略）以便统一路由。"""

    inject = {'dropout': 'dropout'}

    def __init__(self, c1: int, c2: int, dropout: float = 0.5, act: str = 'relu'):
        super().__init__()
        self.lin = nn.Linear(int(c1), int(c2))
        self.dropout, self.act = dropout, ACTS[act]
        _tag(self, c2)

    def forward(self, x: Tensor, edge_index: Tensor = None) -> Tensor:
        h = self.act(self.lin(x))
        return F.dropout(h, p=self.dropout, training=self.training)
