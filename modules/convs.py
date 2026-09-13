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

注册名与类名可以不同（``GATConv`` ↔ :class:`GATBlock`）：YAML 用 PyG 风味的
名字，是为了让基线行读起来就是「这里是一层 GAT」；``m.type`` 存类名，日志与
``test_model_builder`` 对的是类名。两边不一致是故意的，故 ``@register_module``
里显式写了名字。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import degree

from modules.base import GraphOp, register_module
from modules.oca import OCAConfig, OCALayer

__all__ = ['OCAConv', 'GATBlock', 'GCNBlock', 'LinearBlock', 'DIAConv',
           'ACTS', 'ACT_MODULES']

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


# ============================================================================
# DIA 消息传递层
# ============================================================================


@register_module('DIAConv')
class DIAConv(GraphOp):
    r"""DIA 消息传递层（技术文档 §2.5）。``c1`` 是输入宽度，``c2`` 是输出宽度。

    .. math::

        h_i' = h_i + \sigma\Big(\mathrm{mean}_{j\in\mathcal N(i)} m_{ij}\Big),\qquad
        m_{ij}=\tilde h_{i|j}\odot U\big(V^\top\tilde h_{j|i}\big)

    与 :class:`~modules.dia.DIALayer` 相比，这一层多做的只有**聚合与融合**，DIA 的
    数学全在 ``self.layer`` 里。这样分是为了让边分类的 ``EdgeScore`` 能复用同一份
    三层实现而不带聚合。

    两处刻意的偏离：

    1. 融合用 ``norm(skip(x) + out_mlp([h_i ; mean_j m_ij]))``（OCA 风格），
       而不是裸残差：结构表要支持 ``scale=n/l/x``（宽度会变），裸残差要求
       ``c1 == out_dim``，宽度一缩放就崩。
    2. 聚合前**去掉自环**。DIA 的 :math:`\gamma` 本来就能表达「这条边没用」，
       自环只会让 :math:`m_{ii}` 恒等于自己，把 mean 稀释掉。
    """

    takes_batch = True                 # 前向还要吃 batch（rel/edge_attr/node_type）
    inject = {'dia': 'dia', 'dropout': 'dropout'}
    level = 'node'

    def __init__(self, c1: int, c2: int, dia: Optional[Dict[str, Any]] = None,
                 dropout: float = 0.5):
        super().__init__()
        from modules.dia import DIAConfig, DIALayer
        from utils.graph import scatter_add, unpack_batch
        # out_dim 由结构表的 c2 决定（要被 width 缩放），dia 块里写了也一律覆盖
        cfg = DIAConfig(**(dia or {})).replaced(out_dim=int(c2))
        if not cfg.use_projection and int(c1) != int(c2):
            raise ValueError(
                f'dia.use_projection=false 要求 c1 == c2（U 的行才是原始特征维度），'
                f'收到 c1={c1}, c2={c2}；要么让这一行不改宽度，要么打开投影')
        self.cfg = cfg
        d = int(c2)
        self.proj_i = nn.Linear(int(c1), d) if cfg.use_projection else nn.Identity()
        self.proj_j = nn.Linear(int(c1), d) if cfg.use_projection else nn.Identity()
        self.layer = DIALayer(d, d, cfg)
        self.out_mlp = nn.Sequential(
            nn.Linear(2 * d, 2 * d), nn.ELU(), nn.Dropout(float(dropout)),
            nn.Linear(2 * d, d))
        self.skip = nn.Linear(int(c1), d) if int(c1) != d else nn.Identity()
        self.norm = nn.LayerNorm(d)
        _tag(self, d)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None,
                batch: Optional[Dict[str, Any]] = None) -> Tensor:
        from utils.graph import scatter_add, unpack_batch
        assert edge_index is not None, 'DIAConv 需要 edge_index'
        cfg = self.cfg
        N = x.size(0)
        ei, rel, attr, r_i, r_j = unpack_batch(
            batch, edge_index, N, cfg.d_edge, cfg.n_rel, cfg.n_node_types,
            symmetrize=cfg.symmetrize, dtype=x.dtype)
        if not cfg.symmetrize:
            keep = ei[0] != ei[1]
            if not bool(keep.all()):
                ei = ei[:, keep]
                if rel is not None:
                    rel = rel[keep]
                if attr is not None:
                    attr = attr[keep]
                if r_i is not None:
                    r_i, r_j = r_i[keep], r_j[keep]
        src, dst = ei[0], ei[1]

        h_i, h_j = self.proj_i(x), self.proj_j(x)
        out = self.layer(h_i[src], h_j[dst], attr, rel, r_i, r_j,
                         src=src, dst=dst, want=('score', 'message'))
        deg = degree(src, num_nodes=N, dtype=x.dtype).clamp(min=1.0).unsqueeze(-1)
        agg = scatter_add(out['m_ij'], src, N) / deg
        fused = self.out_mlp(torch.cat([h_i, agg], dim=-1))
        y = fused if not cfg.residual else self.skip(x) + fused
        return self.norm(y)
