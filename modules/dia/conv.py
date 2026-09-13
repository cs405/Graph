r"""DIA 的消息传递层（技术文档 §2.5）：``cfg/models/*.yaml`` 里写 ``DIAConv``。

.. math::

    h_i' = h_i + \sigma\Big(\mathrm{mean}_{j\in\mathcal N(i)} m_{ij}\Big),\qquad
    m_{ij}=\tilde h_{i|j}\odot U\big(V^\top\tilde h_{j|i}\big)

与文档 §4.4 的 ``DIALayer`` 相比，这一层多做的只有**聚合与融合**，DIA 的数学全在
:class:`~modules.dia.layer.DIALayer` 里（``self.layer``）。这样分是为了让
「打分头」（:class:`~modules.dia.head.EdgeScore`）能复用同一份三层实现而不带聚合 ——
边分类要的是 :math:`s_{ij}`，不是 :math:`h_i'`。

两处刻意的偏离，都在 docstring 里点名：

1. 融合不是 §2.5 的裸 ``h_i + sigma(mean m)``，而是 OCA 那套
   ``norm(skip(x) + out_mlp([h_i ; mean_j m_ij]))``。原因：结构表要支持
   ``scale=n/l/x``（宽度会变），裸残差要求 ``c1 == out_dim``，宽度一缩放就崩；
   而且拼接 ``h_i`` 让「不传消息」这条路（``agg=0``）也能表达自身特征。
   ``dia.residual=false`` 时退化成无 skip 的纯 MLP，用来量残差值多少分。
2. 聚合前**去掉自环**。OCA 需要自环（它的竞争场显式含中心
   :math:`\\mathcal V_i=\\mathcal N_i\\cup\\{i\\}`）；DIA 的 :math:`\\gamma` 本来就能表达
   「这条边没用」，自环只会让 :math:`m_{ii}` 恒等于自己，把 mean 稀释掉。
   两条路径都去：``symmetrize=true`` 时在 :func:`unpack_batch` 里连同
   ``to_undirected`` 一起做；``symmetrize=false`` 时在这一层做，**并且必须同时裁
   ``rel``/``edge_attr``** —— 只裁 ``edge_index`` 的话边级属性会整体错位一位，
   前向照跑、指标照降，错得无声无息。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import degree

from modules.base import GraphOp, register_module
from modules.convs import _tag
from modules.dia.config import DIAConfig
from modules.dia.edges import unpack_batch
from modules.dia.layer import DIALayer
from utils.graph import scatter_add

__all__ = ['DIAConv']


@register_module('DIAConv')
class DIAConv(GraphOp):
    """DIA 消息传递层。``c1`` 是输入宽度，``c2`` 是输出宽度（按 scale 缩放）。"""

    takes_batch = True                 # 前向还要吃 batch（rel/edge_attr/node_type）
    inject = {'dia': 'dia', 'dropout': 'dropout'}
    level = 'node'

    def __init__(self, c1: int, c2: int, dia: Optional[Dict[str, Any]] = None,
                 dropout: float = 0.5):
        super().__init__()
        # out_dim 由结构表的 c2 决定（要被 width 缩放），dia 块里写了也一律覆盖
        # —— 与 OCAConv 对 'out_dim' 的处理一致，免得「scale 改了宽度但层没跟着改」
        cfg = DIAConfig(**(dia or {})).replaced(out_dim=int(c2))
        if not cfg.use_projection and int(c1) != int(c2):
            # use_projection=False 是可解释模式：U 的行就是原始特征维度，
            # 于是「维度槽」与「特征维度」必须是同一套编号，宽度不能变。
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
        assert edge_index is not None, 'DIAConv 需要 edge_index'
        cfg = self.cfg
        N = x.size(0)
        ei, rel, attr, r_i, r_j = unpack_batch(
            batch, edge_index, N, cfg.d_edge, cfg.n_rel, cfg.n_node_types,
            symmetrize=cfg.symmetrize, dtype=x.dtype)
        if not cfg.symmetrize:
            # 数据侧给的边集可能含自环；这里去掉（理由见模块 docstring 第 2 条）。
            # 边级属性与两端类型都由 unpack_batch 保证与 edge_index 同长，必须一起裁。
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
        agg = scatter_add(out['m_ij'], src, N) / deg          # mean_j m_ij
        fused = self.out_mlp(torch.cat([h_i, agg], dim=-1))
        y = fused if not cfg.residual else self.skip(x) + fused
        return self.norm(y)
