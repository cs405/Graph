r"""边级读出头（对应 yolov8 的 decoupled head，这里是 ``EdgeScore``）。

节点分类的 ``Classify`` 吃 :math:`h_i` 出 :math:`[N,nc]`；边分类要的是**一条边**的
logit，而 DIA 的边强度 :math:`s_{ij}` 本来就是边级的量，于是这一头做三件事：

1. 两端各自投影到同一套「维度槽」（:math:`d`），投影是**两条独立分支**——
   共享一条就等于假设 :math:`i` 与 :math:`j` 在同一空间里可比，异构图上不成立；
2. 跑 ``dia.n_layers`` 层 :class:`~modules.dia.layer.DIALayer`，层间按 §2.5 更新
   节点嵌入（:math:`H\leftarrow H+\sigma(\mathrm{mean}_j m_{ij})`，这里两端同维，
   所以能像文档那样直接相加，不必套 ``DIAConv`` 的 ``out_mlp``）；
3. 读出 ``MLP([H_i ; H_j ; s_{ij}]) -> nc``。

第 3 步把 :math:`s_{ij}` 显式拼进去是刻意的：它是**可解释的那个标量**
（由 :math:`U,V,\gamma,\alpha` 完全决定），拼进去之后「logit 里有多少来自可解释量」
是可以直接量出来的（``explain.py::score_attribution``），而不是只能相信它有用。

结构表约定：``EdgeScore`` 必须是**最后一行**。它的输出是 ``[E, nc]``（边级），
后面接任何节点级层都会静默错位；``GraphSequential.level`` 会把它标成 ``'edge'``，
``model.py`` 拿这个与数据集的 ``supervision`` 对齐（对不上就报错，不猜）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import degree

from modules.base import GraphOp, register_module
from modules.convs import ACT_MODULES, _tag
from modules.dia.config import DIAConfig
from modules.dia.edges import unpack_batch
from modules.dia.layer import DIALayer, K_SCORE
from utils.graph import scatter_add

__all__ = ['EdgeScore']

Feats = Union[Tensor, Sequence[Tensor]]


@register_module('EdgeScore')
class EdgeScore(GraphOp):
    """``[N, dim_in]`` 节点嵌入 + 边集 -> ``[E, nc]`` 边 logits。"""

    scale_out = False                  # args[0] 是类别数，不是通道数，不参与缩放
    takes_batch = True
    level = 'edge'
    inject = {'dia': 'dia', 'dropout': 'dropout'}

    def __init__(self, c1: Union[int, List[int]], nc: int,
                 dia: Optional[Dict[str, Any]] = None, dropout: float = 0.5,
                 act: str = 'relu'):
        super().__init__()
        dims = [c1] if isinstance(c1, int) else list(c1)
        assert nc and int(nc) > 0, \
            f'EdgeScore 需要正整数 nc，收到 {nc!r}（结构 YAML 里的 nc 需由数据集注入）'
        cfg = DIAConfig(**(dia or {}))
        dim_in = sum(dims)
        if not cfg.use_projection:
            # 可解释模式：不建投影分支，维度槽就是原始特征维度（U 的行可直接读）
            d = dim_in
        else:
            d = int(cfg.out_dim)
        cfg = cfg.replaced(out_dim=d)
        self.cfg, self.dims, self.num_classes = cfg, dims, int(nc)
        self.act = ACT_MODULES[act]()
        self.proj_i = nn.Linear(dim_in, d) if cfg.use_projection else nn.Identity()
        self.proj_j = nn.Linear(dim_in, d) if cfg.use_projection else nn.Identity()
        self.layers = nn.ModuleList([DIALayer(d, d, cfg) for _ in range(cfg.n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(cfg.n_layers - 1)])
        self.net = nn.Sequential(
            nn.Linear(2 * d + 1, d), self.act, nn.Dropout(float(dropout)),
            nn.Linear(d, self.num_classes))
        _tag(self, self.num_classes)

    # ------------------------------------------------------------------
    def forward(self, x: Feats, edge_index: Optional[Tensor] = None,
                batch: Optional[Dict[str, Any]] = None) -> Tensor:
        return self.forward_with(x, edge_index, batch)

    def forward_with(self, x: Feats, edge_index: Optional[Tensor] = None,
                     batch: Optional[Dict[str, Any]] = None,
                     zero_score: bool = False) -> Tensor:
        """``zero_score=True`` 时把读出里的 :math:`s_{ij}` 那一列置零。

        只给 :func:`modules.dia.explain.score_attribution` 用：它要回答「logit 里
        有多少来自可解释量」。做成参数而不是模块开关，是因为开关会在两次前向之间
        漏下来（训练时忘复位 = 静默地把可解释量关了）。
        """
        assert edge_index is not None, 'EdgeScore 需要 edge_index'
        if isinstance(x, (list, tuple)):
            x = torch.cat(list(x), -1)
        cfg = self.cfg
        N = x.size(0)
        ei, rel, attr, r_i, r_j = unpack_batch(
            batch, edge_index, N, cfg.d_edge, cfg.n_rel, cfg.n_node_types,
            symmetrize=False, dtype=x.dtype)
        src, dst = ei[0], ei[1]
        # 两个方向的均值各用自己的分母：m_ij 按 src 聚合（出度），m_ji 按 dst
        # 聚合（入度）。对称边集下两者相等，不对称时写成一个就会错归一化。
        deg = degree(src, num_nodes=N, dtype=x.dtype).clamp(min=1.0).unsqueeze(-1)
        deg_in = degree(dst, num_nodes=N, dtype=x.dtype).clamp(min=1.0).unsqueeze(-1)

        # 两端各自的投影（docstring 第 1 条）：两条分支、两个节点级空间，
        # HI[i] 是 i 作为主体时的表示，HJ[j] 是 j 作为邻居时的表示。
        # 可解释模式（use_projection=false）下两者都是原始特征本身。
        HI, HJ = self.proj_i(x), self.proj_j(x)

        L = len(self.layers)
        out: Dict[str, Tensor] = {}
        for t, layer in enumerate(self.layers):
            out = layer(HI[src], HJ[dst], attr, rel, r_i, r_j, src=src, dst=dst,
                        want=('score', 'message') if t < L - 1 else ('score',))
            if t < L - 1:
                # §2.5 的节点更新，但**各更新各的空间**：m_ij 落在 i 的空间、
                # m_ji 落在 j 的空间。把两者平均成一个节点级载体是最省事的写法，
                # 但那等于把两条分支又缝回一条，第 1 条的解耦就白说了。
                agg_i = scatter_add(out['m_ij'], src, N) / deg
                agg_j = scatter_add(out['m_ji'], dst, N) / deg_in
                HI = self.norms[t](HI + self.act(agg_i))
                HJ = self.norms[t](HJ + self.act(agg_j))
        score = out[K_SCORE]
        if zero_score:
            score = torch.zeros_like(score)
        return self.net(torch.cat([HI[src], HJ[dst], score.unsqueeze(-1)], -1))
