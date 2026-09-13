"""读出层（对应 yolov8 的 ``Classify``）：特征 -> logits。

节点级读出（:class:`Classify`）与边级读出（:class:`EdgeScore`）集中在这里，
与 ``modules/convs.py`` 的结构层对称：框架侧只 import 本文件就能拿到全部读出头。

**多输入自己拼**：``Classify`` 收到 list 就沿特征维 cat，收到单个 tensor 就退化成
普通 MLP 头（基线的 ``- [-1, 1, Classify, [nc]]`` 走的就是这条路）。这样「多尺度读出」
在结构表里是一行而不是两行。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import degree

from modules.base import GraphOp, register_module
from modules.convs import ACT_MODULES, _tag

__all__ = ['Classify', 'EdgeScore']

Feats = Union[Tensor, Sequence[Tensor]]


# ============================================================================
# 节点级读出
# ============================================================================


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


# ============================================================================
# 边级读出
# ============================================================================


@register_module('EdgeScore')
class EdgeScore(GraphOp):
    r"""边级读出头：``[N, dim_in]`` 节点嵌入 + 边集 -> ``[E, nc]`` 边 logits。

    节点分类的 ``Classify`` 吃 :math:`h_i` 出 :math:`[N,nc]`；边分类要的是**一条边**的
    logit，而 DIA 的边强度 :math:`s_{ij}` 本来就是边级的量，于是这一头做三件事：

    1. 两端各自投影到同一套「维度槽」（:math:`d`），投影是**两条独立分支**；
    2. 跑 ``dia.n_layers`` 层 :class:`~modules.dia.DIALayer`，层间按 §2.5 更新
       节点嵌入；
    3. 读出 ``MLP([H_i ; H_j ; s_{ij}]) -> nc``。

    第 3 步把 :math:`s_{ij}` 显式拼进去是刻意的：它是**可解释的那个标量**，拼进去
    之后「logit 里有多少来自可解释量」是可以直接量出来的
    （:func:`modules.dia.score_attribution`）。

    结构表约定：``EdgeScore`` 必须是**最后一行**。
    """

    scale_out = False                  # args[0] 是类别数，不是通道数，不参与缩放
    takes_batch = True
    level = 'edge'
    inject = {'dia': 'dia', 'dropout': 'dropout'}

    def __init__(self, c1: Union[int, List[int]], nc: int,
                 dia: Optional[Dict[str, Any]] = None, dropout: float = 0.5,
                 act: str = 'relu'):
        super().__init__()
        from modules.dia import DIAConfig, DIALayer, K_SCORE
        dims = [c1] if isinstance(c1, int) else list(c1)
        assert nc and int(nc) > 0, \
            f'EdgeScore 需要正整数 nc，收到 {nc!r}（结构 YAML 里的 nc 需由数据集注入）'
        cfg = DIAConfig(**(dia or {}))
        dim_in = sum(dims)
        if not cfg.use_projection:
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

        只给 :func:`modules.dia.score_attribution` 用。
        """
        from modules.dia import K_SCORE
        from utils.graph import scatter_add, unpack_batch
        assert edge_index is not None, 'EdgeScore 需要 edge_index'
        if isinstance(x, (list, tuple)):
            x = torch.cat(list(x), -1)
        cfg = self.cfg
        N = x.size(0)
        ei, rel, attr, r_i, r_j = unpack_batch(
            batch, edge_index, N, cfg.d_edge, cfg.n_rel, cfg.n_node_types,
            symmetrize=False, dtype=x.dtype)
        src, dst = ei[0], ei[1]
        deg = degree(src, num_nodes=N, dtype=x.dtype).clamp(min=1.0).unsqueeze(-1)
        deg_in = degree(dst, num_nodes=N, dtype=x.dtype).clamp(min=1.0).unsqueeze(-1)

        HI, HJ = self.proj_i(x), self.proj_j(x)

        L = len(self.layers)
        out: Dict[str, Tensor] = {}
        for t, layer in enumerate(self.layers):
            out = layer(HI[src], HJ[dst], attr, rel, r_i, r_j, src=src, dst=dst,
                        want=('score', 'message') if t < L - 1 else ('score',))
            if t < L - 1:
                agg_i = scatter_add(out['m_ij'], src, N) / deg
                agg_j = scatter_add(out['m_ji'], dst, N) / deg_in
                HI = self.norms[t](HI + self.act(agg_i))
                HJ = self.norms[t](HJ + self.act(agg_j))
        score = out[K_SCORE]
        if zero_score:
            score = torch.zeros_like(score)
        return self.net(torch.cat([HI[src], HJ[dst], score.unsqueeze(-1)], -1))
