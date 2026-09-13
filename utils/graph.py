"""图侧通用工具：竞争场边集构造、scatter-add、按中心分组。

这里的东西与 OCA 的数学内容无关 —— 任何「在增广边集上按中心分组做计算」的算子
都要用，故放 utils，不与 modules 耦合。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import (add_self_loops, coalesce, degree,
                                   remove_self_loops, to_undirected)

__all__ = ['augment_edge_index', 'scatter_add', 'slots_by_center',
           'field_degrees']


def augment_edge_index(edge_index: Tensor, num_nodes: int,
                       symmetrize: bool = True) -> Tuple[Tensor, Tensor]:
    r"""构造竞争场边集 :math:`\hat{\mathcal{E}}=\mathcal{E}\cup\{(i,i)\}`。

    先 ``remove_self_loops`` 再 ``add_self_loops`` 是必须的：
    ``add_self_loops`` 不会为已有自环去重，直接加会重复计数。

    约定 ``row = 中心 i``、``col = 场内槽位 a``（与 PyG ``MessagePassing``
    默认的「col 为聚合目标」相反）。

    Returns:
        edge_index_hat: [2, E_hat]
        is_self: [E_hat] bool，标记该槽位即中心本身
    """
    ei, _ = remove_self_loops(edge_index)
    if symmetrize:
        ei = to_undirected(ei, num_nodes=num_nodes)
    ei = coalesce(ei, num_nodes=num_nodes)
    ei, _ = add_self_loops(ei, num_nodes=num_nodes)
    return ei, ei[0] == ei[1]


def scatter_add(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """dim=0 的 scatter_add（用 ``index_add_``，不依赖 torch_scatter）。"""
    return src.new_zeros((dim_size,) + tuple(src.shape[1:])).index_add_(
        0, index, src)


def field_degrees(edge_index_hat: Tensor, num_nodes: int,
                  dtype: torch.dtype = None) -> Tensor:
    """每个竞争场的槽位数 :math:`|\\mathcal{V}_i| = \\deg_i + 1`，形状 [N, 1]。"""
    d = degree(edge_index_hat[0], num_nodes=num_nodes, dtype=dtype)
    return d.unsqueeze(-1)


def slots_by_center(edge_index_hat: Tensor, is_self: Tensor, num_nodes: int,
                    center_first: bool = True) -> List[Tensor]:
    """把边按中心 ``row`` 分组，返回每个中心的槽位下标列表。

    ``center_first=True`` 时把中心槽位置于首位 —— 稠密参考实现与分析代码用它
    显式建 :math:`|\\mathcal{V}_i|\\times|\\mathcal{V}_i|` 矩阵，槽位序必须与
    ``test/support.py::field_kernel`` 的独立重建一致，否则比对无意义。
    """
    src = edge_index_hat[0]
    cnt = degree(src, num_nodes=num_nodes, dtype=torch.long)
    ptr = torch.cat([cnt.new_zeros(1), cnt.cumsum(0)])
    order = torch.argsort(src, stable=True)
    out: List[Tensor] = []
    for i in range(num_nodes):
        g = order[ptr[i]:ptr[i + 1]]
        if center_first and g.numel() > 0:
            g = torch.cat([g[is_self[g]], g[~is_self[g]]])   # 中心在前
        out.append(g)
    return out
