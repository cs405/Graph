"""图侧通用工具：竞争场边集构造、scatter-add、按中心分组、batch 解包。

这里的东西与具体算法的数学内容无关 —— 任何「在增广边集上按中心分组做计算」的算子
都要用 ``augment_edge_index`` / ``scatter_add`` / ``slots_by_center``；任何需要
解包 ``batch``（rel/edge_attr/node_type）的结构层都用 ``unpack_batch``。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import (add_self_loops, coalesce, degree,
                                   remove_self_loops, to_undirected)

__all__ = ['augment_edge_index', 'scatter_add', 'slots_by_center',
           'field_degrees', 'unpack_batch', 'BATCH_KEYS']


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


# ============================================================================
# batch 解包：结构层与数据集之间唯一的一份边级数据契约
# ============================================================================

BATCH_KEYS = ('rel', 'edge_attr', 'node_type')


def unpack_batch(batch: Optional[Dict[str, Any]], edge_index: Tensor,
                 num_nodes: int, d_edge: int, n_rel: int, n_node_types: int,
                 symmetrize: bool = False, dtype: torch.dtype = torch.float32
                 ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor],
                            Optional[Tensor], Optional[Tensor]]:
    """解包 ``batch`` -> ``(edge_index, rel, edge_attr, node_type_src, node_type_dst)``。

    结构层的前向签名是 ``forward(x, edge_index, batch=None)``；``batch`` 是一个 dict：

    | 键 | 形状 | 含义 | 缺省 |
    | :-- | :-- | :-- | :-- |
    | ``rel``       | ``[E]``   | 关系类型 id | 全 0 |
    | ``edge_attr`` | ``[E,d_e]`` | 边特征 | 不参与 |
    | ``node_type`` | ``[N]``   | 节点类型 | 全 0 |

    ``rel``/``node_type_*`` 在数据集没给时返回 ``None``（而不是全零张量）：
    下游的 ``n_rel=1``/``n_node_types=1`` 快路径据此跳过 gather 与偏置查表。

    **为什么对称化不在这里做。** ``to_undirected`` 只搬 ``edge_index``，边级属性得跟着
    一起搬，而搬的时候必须回答「反向边是什么关系」：这是数据集的语义，框架不该猜。
    所以这里的规则是：**边级属性的长度必须等于 ``edge_index`` 的列数**，不等就报错。

    Raises:
        ValueError: 边级属性与边集不同长；或 ``d_edge``/``n_rel`` 与数据对不上。
    """
    ei = edge_index
    rel = _batch_get(batch, 'rel')
    attr = _batch_get(batch, 'edge_attr')
    ntype = _batch_get(batch, 'node_type')

    if attr is None and rel is None:
        if symmetrize:
            ei, _ = remove_self_loops(ei)
            ei = coalesce(to_undirected(ei, num_nodes=num_nodes),
                          num_nodes=num_nodes)
    else:
        E = ei.size(1)
        if rel is not None and rel.numel() != E:
            raise ValueError(_mismatch('rel', rel.numel(), E, symmetrize))
        if attr is not None and attr.size(0) != E:
            raise ValueError(_mismatch('edge_attr', attr.size(0), E, symmetrize))
        if symmetrize:
            raise ValueError(
                'symmetrize=True 与边级属性同时出现：对称化会把 rel/edge_attr 与边'
                '错位（反向边是什么关系只有数据集知道）。请在数据侧建对称边集'
                '（dataset.relation.make_relation_bundle），再设 dia.symmetrize=false')
    if attr is not None:
        if attr.size(-1) != d_edge:
            raise ValueError(
                f'边特征维度对不上：数据给的是 {attr.size(-1)}，dia.d_edge={d_edge}。'
                f'改 cfg/models/*.yaml 的 dia.d_edge（或关掉边特征）')
        attr = attr.to(dtype)
    if rel is not None:
        rel = rel.long()
        if int(rel.max()) >= n_rel:
            raise ValueError(
                f'关系类型 id 最大 {int(rel.max())}，但 dia.n_rel={n_rel}：'
                f'配对矩阵的数量不够，多出来的关系会静默套错 U/V')
    if ntype is not None:
        ntype = ntype.long()
        if ntype.numel() != num_nodes:
            raise ValueError(f'node_type 长度 {ntype.numel()} != 节点数 {num_nodes}')
        if int(ntype.max()) >= n_node_types:
            raise ValueError(
                f'节点类型 id 最大 {int(ntype.max())}，但 dia.n_node_types='
                f'{n_node_types}：类型偏置 b_(r_i->r_j) 会索引越界')
    r_i = r_j = None
    if ntype is not None:
        r_i, r_j = ntype[ei[0]], ntype[ei[1]]
    return ei, rel, attr, r_i, r_j


def _batch_get(batch: Optional[Dict[str, Any]], key: str) -> Optional[Tensor]:
    if not batch:
        return None
    unknown = set(batch) - set(BATCH_KEYS)
    if unknown:
        raise ValueError(
            f'batch 里有本层不认识的键 {sorted(unknown)}（认识的是 {list(BATCH_KEYS)}）；'
            f'拼错键名的后果是这一路信息静默不参与计算')
    return batch.get(key)


def _mismatch(what: str, got: int, want: int, symmetrize: bool) -> str:
    hint = ('（symmetrize=True 时框架会改边集，边级属性必须由数据集一起建对称）'
            if symmetrize else '（边集与边级属性必须同长）')
    return f'{what} 的长度 {got} != 边数 {want}{hint}'
