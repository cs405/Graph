r"""``batch`` 的解包：结构层与数据集之间唯一的一份**边级数据契约**。

结构层（:class:`~modules.dia.conv.DIAConv` / :class:`~modules.dia.head.EdgeScore`）
的前向签名是 ``forward(x, edge_index, batch=None)``；``batch`` 是一个 dict，键与含义：

| 键 | 形状 | 含义 | 缺省 |
| :-- | :-- | :-- | :-- |
| ``rel``       | ``[E]``   | 关系类型 id（选哪一套 :math:`U,V`） | 全 0 |
| ``edge_attr`` | ``[E,d_e]`` | 边特征 :math:`e_{ij}` | 不参与 |
| ``node_type`` | ``[N]``   | 节点类型（:math:`b_{r_i\to r_j}`） | 全 0 |

**为什么对称化不在这里做。** ``to_undirected`` 只搬 ``edge_index``，边级属性得跟着
一起搬，而搬的时候必须回答「反向边是什么关系」：DBLP 的 ``ap`` 反过来是 ``pa``，
ACM 的 ``pap`` 反过来还是 ``pap``。这是数据集的语义，框架不该猜（猜错的后果是
``rel`` 与边错位，前向照跑、指标照降，只是学到的配对矩阵不是你以为的那个）。
所以这里的规则是：**边级属性的长度必须等于 ``edge_index`` 的列数**，不等就报错并
指向 :func:`dataset.relation.make_relation_bundle`（它按关系分别建正反两向，
天然对称且对齐）。同构图没有属性，``symmetrize=True`` 才由这一层代劳。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

__all__ = ['unpack_batch', 'BATCH_KEYS']

BATCH_KEYS = ('rel', 'edge_attr', 'node_type')


def unpack_batch(batch: Optional[Dict[str, Any]], edge_index: Tensor,
                 num_nodes: int, d_edge: int, n_rel: int, n_node_types: int,
                 symmetrize: bool = False, dtype: torch.dtype = torch.float32
                 ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor],
                            Optional[Tensor], Optional[Tensor]]:
    """-> ``(edge_index, rel, edge_attr, node_type_src, node_type_dst)``。

    ``rel``/``node_type_*`` 在数据集没给时返回 ``None``（而不是全零张量）：
    下游的 ``n_rel=1``/``n_node_types=1`` 快路径据此跳过 gather 与偏置查表，
    同构单关系图不为异构能力付一分钱。

    Raises:
        ValueError: 边级属性与边集不同长；或 ``d_edge``/``n_rel`` 与数据对不上。
    """
    ei = edge_index
    rel = _get(batch, 'rel')
    attr = _get(batch, 'edge_attr')
    ntype = _get(batch, 'node_type')

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


def _get(batch: Optional[Dict[str, Any]], key: str) -> Optional[Tensor]:
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
