"""合成图：给测试和「离线跑不通真实数据集」时的 fallback 用。

两类用途分开：

* :func:`ba_graph` / :func:`dup_directed_graph` / :func:`star_graph` —— 只要
  ``(x, edge_index)``，供 ``test/`` 校验算子的数值性质（BA 天然带 hub，正好卡
  「无度数截断」与稳定性）。
* :func:`hetero_bundle` —— 带标签、可控同质性的可训练图，供 ``training`` 的
  端到端冒烟测试与 λ-vs-同质性 的**受控**验证。

.. warning::
   合成图上的准确率**不能**当作论文卖点。它的特征是按类中心 + 高斯噪声造的，
   MLP 就能刷很高；它只能说明「管线跑通、趋势方向对」，不说明方法在真实数据上有效。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import (barabasi_albert_graph, coalesce,
                                   remove_self_loops, to_undirected)

from dataset.base import GraphBundle
from dataset.splits import random_split
from metrics.homophily import edge_homophily

__all__ = ['ba_graph', 'dup_directed_graph', 'star_graph', 'hetero_bundle']


# ---------------------------------------------------------------------------
# 无标签：算子数值性质用
# ---------------------------------------------------------------------------


def ba_graph(n: int = 97, m: int = 3, seed: int = 1,
             feat_dim: int = 12) -> Tuple[Tensor, Tensor]:
    """BA 无标度图（天然带 hub）。

    必须同时钉住 numpy 的种子：PyG 的 ``barabasi_albert_graph`` 内部用
    ``np.random.choice`` 选挂载点（只有 ``torch.randperm`` 走 torch），
    只调 ``torch.manual_seed`` 得到的是一张张都不一样的“可复现”图。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    ei = barabasi_albert_graph(n, m)                  # 已 to_undirected
    ei, _ = remove_self_loops(ei)
    ei = coalesce(ei, num_nodes=n)
    return torch.randn(n, feat_dim), ei


def dup_directed_graph(n: int = 40, seed: int = 2,
                       feat_dim: int = 8) -> Tuple[Tensor, Tensor]:
    """含自环 + 重边 + 单向边：验证入口归一化（原实现会静默算错）。"""
    torch.manual_seed(seed)
    src = torch.randint(0, n, (200,))
    dst = torch.randint(0, n, (200,))
    ei = torch.stack([src, dst])
    ei = torch.cat([ei, ei[:, :30], torch.stack([src[:20], src[:20]])], 1)
    return torch.randn(n, feat_dim), ei


def star_graph(spokes: int = 2000, seed: int = 0,
               feat_dim: int = 8) -> Tuple[Tensor, Tensor]:
    """单中心极端 hub：最坏情形的稳定性压力测试。"""
    torch.manual_seed(seed)
    idx = torch.arange(1, spokes + 1)
    ei = torch.stack([torch.cat([idx, torch.zeros_like(idx)]),
                      torch.cat([torch.zeros_like(idx), idx])])
    n = spokes + 1
    return torch.randn(n, feat_dim), ei


# ---------------------------------------------------------------------------
# 带标签：端到端训练用
# ---------------------------------------------------------------------------


def hetero_bundle(num_classes: int = 5, n_per_class: int = 60,
                  avg_deg: int = 8, homophily: float = 0.3,
                  feat_dim: int = 24, sep: float = 1.5,
                  train_per_class: Optional[int] = None,
                  val_per_class: Optional[int] = None,
                  seed: int = 0, name: str = 'synth') -> GraphBundle:
    r"""可控边同质性的合成节点分类图。

    Args:
        homophily: 目标边同质性 :math:`h`。每条边先决定「同类/异类」再随机取端点，
            去重后实测值会略偏离目标（写进 ``meta['edge_homophily']``，报结果用实测）。
        sep: 类中心间的欧氏距离尺度，控制特征可分性（与 :math:`h` 独立调节）。
        train_per_class: 每类训练节点数，默认取 :code:`n_per_class // 3`（剩下均分给
            val/test）—— 写死 20 会在小图（每类 40 个）上直接切不出划分。
    """
    train_per_class = n_per_class // 3 if train_per_class is None \
        else train_per_class
    val_per_class = train_per_class if val_per_class is None else val_per_class
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n = num_classes * n_per_class
    y = torch.arange(num_classes).repeat_interleave(n_per_class)

    centroids = torch.randn(num_classes, feat_dim) * sep
    x = centroids[y] + torch.randn(n, feat_dim)

    by_class = [(y == c).nonzero().view(-1) for c in range(num_classes)]
    others = [torch.cat([by_class[c] for c in range(num_classes)
                         if c != k]) for k in range(num_classes)]
    pairs = set()
    for i in range(n):
        own = int(y[i])
        for _ in range(avg_deg):
            # 先决定「同类/异类」再取端点：这样实测 h 才能贴近目标值
            pool = by_class[own] if torch.rand(1, generator=g).item() < homophily \
                else others[own]
            j = int(pool[torch.randint(pool.numel(), (1,), generator=g)].item())
            if j != i:
                pairs.add((min(i, j), max(i, j)))
    ei = torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()
    ei = to_undirected(ei, num_nodes=n)
    ei = coalesce(ei, num_nodes=n)

    masks = random_split(y, num_classes, train_per_class, val_per_class,
                         seed=seed)
    ds = GraphBundle(name=name, x=x, edge_index=ei, y=y, num_classes=num_classes,
                     source='synthetic', train_mask=masks['train'],
                     val_mask=masks['val'], test_mask=masks['test'],
                     meta={'target_homophily': f'{homophily:.3f}',
                           'avg_deg': str(avg_deg), 'sep': str(sep)})
    ds.meta['edge_homophily'] = f'{edge_homophily(ei, y):.4f}'
    return ds
