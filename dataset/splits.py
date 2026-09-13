"""节点分类的 train/val/test 划分。

两条路径：
1. 数据集自带 split（Planetoid public、WebKB 的 10 折）—— 优先用，否则与文献不可比；
2. :func:`random_split` —— 每类固定 ``train_per_class`` 个，异配数据集通用做法。

随机划分必须**按类分层**且**种子显式**：不控制种子时，同一份代码两次跑出的
val 曲线会差过 1 个点，早停挑到的模型也随之不同，任何「OCA 比 GAT 高 0.5」的
结论都会在这种噪声下失去意义。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

__all__ = ['as_bool_mask', 'random_split', 'split_from_column']


def as_bool_mask(m: Tensor, num_nodes: int, name: str = '') -> Tensor:
    """把 bool / 0-1 long 统一成 [N] bool。

    PyG 不同数据集的 mask dtype 不一致（bool 与 long 混用），早期实现里
    ``mask.sum()`` 在 long 上得到计数、在 bool 上被当逻辑或，是个静默差异源。
    """
    if m.dtype != torch.bool:
        m = m.bool()
    assert m.numel() == num_nodes, f'{name}: mask 长度 {m.numel()} != N {num_nodes}'
    return m


def split_from_column(col: Tensor, num_nodes: int) -> Dict[str, Tensor]:
    """WebKG/Geom-GCN 风格：一列里 0=train, 1=val, 2=test。"""
    col = col.long()
    return {k: as_bool_mask(col == i, num_nodes, k)
            for i, k in enumerate(('train', 'val', 'test'))}


def random_split(y: Tensor, num_classes: int, train_per_class: int = 20,
                 val_per_class: Optional[int] = None,
                 test_ratio: Optional[float] = None,
                 seed: int = 0) -> Dict[str, Tensor]:
    r"""分层随机划分。

    给了 ``test_ratio`` 就按比例切 test、其余全给 train/val（适合合成数据）；
    否则「剩下的都是 test」，即 Planetoid 的 ``num_train_per_class`` 语义。

    ``train_per_class`` 超过某类的样本数时会抛错而不是静默少给 —— 静默会让不同
    数据集的「同配置」实际训练量不同，消融表就废了。
    """
    n = int(y.numel())
    val_per_class = train_per_class if val_per_class is None else val_per_class
    tr = torch.zeros(n, dtype=torch.bool)
    va = torch.zeros(n, dtype=torch.bool)
    te = torch.zeros(n, dtype=torch.bool)
    g = torch.Generator().manual_seed(seed)
    for c in range(num_classes):
        idx = (y == c).nonzero().view(-1)
        if idx.numel() == 0:
            continue
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        need = train_per_class + val_per_class
        if test_ratio is not None:
            n_test = int(round(idx.numel() * test_ratio))
            if idx.numel() - n_test < need:
                raise ValueError(
                    f'类 {c} 只有 {idx.numel()} 个节点，切不出 '
                    f'{need} 个 train+val（test_ratio={test_ratio}）')
            n_tr = int(round((idx.numel() - n_test) * train_per_class / need))
            n_va = idx.numel() - n_test - n_tr
            tr[idx[:n_tr]] = True
            va[idx[n_tr:n_tr + n_va]] = True
            te[idx[n_tr + n_va:]] = True
        else:
            if idx.numel() <= need:
                raise ValueError(
                    f'类 {c} 只有 {idx.numel()} 个节点，不够 '
                    f'train({train_per_class})+val({val_per_class})+1 个 test')
            tr[idx[:train_per_class]] = True
            va[idx[train_per_class:need]] = True
            te[idx[need:]] = True
    return {'train': tr, 'val': va, 'test': te}
