"""数据层入口。

``load_dataset('cora')`` 拿真实数据，``load_dataset('synth')`` 拿可控合成图，
``load_dataset('relsynth')`` 拿可控合成**关系图**（边级监督），
``load_dataset('cora-rel')`` 把任意一张图转成「这条边存在吗」的关系图。

真实数据下载失败时，只有显式传 ``fallback_to_synth=True`` 才会退回合成图，并在
``meta`` 里标 ``source='synthetic'`` —— 绝不允许「以为在 Cora 上跑、其实在合成图」。

``supervision`` 由 bundle 自己声明（``'node'``/``'edge'``），任务分派
（:func:`tasks.resolve_task`）与损失选择都读它，不在这里判断名字。
"""

from __future__ import annotations

from typing import Optional

from dataset.base import GraphBundle, row_normalize
from dataset.real import REAL_DATASETS, DatasetUnavailable, load_real
from dataset.relation import (RelationBundle, make_relation_bundle,
                              relation_from_bundle)
from dataset.splits import as_bool_mask, random_split, split_from_column
from dataset.synthetic import (ba_graph, dup_directed_graph, hetero_bundle,
                               star_graph)

__all__ = ['GraphBundle', 'RelationBundle', 'row_normalize', 'load_dataset',
           'load_real', 'DatasetUnavailable', 'REAL_DATASETS', 'random_split',
           'as_bool_mask', 'split_from_column', 'hetero_bundle', 'ba_graph',
           'dup_directed_graph', 'star_graph', 'make_relation_bundle',
           'relation_from_bundle', 'RELATION_SYNTH']

#: 走 :func:`make_relation_bundle` 的名字（边级监督的合成关系图）
RELATION_SYNTH = ('relsynth', 'relation', 'rel-synth', 'rel_synth')
#: ``<name>-rel`` 后缀：先按 ``<name>`` 装载，再转成边级监督
REL_SUFFIX = '-rel'


def load_dataset(name: str = 'synth', root: str = 'data', split_idx: int = 0,
                 seed: int = 0, train_per_class: Optional[int] = 20,
                 homophily: float = 0.3, n_per_class: int = 60,
                 fallback_to_synth: bool = False,
                 edge_neg_ratio: float = 1.0) -> GraphBundle:
    """统一装载 + 立刻体检（split 互斥、mask 非空在这里就断掉）。

    Args:
        edge_neg_ratio: 仅关系图用（``cfg.task.edge_neg_ratio``）。负例数 =
            该比例 × 正例数；负例太少时模型只要恒判「成立」就能拿高准确率。
    """
    key = name.lower()
    if key in RELATION_SYNTH:
        ds = make_relation_bundle(neg_ratio=edge_neg_ratio, seed=seed)
    elif key.endswith(REL_SUFFIX) and len(key) > len(REL_SUFFIX):
        base = load_dataset(key[:-len(REL_SUFFIX)], root=root, split_idx=split_idx,
                            seed=seed, train_per_class=train_per_class,
                            homophily=homophily, n_per_class=n_per_class,
                            fallback_to_synth=fallback_to_synth)
        ds = relation_from_bundle(base, neg_ratio=edge_neg_ratio, seed=seed)
    elif key in ('synth', 'synthetic', 'synth-het'):
        ds = hetero_bundle(homophily=homophily, n_per_class=n_per_class,
                           train_per_class=train_per_class, seed=seed)
    else:
        try:
            ds = load_real(key, root=root, split_idx=split_idx, seed=seed,
                           train_per_class=train_per_class)
        except DatasetUnavailable:
            if not fallback_to_synth:
                raise
            ds = hetero_bundle(homophily=homophily, n_per_class=n_per_class,
                               train_per_class=train_per_class, seed=seed,
                               name=f'{key}->synth')
            ds.meta['fallback_reason'] = 'real dataset unavailable'
    ds.assert_split_disjoint()
    return ds
