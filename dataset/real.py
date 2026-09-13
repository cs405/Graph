"""真实数据集装载（PyG 的 Planetoid / WebKB / WikipediaNetwork）。

统一成一个 :class:`~dataset.base.GraphBundle`，并把「split 从哪来」写进 ``meta``：
Planetoid 走 public（20/类），WebKB 自带 10 折 ``48/32/20``，WikipediaNetwork 走
geom-gcn 的 10 折；都没有时才退到分层随机划分 —— 混用 split 是让异配论文数字
失去可比性的头号原因，所以必须显式记录。

下载失败（无网 / 源站挂）时抛 :class:`DatasetUnavailable`，由调用方决定是退到
合成图还是直接报错，绝不在这里静默伪造数据。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

from dataset.base import GraphBundle, row_normalize
from dataset.splits import as_bool_mask, random_split

__all__ = ['DatasetUnavailable', 'REAL_DATASETS', 'load_real']


class DatasetUnavailable(RuntimeError):
    """数据集拿不到（未注册 / 下载失败 / 无特征）。"""


# name -> (PyG 数据集类, 该类的 name)
#
# 三条异配图的归类修正（本 PyG 版本 2.6.x）：`WebKB` 只收 cornell/texas/wisconsin
# （webkb.py:74 那句 assert 就是它抛的），Chameleon / Squirrel 属 `WikipediaNetwork`（geom-gcn
# 那批，带 10 折 split），Actor 另有 `Actor` 类。旧表把这三个主名指向 WebKB ⇒ 它们
# 必然在处理参数之前就挂掉（连下载请求都发不出去），报出来的却是「加载失败（无网时…）」，
# 误导排障方向。第二个元素是传给该类的 ``name``，None 表示这个类压根没有该形参
# （``Actor.__init__`` 只有 root/transform/pre_transform/force_reload，传 name 会 TypeError）。
REAL_DATASETS: Dict[str, Tuple[str, Optional[str]]] = {
    'cora': ('Planetoid', 'Cora'),
    'citeseer': ('Planetoid', 'Citeseer'),
    'pubmed': ('Planetoid', 'Pubmed'),
    'chameleon': ('WikipediaNetwork', 'chameleon'),
    'squirrel': ('WikipediaNetwork', 'squirrel'),
    'actor': ('Actor', None),                        # 无 name 形参，见上
    'texas': ('WebKB', 'Texas'),
    'cornell': ('WebKB', 'Cornell'),
    'wisconsin': ('WebKB', 'Wisconsin'),
    'chameleon_wiki': ('WikipediaNetwork', 'chameleon'),
    'squirrel_wiki': ('WikipediaNetwork', 'squirrel'),
    # crocodile 不列：本版本的 WikipediaNetwork 默认 geom_gcn_preprocess=True，
    # 只认 chameleon / squirrel（取它会在构造里报 AttributeError，不是网络错）。
}


def _pick(m: Tensor, split_idx: int, num_nodes: int, what: str) -> Tensor:
    """[N] 或 [N, 10] 的 mask 统一取出指定一折。"""
    if m.dim() == 2:
        m = m[:, split_idx % m.size(1)]
    return as_bool_mask(m, num_nodes, what)


def _masks(data, num_nodes: int, split_idx: int, y: Tensor, num_classes: int,
           seed: int, train_per_class: Optional[int]
           ) -> Tuple[Dict[str, Tensor], str]:
    have = [getattr(data, k, None) for k in
            ('train_mask', 'val_mask', 'test_mask')]
    if all(m is not None for m in have):
        masks = {name: _pick(m, split_idx, num_nodes, name)
                 for name, m in zip(('train', 'val', 'test'), have)}
        kind = 'official' if have[0].dim() == 1 else 'official-10fold'
        return masks, kind
    if train_per_class is None:
        raise DatasetUnavailable(
            '该数据集不带 split，且 train_per_class=None，无法随机划分')
    return random_split(y, num_classes, train_per_class, train_per_class,
                        seed=seed), 'random'


def load_real(name: str, root: str = 'data', split_idx: int = 0,
              train_per_class: Optional[int] = 20, normalize_x: bool = True,
              seed: int = 0) -> GraphBundle:
    key = name.lower()
    if key not in REAL_DATASETS:
        raise DatasetUnavailable(
            f'未知数据集 {name!r}，可选：{sorted(REAL_DATASETS)} 或 synth')
    cls_name, ds_name = REAL_DATASETS[key]
    try:
        from torch_geometric import datasets as P
        kw = {'split': 'public'} if cls_name == 'Planetoid' else {}
        if ds_name is not None:
            kw['name'] = ds_name
        ds = getattr(P, cls_name)(root=root, **kw)
        data = ds[0]
    except DatasetUnavailable:
        raise
    except Exception as e:                             # 下载 / 解压 / 反序列化失败
        raise DatasetUnavailable(
            f'加载 {name} 失败（{type(e).__name__}: {str(e)[:200]}）。'
            '无网时请用 --dataset synth。') from e

    num_nodes = int(data.num_nodes)
    x, y = data.x, data.y.long()
    if x is None:
        raise DatasetUnavailable(f'{name} 没有节点特征，本管线不支持')
    # 跟全局默认 dtype 走（而不是写死 float32）：测试在 float64 下跑，
    # 特征若是 float32 会与 float64 权重在首次 matmul 直接报 dtype mismatch。
    x = x.to(torch.get_default_dtype())
    if normalize_x:
        x = row_normalize(x)

    # 自环统一由算子添加（utils.graph.augment_edge_index）。这里先删干净，
    # 否则真实数据里残留的自环会让「中心槽位」被算两遍，alpha/温度全部错位。
    ei = remove_self_loops(data.edge_index.long())[0]
    ei = coalesce(to_undirected(ei, num_nodes=num_nodes), num_nodes=num_nodes)

    num_classes = int(ds.num_classes)
    masks, kind = _masks(data, num_nodes, split_idx, y, num_classes, seed,
                         train_per_class)
    return GraphBundle(
        name=key, x=x, edge_index=ei, y=y, num_classes=num_classes,
        source='real', train_mask=masks['train'], val_mask=masks['val'],
        test_mask=masks['test'],
        meta={'split_source': f'{kind}@fold{split_idx}',
              'raw_name': f'{cls_name}/{ds_name}',
              'normalized': str(normalize_x)})
