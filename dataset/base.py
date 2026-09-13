r"""数据集统一出口：一个 ``GraphBundle`` 装下「一张全图 + 三个 mask + 元信息」。

不做 ``torch_geometric.data.DataLoader`` 式的 mini-batch：本项目的算子是全图稀疏
实现，节点数最大的真实数据集（Actor 7600）也放得下，硬上采样只会让
「:math:`\lambda_i` 与局部同质性的关系」这种逐节点分析失去意义。
真需要 subgraph 训练时再在 ``dataset/`` 里加，不提前设计。

两个对训练层公开的契约（训练层只认这两个，不认具体字段）：

* :attr:`GraphBundle.supervision` —— ``'node'`` 还是 ``'edge'``。它决定
  ``y``/``train_mask``/``val_mask``/``test_mask`` 索引的是**什么**：节点级任务里是
  ``[N]``，边级任务里是 ``[E]``。结构表最后一层的 ``level`` 必须与它一致
  （:meth:`model.GraphModel.build` 里校验）—— 两边都是二维张量，接错了不报错，
  只会静默错位；
* :meth:`GraphBundle.forward_kwargs` —— 前向的全部输入。节点级是
  ``x/edge_index``，关系图还要带 ``batch``（关系类型/边特征/节点类型）。
  训练与诊断一律写 ``model(**ds.forward_kwargs())``，不要手写 ``model(ds.x, ds.edge_index)``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from torch import Tensor

from metrics.homophily import edge_homophily

__all__ = ['GraphBundle', 'row_normalize']


def row_normalize(x: Tensor) -> Tensor:
    """行 L1 归一化（Bag of Words 特征的标准预处理）。

    全零行会给出 0 向量而不是 NaN —— 真实数据集里确实存在空特征节点。
    """
    return x / x.abs().sum(-1, keepdim=True).clamp(min=1e-12)


@dataclass
class GraphBundle:
    name: str
    x: Tensor
    edge_index: Tensor
    y: Tensor
    train_mask: Tensor
    val_mask: Tensor
    test_mask: Tensor
    num_classes: int
    supervision: str = 'node'        # 'node' | 'edge'：y 与三个 mask 索引的是什么
    source: str = 'unknown'          # 'real' / 'synthetic'，报结果时必须区分
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_features(self) -> int:
        return int(self.x.size(1))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))

    @property
    def num_units(self) -> int:
        """被监督的单元数：节点级 = ``N``，边级 = ``E``（``y`` 的长度）。"""
        return int(self.y.size(0))

    @property
    def device(self) -> torch.device:
        return self.x.device

    def forward_kwargs(self) -> Dict[str, Any]:
        """前向的全部输入。训练/诊断一律 ``model(**ds.forward_kwargs())``。

        ``batch`` 恒存且默认为 ``None``：:class:`model_builder.GraphSequential`
        的签名里它是关键字参数，而只声明了 ``takes_batch`` 的层会真正收到它 ——
        这样基线层（GAT/GCN/MLP）的 ``forward(x, edge_index)`` 不必跟着改。
        """
        return {'x': self.x, 'edge_index': self.edge_index, 'batch': None}

    def to(self, device) -> 'GraphBundle':
        for k in ('x', 'edge_index', 'y', 'train_mask', 'val_mask',
                  'test_mask'):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def masks(self) -> Dict[str, Tensor]:
        return {'train': self.train_mask, 'val': self.val_mask,
                'test': self.test_mask}

    def stats(self) -> Dict[str, float]:
        """数据体检表。训练前打印一次，能当场抓住「val 与 train 重叠」这类事故。"""
        from torch_geometric.utils import degree
        d = degree(self.edge_index[0], num_nodes=self.num_nodes)
        tr, va, te = self.train_mask, self.val_mask, self.test_mask
        return {
            'num_nodes': float(self.num_nodes),
            'num_features': float(self.num_features),
            'num_classes': float(self.num_classes),
            'num_edges': float(self.edge_index.size(1)),
            'avg_deg': float(d.mean()),
            'max_deg': float(d.max()) if d.numel() else 0.0,
            'edge_homophily': edge_homophily(self.edge_index, self.y),
            'train_nodes': float(tr.sum()),
            'train_per_class': float(tr.sum() / max(self.num_classes, 1)),
            'val_nodes': float(va.sum()),
            'test_nodes': float(te.sum()),
            # 三个 mask 必须互斥；重叠时差值非 0，直接判失败
            'mask_overlap': float((int(tr.sum()) + int(va.sum()) + int(te.sum()))
                                  - int((tr | va | te).sum())),
        }

    def assert_split_disjoint(self) -> None:
        s = self.stats()
        assert s['mask_overlap'] == 0.0, (
            f'{self.name}: train/val/test 有重复节点，泄漏了')
        assert s['train_nodes'] > 0 and s['val_nodes'] > 0 and s['test_nodes'] > 0, \
            f'{self.name}: 某个 split 为空'
