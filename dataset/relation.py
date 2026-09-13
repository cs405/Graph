r"""边级监督的数据集：``RelationBundle`` + 两个构造器（关系图版的 ``dataset/synthetic.py``）。

**为什么边级任务要单独一个 bundle 类型**，而不是给 ``GraphBundle`` 加几个可选字段：
``y``/``train_mask`` 这些名字在两种任务里索引的是**不同的东西**（节点 vs 边），
混在一个类里就得靠注释提醒「这里其实是边」—— 而注释不会报错。这里用
``supervision='edge'`` 把语义钉在类型上，并覆盖 :meth:`stats` / :meth:`forward_kwargs` /
:meth:`to`，让「按节点对齐」的旧代码在边级 bundle 上当场失效而不是静默算错。

任务形态（务必读，否则会觉得标签很奇怪）：**给定关系 r 与一对节点 (i,j)，判断
r 在 (i,j) 上成立吗**。所以

* ``num_classes == 1``：单 logit 的二分类（判决走 ``cfg.task.threshold``）；
* ``rel`` 是**输入**不是标签：把「关系 id」当标签会让模型直接从 ``batch['rel']``
  读出答案（``unpack_batch`` 会把 rel 喂给 DIA），指标刷到 1.0 而什么都没学；
* 正例 = 真边，负例 = 采出来的非边，两者带同一个 ``rel``（「查询的是哪个关系」）。

划分按**无向对**而不是按有向边：同一条事实的两个方向必须落在同一个 split 里，
否则 ``(i,j)`` 在 train、``(j,i)`` 在 test，测试集里有一半样本训练时见过 ——
这是关系图上最容易犯、且指标看不出来的泄漏。:meth:`RelationBundle.assert_split_disjoint`
里的 ``pair_leak`` 就是钉这件事的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from dataset.base import GraphBundle

__all__ = ['RelationBundle', 'make_relation_bundle', 'relation_from_bundle']


@dataclass
class RelationBundle(GraphBundle):
    """带关系类型/边特征/节点类型的图，监督落在**边**上。

    字段语义（与父类的差别）：

    * ``y`` ``[E]`` —— 每条**有向边**的标签（这里是 0/1：该关系成不成立）；
    * ``train_mask``/``val_mask``/``test_mask`` ``[E]`` —— 边级划分；
    * ``num_classes`` —— 边标签的类别数（二分类 = 1）；
    * ``rel`` ``[E]`` —— 每条边查询的关系 id（DIA 按它选 :math:`U,V`）；
    * ``edge_attr`` ``[E,d_e]`` —— 边特征 :math:`e_{ij}`（L1 维度注意力的输入之一）；
    * ``node_type`` ``[N]`` —— 节点类型（L3 的 :math:`b_{r_i\to r_j}`）；
    * ``pair_id`` ``[E]`` —— 该边所属的**无向对**编号（划分与泄漏检查都按它）。
    """

    rel: Optional[Tensor] = None
    edge_attr: Optional[Tensor] = None
    node_type: Optional[Tensor] = None
    pair_id: Optional[Tensor] = None
    node_y: Optional[Tensor] = None        # 造图时用的节点标签；边级任务**不**监督它
    n_rel: int = 1
    supervision: str = 'edge'              # 覆盖父类默认（位置不变，只是改默认值）

    # ------------------------------------------------------------------ 契约
    def forward_kwargs(self) -> Dict[str, Any]:
        """``batch`` 里只放 :data:`utils.graph.BATCH_KEYS` 认识的三个键。

        多放一个键（比如顺手把 ``pair_id`` 塞进去）会被 ``unpack_batch`` 当场拒；
        少放一个则是静默少一层信息 —— 所以 ``None`` 也照传，让下游自己决定快路径。
        """
        return {'x': self.x, 'edge_index': self.edge_index,
                'batch': {'rel': self.rel, 'edge_attr': self.edge_attr,
                          'node_type': self.node_type}}

    def to(self, device) -> 'RelationBundle':
        super().to(device)
        for k in ('rel', 'edge_attr', 'node_type', 'pair_id', 'node_y'):
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self

    # ------------------------------------------------------------------ 体检
    def stats(self) -> Dict[str, float]:
        from torch_geometric.utils import degree
        d = degree(self.edge_index[0], num_nodes=self.num_nodes)
        tr, va, te = self.train_mask, self.val_mask, self.test_mask
        out = {
            'num_nodes': float(self.num_nodes),
            'num_features': float(self.num_features),
            'num_edges': float(self.num_edges),
            'num_pairs': float(int(self.pair_id.max()) + 1)
            if self.pair_id is not None else float('nan'),
            'n_rel': float(self.n_rel),
            'num_classes': float(self.num_classes),
            'avg_deg': float(d.mean()) if d.numel() else 0.0,
            'max_deg': float(d.max()) if d.numel() else 0.0,
            'pos_frac': float(self.y.float().mean()) if self.y.numel() else 0.0,
            'train_edges': float(tr.sum()),
            'train_per_class': float(tr.sum() / max(self.num_classes, 1)),
            'val_edges': float(va.sum()),
            'test_edges': float(te.sum()),
            'mask_overlap': float((int(tr.sum()) + int(va.sum()) + int(te.sum()))
                                  - int((tr | va | te).sum())),
            'uncovered_edges': float(self.num_edges
                                     - int((tr | va | te).sum())),
            'pair_leak': float(self.pair_leak()),
        }
        if self.edge_attr is not None:
            out['edge_dim'] = float(self.edge_attr.size(-1))
        return out

    def pair_leak(self) -> int:
        """跨越两个以上 split 的**无向对**个数（必须是 0，见模块 docstring）。

        用 ``bincount`` 而不是逐边循环：边数上千时 Python 循环会把 ``stats()``
        （每次装载都要跑）拖成可感知的开销。
        """
        if self.pair_id is None:
            return 0
        P = int(self.pair_id.max()) + 1
        cnt = [torch.bincount(self.pair_id[m], minlength=P)
               for m in (self.train_mask, self.val_mask, self.test_mask)]
        present = sum((c > 0).long() for c in cnt)
        return int((present > 1).sum())

    def assert_split_disjoint(self) -> None:
        s = self.stats()
        assert s['mask_overlap'] == 0.0, f'{self.name}: train/val/test 有重复边'
        assert s['pair_leak'] == 0.0, (
            f'{self.name}: 同一条无向对的两个方向落在不同 split 里（泄漏）')
        assert s['train_edges'] > 0 and s['val_edges'] > 0 and s['test_edges'] > 0, \
            f'{self.name}: 某个 split 为空'


# ---------------------------------------------------------------------------
# 构造器一：可控合成关系图
# ---------------------------------------------------------------------------

def make_relation_bundle(num_classes: int = 3, n_per_class: int = 40,
                         n_rel: int = 3, avg_deg: int = 6, feat_dim: int = 16,
                         sep: float = 1.2, edge_dim: int = 4,
                         edge_noise: float = 1.0, n_node_types: int = 2,
                         pref_density: float = 0.4, directed_rel: int = 1,
                         neg_ratio: float = 1.0, train_frac: float = 0.6,
                         val_frac: float = 0.2, seed: int = 0,
                         name: str = 'relsynth') -> RelationBundle:
    r"""造一张多关系图 + 边级二分类标签。

    生成规则（每条都对应一个「模型必须学到什么」）：

    1. 节点特征 = 类中心 + 高斯噪声（与 :func:`dataset.synthetic.hetero_bundle` 同口径），
       类中心间距由 ``sep`` 控制；
    2. 每个关系 :math:`r` 持有一张**类对偏好矩阵** ``pref[r]``（Bernoulli，
       密度 ``pref_density``）：只有 ``(y_i, y_j)`` 命中偏好的对才可能成为 :math:`r` 的边。
       于是「关系 r 成立吗」等价于「i,j 的类对是否在 pref[r] 里」——
       这正是 DIA 的低秩非负 :math:`W_r=U_rV_r^\top` 该学到的东西
       （不同关系 ⇒ 不同的列支撑集，可被 :func:`modules.dia.supports_disjoint` 检验）；
    3. 最后 ``directed_rel`` 个关系只建单向边：L3 的非对称主体贡献
       （:math:`\alpha_i \ne \alpha_j`）只有在有向关系上才学得到东西，
       全对称的图会让那一层退化成常数 0.5；
    4. 负例是**采出来的非边**，带一个随机 ``rel``；它的 ``edge_attr`` 从与正例
       同一个分布采（``proto[rel] + noise``）。让负例的边特征分布与正例不同，
       等于把答案写进输入 —— 模型只要看 ``edge_attr`` 的范数就能判对。

    Args:
        neg_ratio: 负例数 = ``neg_ratio`` × 正例数（``cfg.task.edge_neg_ratio``）。
        edge_noise: 边特征的噪声尺度。设成 0 会让 ``edge_attr`` 变成 ``rel`` 的
            无损副本（信息重复），太大则 L1 的 :math:`\gamma` 无信号可用。
        train_frac/val_frac: 按**无向对**划分的比例，剩下的是 test。
    """
    assert 0.0 < pref_density < 1.0, f'pref_density 需在 (0,1)：{pref_density}'
    assert 0.0 <= train_frac and 0.0 <= val_frac and train_frac + val_frac < 1.0, \
        f'train_frac+val_frac 必须 <1（剩下的是 test）：{train_frac}+{val_frac}'
    assert directed_rel <= n_rel, f'directed_rel={directed_rel} 超过关系数 {n_rel}'
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    n = num_classes * n_per_class
    node_y = torch.arange(num_classes).repeat_interleave(n_per_class)
    centroids = torch.randn(num_classes, feat_dim, generator=g) * sep
    x = centroids[node_y] + torch.randn(n, feat_dim, generator=g)
    # 节点类型与节点类别**独立**：L3 的 b_(r_i->r_j) 要学的是类型效应，
    # 让两者重合等于把类别信息从另一条路喂进去，读不出类型偏置到底学到没有。
    perm = torch.randperm(n, generator=g)
    node_type = torch.zeros(n, dtype=torch.long)
    node_type[perm] = torch.arange(n, generator=g) % max(n_node_types, 1)

    by_class = torch.stack([(node_y == c).nonzero().view(-1)
                            for c in range(num_classes)])           # [C, n_per_class]

    # ---- 每个关系一张类对偏好矩阵 ----------------------------------------
    pref = (torch.rand(n_rel, num_classes, num_classes, generator=g)
            < pref_density)
    for r in range(n_rel):
        if not pref[r].any():                    # 全 0 的关系一条边都造不出来
            pref[r, 0, 0] = True
    proto = torch.randn(n_rel, max(edge_dim, 1), generator=g)

    pairs: Dict[Tuple[int, int], int] = {}       # (i, j) 有向对 -> rel
    per_rel = max(n * avg_deg // max(n_rel, 1), num_classes * num_classes)
    for r in range(n_rel):
        allowed = pref[r].nonzero()              # [K, 2] 命中的类对
        pick = allowed[torch.randint(allowed.size(0), (per_rel,), generator=g)]
        ci, cj = pick[:, 0], pick[:, 1]
        i = by_class[ci, torch.randint(n_per_class, (per_rel,), generator=g)]
        j = by_class[cj, torch.randint(n_per_class, (per_rel,), generator=g)]
        keep = i != j
        for a, b in zip(i[keep].tolist(), j[keep].tolist()):
            pairs[(a, b)] = r
            if r < n_rel - directed_rel:         # 对称关系：反向边同一个 rel
                pairs.setdefault((b, a), r)

    pos = sorted(pairs.items())
    n_pos = len(pos)
    n_neg = int(round(n_pos * max(neg_ratio, 0.0)))
    neg: List[Tuple[Tuple[int, int], int]] = []
    seen_neg: set = set()
    guard = 0
    while len(neg) < n_neg and guard < 50 * (n_neg + 1):
        guard += 1
        r = int(torch.randint(n_rel, (1,), generator=g).item())
        ci = int(torch.randint(num_classes, (1,), generator=g).item())
        cj = int(torch.randint(num_classes, (1,), generator=g).item())
        a = int(by_class[ci, torch.randint(n_per_class, (1,), generator=g)].item())
        b = int(by_class[cj, torch.randint(n_per_class, (1,), generator=g)].item())
        if a == b or (a, b) in seen_neg:
            continue
        # 任一方向已经是正例就跳过：否则同一条边上会同时出现 label=1 与 label=0，
        # 模型看到的是一个自相矛盾的训练集（loss 降不下去，但看不出原因）。
        if (a, b) in pairs or (b, a) in pairs:
            continue
        seen_neg.add((a, b))
        neg.append(((a, b), r))

    rows = [(ij, r, 1) for ij, r in pos] + [(ij, r, 0) for ij, r in neg]
    src = torch.tensor([r[0][0] for r in rows], dtype=torch.long)
    dst = torch.tensor([r[0][1] for r in rows], dtype=torch.long)
    rel = torch.tensor([r[1] for r in rows], dtype=torch.long)
    y = torch.tensor([r[2] for r in rows], dtype=torch.long)
    edge_index = torch.stack([src, dst])

    # ---- 无向对编号 + 按对划分 -------------------------------------------
    lo, hi = torch.minimum(src, dst), torch.maximum(src, dst)
    key = lo * n + hi
    uniq, pair_id = torch.unique(key, return_inverse=True)
    n_pairs = uniq.numel()
    order = torch.randperm(n_pairs, generator=g)
    n_tr = int(round(n_pairs * train_frac))
    n_va = int(round(n_pairs * val_frac))
    split = torch.full((n_pairs,), 2, dtype=torch.long)     # 2 = test
    split[order[:n_tr]] = 0
    split[order[n_tr:n_tr + n_va]] = 1
    which = split[pair_id]
    train_mask, val_mask, test_mask = which == 0, which == 1, which == 2

    edge_attr = (proto[rel] + edge_noise * torch.randn(rel.size(0), edge_dim,
                                                       generator=g)
                 if edge_dim > 0 else None)

    ds = RelationBundle(
        name=name, x=x, edge_index=edge_index, y=y, num_classes=1,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        source='synthetic', rel=rel, edge_attr=edge_attr, node_type=node_type,
        pair_id=pair_id, node_y=node_y, n_rel=n_rel,
        meta={'kind': 'relation', 'num_node_classes': str(num_classes),
              'n_rel': str(n_rel), 'directed_rel': str(directed_rel),
              'neg_ratio': f'{neg_ratio:.2f}', 'sep': str(sep),
              'pref_density': str(pref_density),
              'edge_noise': str(edge_noise),
              'pos_edges': str(n_pos), 'neg_edges': str(len(neg)),
              'split_source': 'random-pairwise'})
    ds.assert_split_disjoint()
    return ds


# ---------------------------------------------------------------------------
# 构造器二：把任意节点数据集转成边级二分类
# ---------------------------------------------------------------------------

def relation_from_bundle(ds: GraphBundle, neg_ratio: float = 1.0,
                         seed: int = 0, name: Optional[str] = None,
                         max_neg_candidates: int = 200000) -> RelationBundle:
    """把一张普通图转成「这条边存在吗」的关系图（真实数据上的边级监督）。

    两个关系：``rel=0`` 问「同类之间有边吗」，``rel=1`` 问「异类之间有边吗」。
    于是正例来自原图的真边（按两端标签是否相同分派 rel），负例是采出来的非边
    （同样按两端标签分派 rel）—— 标签与 ``rel`` 不重合，模型不能靠读 ``rel`` 作弊。

    这不是「新数据集」，是把已有数据换一种监督口径：它能让 DIA 的边级头在
    **真实图**上跑通，而不必先去下载一个异构关系数据集。``edge_attr`` 为
    ``None``（原图没有边特征），``node_type`` 全 0（同构），所以
    ``dia.d_edge`` 要设成 0、``n_rel=2``、``n_node_types=1``。
    """
    g = torch.Generator().manual_seed(seed)
    n = ds.num_nodes
    src, dst = ds.edge_index[0], ds.edge_index[1]
    keep = src != dst
    src, dst = src[keep], dst[keep]
    same = (ds.y[src] == ds.y[dst]).long()
    rel_pos = same                                     # 0 = 同类查询, 1 = 异类查询
    pos = set(zip(src.tolist(), dst.tolist()))
    n_neg = int(round(len(pos) * max(neg_ratio, 0.0)))

    ns, nd, nr = [], [], []
    seen: set = set()
    guard = 0
    while len(ns) < n_neg and guard < max(max_neg_candidates, 10 * (n_neg + 1)):
        guard += 1
        a = int(torch.randint(n, (1,), generator=g).item())
        b = int(torch.randint(n, (1,), generator=g).item())
        if a == b or (a, b) in seen or (a, b) in pos or (b, a) in pos:
            continue
        seen.add((a, b))
        ns.append(a), nd.append(b)
        nr.append(0 if int(ds.y[a]) == int(ds.y[b]) else 1)

    ei = torch.stack([torch.cat([src, torch.tensor(ns, dtype=torch.long)]),
                      torch.cat([dst, torch.tensor(nd, dtype=torch.long)])])
    rel = torch.cat([rel_pos, torch.tensor(nr, dtype=torch.long)])
    y = torch.cat([torch.ones(len(pos), dtype=torch.long),
                   torch.zeros(len(ns), dtype=torch.long)])
    lo, hi = torch.minimum(ei[0], ei[1]), torch.maximum(ei[0], ei[1])
    _, pair_id = torch.unique(lo * n + hi, return_inverse=True)
    n_pairs = int(pair_id.max()) + 1
    order = torch.randperm(n_pairs, generator=g)
    n_tr = int(round(n_pairs * 0.6))
    n_va = int(round(n_pairs * 0.2))
    split = torch.full((n_pairs,), 2, dtype=torch.long)
    split[order[:n_tr]] = 0
    split[order[n_tr:n_tr + n_va]] = 1
    which = split[pair_id]
    out = RelationBundle(
        name=name or f'{ds.name}-rel', x=ds.x, edge_index=ei, y=y, num_classes=1,
        train_mask=which == 0, val_mask=which == 1, test_mask=which == 2,
        source=ds.source, rel=rel, edge_attr=None,
        node_type=torch.zeros(n, dtype=torch.long), pair_id=pair_id,
        node_y=ds.y, n_rel=2, supervision='edge',
        meta={'kind': 'relation-from-nodes', 'base': ds.name,
              'n_rel': '2', 'neg_ratio': f'{neg_ratio:.2f}',
              'split_source': 'random-pairwise'})
    out.assert_split_disjoint()
    return out
