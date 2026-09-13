r"""同质性度量：全局 / 逐节点，以及与门控的相关性。

这是论文最可能的 headline（「:math:`\lambda_i` 自动随局部异配度上升」），
所以必须与 :mod:`modules.oca` 完全无关地独立实现 —— 若用它自己的输出反推自己，
结论无价值。

逐节点同质性只对**有邻居**的节点有定义，孤立节点的局部同质性没有意义，
一律通过返回的 ``valid`` mask 剔除，不做任何填充。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import degree

__all__ = ['edge_homophily', 'local_homophily', 'neighbor_label_entropy',
           'pearson', 'spearman', 'gate_homophily_report']


def edge_homophily(edge_index: Tensor, y: Tensor,
                   node_mask: Optional[Tensor] = None) -> float:
    r"""全局边同质性 :math:`h=\frac{1}{|\mathcal{E}|}\sum_{(u,v)}\mathbb{1}[y_u=y_v]`。

    ``node_mask`` 是**按中心端**（``edge_index[0]``）筛选的 bool，用于只统计
    有标签节点发出的边；不传则全体边。
    """
    if edge_index.numel() == 0:
        return float('nan')
    same = y[edge_index[0]] == y[edge_index[1]]
    if node_mask is not None:
        same = same[node_mask[edge_index[0]]]
    return same.to(torch.float64).mean().item()


def local_homophily(edge_index: Tensor, y: Tensor, num_nodes: int,
                    ) -> Tuple[Tensor, Tensor]:
    """逐节点局部同质性 :math:`h_i` = 邻居中与中心节点同类者的占比。

    Returns:
        h: [N] float64，孤立节点处为 0（值无意义，看 valid）
        valid: [N] bool，``deg_i > 0``

    两个返回值都落在 ``y`` 的同一设备上（包括 CUDA），否则 :meth:`index_add_`
    会报 device mismatch —— 这类错误只在真实数据上才暴露，合成图跑在 CPU 上测不出。
    """
    h = torch.zeros(num_nodes, dtype=torch.float64, device=y.device)
    if edge_index.numel() == 0:
        return h, h > 0                      # 全 False
    src, dst = edge_index[0], edge_index[1]
    same = (y[src] == y[dst]).to(torch.float64)
    h = h.index_add_(0, src, same)
    cnt = degree(src, num_nodes=num_nodes, dtype=torch.float64)
    valid = cnt > 0
    h[valid] = h[valid] / cnt[valid]
    return h, valid


def neighbor_label_entropy(edge_index: Tensor, y: Tensor, num_nodes: int,
                           num_classes: int) -> Tensor:
    r"""邻居标签分布的归一化熵（H2GCN 意义的 class overlap，与 :math:`h_i` 不同轴）。

    异配 = 「邻居多为异类」，类别重叠 = 「邻居跨多个类」；一个节点可以两者兼有，
    也可以只有前者（二分类交替）。只用一个量代理「该不该竞争」会混淆二者。
    """
    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, dtype=torch.float64, device=y.device)
    cnt = torch.zeros(num_nodes, num_classes, dtype=torch.float64,
                      device=y.device)
    cnt.index_put_((edge_index[0], y[edge_index[1]]),
                   torch.ones(edge_index.size(1), dtype=torch.float64,
                              device=y.device),
                   accumulate=True)
    p = cnt / cnt.sum(1, keepdim=True).clamp(min=1.0)
    ent = -(p * torch.log(p.clamp(min=1e-12))).sum(1)
    scale = torch.log(torch.tensor(float(max(num_classes, 2)), device=y.device))
    return ent / scale


def pearson(x: Tensor, y: Tensor) -> float:
    x = x.to(torch.float64).reshape(-1)
    y = y.to(torch.float64).reshape(-1)
    assert x.numel() == y.numel() and x.numel() > 1
    xc, yc = x - x.mean(), y - y.mean()
    den = (xc.norm() * yc.norm()).item()
    return float('nan') if den < 1e-12 else (xc @ yc).item() / den


def spearman(x: Tensor, y: Tensor) -> float:
    """秩相关。并列值用 argsort 的顺序打破（样本量足够时影响可忽略，
    但严格讲不是 tie-corrected —— 报数字时要注明）。
    """
    def rank(v: Tensor) -> Tensor:
        idx = torch.argsort(v.to(torch.float64))
        r = torch.empty_like(idx, dtype=torch.float64)
        r[idx] = torch.arange(idx.numel(), dtype=torch.float64)
        return r

    return pearson(rank(x), rank(y))


def gate_homophily_report(gate: Tensor, h: Tensor, valid: Tensor,
                          extra: Optional[Dict[str, Tensor]] = None,
                          ) -> Dict[str, float]:
    r""":math:`\lambda`（或 :math:`\alpha`）与局部同质性的相关性汇总。

    ``extra`` 可再塞几路门控（如 ``{'alpha': ...}``），一并报。
    显著性只给正态近似的双侧 z（大样本够用），不是精确 t 检验 —— 报进论文前
    需要换成置换检验，这里不假装它等价。
    """
    # 统一拉回 CPU：相关分析只产出几个标量，没必要在 GPU 上算，
    # 而且门控（已在 CPU）与图张量（可能在 CUDA）混着算会直接报 device mismatch
    gate = gate.to(torch.float64).reshape(-1).cpu()
    h = h.to(torch.float64).cpu()
    valid = valid.cpu()
    keep = valid & torch.isfinite(gate) & torch.isfinite(h)
    g, hh = gate[keep], h[keep]
    n = int(g.numel())
    out: Dict[str, float] = {'n': float(n)}
    if n < 3:
        out.update(r_pearson=float('nan'), r_spearman=float('nan'),
                   z_pearson=float('nan'))
        return out
    r = pearson(g, hh)
    rho = spearman(g, hh)
    out['r_pearson'] = r
    out['r_spearman'] = rho
    out['gate_mean'] = g.mean().item()
    out['gate_std'] = g.std(unbiased=True).item()
    out['homophily_mean'] = hh.mean().item()
    # 按同质性中位数分组的门控均值差：比相关系数更直观，也更好写进正文
    med = hh.median()
    lo, hi = hh <= med, hh > med
    out['gate_low_h'], out['gate_high_h'] = g[lo].mean().item(), g[hi].mean().item()
    out['gate_gap'] = out['gate_high_h'] - out['gate_low_h']
    z = r * (n - 1) ** 0.5 / max((1 - r * r) ** 0.5, 1e-12)
    out['z_pearson'] = z
    if extra:
        for k, v in extra.items():
            v = v.to(torch.float64).reshape(-1).cpu()
            v = v[keep] if v.numel() == gate.numel() else v
            out[f'{k}_mean'], out[f'{k}_std'] = (v.mean().item(),
                                                v.std(unbiased=True).item())
    return out
