r"""DIA 的可解释性出口（技术文档 §4.8 ``visualize_edge``、§五 对比表的「可解释」一栏）。

**这里只产数据，不画图。** 文档 §4.8 的 ``visualize_edge`` 直接调 matplotlib，
于是「跑一次解释」变成「必须有显示后端」，CI 里跑不了、批处理里跑不了。这里一律
返回 dict/list，画图是调用方（notebook、``docs`` 里的脚本）的事。

三个读数，分别对应文档的三条主张，都能被测试直接断言：

| 函数 | 对应主张 | 怎么读 |
| :-- | :-- | :-- |
| :func:`column_supports` / :func:`supports_disjoint` | §2.7 可识别性（非负+列正交 ⇒ 支撑集不相交） | 每个维度槽管着哪几行原始维度；两个槽有没有重叠 |
| :func:`edge_explanation` | §2.2/§2.4 逐维筛选与非对称主导 | 一条边上的 :math:`\gamma` top-k、:math:`M` 的 top 维度对、:math:`\alpha` 偏斜 |
| :func:`score_attribution` | §六 创新点 3（分数由可解释量决定） | 把 :math:`s_{ij}` 置零后 logit 变了多少 |

可解释性的前提（务必记住，否则读数没有意义）：``dia.use_projection=false``。
投影模式下 :math:`U` 的行是 ``Linear`` 之后的隐维度，「第 7 行」不对应任何原始特征，
:func:`pairing_report` 里那些维度下标只能说明稀疏结构，不能说明「哪两个特征在配对」。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from modules.base import iter_impl
from modules.dia.layer import (DIALayer, K_ALPHA_I, K_ALPHA_J, K_GAMMA_I,
                               K_GAMMA_J, K_S_IJ, K_S_JI, K_SCORE)
from modules.dia.pairing import LowRankNonNegPairing

__all__ = ['dia_layers', 'pairings', 'column_supports', 'supports_disjoint',
           'pairing_report', 'edge_explanation', 'score_attribution',
           'dump_pairings']


def dia_layers(model) -> List[DIALayer]:
    """模型里全部 :class:`DIALayer`（含 ``DIAConv``/``EdgeScore`` 内部的那些）。"""
    return iter_impl(model, DIALayer)                    # type: ignore[arg-type]


def pairings(model) -> List[Tuple[str, LowRankNonNegPairing]]:
    """``[(路径, 配对模块)]``，路径形如 ``'model.0.layer.pairing_ij'``。"""
    out: List[Tuple[str, LowRankNonNegPairing]] = []
    for name, m in model.named_modules():
        if isinstance(m, LowRankNonNegPairing):
            out.append((name, m))
    return out


def column_supports(p: LowRankNonNegPairing, eps: float = 1e-3,
                    which: str = 'U') -> List[List[int]]:
    r"""每个维度槽（列）的支撑集：``U`` 的哪几行非零。

    ``which='U'`` 读源端维度，``'V'`` 读邻端维度。返回 list-of-list（不是张量），
    因为各列长度不同 —— 这正是「稀疏 + 支撑集」该有的样子。
    """
    M = p.U if which == 'U' else p.V
    with torch.no_grad():
        nz = (M > eps)                                    # [R, d, k]
        out: List[List[int]] = []
        for r in range(M.size(0)):
            for k in range(M.size(2)):
                out.append(torch.nonzero(nz[r, :, k]).flatten().tolist())
        return out


def supports_disjoint(p: LowRankNonNegPairing, eps: float = 1e-3,
                      which: str = 'U') -> bool:
    """分离条件是否成立：任意两列的支撑集不相交（§2.7 推论的可检验形式）。"""
    sup = column_supports(p, eps, which)
    seen: set = set()
    for s in sup:
        if seen.intersection(s):
            return False
        seen.update(s)
    return True


def pairing_report(model, eps: float = 1e-3, max_pairs: int = 5
                   ) -> Dict[str, Any]:
    """逐配对模块的稀疏度/分离性/top 维度对（``results.json`` 里能直接落盘）。"""
    rep: Dict[str, Any] = {}
    for name, p in pairings(model):
        with torch.no_grad():
            W = p.W()                                     # [R, d_src, d_dst] | [d,d]
            W = W.unsqueeze(0) if W.dim() == 2 else W
            top: List[Any] = []
            for r in range(W.size(0)):
                flat = W[r].flatten()
                v, i = flat.topk(min(max_pairs, flat.numel()))
                d_dst = W.size(2)
                top.append([{'src': int(x // d_dst), 'dst': int(x % d_dst),
                             'w': float(w)} for w, x in zip(v.tolist(), i.tolist())])
        rep[name] = {**p.stats(eps), 'rank': p.rank, 'n_rel': p.n_rel,
                     'disjoint_U': supports_disjoint(p, eps, 'U'),
                     'disjoint_V': supports_disjoint(p, eps, 'V'),
                     'top_pairs': top}
    return rep


def edge_explanation(layer: DIALayer, h_i: Tensor, h_j: Tensor,
                     e_ij: Optional[Tensor] = None, rel: Optional[Tensor] = None,
                     r_i: Optional[Tensor] = None, r_j: Optional[Tensor] = None,
                     topk: int = 8) -> Dict[str, Any]:
    r"""单条（或一小批）边的完整解释：文档 §4.8 的无绘图版。

    Args:
        h_i/h_j: ``[E, d]`` 两端特征（**未经筛选**，筛选是层内部的事）
        topk: 每个读数保留前几项

    Returns:
        dict：``gamma_i_top``（被点亮最多的源端维度）、``gamma_j_top``、
        ``M_top``（:math:`M_{ij}` 里最大的维度对 = 「哪一维与哪一维在配对」）、
        ``s_ij``/``s_ji``/``alpha``/``score``、``factors``（哪些维度槽被点亮）。
    """
    was_training = layer.training
    layer.eval()
    with torch.no_grad():
        out = layer(h_i, h_j, e_ij, rel, r_i, r_j,
                    want=('score', 'message', 'matrix', 'factors'))
        aux = dict(layer.aux)
        E = h_i.size(0)
        k = min(topk, h_i.size(-1))
        rep: Dict[str, Any] = {
            'E': int(E),
            's_ij': aux[K_S_IJ].tolist(), 's_ji': aux[K_S_JI].tolist(),
            'alpha_i': aux[K_ALPHA_I].tolist(), 'alpha_j': aux[K_ALPHA_J].tolist(),
            'score': aux[K_SCORE].tolist(),
            'gamma_i_top': [torch.topk(aux[K_GAMMA_I][e], k).indices.tolist()
                            for e in range(E)] if K_GAMMA_I in aux else None,
            'gamma_j_top': [torch.topk(aux[K_GAMMA_J][e], k).indices.tolist()
                            for e in range(E)] if K_GAMMA_J in aux else None,
            'factors_ij': [f.tolist() for f in out['factors_ij']]
            if 'factors_ij' in out else None,
        }
        if 'M_ij' in out:
            M = out['M_ij']                                # [E, d_src, d_dst]
            d_dst = M.size(-1)
            flat = M.flatten(1)
            v, i = flat.topk(min(topk, flat.size(1)), dim=1)
            rep['M_top'] = [[{'src': int(a // d_dst), 'dst': int(a % d_dst),
                              'm': float(w)} for w, a in zip(vv.tolist(), ii.tolist())]
                            for vv, ii in zip(v, i)]
    if was_training:
        layer.train()
    return rep


def score_attribution(head, x: Tensor, edge_index: Tensor,
                      batch: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    r""":math:`s_{ij}` 对 logit 的贡献有多大（把 :math:`s_{ij}` 置零后重算一遍）。

    ``delta_l1_mean`` 接近 0 说明这个头基本没用可解释量（读数是装饰）；
    接近 logit 本身的量级说明结论主要由 :math:`U,V,\gamma,\alpha` 决定。
    """
    was_training = head.training
    head.eval()
    with torch.no_grad():
        a = head.forward_with(x, edge_index, batch)
        b = head.forward_with(x, edge_index, batch, zero_score=True)
        d = (a - b).abs()
        rep = {'logit_abs_mean': float(a.abs().mean()),
               'delta_l1_mean': float(d.mean()),
               'delta_l1_max': float(d.max()),
               'argmax_flip_frac': float((a.argmax(-1) != b.argmax(-1)).float().mean())}
    if was_training:
        head.train()
    return rep


def dump_pairings(path: str, model, eps: float = 1e-3) -> str:
    """把学到的 :math:`U,V,W` 存成 ``.npz``（画热图/写论文表格的输入）。

    与 ``training.diagnostics.dump_gates`` 的分工：那边存**逐次前向**的门控统计
    （随边集变），这边存**参数本身**（只随训练变），两者不要混在一个文件里。
    """
    arrays: Dict[str, np.ndarray] = {}
    for name, p in pairings(model):
        key = name.replace('.', '_')
        with torch.no_grad():
            arrays[f'{key}__U'] = p.U.detach().cpu().numpy()
            arrays[f'{key}__V'] = p.V.detach().cpu().numpy()
            arrays[f'{key}__W'] = p.W().detach().cpu().numpy()
        arrays[f'{key}__meta'] = np.array([p.rank, p.n_rel, p.d_src, p.d_dst, eps])
    np.savez_compressed(path, **arrays)
    return path
