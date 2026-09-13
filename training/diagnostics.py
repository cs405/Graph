r"""门控诊断：把 :math:`\lambda_i/\alpha_i/\tau_i` 与逐节点竞争强度从 ``aux`` 里取出来。

这是「自适应」主张唯一可被检验的形式：门控必须与**独立算出的**局部同质性相关，
而不是与模型自己的预测相关。因此本文件只负责「取数」，相关性一律走
:func:`metrics.homophily.gate_homophily_report`（其输入由 :mod:`dataset` 的
``y`` 与 ``edge_index`` 决定，与模型无关）。

本文件不 import :mod:`modules.oca`：取层走 :func:`modules.base.explainable_layers`
的 ``family='oca'`` 标签。上一版写的是 ``isinstance(m, OCALayer)``，那就意味着
「诊断代码认识具体算子」—— 而 DIA 一进来它就应该只认协议。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor, nn

from dataset.base import GraphBundle
from metrics.homophily import (edge_homophily, gate_homophily_report,
                               local_homophily)
from modules.base import explainable_layers
from utils.graph import scatter_add

__all__ = ['collect_gates', 'gate_report', 'dump_gates']


def _oca_layers(model: nn.Module) -> List[nn.Module]:
    """OCA 那一族的可解释层（按 ``family`` 标签筛，不 import 算子）。"""
    return list(explainable_layers(model, family='oca'))


@torch.no_grad()
def collect_gates(model: nn.Module, ds: GraphBundle, forward: bool = True
                  ) -> Dict[str, Tensor]:
    r"""返回 ``{'lambda': [L,N], 'alpha': [L,N], 'tau': [L,N], 's_abs': [L,N]}``。

    ``s_abs`` 是竞争场里 :math:`|s|` 的按中心平均（不含中心槽位本身），
    它比 :math:`\lambda` 更靠近「这个节点实际被抑制了多少」。
    """
    model.eval()
    if forward:
        model(**ds.forward_kwargs())
    layers = _oca_layers(model)
    assert layers, '模型里没有 family=oca 的可解释层，无法取门控'
    N = ds.num_nodes
    out: Dict[str, list] = {}
    for lay in layers:
        aux = lay.aux
        t = aux.get('t')
        for key, src_key in (('lambda', 'lambda'), ('alpha', 'alpha'),
                             ('tau', 'tau')):
            v = aux.get(src_key)
            if v is None:
                continue
            v = v.reshape(N, -1).mean(-1) if v.dim() > 1 else v.reshape(N)
            out.setdefault(key, []).append(v.double().cpu())
        if t is not None and 's' in aux:
            s = aux['s']                                        # [E_hat, H]
            keep = t['nb'].reshape(-1).bool()                  # 只留邻居槽位（中心不参与）
            idx = t['src'][keep]
            cnt = scatter_add(torch.ones_like(idx, dtype=s.dtype), idx, N)
            acc = scatter_add(s[keep].abs().mean(-1), idx, N)
            out.setdefault('s_abs', []).append((acc / cnt.clamp(min=1.0))
                                               .double().cpu())
    return {k: torch.stack(v) for k, v in out.items()}         # [L, N]


def gate_report(model: nn.Module, ds: GraphBundle, layer: Optional[int] = None,
                ) -> Dict[str, Any]:
    r"""一行摘要：门控统计 + 与局部同质性的相关（相关用第 ``layer`` 层，默认最深层）。

    ``layer=None`` 时取**最后一层**：多层骨架里各层的 :math:`\lambda` 各自为政，
    混在一起平均会掩盖「浅层不竞争、深层竞争」这种真实结构，那种结构该分开报。

    返回值大多是标量，唯独 ``lambda_r_by_layer`` 是一个逐层列表（故类型是 Any）。
    """
    gates = collect_gates(model, ds)
    h, valid = local_homophily(ds.edge_index, ds.y, ds.num_nodes)
    rep: Dict[str, Any] = {'edge_homophily': edge_homophily(ds.edge_index, ds.y),
                           'num_layers': float(len(_oca_layers(model)))}
    rho_by_layer = []
    for key in ('lambda', 'alpha', 'tau', 's_abs'):
        if key not in gates:
            continue
        L = gates[key].size(0)
        li = (L - 1) if layer is None else layer
        for j in range(L):
            if key == 'lambda':
                rho_by_layer.append(
                    gate_homophily_report(gates[key][j], h, valid)['r_pearson'])
        r = gate_homophily_report(gates[key][li], h, valid)
        prefix = f'{key}_l{li}'
        rep[f'{prefix}_mean'] = r['gate_mean']
        rep[f'{prefix}_std'] = r['gate_std']
        rep[f'{prefix}_r_homo'] = r['r_pearson']
        rep[f'{prefix}_rho_homo'] = r['r_spearman']
        rep[f'{prefix}_gap_hilo'] = r['gate_gap']
        rep[f'{prefix}_n'] = r['n']
    if rho_by_layer:
        # 逐层皮尔逊 r：「浅层不竞争、深层竞争」的结构只能在这里看到
        rep['lambda_r_by_layer'] = rho_by_layer
    return rep


def dump_gates(path: str, model: nn.Module, ds: GraphBundle) -> str:
    """存 npz，供画图脚本（λ vs 局部同质性散点）离线使用。

    画图的活儿留给分析脚本；这里只保证**数据出得去**，并且 labels 与门控按节点
    对齐（同一下标 = 同一节点），避免事后拼图时错位。
    """
    gates = collect_gates(model, ds)
    h, valid = local_homophily(ds.edge_index, ds.y, ds.num_nodes)
    with torch.no_grad():                 # 取预测不应保留计算图
        pred = model(**ds.forward_kwargs()).argmax(-1)
    payload = {k: v.numpy() for k, v in gates.items()}
    payload.update({'h_local': h.cpu().numpy(), 'h_valid': valid.cpu().numpy(),
                    'y': ds.y.cpu().numpy(), 'pred': pred.cpu().numpy()})
    np.savez(path, **payload)
    return path
