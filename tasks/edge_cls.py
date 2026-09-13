r"""边级任务：关系分类 / 链接存在性（对应 ultralytics ``models/yolo/detect/`` 那一层的
另一个 task，本项目里就是 ``EdgeScore`` 收尾的结构表）。

与节点分类的三点不同，全部落在 hooks 里，trainer 一行都不用改：

1. **诊断报的不是门控而是配对**：:math:`U,V` 的稀疏度、支撑集是否分离
   （§2.7 可识别性的可检验形式）、:math:`\alpha` 的偏斜（非对称到底学到没有）；
2. **落盘的是参数而不是前向统计**：``pairings.npz`` 里是训练完的 :math:`U,V,W`，
   与 ``gates.npz``（逐次前向的 λ）分工不同，不混在一个文件里；
3. ``y``/三个 mask 索引的是**边**，这件事由 ``ds.supervision='edge'`` 声明，
   损失那边（:mod:`training.losses`）不需要知道 —— 它只做 ``logits[mask]``。

诊断值必须是标量或数值列表：``run_experiment`` 会把 ``results.json`` 里的
``diagnostics`` 逐项 ``float()``，嵌套 dict 会当场 TypeError，所以这里摊平。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch import nn

from config import TrainConfig
from dataset.base import GraphBundle
from modules.base import explainable_layers
from modules.dia.explain import dump_pairings, pairing_report
from training.base import DiagFn, TrainHooks

__all__ = ['build_hooks', 'diagnostics_for', 'pairing_diagnostics']


def _flat(prefix: str, d: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, Any]:
    """只挑标量与数值列表；嵌套 dict 交给专门的函数处理（否则 JSON 里全是 str）。"""
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, (list, tuple)) and v and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            out[key] = [float(x) for x in v]
    return out


@torch.no_grad()
def pairing_diagnostics(model: nn.Module, ds: GraphBundle,
                        eps: float = 1e-3) -> Dict[str, Any]:
    r"""配对矩阵与三层读数的扁平摘要。

    必须先跑一次前向：:math:`\gamma/\alpha/s` 是**边集相关**的量，只存在于
    ``aux`` 里；而 :math:`U,V` 是参数，任何时候都能读。两类量在这里合并成一份报告，
    但键名分得开（``l*`` = 逐层前向量，``p*`` = 参数）。
    """
    was_training = model.training
    model.eval()
    model(**ds.forward_kwargs())                  # 只为填 aux，不取输出
    layers = explainable_layers(model, family='dia')
    out: Dict[str, Any] = {'num_dia_layers': float(len(layers))}
    skew, gamma, disjoint = [], [], []
    for t, lay in enumerate(layers):
        rep = lay.explain()
        _flat(f'l{t}_', rep, out)
        if 'alpha_skew' in rep:
            skew.append(float(rep['alpha_skew']))
        if 'gamma_i_mean' in rep:
            gamma.append(float(rep['gamma_i_mean']))
    for q, r in enumerate(pairing_report(model, eps).values()):
        # 路径名不落进诊断：``results.json`` 那一层会对每个值做 ``float()``，
        # 字符串会当场 ValueError。顺序本身就是 ``named_modules()`` 序（= 构建序），
        # 想对上名字去 ``pairings.npz`` 的键里看。
        _flat(f'p{q}_', r, out)
        disjoint.append(float(bool(r['disjoint_U']) and bool(r['disjoint_V'])))
    if skew:
        out['alpha_skew_mean'] = sum(skew) / len(skew)
    if gamma:
        out['gamma_mean'] = sum(gamma) / len(gamma)
    if disjoint:
        # 分离条件成立的比例：0 = 完全没学到可识别的分解，1 = 每个槽管不相交的维度
        out['disjoint_frac'] = sum(disjoint) / len(disjoint)
    if was_training:
        model.train()
    return out


def _pairings(model: nn.Module):
    from modules.dia.explain import pairings
    return pairings(model)


def diagnostics_for(model: nn.Module) -> Optional[DiagFn]:
    """有 DIA 层才做配对诊断（换个边级头就没有 U/V 可报）。"""
    return pairing_diagnostics if explainable_layers(model, family='dia') else None


def _dump(path: str, model: nn.Module, ds: GraphBundle) -> str:
    """适配 :data:`training.base.DumpFn` 的签名（``dump_pairings`` 不需要 ds）。"""
    return dump_pairings(path, model)


def build_hooks(cfg: TrainConfig, ds: GraphBundle, model: nn.Module,
                device: Optional[Any] = None) -> TrainHooks:
    dumps: Dict[str, Callable] = {}
    if cfg.run.dump_pairings and _pairings(model):
        dumps['pairings'] = _dump
    return TrainHooks.default(cfg, ds, device,
                              diagnostics=diagnostics_for(model), dumps=dumps)
