"""分类打分：accuracy / macro-F1 / 多种子汇总。

环境里没有 sklearn，所以手写 —— 关键是 macro-F1 的分母：只对**出现过**的类求平均，
否则 tiny split（每类 20 个训练节点）下某些类一个都没预测出来时会把 0 记进均值，
macro-F1 会被系统性压低，与文献里的数字不可比。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

__all__ = ['accuracy', 'macro_f1', 'prf1', 'summarize']


def _select(pred: Tensor, y: Tensor,
            mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    if mask is not None:
        pred, y = pred[mask], y[mask]
    return pred, y


def _ratio(pred: Tensor, y: Tensor) -> float:
    """用 Python 除法而不是 ``float32.mean()``：后者会把 5/6 截成 0.83333331，
    手算期望值对不上，累加多个种分时也会引入无意义的舍入差。"""
    n = int(y.numel())
    return float('nan') if n == 0 else int((pred == y).sum().item()) / n


def accuracy(pred: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> float:
    return _ratio(*_select(pred, y, mask))


def prf1(pred: Tensor, y: Tensor, mask: Optional[Tensor] = None,
         num_classes: Optional[int] = None) -> Dict[str, float]:
    """返回 ``acc`` 与宏平均 ``precision/recall/f1``。"""
    pred, y = _select(pred, y, mask)
    if y.numel() == 0:
        return {'acc': float('nan'), 'precision': float('nan'),
                'recall': float('nan'), 'f1': float('nan')}
    C = int(max(int(y.max()), int(pred.max()))) + 1 if num_classes is None \
        else num_classes
    # 混淆矩阵：行 = 真类，列 = 预测类（必须在 y 的设备上，否则 CUDA 下 index_put_ 报错）
    cm = torch.zeros(C, C, dtype=torch.long, device=y.device)
    cm.index_put_((y, pred), torch.ones_like(y, dtype=torch.long),
                  accumulate=True)
    tp = cm.diagonal()
    pred_pos = cm.sum(0)
    true_pos = cm.sum(1)
    seen = true_pos > 0                        # 只统计真实标签里出现过的类
    prec = (tp / pred_pos.clamp(min=1)).to(torch.float64)
    rec = (tp / true_pos.clamp(min=1)).to(torch.float64)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-12)
    n = int(seen.sum())
    return {
        'acc': _ratio(pred, y),
        'precision': (prec[seen].sum() / n).item(),
        'recall': (rec[seen].sum() / n).item(),
        'f1': (f1[seen].sum() / n).item(),
    }


def macro_f1(pred: Tensor, y: Tensor, mask: Optional[Tensor] = None,
             num_classes: Optional[int] = None) -> float:
    return prf1(pred, y, mask, num_classes)['f1']


def summarize(runs: Sequence[float]) -> Dict[str, float]:
    """多种子汇总：报 mean ± std（无偏），论文里的「±」就是这里的 std。"""
    v = torch.as_tensor(list(runs), dtype=torch.float64)
    if v.numel() == 0:
        raise ValueError('runs 为空')
    return {'mean': v.mean().item(),
            'std': (v.std(unbiased=True) if v.numel() > 1 else v.new_zeros(()))
            .item(),
            'n': int(v.numel()), 'min': v.min().item(),
            'max': v.max().item()}
