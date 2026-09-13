"""指标层：macro-F1 / 相关系数必须与手算一致。

没有 sklearn，所以这些实现全部手写 —— 手写的东西必须被钉住，否则论文表格里的
F1 与别人的不可比（最常见的坑：把真实标签里没出现的类也算进宏平均）。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                    # noqa: E402

from metrics.homophily import (edge_homophily, gate_homophily_report,   # noqa: E402
                               local_homophily, pearson, spearman)
from metrics.scoring import accuracy, prf1, summarize             # noqa: E402
from test.support import case, run_registered                     # noqa: E402


@case
def test_prf1_matches_hand_computed():
    y = torch.tensor([0, 0, 0, 1, 1, 2])
    pred = torch.tensor([0, 0, 1, 1, 1, 2])
    out = prf1(pred, y)
    # 类0: P=2/2, R=2/3 -> F1=0.8；类1: P=2/3, R=2/2 -> F1=0.8；类2: 1.0
    assert abs(out['acc'] - 5 / 6) < 1e-12, out
    assert abs(out['f1'] - (0.8 + 0.8 + 1.0) / 3) < 1e-12, out
    assert abs(out['precision'] - (1.0 + 2 / 3 + 1.0) / 3) < 1e-12, out
    assert abs(out['recall'] - (2 / 3 + 1.0 + 1.0) / 3) < 1e-12, out
    print(f'  P/R/F1 与手算一致 OK   acc={out["acc"]:.4f} f1={out["f1"]:.4f}')


@case
def test_macro_f1_ignores_absent_classes():
    """真实标签里没出现的类不得进宏平均，否则 tiny split 下被系统性压低。"""
    y = torch.tensor([0, 0, 1])
    pred = torch.tensor([0, 1, 1])
    mask = torch.ones(3, dtype=torch.bool)
    out = prf1(pred, y, mask, num_classes=5)
    # 类0: P=1, R=1/2 -> F1=2/3；类1: P=1/2, R=1 -> F1=2/3；类 2/3/4 未出现 => 不计
    assert abs(out['f1'] - 2 / 3) < 1e-12, out
    assert abs(out['acc'] - 2 / 3) < 1e-12, out
    # mask 生效：只算被选中的节点
    half = torch.tensor([True, True, False])
    assert abs(prf1(pred, y, half)['acc'] - 0.5) < 1e-12
    assert accuracy(pred, y, half) == 0.5
    print('  宏平均只计出现过的类 OK；mask 正确切片')


@case
def test_summarize_uses_unbiased_std():
    s = summarize([1.0, 2.0, 3.0])
    assert s['mean'] == 2.0 and abs(s['std'] - 1.0) < 1e-12, s
    assert summarize([0.5])['std'] == 0.0
    try:
        summarize([])
    except ValueError:
        pass
    else:
        raise AssertionError('空 runs 应报错')
    print('  多种子汇总 OK（无偏 std，单种子 std=0）')


@case
def test_correlations():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert abs(pearson(x, 2 * x + 1) - 1.0) < 1e-12
    assert abs(pearson(x, -x) + 1.0) < 1e-12
    y = x ** 3                                    # 单调但非线性
    assert abs(spearman(x, y) - 1.0) < 1e-12, spearman(x, y)
    assert pearson(x, y) < 1.0                    # 皮尔逊惩罚非线性
    assert pearson(x, torch.ones_like(x)) != pearson(x, torch.ones_like(x))  # nan
    print('  Pearson/Spearman OK（线性=1、单调非线性 ρ=1 而 r<1、常数=>NaN）')


@case
def test_gate_homophily_report_direction():
    """门控与同质性完全负相关时，r 必须 ≈ -1，且高/低同质性分组差为负。"""
    ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    y = torch.tensor([0, 0, 1, 1])
    h, valid = local_homophily(ei, y, 4)
    gate = 1.0 - h                                   # 越异配越竞争：这才是想要的方向
    rep = gate_homophily_report(gate, h, valid, extra={'alpha': h})
    assert abs(rep['r_pearson'] + 1.0) < 1e-12, rep
    assert rep['gate_gap'] < 0, rep                  # 高同质组的门控更低
    assert rep['alpha_mean'] is not None
    assert edge_homophily(ei, y) > 0
    print(f'  门控-同质性报告 OK   r={rep["r_pearson"]:.3f} '
          f'gap={rep["gate_gap"]:.3f} n={int(rep["n"])}')


if __name__ == '__main__':
    sys.exit(run_registered('== 指标层 =='))
