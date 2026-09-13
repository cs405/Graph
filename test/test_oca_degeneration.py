"""严格退化到 GAT（docs §三 的七条件）+ B/C 模式的解耦判据。

这组用例是「原算子 vs 插件」答辩的硬证据：退化必须是**逐元素相等**，
不是「差不多」。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                  # noqa: E402
from torch_geometric.utils import degree                      # noqa: E402

from dataset.synthetic import ba_graph                          # noqa: E402
from modules.oca import OCAConfig, OCALayer                     # noqa: E402
from test.support import case, field_kernel, gat_reference, run_registered   # noqa: E402
from utils.graph import augment_edge_index                      # noqa: E402


@case
def test_gat_degeneration_is_exact():
    """主张：λ=0 ∧ φ≡0 ∧ α=0 ∧ τ=1 ∧ T=0 ∧ 加法式打分 ∧ GAT 输出 => 逐元素等于 GAT。"""
    x, ei = ba_graph(n=61, m=2, seed=1, feat_dim=12)
    n = x.size(0)
    eh, is_self = augment_edge_index(ei, n, True)
    nb = eh[:, ~is_self]
    worst = 0.0
    for heads, out in ((1, 16), (4, 16), (8, 32)):
        layer = OCALayer.gat_equivalent(x.size(1), out_dim=out,
                                        heads=heads).eval()
        with torch.no_grad():
            y = layer(x, ei)
            ref = gat_reference(x, nb, layer.W, layer.gat_att, heads)
        d = (y - ref).abs().max().item()
        assert d < 1e-12, (heads, out, d)
        assert y.shape == (n, out)
        worst = max(worst, d)
    print(f'  GAT 严格退化 OK   heads=1/4/8, max|diff|={worst:.2e}（含 8 头 concat）')


@case
def test_gat_mode_has_no_dead_branches():
    """gat_equivalent 不应分配用不到的参数，否则与 GAT 的参数量不可比。"""
    layer = OCALayer.gat_equivalent(12, out_dim=16, heads=4)
    for name in ('phi', 'q_proj', 'w_lambda', 'w_alpha', 'w_tau', 'out_mlp',
                 'skip', 'norm', 'W_s', 'W_q', 'W_k'):
        assert not hasattr(layer, name), f'gat 模式下不应存在 {name}'
    n = sum(p.numel() for p in layer.parameters())
    assert n == 12 * 16 + 16 + 16 * 8 + 8, n          # W(+bias) 与 gat_att(+bias)
    print(f'  gat 模式参数量 {n}（无死分支，逐项对得上）')


@case
def test_b_mode_decouples_center():
    """设计修正的判据：alpha -> 0 时中心的 kappa 行列必须恒为 0。

    v1 里 alpha 只缩放提案 (1-alpha)b_a，中心恒在场内，所以 §3 表格声称的
    「alpha -> 0 得到 B 模式」并不成立。这里直接检验修正后的归一化核结构：
    K[0,:] 与 K[:,0] 是否随 alpha 消失。
    """
    x, ei = ba_graph(n=55, m=3, seed=1, feat_dim=12)
    n = x.size(0)
    stats = {}
    for ab, tag in ((-40.0, 'B (alpha->0)'), (0.0, 'mid'),
                    (40.0, 'C (alpha->1)')):
        cfg = OCAConfig(out_dim=16, heads=2, T=4, alpha_bias_init=ab,
                        lambda_bias_init=-1.0)
        layer = OCALayer(x.size(1), cfg).eval()
        with torch.no_grad():
            layer(x, ei)
        i = int(torch.argmax(degree(ei[0], num_nodes=n)))     # 度数最大的 hub 场
        e, K, c, lam = field_kernel(layer, i)
        a = float(layer.aux['alpha'].min())
        # 中心所在行/列（槽位 0）对邻居的影响
        influence = max(K[0, 1:].abs().max().item(), K[1:, 0].abs().max().item())
        offdiag = K[1:, 1:].abs().max().item()
        stats[tag] = (a, influence, offdiag)
        assert offdiag > 0, '邻居间抑制必须存在，否则用例无意义'
    assert stats['B (alpha->0)'][1] < 1e-12, stats
    assert stats['mid'][1] > 1e-3, stats
    assert stats['C (alpha->1)'][1] > 1e-2, stats
    print('  B/C 解耦判据 OK   ' + '  '.join(
        f'{k}: alpha_min={v[0]:.2g} 中心->邻居影响={v[1]:.2e}'
        for k, v in stats.items()))


if __name__ == '__main__':
    sys.exit(run_registered('== 严格退化 / B-C 解耦 =='))
