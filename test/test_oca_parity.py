"""等价性与入口归一化：稀疏因子化必须逐元素等于稠密显式建矩阵。

P0-2 的核心主张（O(|E|d) 且**无度数截断**）全部落在这里。

跑：``python test/test_oca_parity.py``
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                               # noqa: E402
from torch_geometric.utils import (coalesce, degree, remove_self_loops,   # noqa: E402
                                   to_undirected)

from dataset.synthetic import ba_graph, dup_directed_graph   # noqa: E402
from modules.oca import OCAConfig, OCALayer                  # noqa: E402
from test.support import case, gate_stats, run_registered    # noqa: E402
from utils.graph import augment_edge_index                   # noqa: E402


@case
def test_sparse_equals_dense():
    """覆盖 heads ∈ {1,2}、T ∈ {1,3}、alpha -> {~0 纯B, 中间, ~1 纯C}。"""
    x, ei = ba_graph(n=97, m=3, seed=1, feat_dim=12)
    n = x.size(0)
    worst = 0.0
    for heads in (1, 2):
        for T in (1, 3):
            for ab in (-40.0, -0.4, 40.0):
                cfg = OCAConfig(out_dim=16, heads=heads, T=T, beta=0.5,
                                alpha_bias_init=ab, lambda_bias_init=-1.0)
                layer = OCALayer(x.size(1), cfg).eval()
                with torch.no_grad():
                    y_sp = layer(x, ei)
                    y_de = layer.forward_dense(x, ei)
                assert y_sp.shape == y_de.shape == (n, 16)
                worst = max(worst, (y_sp - y_de).abs().max().item())
                mean_a, min_a, max_a = gate_stats(layer, 'alpha')
                if ab == -40.0:
                    assert max_a < 1e-15, (min_a, max_a)     # 门控确实关掉所有中心
                elif ab == 40.0:
                    assert min_a > 1 - 1e-15, (min_a, max_a)
    assert worst < 1e-10, worst
    print(f'  sparse == dense，最大绝对误差 {worst:.2e}（12 组配置）')


@case
def test_directed_dupe_input():
    """自环 / 重边 / 单向边输入不得产生 NaN 或重复计数。"""
    x, ei = dup_directed_graph(n=40, seed=2, feat_dim=8)
    n = x.size(0)
    assert (ei[0] == ei[1]).any(), '用例需自带自环'
    eh, is_self = augment_edge_index(ei, n)
    assert eh.size(1) == to_undirected(
        coalesce(remove_self_loops(ei)[0], num_nodes=n),
        num_nodes=n).size(1) + n, '增广后应无重边、无残留自环'
    assert is_self.sum().item() == n, '每个节点恰一个中心槽位'
    degp = degree(eh[0], num_nodes=n)
    assert (degp >= 1).all(), 'degp = deg+1 恒 >= 1（归一化子有下界）'
    layer = OCALayer(x.size(1), OCAConfig(out_dim=16, T=3)).eval()
    with torch.no_grad():
        assert torch.isfinite(layer(x, ei)).all()
    print(f'  自环/重边/单向输入 OK   E_hat={eh.size(1)}  '
          f'max_deg={int(degp.max())}')


@case
def test_no_degree_truncation():
    """主张：不再有 K=8 静默截断，全部邻居进入竞争场。"""
    x, ei = ba_graph(n=400, m=20, seed=3, feat_dim=12)
    n = x.size(0)
    layer = OCALayer(x.size(1), OCAConfig(out_dim=32, heads=4, T=2)).eval()
    eh, is_self = augment_edge_index(ei, n)
    raw = degree(to_undirected(coalesce(remove_self_loops(ei)[0], num_nodes=n),
                               num_nodes=n)[0], num_nodes=n)
    got = degree(eh[0], num_nodes=n) - 1.0                    # 去掉自环
    assert torch.equal(raw, got), '场内槽位数 == 原图度数，无截断'
    assert got.max() > 8, '用例需含 degree > 8 的节点才有意义'
    with torch.no_grad():
        assert torch.isfinite(layer(x, ei)).all()
    print(f'  无截断 OK   max_deg={int(got.max())}'
          '（原实现在 K=8 处静默丢弃）')


@case
def test_dense_relu_ablation_path():
    """docs §八 的「OCA 核消融」行声称可跑，就得有用例撑着。

    注意：dense 的 relu 分支同样经过归一化，所以它稳；v1 真正的问题是
    「无归一化」（见 test_oca_stability.test_old_unnormalized_kernel_diverges）。
    两者不能混为一谈。
    """
    x, ei = ba_graph(n=55, m=3, seed=1, feat_dim=12)
    n = x.size(0)
    outs = {}
    for kern in ('shifted_cosine', 'relu'):
        cfg = OCAConfig(out_dim=16, heads=2, T=3, backend='dense', kernel=kern)
        torch.manual_seed(0)                       # 两分支共用同一份初始化
        layer = OCALayer(x.size(1), cfg).eval()
        with torch.no_grad():
            outs[kern] = layer.forward_dense(x, ei)
        assert torch.isfinite(outs[kern]).all(), kern
        assert outs[kern].shape == (n, 16)
    assert not torch.allclose(outs['shifted_cosine'], outs['relu']), \
        '换核未生效，消融行是空的'
    print('  dense 核消融路径 OK（两种核均有限且结果不同）')


@case
def test_invalid_configs_rejected():
    """非法配置必须在构造期就炸，不能等到 forward 才发现。"""
    for kw, exc in ((dict(beta=1.0), AssertionError),
                    (dict(beta=0.0), AssertionError),
                    (dict(out_dim=16, heads=3), AssertionError),
                    (dict(backend='sparse', kernel='relu'),
                     NotImplementedError)):
        try:
            OCAConfig(**kw)
        except exc:
            pass
        else:
            raise AssertionError(f'{kw} 应被拒绝')
    print('  非法配置被拒绝 OK（beta 越界 / heads 不整除 / sparse+relu）')


if __name__ == '__main__':
    sys.exit(run_registered('== 等价性 / 入口归一化 =='))
