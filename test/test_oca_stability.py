r"""稳定性（P0-1）：hub + :math:`\lambda\to1` 不发散，且反向对照证明归一化确为必需。

两个方向都要验：

* 正面：度数 ~2000 的 hub、:math:`\lambda\approx 1`、:math:`\alpha\approx1`、T=40
  下 :math:`|s|` 仍有界（Lemma 2 与度数无关的断言就靠它）；
* 反向：把核换回 v1 的原式 :math:`\mathrm{ReLU}(\langle q_a,q_b\rangle)/\sqrt{d}`
  （无归一化）必须在 hub 上炸。若这个不炸，说明 P0-1 的严重性判断有误。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                  # noqa: E402
import torch.nn.functional as F                                # noqa: E402
from torch_geometric.utils import degree                       # noqa: E402

from dataset.synthetic import ba_graph                          # noqa: E402
from modules.oca import OCAConfig, OCALayer                     # noqa: E402
from test.support import case, run_registered                   # noqa: E402


@case
def test_stability_extreme_hub_lambda1():
    n, m = 2500, 16
    x, ei = ba_graph(n=n, m=m, seed=7, feat_dim=12)
    hub_deg = int(degree(ei[0], num_nodes=n).max())
    layer = OCALayer(x.size(1), OCAConfig(
        out_dim=32, heads=4, T=40, beta=0.5)).eval()
    with torch.no_grad():
        layer.w_lambda.weight.zero_()
        layer.w_lambda.bias.fill_(30.0)       # lam ≈ 1
        layer.w_alpha.weight.zero_()
        layer.w_alpha.bias.fill_(30.0)        # alpha ≈ 1（纯 C）
        y = layer(x, ei)
        s, lam = layer.aux['s'], layer.aux['lambda']
    assert torch.isfinite(y).all() and torch.isfinite(s).all()
    assert s.abs().max() < 1e3, s.abs().max()
    assert lam.min() > 1 - 1e-10, lam.min()   # 确实验到了 λ=1，不是 λ 很小
    print(f'  hub 稳定 OK   max_deg={hub_deg}  lam_min={lam.min():.6f}  T=40  '
          f'max|s|={s.abs().max():.3f}')


@case
def test_old_unnormalized_kernel_diverges():
    """反向验证：v1 的无归一化 ReLU 核在 hub 上必炸。"""
    n, m = 2500, 16
    x, ei = ba_graph(n=n, m=m, seed=7, feat_dim=12)
    layer = OCALayer(x.size(1), OCAConfig(out_dim=32, heads=4, T=40)).eval()
    with torch.no_grad():
        q = layer.q_proj(layer.W(x)).view(n, 4, 8)             # 故意不归一化
        hub = int(torch.argmax(degree(ei[0], num_nodes=n)))
        g = (ei[0] == hub).nonzero().view(-1)
        qh = q[ei[1][g]]                                       # 场内槽位
        kap = F.relu(torch.einsum('ahi,bhi->abh', qh, qh)) / (8 ** 0.5)
        idx = torch.arange(g.numel())
        kap[idx, idx] = 0.0
        s = torch.randn(g.numel(), 4)
        blew, it = False, -1
        for it in range(40):
            s = 0.5 * s + 0.5 * (-1.0 * torch.einsum('abh,bh->ah', kap, s))
            s = s - s.mean(0, keepdim=True)
            if not torch.isfinite(s).all() or s.abs().max() > 1e12:
                blew = True
                break
    assert blew, '未归一化核在 hub 上未发散 —— P0-1 的必要性需重新评估'
    print(f'  原式（无归一化）第 {it + 1} 轮即失控  '
          f'max|s|={s.abs().max():.3e}  -> 归一化确为必需')


@case
def test_isolated_nodes():
    """孤立节点：场 = {i}，无邻居可聚合 => 输出有限、无 NaN。"""
    n = 30
    ei = torch.stack([torch.arange(0, 20), torch.arange(1, 21)])
    x = torch.randn(n, 8)
    layer = OCALayer(8, OCAConfig(out_dim=16, T=3)).eval()
    with torch.no_grad():
        y = layer(x, ei)
    assert torch.isfinite(y).all(), '孤立节点 25..29 输出必须有限'
    print('  孤立节点 OK')


@case
def test_empty_edge_index():
    x = torch.randn(5, 8)
    layer = OCALayer(8, OCAConfig(out_dim=16, T=2)).eval()
    with torch.no_grad():
        y = layer(x, torch.empty(2, 0, dtype=torch.long))
    assert torch.isfinite(y).all() and y.shape == (5, 16)
    print('  空边集 OK')


if __name__ == '__main__':
    sys.exit(run_registered('== 稳定性 / 边界情形 =='))
