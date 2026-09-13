"""能量一致性（Lemma 1）与收敛到闭式极小点（P0-1 的正面证据）。

这两个用例是「有能量函数解释」这一卖点的**唯一定量**支撑：

* 代码的一步更新必须等于对 E 的投影梯度下降一步（autograd 数值梯度对拍）；
* 迭代必须收敛到 KKT 闭式解，且速率不超过 :math:`\\rho^T`。

能量梯度一致性对**全图每个场**都验，只验一个方便的 hub 场可能碰巧成立。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                   # noqa: E402

from dataset.synthetic import ba_graph                           # noqa: E402
from modules.oca import OCAConfig, OCALayer                      # noqa: E402
from test.support import case, field_kernel, run_registered      # noqa: E402


@case
def test_energy_gradient_matches_update():
    """Lemma 1：代码的更新式 == 对 E(s) 的（投影）梯度下降一步。"""
    x, ei = ba_graph(n=41, m=3, seed=1, feat_dim=12)
    n = x.size(0)
    cfg = OCAConfig(out_dim=8, heads=2, T=3, beta=0.37,
                    lambda_bias_init=-1.0, alpha_bias_init=-0.3)
    layer = OCALayer(x.size(1), cfg).eval()
    with torch.no_grad():
        layer(x, ei)
    t, ctx, s_full = layer.aux['t'], layer.aux['ctx'], layer.aux['s']

    # 一步迭代在全图上跑（_competition_step 作用于边级张量 [E_hat, H]）
    s1_full = layer._competition_step(s_full, t, ctx, layer.aux['z'])

    def E(sv, P, c, K, lam):
        # 约束在零均值子空间：E(sv) = Etilde(P sv) => ∇E = P ∇Etilde
        u = torch.einsum('ab,bh->ah', P, sv)
        return ((-c * u).sum()
                + 0.5 * lam * torch.einsum('abh,ah,bh->', K, u, u)
                + 0.5 * (u * u).sum())

    worst_g = worst_step = 0.0
    checked = 0
    for i in range(n):
        g = (t['src'] == i).nonzero().view(-1)
        if g.numel() < 3:
            continue
        checked += 1
        e, K, c, lam = field_kernel(layer, i)
        s = s_full[e].contiguous()                             # [m,H]
        m_ = e.numel()
        P = torch.eye(m_) - 1.0 / m_
        assert (s - torch.einsum('ab,bh->ah', P, s)).abs().max() < 1e-12, \
            's 应已零均值（中心化是投影步）'

        sv = s.clone().requires_grad_(True)
        g_num = torch.autograd.grad(E(sv, P, c, K, lam), sv)[0]
        g_ana = -c + lam * torch.einsum('abh,bh->ah', K, s) + s
        worst_g = max(worst_g, (g_num - torch.einsum('ab,bh->ah', P, g_ana)
                                ).abs().max().item())
        # 一步迭代 = PGD：s - β∇E 再投影（P 幂等且 Ps=s，故用 g_ana 等价）
        s1_ref = torch.einsum('ab,bh->ah', P, s - cfg.beta * g_ana)
        worst_step = max(worst_step,
                         (s1_full[e] - s1_ref).abs().max().item())

    assert checked >= 20, checked
    assert worst_g < 1e-10, worst_g
    assert worst_step < 1e-10, worst_step
    print(f'  能量梯度一致性 OK（{checked} 个场全查）  '
          f'max|∇_num - P∇_ana|={worst_g:.2e}  '
          f'max|step - PGD|={worst_step:.2e}')


@case
def test_converges_to_closed_form_minimizer():
    """极小点由 KKT 线性方程组闭式给出：
    :math:`[I+\\lambda K, -\\mathbf{1}; \\mathbf{1}^T, 0][s; \\nu] = [c; 0]`
    """
    x, ei = ba_graph(n=41, m=3, seed=1, feat_dim=12)
    n = x.size(0)
    beta = 0.5
    cfg = OCAConfig(out_dim=8, heads=1, T=3, beta=beta, lambda_bias_init=-1.0,
                    alpha_bias_init=-0.3)
    layer = OCALayer(x.size(1), cfg).eval()

    def run_T(T):
        """同一份权重下只改 T，否则三次构造的随机初始化不同，比对无意义。"""
        with torch.no_grad():
            layer.cfg.T = T
            layer(x, ei)
        return layer.aux['s'].clone()

    s_by_T = {T: run_T(T) for T in (3, 6, 300)}
    ratios, err300 = [], 0.0
    checked = 0
    for i in range(n):
        if int((layer.aux['t']['src'] == i).sum()) < 3:
            continue
        checked += 1
        e, K, c, lam = field_kernel(layer, i)
        m_ = e.numel()
        A = torch.zeros(m_ + 1, m_ + 1)
        A[:m_, :m_] = torch.eye(m_) + lam.item() * K[:, :, 0]
        A[:m_, m_] = -1.0
        A[m_, :m_] = 1.0
        s_star = torch.linalg.solve(A, torch.cat([c[:, 0], torch.zeros(1)]))[:m_]

        def err(T):
            s = s_by_T[T][e, 0]
            return (s - s.mean() - s_star).abs().max().item()

        er, e3, e6 = err(300), err(3), err(6)
        err300 = max(err300, er)
        rho = max(1 - beta + beta * lam.item(), abs(1 - beta - beta * lam.item()))
        assert rho < 1.0
        # ‖e^{2T}‖ <= ρ^T √m ‖e^T‖（2 范数 vs inf 范数的换算因子 √m 已在界内）
        ratios.append(e6 / (rho ** 3 * (m_ ** 0.5) * e3 + 1e-12))
    assert checked >= 20, checked
    assert err300 < 1e-6, err300
    assert max(ratios) <= 1.0, max(ratios)
    print(f'  收敛到闭式极小点 OK（{checked} 个场）  '
          f'err(T=300) < {err300:.2e}；速率满足 ‖e^2T‖ ≤ ρ^T √m ‖e^T‖'
          f'（最差比 {max(ratios):.3f} ≤ 1）')


if __name__ == '__main__':
    sys.exit(run_registered('== 能量一致性 / 收敛 =='))
