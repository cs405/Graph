"""测试公共件：用例注册器 + **独立重建**的参考实现。

为什么参考实现放在这里而不是复用被测代码：

* :func:`field_kernel` 用稠密方式显式建出节点 i 的归一化抑制核，槽位序（中心在
  首位）与 ``utils.graph.slots_by_center`` 一致，但**不复用**它的分组逻辑；
* :func:`gat_reference` 用 ``[N, N]`` 掩码 + ``torch.softmax`` 实现 GAT，不复用
  ``OCALayer`` 里的 ``scatter``/``softmax``。

两边共用的是数学定义，不是代码路径 —— 否则「稀疏 == 稠密」这种等价性测试会退化成
自己等于自己。

跑法（环境里没有 pytest，所以自带注册器）：

    python -m test.run_all            # 全部
    python test/test_oca_energy.py    # 单文件
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

# 全部数值比对都在 CPU + float64 下做：index_add_ 在 CUDA 上不满足结合律，
# 位级不一致会让 1e-16 量级的断言随机失败。
torch.set_default_dtype(torch.float64)
# Windows 控制台/重定向默认 GBK，打印 ∇ ‖ λ 等符号会 UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

__all__ = ['RESULTS', 'case', 'run_registered', 'field_kernel',
           'gat_reference', 'gate_stats']

RESULTS: List[Callable[[], None]] = []


def case(fn: Callable[[], None]) -> Callable[[], None]:
    """登记一个用例。函数名以 ``test_`` 开头，装上 pytest 后也能直接被收集。"""
    RESULTS.append(fn)
    return fn


def run_registered(header: str = '') -> int:
    """按登记顺序执行，返回退出码。失败不中断，末尾汇总。"""
    if header:
        print(header)
    fails = 0
    for fn in RESULTS:
        try:
            fn()
            print(f'[PASS] {fn.__name__}\n')
        except Exception:
            fails += 1
            print(f'[FAIL] {fn.__name__}')
            traceback.print_exc()
            print()
    total = len(RESULTS)
    print('=' * 66)
    print(f'{total - fails}/{total} 通过')
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# 独立重建
# ---------------------------------------------------------------------------


def field_kernel(layer, i: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """按稠密方式独立重建节点 i 的归一化抑制核与驱动项。

    Returns:
        e: 本场槽位对应的全局边下标（中心在首位）
        K: [m, m, H] 归一化抑制核（已去对角）
        c: [m, H] 驱动力 :math:`\\phi_a + (1-\\alpha_i)b_a`
        lam: 标量 :math:`\\lambda_i`
    """
    aux = layer.aux
    t, ctx = aux['t'], aux['ctx']
    src, is_self = t['src'], t['is_self']
    g = (src == i).nonzero().view(-1)
    e = torch.cat([g[is_self[g]][:1], g[~is_self[g]]])          # 中心置首位
    q = ctx['q'][t['dst'][e]]                                   # [m,H,dh]
    kap = torch.einsum('ahi,bhi->abh', q, q) + 1.0              # shifted cosine
    m_ = e.numel()
    c = torch.zeros(m_)
    c[0] = 1.0
    # 必须取本场的 alp_e/lam_e（e[0] 是本场的中心槽位），而不是全局边 0
    a = t['alp_e'][e[0], 0]
    M = 1.0 - (1.0 - a) * (c[:, None] + c[None, :] - c[:, None] * c[None, :])
    D = (kap * M[..., None]).sum(1).clamp(min=1.0)              # [m,H] 行和
    isd = D.rsqrt()
    K = kap * M[..., None] * isd[:, None, :] * isd[None, :, :]
    K[torch.arange(m_), torch.arange(m_)] = 0.0                 # b != a
    return e, K, t['c'][e].contiguous(), t['lam_e'][e[0], 0]


def gat_reference(x: Tensor, ei_nb: Tensor, W, att, H: int,
                  neg_slope: float = 0.2) -> Tensor:
    """独立实现的稠密 GAT（无自环、中心不参与）。

    ``e_ij = LeakyReLU(a_src^T z_i + a_dst^T z_j)``，``z = Wx``，
    对 :math:`\\mathcal{N}(i)` 做行 softmax，输出 ``ELU(concat_h Σ p z)``。
    """
    N, d = x.size(0), W.out_features
    z = F.linear(x, W.weight, W.bias)
    u = F.linear(z, att.weight, att.bias)                       # [N, 2H]
    A = torch.zeros(N, N, dtype=torch.bool)
    A[ei_nb[0], ei_nb[1]] = True
    # e[i, j, h] = a_src^T z_i + a_dst^T z_j：前者按行广播，后者按列广播
    e = F.leaky_relu(u[:, :H].unsqueeze(1) + u[:, H:].unsqueeze(0), neg_slope)
    e = e.masked_fill(~A.unsqueeze(-1), float('-inf'))
    has_nb = A.sum(1) > 0
    e = torch.where(has_nb.unsqueeze(-1), e, torch.zeros_like(e))
    p = torch.softmax(e, dim=1) * has_nb.unsqueeze(-1)          # 孤立行 => 全 0
    m = torch.einsum('ijh,jhd->ihd', p, z.view(N, H, -1))
    return F.elu(m.reshape(N, d))


def gate_stats(layer, key: str = 'alpha') -> Tuple[float, float, float]:
    """节点级门控的 (mean, min, max)。

    用 :data:`OCALayer.aux` 里的节点级量，而不是某条边上的 ``alp_e``
    —— 后者只是某个中心的值，拿它断言「alpha 全局被关掉」是偷懒。
    """
    v: Dict[str, Tensor] = layer.aux
    assert key in v, f'aux 里没有 {key}（对应开关未开？）'
    g = v[key].reshape(-1)
    return g.mean().item(), g.min().item(), g.max().item()
