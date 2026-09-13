r"""OCA: Omni-Competition Attention —— 稀疏、可证明稳定、可严格退化到 GAT。

本文件只放**算子**；多尺度拓扑写在 ``cfg/models/*.yaml`` 里，由 ``model_builder.py`` 展开；
图侧通用件在 ``utils/graph.py``。
设计主张与勘误见 ``docs/OCA.md``，逐条验证见 ``test/``。

相对 docs/OCA.md v1 的 4 处实质修改（详表在文档顶部）：

[P0-1 稳定性]  原式 kappa = ReLU(<q_a,q_b>)/sqrt(d) 无界，迭代矩阵
    (1-beta)I - beta*lam*K 谱半径可 > 1，hub 节点若干轮内发散。
    新核 kappa_ab = <qhat_a, qhat_b> + 1 in [0, 2]：对称、非负，可做 GCN 式归一化。
    ||K||_2 <= 1 的证明走「相似于随机游走矩阵」（不是「归一化后行和 <= 1」——
    混合分母 sqrt(D_a D_b) 控制不住行和，那个写法是错的），从而
        (a) Hessian = I + lam*K >= (1-lam)I > 0        =>  E 强凸
        (b) 谱半径 <= max(1-b+b*lam, |1-b-b*lam|) < 1   =>  线性收敛
    前提仅 beta in (0,1), lam in (0,1) —— 与节点度数完全无关。

[P0-2 可扩展]  删除 max_deg / K=8 稠密 padding 与 `for i in range(N)` 循环。
    关键观察：分数 s 本质是「增广边」上的量 s_{i,a}（同一节点 a 在不同中心 i 的
    竞争场里分数不同），故在 E_hat = E ∪ {(i,i)} 上张量即可；竞争项精确因子化为
    两次 index_add（见 _competition_step 的推导），复杂度 O(|E_hat| d)，无截断。
    代价：kappa 必须线性可因子化 => 稀疏路径放弃 ReLU（+1 偏移保非负）。
    ReLU 分支仅 backend='dense' 参考实现提供，供小图核消融。

[设计修正]     alpha 的语义。v1 中 alpha 只缩放纵向提案，中心恒在场内
    => 「alpha->0 得到 B 模式」并未实现。现在 alpha 同时：
        (1) 衰减中心的 kappa 行列： M_ab = alpha + (1-alpha)(1-c_a)(1-c_b)
        (2) 反向缩放纵向提案：     (1-alpha) * b_a
    alpha=0 => 中心与场完全解耦 = 真 B；alpha=1 => 中心全场参与 = 纯 C。

[理论精确化]   E 严格存在（归一化核对称），但准确表述是「在零均值子空间
    1^T s = 0 上对 E 做投影梯度下降」（§2.5 中心化 = 投影步）。中心化对最终
    readout 的 softmax 是 no-op（平移不变），唯一作用是数值防漂移。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import degree, softmax

from modules.base import Explorable
from utils.graph import augment_edge_index, scatter_add, slots_by_center

__all__ = ['OCAConfig', 'OCALayer']


# ---------------------------------------------------------------------------


@dataclass
class OCAConfig:
    """OCA 层配置。docs/OCA.md §八 的消融表 = 本结构的一组开关组合。"""

    out_dim: int = 64
    heads: int = 1

    # ---- 竞争动力学 ----
    T: int = 2                      # 迭代轮数；T=0 => 跳过竞争与中心化
    beta: float = 0.5               # 梯度下降步长，需 ∈ (0,1)
    kernel: str = 'shifted_cosine'  # 'shifted_cosine'（稀疏必需）| 'relu'（仅 dense）

    # ---- 组件开关（消融用）----
    score_mode: str = 'dot'         # 'dot'（§2.2 纵向提案）| 'gat'（GAT 加法式打分）
    use_proposal: bool = True       # b_j = <W_q h_i, W_k z_j>/sqrt(d)
    use_phi: bool = True            # 自身嗓门 phi(z_a)
    use_competition: bool = True    # 横向抑制 lam；False => lam := 0
    use_center_field: bool = True   # 中心进场 alpha；False => alpha := 0（纯 B）
    use_temperature: bool = True    # 中心调温；False => tau := 1

    # ---- 读出 / 输出 ----
    center_stat: str = 'zscore'     # 喂给温度头的中心统计量 'raw' | 'zscore'
    # ↑ 'zscore' 是历史别名：分母多乘了一个 sqrt(deg_i)，不是标准 z-score。
    #   它的影响 §9.8 已实测：数值不动，只改措辞（详见 _temperature）。
    out_mode: str = 'fusion'        # 'fusion'（§2.9）| 'gat'（严格 GAT 输出）
    gat_negative_slope: float = 0.2
    gat_activation: str = 'elu'
    tau_min: float = 1e-3

    # ---- 门控初始化（P1-3：默认不竞争，竞争要靠挣）----
    lambda_bias_init: float = -2.0  # sigmoid(-2) ~= 0.12
    alpha_bias_init: float = 0.0

    # ---- 工程 ----
    backend: str = 'sparse'          # 'sparse' O(|E|d) | 'dense' 逐节点参考实现
    detach_iterations: bool = False  # §7 的 stop-gradient 技巧
    symmetrize: bool = True

    def __post_init__(self):
        assert self.out_dim % self.heads == 0, \
            f'out_dim({self.out_dim}) 需能被 heads({self.heads}) 整除'
        assert 0.0 < self.beta < 1.0, 'beta 必须 ∈ (0,1) 才是收缩映射'
        assert self.kernel in ('shifted_cosine', 'relu')
        assert self.score_mode in ('dot', 'gat')
        assert self.out_mode in ('fusion', 'gat')
        assert self.backend in ('sparse', 'dense')
        assert self.center_stat in ('raw', 'zscore')
        if self.backend == 'sparse' and self.kernel == 'relu':
            raise NotImplementedError(
                'ReLU 抑制核不可因子化（需 O(sum_i deg_i^2)），'
                "仅 backend='dense' 支持。")


# ---------------------------------------------------------------------------


class OCALayer(nn.Module, Explorable):
    r"""Omni-Competition Attention layer.

    竞争场 :math:`\mathcal{V}_i=\mathcal{N}(i)\cup\{i\}`，场上分数
    :math:`s\in\mathbb{R}^{\hat{\mathcal{E}}\times H}` 是逐 (中心, 槽位) 的。

    每轮迭代 = 在 :math:`\mathbf{1}^\top s = 0` 上对 :math:`E_i` 做一步投影梯度下降::

        s_a <- (1-beta) s_a + beta ( phi_a + (1-alpha_i) b_a 1_{a!=i}
                                    - lambda_i \sum_{b!=a} K^{(i)}_{ab} s_b )
        s   <- s - mean_{b in V_i}(s_b)

    能量函数（x 固定时是 s 的二次型）::

        E_i(s) = - \sum_a phi_a s_a
                 - (1-alpha_i) \sum_{j in N(i)} b_j s_j
                 + (lambda_i/2) s^T K^{(i)} s
                 + 1/2 ||s||^2

    **Lemma 1（E 良定义且与更新式逐项吻合）.**
    :math:`K^{(i)} = (D'^{-1/2}(M^{(i)}\!\circ\!\kappa^{(i)})D'^{-1/2})\big|_{\text{去对角}}`
    对称，故 :math:`E_i` 存在且
    :math:`\nabla_s E_i = -c + \lambda_i K s + s`，
    :math:`c_a := \phi_a + (1-\alpha_i)b_a\mathbb{1}[a\neq i]`；
    上式恰为 :math:`s \leftarrow s - \beta\nabla_s E_i` 后再投影。
    （``test/test_oca_energy.py`` 用 autograd 逐场对拍。）

    **Lemma 2（与度数无关的一致稳定性）.**
    设 :math:`\kappa_{ab}=\langle\hat q_a,\hat q_b\rangle + 1 \ge 0` 对称，
    :math:`M` 对称、:math:`0\le M\le 1`，:math:`A:=M\circ\kappa\ (\ge 0)`，
    :math:`D_a:=\sum_b A_{ab}`、:math:`D'_a:=\max(D_a,1)`。则

    .. math:: \|K\|_2 \le \|D^{-1/2}AD^{-1/2}\|_2 = \rho(D^{-1}A)
            \le \|D^{-1}A\|_\infty = 1 .

    三步依据：(a) :math:`0\le K\le D^{-1/2}AD^{-1/2}` 元素级（抬 :math:`D'` 与
    去对角都只让非负元变小），而对称非负矩阵的谱范数等于其 Perron 根，Perron 根
    对元素级序单调；(b) :math:`D^{-1/2}AD^{-1/2}` 与 :math:`D^{-1}A` 相似 => 同谱；
    (c) :math:`D^{-1}A` 行和恒为 1。:math:`D_a=0` 的行在 :math:`K` 中整行为零，
    不等式平凡成立，故可限制在 :math:`D_a>0` 的子图上用 (b)。

    推论：(i) :math:`\nabla^2 E_i = I+\lambda_i K \succeq (1-\lambda_i)I \succ 0`，
    :math:`E_i` 在 :math:`\mathbf 1^\top s=0` 上有唯一极小点 :math:`s^\star`；
    (ii) 每轮 = 先 :math:`s\leftarrow[(1-\beta)I-\beta\lambda_i K]s+\beta Pc` 再投影，
    :math:`\|P\|_2=1` 且方括号内对称，故
    :math:`\|s^{(T)}-s^\star\| \le \rho^T \|s^{(0)}-s^\star\|`，
    :math:`\rho=\max(1-\beta+\beta\lambda,\,|1-\beta-\beta\lambda|) < 1`。
    只需 :math:`\beta\in(0,1),\lambda\in(0,1)`，与 :math:`|\mathcal{N}(i)|` 无关。
    """

    # ------------------------------------------------------------------

    #: :class:`modules.base.Explorable` 的家族标签。框架侧靠它而不是靠 import
    #: 本模块来挑「哪些可解释层是 OCA 的」（见 ``training/diagnostics.py``）。
    family = 'oca'

    def __init__(self, in_dim: int, cfg: OCAConfig):
        super().__init__()
        self.in_dim = in_dim
        self.cfg = cfg
        self.H = cfg.heads
        self.d = cfg.out_dim
        self.dh = cfg.out_dim // cfg.heads

        self.W = nn.Linear(in_dim, self.d)          # z = W h，打分与取值共享

        if cfg.score_mode == 'dot':
            if cfg.use_proposal:
                self.W_q = nn.Linear(in_dim, self.d)
                self.W_k = nn.Linear(self.d, self.d)
        elif cfg.use_proposal:                       # GAT: a^T[z_i || z_j]
            self.gat_att = nn.Linear(self.d, 2 * self.H)

        if cfg.use_phi:
            self.phi = nn.Sequential(nn.Linear(self.d, self.d), nn.ReLU(),
                                     nn.Linear(self.d, self.H))

        if cfg.use_competition:
            self.q_proj = nn.Linear(self.d, self.d)
            self.w_lambda = nn.Linear(in_dim, 1)
            nn.init.constant_(self.w_lambda.bias, cfg.lambda_bias_init)

        if cfg.use_center_field:
            self.w_alpha = nn.Linear(in_dim, 1)
            nn.init.constant_(self.w_alpha.bias, cfg.alpha_bias_init)

        if cfg.use_temperature:
            self.w_tau = nn.Linear(in_dim + 1, 1)
            nn.init.constant_(self.w_tau.bias, 1.0)   # tau0 ~ softplus(1) ~ 0.72

        if cfg.out_mode == 'gat':
            assert cfg.gat_activation in ('elu', 'relu')
        else:
            self.W_s = nn.Linear(in_dim, self.d)      # MLP 的自身视图（§2.9）
            self.skip = nn.Linear(in_dim, self.d)     # 残差投影（修 shape bug）
            self.out_mlp = nn.Sequential(nn.Linear(2 * self.d, self.d), nn.ReLU(),
                                         nn.Linear(self.d, self.d))
            self.norm = nn.LayerNorm(self.d)

        self.aux: Dict[str, Tensor] = {}              # 供 λ-vs-同质性 等分析

    # ------------------------------------------------------------------

    @classmethod
    def gat_equivalent(cls, in_dim: int, out_dim: int = 64, heads: int = 1,
                       **kw) -> 'OCALayer':
        r"""严格退化到 GAT（docs §三；``test/test_oca_degeneration.py`` 逐元素校验）。

        需同时满足：\(\lambda=0,\ \phi\equiv0,\ \alpha=0\)（中心不进任何计算）,
        \(\tau\equiv1\)，\(T=0\)（连中心化都不做，尽管它对 softmax 是 no-op）,
        加法式 GAT 打分 + GAT 式输出（无残差、无 LayerNorm、逐头拼接后过 ELU）。
        """
        cfg = OCAConfig(out_dim=out_dim, heads=heads, T=0, score_mode='gat',
                        use_proposal=True, use_phi=False, use_competition=False,
                        use_center_field=False, use_temperature=False,
                        out_mode='gat', **kw)
        return cls(in_dim, cfg)

    # ------------------------------------------------------------------

    def explain(self) -> Dict[str, Any]:
        r"""一行摘要：三个门控的均值/方差 + 竞争强度的量级。

        只报标量，不报 ``aux['s']`` 本体：后者是 :math:`[\hat E, H]`，往
        ``results.json`` 里塞一个大数组只会让文件没法读。逐节点的门控取数走
        :func:`training.diagnostics.collect_gates`（它需要 ``edge_index`` 才能把
        边级量按中心聚合，而 ``explain()`` 的签名里没有图）。
        """
        out: Dict[str, Any] = {
            'in_dim': self.in_dim, 'out_dim': self.d, 'heads': self.H,
            'T': self.cfg.T, 'backend': self.cfg.backend,
            'score_mode': self.cfg.score_mode,
            'on': ''.join(str(int(v)) for v in (
                self.cfg.use_proposal, self.cfg.use_phi, self.cfg.use_competition,
                self.cfg.use_center_field, self.cfg.use_temperature)),
        }
        for key in ('lambda', 'alpha', 'tau'):
            v = self.aux.get(key)
            if v is None or not v.numel():
                continue
            v = v.detach()
            out[f'{key}_mean'] = float(v.mean())
            out[f'{key}_std'] = float(v.std()) if v.numel() > 1 else 0.0
        s, t = self.aux.get('s'), self.aux.get('t')
        if s is not None and t is not None:
            keep = t['nb'].reshape(-1).bool()          # 中心槽位不参与竞争强度
            if bool(keep.any()):
                out['s_abs_mean'] = float(s.detach()[keep].abs().mean())
        return out

    # ------------------------------------------------------------------

    def _gates(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        cfg = self.cfg
        z1 = x.new_zeros(x.size(0), 1)
        lam = torch.sigmoid(self.w_lambda(x)) if cfg.use_competition else z1
        alp = torch.sigmoid(self.w_alpha(x)) if cfg.use_center_field else z1
        return lam, alp

    def _edge_terms(self, x: Tensor, z2: Tensor, ei: Tensor, is_self: Tensor,
                    lam: Tensor, alp: Tensor) -> Dict[str, Tensor]:
        """所有「与 s 无关」的边上量（保证 E 是 s 的二次型）。

        z2 为 [N, d] 的二维形式；多头视图在本函数内部按需 reshape。
        """
        cfg = self.cfg
        src, dst = ei[0], ei[1]
        E = src.numel()
        H, dh = self.H, self.dh
        nb = (~is_self).to(z2.dtype).unsqueeze(-1)      # [E,1] 中心槽位无纵向提案

        if cfg.use_proposal:
            if cfg.score_mode == 'dot':
                qi = self.W_q(x).view(-1, H, dh)[src]
                kj = self.W_k(z2).view(-1, H, dh)[dst]
                b = (qi * kj).sum(-1) / (dh ** 0.5) * nb
            else:
                a = self.gat_att(z2)                   # [N, 2H]
                b = F.leaky_relu(a[src, :H] + a[dst, H:],
                                 cfg.gat_negative_slope) * nb
        else:
            b = torch.zeros(E, H, device=z2.device, dtype=z2.dtype)

        if cfg.use_phi:
            phi = self.phi(z2)[dst]                    # [E,H] 中心槽位同样有 phi
        else:
            phi = torch.zeros_like(b)

        return dict(src=src, dst=dst, is_self=is_self, nb=nb, b=b, phi=phi,
                    c=phi + (1.0 - alp[src]) * b,      # 驱动力 c_a（Lemma 1）
                    lam_e=lam[src], alp_e=alp[src])

    def _kernel_ctx(self, z2: Tensor,
                    t: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """归一化抑制核的静态量：qhat / C_i / degp / kappa_{a,i} / Draw。"""
        src, dst, nb = t['src'], t['dst'], t['nb']
        H, dh = self.H, self.dh
        N = z2.size(0)
        q = F.normalize(self.q_proj(z2).view(-1, H, dh), dim=-1)  # 单位向量
        # C_i = sum_{b in V_i} qhat_b，|V_i| = degp_i = deg_i + 1
        C = scatter_add(q[dst], src, N)
        degp = degree(src, num_nodes=N, dtype=z2.dtype).unsqueeze(-1)
        qc = q[src]                                   # 场中心 i 的核向量（按边）
        kap_ci = (qc * q[dst]).sum(-1) + 1.0          # [E,H] kappa_{a,i}
        kap_aa = (q[dst] * q[dst]).sum(-1) + 1.0      # [E,H] 对角（单位向量=>2）
        Draw = (q[dst] * C[src]).sum(-1) + degp[src]  # [E,H] 未屏蔽行和
        c_a = (1.0 - nb)                              # [E,1] 槽位即中心？
        return dict(q=q, C=C, degp=degp, kap_ci=kap_ci, kap_aa=kap_aa,
                    Draw=Draw, c_a=c_a)

    # ------------------------------------------------------------------

    def forward(self, x: Tensor, edge_index: Tensor,
                backend: Optional[str] = None) -> Tensor:
        cfg = self.cfg
        backend = backend or cfg.backend
        # 每次 forward 重写：不然读到的是上一次（可能是另一套开关/后端）的残留
        self.aux.clear()
        if backend == 'dense':
            return self._forward_dense(x, edge_index)
        N, H, d = x.size(0), self.H, self.d
        ei, is_self = augment_edge_index(edge_index, N, cfg.symmetrize)

        lam, alp = self._gates(x)
        z2 = self.W(x)                                 # [N, d]
        z = z2.view(N, H, d // H)
        t = self._edge_terms(x, z2, ei, is_self, lam, alp)
        if cfg.T > 0 and cfg.use_competition:
            ctx = self._kernel_ctx(z2, t)
            s = t['c']              # 初值 s(0) = φ + (1-α_i)b，与设计稿 §2.2 一致
            for it in range(cfg.T):
                s = self._competition_step(s, t, ctx, z)
                if cfg.detach_iterations and it < cfg.T - 1:
                    s = s.detach()
        else:
            ctx = None
            s = t['c']              # T=0：不迭代也不中心化（α=0 时 c == φ+b）
        # 暴露竞争轨迹供分析与测试（λ-vs-同质性图需要 aux['s']）
        self.aux.update(s=s, t=t, z=z, ctx=ctx)
        return self._readout(x, z, s, t, N)

    # ------------------------------------------------------------------

    def _competition_step(self, s: Tensor, t: Dict[str, Tensor],
                          ctx: Dict[str, Tensor], z: Tensor) -> Tensor:
        r"""一步竞争迭代（稀疏、精确、:math:`O(|\hat{\mathcal{E}}|d)`）。

        目标：:math:`r_a = \sum_{b\neq a} K_{ab} s_b`，
        :math:`K = D^{-1/2}(M\circ\kappa)D^{-1/2}` 去对角，
        :math:`M_{ab} = 1-(1-\alpha)[c_a + c_b - c_a c_b]`，
        :math:`c_b = \mathbb{1}[b=i]`。

        记 :math:`w_b := s_b/\sqrt{D_b}`，则一切含 :math:`\kappa` 的求和都能因子化::

            sum_b kappa_ab w_b   = <qhat_a, U_i> + Rd_i,
                U_i = sum_b w_b qhat_b  (index_add),  Rd_i = sum_b w_b (index_add)
            D_a = sum_b M_ab kappa_ab   = Draw_a - (1-alpha)[c_a Draw_a
                                                           + (1-c_a) kappa_{a,i}]
            sum_b M_ab kappa_ab w_b
                = num_raw_a - (1-alpha)[c_a num_raw_a + (1-c_a) kappa_{a,i} w_i]
            r_a = (上式 - M_aa kappa_aa w_a) / sqrt(D_a)

        ``+1`` 偏移的非负性在此兜底：Lemma 2 的「非负对称 => 谱范数 = Perron 根」
        与「去对角 / 抬 :math:`D'` 只让元素变小」两步都依赖 :math:`\kappa \ge 0`。
        而 :math:`D_a` 本身可以趋于 0（孤立节点，或 :math:`\alpha\to0` 的中心行），
        所以 ``clamp(min=1)`` 是必需的，不是可选的数值保护。
        """
        cfg = self.cfg
        src, N = t['src'], z.size(0)
        q, c_a = ctx['q'], ctx['c_a']
        one_ma = 1.0 - t['alp_e']                       # (1-alpha_i) [E,1]

        # --- 行和 D_a（含 alpha 对中心行列的衰减）---
        D = ctx['Draw'] - one_ma * (c_a * ctx['Draw']
                                    + (1.0 - c_a) * ctx['kap_ci'])
        D = D.clamp(min=1.0)
        isd = D.rsqrt()

        # --- 因子化求和 ---
        w = s * isd
        U = scatter_add(w.unsqueeze(-1) * q[t['dst']], src, N)   # [N,H,dh]
        Rd = scatter_add(w, src, N)                              # [N,H]
        num_raw = (q[t['dst']] * U[src]).sum(-1) + Rd[src]       # [E,H]

        # --- 中心的 w（沿 is_self 收集到节点，再按 src 展回边）---
        w_i = scatter_add(w * c_a, src, N)[src]
        num = num_raw - one_ma * (c_a * num_raw + (1.0 - c_a)
                                  * ctx['kap_ci'] * w_i)
        m_aa = 1.0 - one_ma * c_a                                # M_aa
        r = (num - m_aa * ctx['kap_aa'] * w) * isd               # [E,H]

        # --- 投影梯度下降 + 中心化（= 投影到 1^T s = 0）---
        s_new = ((1.0 - cfg.beta) * s
                 + cfg.beta * (t['c'] - t['lam_e'] * r))
        mean = scatter_add(s_new, src, N) / ctx['degp'].clamp(min=1.0)
        return s_new - mean[src]

    # ------------------------------------------------------------------

    def _temperature(self, x: Tensor, s: Tensor, t: Dict[str, Tensor],
                     N: int) -> Tensor:
        """§2.6。默认喂「按 1/sqrt(deg_i) 再缩放的中心-场差异统计量」：
        键名 `center_stat='zscore'` 只是历史别名，分母多乘了 sqrt(deg_i)，
        它不是标准 z-score。§9.8 已量过这个因子的代价：对聚合锐度的影响在
        2e-4 量级，训 200 步后 w_tau 的 stat 列自己涨 6.9× 把它补偿掉 ——
        所以裁决是「数值不改、只改名字」（去掉那个因子属行为变更，要重跑 §8.4）。

        用统计量而不是原始 s_i(T) 的理由：
        中心化后 s 的绝对尺度不可识别（softmax 又对平移不变），
        直接喂 s_i(T) 等于喂一个带任意参考点的量。
        """
        cfg = self.cfg
        if not cfg.use_temperature:
            return x.new_ones(N, 1)
        src, c_a = t['src'], (1.0 - t['nb'])            # c_a: 槽位即中心
        nb = 1.0 - c_a
        s_i = scatter_add(s * c_a, src, N)                        # [N,H]
        cnt = scatter_add(nb, src, N).clamp(min=1.0)              # [N,1] deg_i
        if cfg.center_stat == 'raw':
            stat = s_i
        else:
            mu = scatter_add(s * nb, src, N) / cnt
            var = scatter_add(s * s * nb, src, N) / cnt - mu * mu
            # 多乘的 sqrt(deg_i) 见 §2.7 / §9.8：已裁决保留，不当 bug 修
            sd = var.clamp(min=1e-8).sqrt() * cnt.clamp(min=1.0).sqrt()
            stat = (s_i - mu) / (sd + 1e-6)                       # 邻居数=0 时 => 0
        feat = torch.cat([x, stat.mean(-1, keepdim=True)], dim=-1)
        tau = F.softplus(self.w_tau(feat)) + cfg.tau_min
        self.aux.update(tau=tau, s_center=stat)
        return tau

    def _readout(self, x: Tensor, z: Tensor, s: Tensor, t: Dict[str, Tensor],
                 N: int) -> Tensor:
        """§2.7-2.9：中心槽位不参与聚合，只用于调温。"""
        cfg = self.cfg
        src, dst, is_self = t['src'], t['dst'], t['is_self']
        tau = self._temperature(x, s, t, N)

        nb = ~is_self
        p_nb = softmax(s[nb] / tau[src[nb]], src[nb], num_nodes=N)  # 分组 softmax
        m = scatter_add(p_nb.unsqueeze(-1) * z[dst[nb]], src[nb],
                        N).view(N, self.d)

        if cfg.out_mode == 'gat':
            out = F.elu(m) if cfg.gat_activation == 'elu' else F.relu(m)
        else:
            h = self.out_mlp(torch.cat([self.W_s(x), m], dim=-1))
            out = self.norm(self.skip(x) + h)
        if cfg.use_competition:
            self.aux['lambda'] = torch.sigmoid(self.w_lambda(x))
        if cfg.use_center_field:
            self.aux['alpha'] = torch.sigmoid(self.w_alpha(x))
        return out

    # ------------------------------------------------------------------
    # 稠密参考实现：逐节点显式建矩阵。慢，但「显然正确」，
    # 用于 test/ 里对稀疏因子化做逐元素等价性校验，以及 ReLU 核消融。
    # ------------------------------------------------------------------

    def _forward_dense(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """逐节点显式建矩阵的参考实现（O(N * |V_i|^2 H)，仅用于小图校验）。

        注：稠密路径不构造稀疏因子化的 :math:`ctx`，所以调用后 ``aux['ctx']``
        不存在 —— :func:`test.support.field_kernel` 只能接在稀疏 forward 之后。
        """
        cfg = self.cfg
        self.aux.clear()
        N, H, dh = x.size(0), self.H, self.dh
        lam, alp = self._gates(x)
        z2 = self.W(x)
        z = z2.view(N, H, dh)
        ei, is_self = augment_edge_index(edge_index, N, cfg.symmetrize)
        t = self._edge_terms(x, z2, ei, is_self, lam, alp)

        if cfg.use_competition:
            qf = self.q_proj(z2).view(N, H, dh)
            if cfg.kernel == 'relu':
                qf = qf / (dh ** 0.5)

                def kern(a, b):
                    return F.relu(torch.einsum('ahi,bhi->abh', a, b))
            else:
                qf = F.normalize(qf, dim=-1)

                def kern(a, b):
                    return torch.einsum('ahi,bhi->abh', a, b) + 1.0

        # 槽位序必须与 test/support.py::field_kernel 一致（中心在前）
        by_src = slots_by_center(ei, is_self, N, center_first=True)
        s_out = t['c']              # 与稀疏路径的 s(0) 同一口径，改一处必须改两处
        if cfg.T > 0 and cfg.use_competition:
            for i in range(N):
                e = by_src[i]
                m_ = e.numel()
                qs = qf[t['dst'][e]]                             # [m,H,dh]
                kap = kern(qs, qs)                               # [m,m,H]
                c = torch.zeros(m_, device=x.device, dtype=x.dtype)
                c[0] = 1.0
                M = 1.0 - (1.0 - alp[i, 0]) * (
                    c[:, None] + c[None, :] - c[:, None] * c[None, :])
                D = (kap * M[..., None]).sum(1).clamp(min=1.0)   # [m,H] 行和 D_a
                isd = D.rsqrt()
                K = kap * M[..., None] * isd[:, None, :] * isd[None, :, :]
                idx = torch.arange(m_, device=x.device)
                K[idx, idx] = 0.0                                 # b != a
                s = s_out[e].contiguous()                         # [m,H]
                dr = t['c'][e].contiguous()                       # [m,H]
                for _ in range(cfg.T):
                    r = torch.einsum('abh,bh->ah', K, s)
                    s = (1.0 - cfg.beta) * s + cfg.beta * (dr - lam[i] * r)
                    s = s - s.mean(0, keepdim=True)
                s_out[e] = s

        return self._readout(x, z, s_out, t, N)

    def forward_dense(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """显式调用稠密参考实现（等价性测试用）。"""
        return self._forward_dense(x, edge_index)
