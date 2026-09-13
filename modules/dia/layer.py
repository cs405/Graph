r"""三层拼起来：:class:`DIALayer`（技术文档 §2.2–§2.6、§4.4）。

一层做四件事，顺序与文档一致：

.. math::

    \tilde h_{i|j}=h_i\odot\gamma_{i|j},\ \tilde h_{j|i}=h_j\odot\gamma_{j|i}
    \quad\text{(L1)}

    s^{i\to j}=\tilde h_{i|j}^\top UV^\top \tilde h_{j|i},\quad
    m_{ij}=\tilde h_{i|j}\odot U(V^\top\tilde h_{j|i})
    \quad\text{(L2, 两个方向各一套 }U,V\text{)}

    s_{ij}=\alpha_{i|j}s^{i\to j}+\alpha_{j|i}s^{j\to i}
    \quad\text{(L3)}

``s_ij`` 是这条边的标量强度（边分类的 logit 来源），``m_ij`` 是落在 :math:`i` 的
:math:`d_i` 维空间里的向量消息（节点更新的来源）—— 两者由**同一次**因子化算出，
所以「打分用的边」与「传消息用的边」永远是同一套 :math:`U,V`。这一点是 §三
（与 GAT/HGT 的差别）能成立的前提：GAT 的注意力系数是标量，消息是 :math:`Wh_j`，
两者不共享任何结构。

两条工程约定（:mod:`modules.base` 的协议在这里落地）：

* :class:`~modules.base.Constraint`：``project_parameters()`` 只对两套配对做
  clamp；训练框架在 ``optimizer.step()`` 之后调它，本层不知道也不关心是谁调的。
* :class:`~modules.base.Regularized`：``penalties()`` 返回未加权的
  ``{'sp','orth','gamma'}``，权重在 ``cfg.loss.lambda_*``。

``aux`` 一律 **detach**（诊断/导出用，不参与反传），而 :math:`\gamma` 的
:math:`L_1` 惩罚需要活的张量，所以另存一份 ``_live``；两个字典都在每次 forward
开头清空 —— 读到上一次（另一套开关、另一批边）的残留是这类代码最常见的静默错。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from modules.base import Constraint, Explorable, Regularized
from modules.dia.asymmetric import AsymmetricContribution
from modules.dia.attention import EdgeConditionedDimAttention
from modules.dia.config import DIAConfig
from modules.dia.pairing import LowRankNonNegPairing

__all__ = ['DIALayer']

# 返回/aux 里用的键名。写成常量是为了让 test 与画图代码引用同一个名字，
# 而不是各自敲一遍 'alpha_i' —— 敲错了不会报错，只会安静地少一列。
K_GAMMA_I, K_GAMMA_J = 'gamma_i', 'gamma_j'
K_S_IJ, K_S_JI = 's_ij', 's_ji'
K_ALPHA_I, K_ALPHA_J = 'alpha_i', 'alpha_j'
K_SCORE, K_M_IJ, K_M_JI = 'score', 'm_ij', 'm_ji'


class DIALayer(nn.Module, Constraint, Regularized, Explorable):
    r"""DIA 的一层。``d_src``/``d_dst`` 是**两端各自的维度**（异构时不相等）。"""

    #: 家族标签（见 :class:`modules.base.Explorable`）
    family = 'dia'

    def __init__(self, d_src: int, d_dst: int,
                 cfg: Optional[DIAConfig] = None, **overrides: Any):
        super().__init__()
        cfg = DIAConfig(**{**(cfg.__dict__ if cfg else {}), **overrides})
        self.cfg = cfg
        self.d_src, self.d_dst = int(d_src), int(d_dst)
        self.aux: Dict[str, Tensor] = {}
        self._live: Dict[str, Tensor] = {}

        need_same_dim = (not cfg.use_pairing) or cfg.symmetric_pairing
        if need_same_dim and d_src != d_dst:
            # W=I 与「反向复用 (V,U)」都要求两端同维：前者是恒等映射的定义域问题，
            # 后者是 (V,U) 的形状对调问题。报错要说清是哪个开关导致的。
            why = 'use_pairing=False（W:=I）' if not cfg.use_pairing else \
                'symmetric_pairing=True（反向复用 (V,U)）'
            raise ValueError(
                f'{why} 要求两端同维，收到 d_src={d_src}, d_dst={d_dst}；'
                f'要么打开 use_projection 把两端投到同一个 out_dim，要么关掉这个开关')
        assert cfg.rank <= min(d_src, d_dst), \
            f'rank={cfg.rank} 超过 min(d_src,d_dst)={min(d_src, d_dst)}'

        # L1 维度注意力（关掉就不建，参数一个都不分配 —— 与 OCAConfig 同一套约定）
        self.dim_attn: Optional[EdgeConditionedDimAttention] = None
        if cfg.use_dim_attention:
            self.dim_attn = EdgeConditionedDimAttention(
                d_src, d_dst, d_edge=cfg.d_edge, hidden=cfg.hidden,
                n_rel=cfg.n_rel, gate_bias_init=cfg.gate_bias_init)
        # L2 配对：两个方向各一套（symmetric_pairing 时只有正向）
        self.pairing_ij: Optional[LowRankNonNegPairing] = None
        self.pairing_ji: Optional[LowRankNonNegPairing] = None
        if cfg.use_pairing:
            kw = dict(rank=cfg.rank, n_rel=cfg.n_rel, nonneg=cfg.nonneg,
                      project=cfg.project)
            self.pairing_ij = LowRankNonNegPairing(d_src, d_dst, **kw)
            if not cfg.symmetric_pairing:
                self.pairing_ji = LowRankNonNegPairing(d_dst, d_src, **kw)
        # L3 主体贡献
        self.asym: Optional[AsymmetricContribution] = None
        if cfg.use_asymmetric:
            self.asym = AsymmetricContribution(
                d_src, d_dst, d_edge=cfg.d_edge, hidden=cfg.hidden,
                n_node_types=cfg.n_node_types, n_rel=cfg.n_rel)

    # ------------------------------------------------------------------ 协议
    def pairings(self) -> List[LowRankNonNegPairing]:
        """本层持有的配对模块（0/1/2 个，取决于开关）。"""
        return [p for p in (self.pairing_ij, self.pairing_ji) if p is not None]

    def project_parameters(self) -> None:
        for p in self.pairings():
            p.project_parameters()

    def penalties(self) -> Dict[str, Tensor]:
        r"""``{'sp','orth','gamma'}``（未加权）。

        ``sp``/``orth`` 来自 :class:`LowRankNonNegPairing`（只依赖参数，随时可算）；
        ``gamma`` 依赖上一次 forward 的 :math:`\gamma`，所以**必须在 forward 之后调**
        —— 没跑过前向就没有这一项（返回的 dict 里缺 'gamma'，调用方按缺省处理）。
        :math:`L_1` 按元素取平均而不是求和：文档写的是 :math:`\|\gamma\|_1`，
        但那样 :math:`\lambda_\gamma` 得随图的边数重新调，消融表在不同数据集间
        就没有可比性了。
        """
        out: Dict[str, Tensor] = {}
        for p in self.pairings():
            for k, v in p.penalties().items():
                out[k] = v if k not in out else out[k] + v
        g = self._live.get(K_GAMMA_I)
        if g is not None:
            pen = g.mean()
            gj = self._live.get(K_GAMMA_J)
            if gj is not None:
                pen = 0.5 * (pen + gj.mean())
            out['gamma'] = pen
        return out

    def explain(self) -> Dict[str, Any]:
        """一行摘要：门控均值、主导权偏斜、分数均值，以及两套配对的稀疏度。"""
        out: Dict[str, Any] = {
            'd_src': self.d_src, 'd_dst': self.d_dst, 'rank': self.cfg.rank,
            'layers_on': ''.join(str(int(v)) for v in (
                self.cfg.use_dim_attention, self.cfg.use_pairing,
                self.cfg.use_asymmetric)),
        }
        for k in (K_GAMMA_I, K_GAMMA_J, K_S_IJ, K_S_JI,
                  K_ALPHA_I, K_ALPHA_J, K_SCORE):
            t = self.aux.get(k)
            if t is not None and t.numel():
                out[f'{k}_mean'] = float(t.mean())
                out[f'{k}_std'] = float(t.std()) if t.numel() > 1 else 0.0
        # alpha 的偏斜：|alpha_i - 0.5| 的均值。0 = 完全对称（非对称没学到东西），
        # 0.5 = 每条边都由单方完全主导。这是 §六 创新点 3 的直接读数。
        a = self.aux.get(K_ALPHA_I)
        if a is not None and a.numel():
            out['alpha_skew'] = float((a - 0.5).abs().mean())
        for name, p in (('ij', self.pairing_ij), ('ji', self.pairing_ji)):
            if p is not None:
                out[f'pairing_{name}'] = p.stats()
        return out

    # ------------------------------------------------------------------ 前向
    def forward(self, h_i: Tensor, h_j: Tensor, e_ij: Optional[Tensor] = None,
                rel: Optional[Tensor] = None, r_i: Optional[Tensor] = None,
                r_j: Optional[Tensor] = None, src: Optional[Tensor] = None,
                dst: Optional[Tensor] = None,
                want: Iterable[str] = ('score', 'message')
                ) -> Dict[str, Tensor]:
        r"""一条边集上的完整 DIA 前向。

        Args:
            h_i: ``[E, d_src]`` 主体（消息要落到的那一端）
            h_j: ``[E, d_dst]`` 邻居（消息来源）
            e_ij: ``[E, d_edge]`` 边特征，可为 ``None``
            rel: ``[E]`` 关系类型 id；``r_i``/``r_j``：两端节点类型 id
            src/dst: ``[E]`` 两端在图里的节点号，**只用于记账**（存进 ``aux``，
                让诊断能把边级量按节点聚合），不参与计算
            want: 见 :meth:`LowRankNonNegPairing.forward`；``'score'`` 恒计算
                （L3 要用它），``'message'`` 决定要不要算 :math:`m`

        Returns:
            至少含 ``score [E]``；``'message' in want`` 时另含 ``m_ij [E,d_src]``、
            ``m_ji [E,d_dst]``。
        """
        self.aux.clear()
        self._live.clear()
        cfg = self.cfg
        want = set(want) | {'score'}
        E = h_i.size(0)
        assert h_j.size(0) == E, f'两端边数不一致：{E} vs {h_j.size(0)}'

        # ---- L1：逐维筛选 -------------------------------------------------
        if self.dim_attn is not None:
            g_i, g_j = self.dim_attn(h_i, h_j, e_ij, rel)
        else:
            g_i = g_j = None
        hs_i = h_i if g_i is None else h_i * g_i
        hs_j = h_j if g_j is None else h_j * g_j
        if g_i is not None:
            self._live[K_GAMMA_I], self._live[K_GAMMA_J] = g_i, g_j
            self.aux[K_GAMMA_I], self.aux[K_GAMMA_J] = g_i.detach(), g_j.detach()

        # ---- L2：两个方向的低秩非负配对 -----------------------------------
        if self.pairing_ij is not None:
            o_ij = self.pairing_ij(hs_i, hs_j, rel, want=want)
            s_ij = o_ij['s']
            if self.pairing_ji is not None:
                o_ji = self.pairing_ji(hs_j, hs_i, rel, want=want)
                s_ji = o_ji['s']
            else:
                # symmetric_pairing：反向复用 (V,U)，即把两端特征对调再走一次正向。
                # 于是 s_ji == s_ij 恒成立（test 钉住），非对称只能来自 L3 的 alpha。
                o_ji = self.pairing_ij(hs_j, hs_i, rel, want=want)
                s_ji = o_ji['s']
        else:
            # use_pairing=False：W := I（逐维内积），消息退化成逐维乘积
            o_ij = o_ji = None
            s_ij = (hs_i * hs_j).sum(-1)
            s_ji = s_ij
        self.aux[K_S_IJ], self.aux[K_S_JI] = s_ij.detach(), s_ji.detach()

        # ---- L3：非对称主体贡献 -------------------------------------------
        if self.asym is not None:
            a_i, a_j = self.asym(h_i, h_j, e_ij, rel, r_i, r_j)
        else:
            half = h_i.new_full((E,), 0.5)
            a_i = a_j = half
        self.aux[K_ALPHA_I], self.aux[K_ALPHA_J] = a_i.detach(), a_j.detach()

        score = a_i * s_ij + a_j * s_ji
        out: Dict[str, Tensor] = {K_SCORE: score}
        self.aux[K_SCORE] = score.detach()
        if 'message' in want:
            if o_ij is not None:
                out[K_M_IJ] = o_ij['m']
                out[K_M_JI] = o_ji['m']
            else:
                # W=I 时 m_ij = h̃_i ⊙ h̃_j（与 s_ij = <h̃_i, h̃_j> 同源）
                prod = hs_i * hs_j
                out[K_M_IJ], out[K_M_JI] = prod, prod
        if 'matrix' in want or cfg.return_matrix:
            p = self.pairing_ij
            out['M_ij'] = (p.matrix(hs_i, hs_j, rel) if p is not None
                           else (hs_i.unsqueeze(-1) * hs_j.unsqueeze(-2)))
            pj = self.pairing_ji or p
            out['M_ji'] = (pj.matrix(hs_j, hs_i, rel) if pj is not None
                           else (hs_j.unsqueeze(-1) * hs_i.unsqueeze(-2)))
        if 'factors' in want and o_ij is not None:
            out['factors_ij'], out['factors_ji'] = o_ij['factors'], o_ji['factors']
        if src is not None:
            self.aux['src'], self.aux['dst'] = src.detach(), dst.detach()
        return out

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        c = self.cfg
        return (f'd_src={self.d_src}, d_dst={self.d_dst}, rank={c.rank}, '
                f'n_rel={c.n_rel}, d_edge={c.d_edge}, '
                f'on=(gamma={c.use_dim_attention}, pair={c.use_pairing}, '
                f'asym={c.use_asymmetric})')
