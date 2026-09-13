r"""DIA：维度级可识别注意力（技术文档《维度级可识别注意力机制（DIA）》的实现）。

与 :mod:`modules.oca` 同一口径：一个文件放完整个算子（config / 三层组件 / 拼装 /
可解释性出口）。结构层包装（:class:`DIAConv`）在 :mod:`modules.convs`，边级读出头
（:class:`EdgeScore`）在 :mod:`modules.head`，``batch`` 解包（:func:`unpack_batch`）
在 :mod:`utils.graph` —— 三处都是框架侧的通用件，不跟算法绑死。

| 组件 | 文档 |
| :-- | :-- |
| :class:`DIAConfig` | §2.2–§2.6 |
| :class:`EdgeConditionedDimAttention`（L1 逐维筛选 :math:`\gamma`） | §2.2 |
| :class:`LowRankNonNegPairing`（L2 :math:`W=UV^\top`，非负+稀疏+正交） | §2.3、§2.7 |
| :class:`AsymmetricContribution`（L3 主导权 :math:`\alpha`） | §2.4 |
| :class:`DIALayer`：三层拼装 + 约束/惩罚/解释三个协议的落地点 | §2.2–§2.6 |
| 可解释性函数（:func:`column_supports` / :func:`edge_explanation` 等） | §4.8、§五 |

框架侧（``model_builder``/``training``/``tasks``）**不 import 本文件**：注册靠
``@register_module``，约束/惩罚/解释靠 :mod:`modules.base` 的三条协议被发现。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from modules.base import Constraint, Explorable, Regularized, iter_impl

__all__ = ['DIAConfig', 'EdgeConditionedDimAttention', 'LowRankNonNegPairing',
           'AsymmetricContribution', 'DIALayer',
           'dia_layers', 'pairings', 'column_supports', 'supports_disjoint',
           'pairing_report', 'edge_explanation', 'score_attribution',
           'dump_pairings']


# ============================================================================
# 配置
# ============================================================================


@dataclass
class DIAConfig:
    r"""DIA 层/头的配置。``cfg/models/*.yaml`` 的 ``dia:`` 块 = 本结构的一组覆盖。

    三条写死在这里的主张（改任何一条都要重跑配对相关测试）：

    1. ``use_pairing=False`` 时配对矩阵取 :math:`W=I`（逐维内积），**不是**「没有消息」——
       消融要的是「去掉低秩配对」这一件事，其它两层保持原样；
    2. ``symmetric_pairing=True`` 时反向复用 :math:`(V,U)`，于是
       :math:`s^{i\to j}=s^{j\to i}`（技术文档 §2.4 明确要求**不**退化到对称，
       这个开关只是用来量「非对称到底值多少分」）；
    3. ``use_projection=False`` 时不建投影分支，:math:`U` 的行**就是原始特征维度**——
       可解释性与可识别性主张只在这个模式下成立（§七 局限 1）。
    """

    # ---- 维度 ----
    out_dim: int = 64               # 投影模式下的「维度槽位数」，由结构表的 c2 注入
    rank: int = 8                   # k：低秩配对的秩（技术文档 §2.3）
    hidden: int = 64                # 三个 MLP 的隐层宽度
    d_edge: int = 0                 # 边特征 e_ij 的维度；0 = 该图没有边特征
    n_rel: int = 1                  # 关系类型数：配对矩阵按关系类型持有
    n_node_types: int = 1           # 主体贡献的类型偏置 b_{r_i->r_j} 的尺寸

    # ---- 三层开关（消融表 = 这三个键的组合）----
    use_dim_attention: bool = True  # L1 边条件维度注意力 gamma
    use_pairing: bool = True        # L2 低秩非负维度配对 U V^T
    use_asymmetric: bool = True     # L3 非对称主体贡献 alpha

    # ---- L2 的约束（可识别性的三个前提）----
    nonneg: bool = True             # U,V >= 0；False 时分解不再可识别（§2.7 定理）
    project: str = 'clamp'          # 'clamp' 投影次梯度 | 'relu' 重参数化（未实现）
    symmetric_pairing: bool = False # True => 反向复用 (V,U)，s 退化为对称

    # ---- 结构 ----
    use_projection: bool = True     # 端点先各自线性投影到 out_dim（两条独立分支）
    n_layers: int = 1               # EdgeScore 里堆几层 DIALayer
    residual: bool = True           # 消息传递的残差（§2.5 的 h_i + sigma(...)）
    symmetrize: bool = True         # 消息传递前是否 to_undirected
    return_matrix: bool = False     # 是否物化 M（O(E d_i d_j)，只对拍/画图用）
    gate_bias_init: float = 0.0     # gamma 的偏置初值（sigmoid 前）
    heads: int = 1                  # >1 未实现：多头维度配对是待办（见 docs/DIA.md §九）

    def __post_init__(self):
        assert self.rank >= 1, f'rank 至少 1（收到 {self.rank}）'
        assert self.n_layers >= 1, f'n_layers 至少 1（收到 {self.n_layers}）'
        assert self.project in ('clamp', 'relu'), \
            f"project 只能是 'clamp' | 'relu'（收到 {self.project!r}）"
        if self.project == 'relu':
            raise NotImplementedError(
                "project='relu'（把 U 重参数化成 relu(U_raw)，从而不需要投影钩子）"
                '还没实现；它与 clamp 的差别只在优化路径，实现前不要写进消融表。')
        if self.heads != 1:
            raise NotImplementedError(
                f'heads={self.heads}：多头维度配对（每个头一套 U/V，分数按头相加）'
                '未实现。它需要先把 rank 与 out_dim 的整除关系定下来，'
                '否则「哪个维度槽属于哪个头」在可解释性上说不清。')
        if not self.use_pairing and not (self.use_dim_attention
                                         or self.use_asymmetric):
            # 三层全关等于「什么都没做」，这种行出现在消融表里只可能是抄漏了
            raise ValueError('三层全关的 DIA 不是消融，是空模型')

    # ------------------------------------------------------------------
    def replaced(self, **kw) -> 'DIAConfig':
        """返回一份改了若干字段的副本（消融表用；不原地改，理由同 ``cfg.patched``）。"""
        return replace(self, **kw)


# ============================================================================
# 通用 MLP 工具（L1 / L3 共用）
# ============================================================================


def _mlp(in_dim: int, hidden: int, out_dim: int,
         bias_init: float = 0.0) -> nn.Sequential:
    """``Linear -> ReLU -> Linear``，末层权重清零（理由见 :class:`EdgeConditionedDimAttention`）。"""
    head = nn.Linear(hidden, out_dim)
    nn.init.zeros_(head.weight)
    nn.init.constant_(head.bias, bias_init)
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), head)


# ============================================================================
# L1：边条件维度注意力（§2.2）
# ============================================================================


class EdgeConditionedDimAttention(nn.Module):
    r"""算 :math:`\gamma_{i|j}` 与 :math:`\gamma_{j|i}`（都是逐维、都在 (0,1)）。

    .. math::

        \gamma_{i|j}=\sigma(\mathrm{MLP}_{r,t}([h_i;h_j;e_{ij}])),\quad
        \gamma_{j|i}=\sigma(\mathrm{MLP}_{r,t}([h_j;h_i;e_{ij}]))
        \qquad \tilde h_{i|j}=h_i\odot\gamma_{i|j}

    三条与「一个 MLP 两个头」不同的取舍，都是为了保住 §2.2 的非对称语义：

    1. **两个 MLP，不是一个 MLP 出 2d 个数**。文档写的 :math:`\gamma_{j|i}` 用的是
       ``[h_j; h_i; e_ij]`` —— 拼接顺序反了。共享主干再切两半，等价于强制
       ``MLP_j([a;b]) = MLP_i([b;a])``，也就是 :math:`\gamma_{j|i}` 与 :math:`\gamma_{i|j}`
       只差一个输入置换，非对称性只剩「参数不同」这一层。既然 §2.4 主张的是
       *两个独立方向*，这里就建两套参数（代价是这一层参数翻倍，d=64/hidden=64 时约 12k）。
    2. **关系条件用加性偏置，不用逐关系一套 MLP**。文档下标写的是
       :math:`\mathrm{MLP}_{r,t}`（按关系 r 与节点类型 t 各一套）。真按关系建 MLP，
       参数量随 R 线性涨，R 大时主干根本喂不动数据。这里折中成主干共享 + 逐关系的
       加性偏置 ``rel_bias[r]``，R=1 时偏置就是一条全零向量。
    3. **sigmoid 而不是 softmax**。softmax 会把「这条边整体不重要」表达不出来
       （它只分配相对权重），而 §2.2 要的是**逐维筛选**：某一维在这条边上完全没用，
       就该是 0，而不是「比别的维度小一点」。

    初始化：末层 ``Linear.weight`` 置零、``bias`` 置 ``gate_bias_init``，于是初始
    :math:`\gamma\equiv\sigma(b_0)`。取 ``gate_bias_init=0`` 时 :math:`\gamma=0.5`
    （半开），取大正值时接近 1（不筛选，等价于关掉这一层）。
    """

    def __init__(self, d_src: int, d_dst: int, d_edge: int = 0, hidden: int = 64,
                 n_rel: int = 1, gate_bias_init: float = 0.0):
        super().__init__()
        assert d_src > 0 and d_dst > 0, f'维度必须为正：({d_src}, {d_dst})'
        assert hidden > 0 and n_rel >= 1, (hidden, n_rel)
        self.d_src, self.d_dst, self.d_edge = int(d_src), int(d_dst), int(d_edge)
        self.n_rel = int(n_rel)
        in_dim = d_src + d_dst + d_edge
        # 两套主干：拼接顺序相反（[h_i;h_j;e] 与 [h_j;h_i;e]），见 docstring 第 1 条
        self.mlp_s = _mlp(in_dim, hidden, d_src, gate_bias_init)
        self.mlp_d = _mlp(in_dim, hidden, d_dst, gate_bias_init)
        # 逐关系加性偏置（docstring 第 2 条）。**n_rel=1 时不分配**：单关系图上
        # 它永远拿不到梯度（没有 rel 可查），白白占一行参数与一份优化器状态。
        self.rel_bias = (nn.Parameter(torch.zeros(n_rel, d_src + d_dst))
                         if n_rel > 1 else None)

    def forward(self, h_s: Tensor, h_d: Tensor, e_ij: Optional[Tensor] = None,
                rel: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """``h_s [E,d_src]``、``h_d [E,d_dst]``、``e_ij [E,d_edge]|None`` -> ``(g_s, g_d)``。"""
        E = h_s.size(0)
        assert h_d.size(0) == E, f'两端边数不一致：{E} vs {h_d.size(0)}'
        d_e = self.d_edge
        if d_e:
            assert e_ij is not None, f'这层建了 d_edge={d_e}，但没给边特征'
            assert e_ij.size(-1) == d_e, f'边特征维度 {e_ij.size(-1)} != {d_e}'
            e = e_ij.to(h_s.dtype)
        else:
            if e_ij is not None:
                # 给了边特征但这层没用：多半是 cfg 的 d_edge 忘了同步数据集
                raise ValueError(
                    f'传了边特征 e_ij（{tuple(e_ij.shape)}）但这层 d_edge=0；'
                    f'把 cfg/models/*.yaml 的 dia.d_edge 设成 {e_ij.size(-1)}')
            e = h_s.new_zeros(E, 0)
        inp_s = torch.cat([h_s, h_d, e], dim=-1)
        inp_d = torch.cat([h_d, h_s, e], dim=-1)
        bias = self._bias(rel, E, h_s)
        g_s = torch.sigmoid(self.mlp_s(inp_s) + bias[..., :self.d_src])
        g_d = torch.sigmoid(self.mlp_d(inp_d) + bias[..., self.d_src:])
        return g_s, g_d

    def _bias(self, rel: Optional[Tensor], E: int, h_s: Tensor) -> Tensor:
        """``[E, d_src+d_dst]`` 的关系偏置；单关系（``rel_bias is None``）时全零。"""
        if self.rel_bias is None:
            return h_s.new_zeros(E, self.d_src + self.d_dst)
        if rel is None:
            raise ValueError(
                f'这层建了 n_rel={self.n_rel} 的关系偏置，但前向没拿到 rel；'
                f'请在 batch 里给 rel（见 utils.graph.unpack_batch 的契约）')
        rel = rel.long()
        assert int(rel.max()) < self.n_rel, \
            f'关系类型 id {int(rel.max())} 超出 n_rel={self.n_rel}'
        return self.rel_bias[rel]

    def extra_repr(self) -> str:
        return (f'd_src={self.d_src}, d_dst={self.d_dst}, d_edge={self.d_edge}, '
                f'n_rel={self.n_rel}')


# ============================================================================
# L2：低秩非负维度配对（§2.3、§2.7）
# ============================================================================


class LowRankNonNegPairing(nn.Module, Constraint, Regularized):
    r"""一个方向（``src -> dst``）的低秩非负配对。反向由 :class:`DIALayer` 另建一份。

    配对矩阵 :math:`W = U V^\top`，:math:`U\in\mathbb R_{\ge0}^{d_i\times k}`、
    :math:`V\in\mathbb R_{\ge0}^{d_j\times k}`，**按关系类型**各持有一份。

    两个实现上的决定：

    1. **不物化 :math:`M`**。:math:`[E,d_i,d_j]` 太贵；分数与消息都能因子化，
       中间量只有 :math:`[E,k]`，代价 :math:`O(E\,d\,k)`。
    2. **先算全部关系类型再 gather，不先 ``U[rel]``**。前者只多算 :math:`R` 倍、
       中间量是 :math:`[E,R,k]`。
    """

    def __init__(self, d_src: int, d_dst: int, rank: int = 8, n_rel: int = 1,
                 nonneg: bool = True, project: str = 'clamp',
                 init_scale: float = 0.1, sparse_penalty: bool = True,
                 orth_penalty: bool = True):
        super().__init__()
        assert d_src > 0 and d_dst > 0, f'维度必须为正：({d_src}, {d_dst})'
        assert n_rel >= 1, f'n_rel 至少 1（收到 {n_rel}）'
        if rank > min(d_src, d_dst):
            raise ValueError(
                f'rank={rank} 超过 min(d_src, d_dst)={min(d_src, d_dst)}：'
                f'那就不是低秩分解了（改成 rank<= {min(d_src, d_dst)}）')
        self.d_src, self.d_dst, self.rank, self.n_rel = \
            int(d_src), int(d_dst), int(rank), int(n_rel)
        self.nonneg, self.project = bool(nonneg), project
        self.sparse_penalty, self.orth_penalty = sparse_penalty, orth_penalty
        self.U = nn.Parameter(torch.rand(n_rel, self.d_src, rank) * init_scale)
        self.V = nn.Parameter(torch.rand(n_rel, self.d_dst, rank) * init_scale)

    # ------------------------------------------------------------------ 协议
    def project_parameters(self) -> None:
        r""":math:`U\leftarrow\max(U,0),\ V\leftarrow\max(V,0)`（§2.6）。幂等。"""
        if not self.nonneg or self.project != 'clamp':
            return
        with torch.no_grad():
            self.U.data.clamp_(min=0.0)
            self.V.data.clamp_(min=0.0)

    def penalties(self) -> Dict[str, Tensor]:
        r"""``{'sp': ||U||_1+||V||_1, 'orth': ||U^\top U-I||_F^2+||V^\top V-I||_F^2}``。"""
        out: Dict[str, Tensor] = {}
        U, V = self.U, self.V
        if self.sparse_penalty:
            out['sp'] = U.abs().sum() + V.abs().sum()
        if self.orth_penalty:
            eye = torch.eye(self.rank, device=U.device, dtype=U.dtype)
            out['orth'] = ((U.transpose(1, 2) @ U - eye).pow(2).sum()
                           + (V.transpose(1, 2) @ V - eye).pow(2).sum())
        return out

    # ------------------------------------------------------------------ 代数
    def W(self, rel: Optional[int] = None) -> Tensor:
        r"""配对矩阵 :math:`W=UV^\top`。``rel=None`` 且只有一种关系时返回 ``[d_src,d_dst]``。"""
        W = self.U @ self.V.transpose(1, 2)
        if rel is None:
            return W[0] if self.n_rel == 1 else W
        return W[int(rel)]

    def _project_bank(self, h: Tensor, bank: Tensor, rel: Tensor) -> Tensor:
        r"""``h [E,d]`` 与 ``bank [R,d,k]`` -> ``[E,k]``。"""
        all_r = torch.einsum('ed,rdk->erk', h, bank)
        idx = rel.view(-1, 1, 1).expand(-1, 1, bank.size(2))
        return all_r.gather(1, idx).squeeze(1)

    def _back_bank(self, c: Tensor, bank: Tensor, rel: Tensor) -> Tensor:
        r"""``c [E,k]`` 与 ``bank [R,d,k]`` -> ``[E,d]``。"""
        all_r = torch.einsum('ek,rdk->erd', c, bank)
        idx = rel.view(-1, 1, 1).expand(-1, 1, bank.size(1))
        return all_r.gather(1, idx).squeeze(1)

    def matrix(self, h_s: Tensor, h_d: Tensor,
               rel: Optional[Tensor] = None) -> Tensor:
        r""":math:`M_{ij}=\mathrm{diag}(\tilde h_i)UV^\top\mathrm{diag}(\tilde h_j)`。

        形状 ``[E, d_src, d_dst]`` —— 只给对拍与画图用，训练路径不要碰。
        """
        rel = self._rel(h_s, rel)
        W = self.U[rel] @ self.V[rel].transpose(1, 2)
        return h_s.unsqueeze(-1) * W * h_d.unsqueeze(-2)

    def _rel(self, h: Tensor, rel: Optional[Tensor]) -> Tensor:
        if rel is None:
            return h.new_zeros(h.size(0), dtype=torch.long)
        rel = rel.long()
        assert rel.size(0) == h.size(0), \
            f'关系类型数与边数不一致：{rel.size(0)} vs {h.size(0)}'
        assert int(rel.max()) < self.n_rel, \
            f'关系类型 id {int(rel.max())} 超出 n_rel={self.n_rel}'
        return rel

    # ------------------------------------------------------------------ 前向
    def forward(self, h_s: Tensor, h_d: Tensor, rel: Optional[Tensor] = None,
                want: Iterable[str] = ('score', 'message')
                ) -> Dict[str, Tensor]:
        r"""一次算完本方向要的所有量。

        Returns:
            ``{'s': [E], 'm': [E,d_src], 'factors': ([E,k], [E,k])}`` 的子集。
        """
        want = set(want)
        rel = self._rel(h_s, rel)
        a = self._project_bank(h_s, self.U, rel)
        b = self._project_bank(h_d, self.V, rel)
        out: Dict[str, Tensor] = {'factors': (a, b)}
        if 'score' in want:
            out['s'] = (a * b).sum(-1)
        if 'message' in want:
            out['m'] = h_s * self._back_bank(b, self.U, rel)
        if 'matrix' in want:
            out['M'] = self.matrix(h_s, h_d, rel)
        return out

    # ------------------------------------------------------------------ 摘要
    def stats(self, eps: float = 1e-3) -> Dict[str, float]:
        """配对矩阵的稀疏度/秩诊断。"""
        with torch.no_grad():
            U, V = self.U, self.V
            W = U @ V.transpose(1, 2)
            col_nnz = (U > eps).to(U.dtype).sum(1)
            return {
                'W_mean': float(W.mean()), 'W_max': float(W.max()),
                'W_nnz_frac': float((W > eps).to(W.dtype).mean()),
                'U_col_nnz_mean': float(col_nnz.mean()),
                'U_col_nnz_max': float(col_nnz.max()),
                'U_min': float(U.min()), 'V_min': float(V.min()),
                'U_l1': float(U.abs().sum()),
            }

    def extra_repr(self) -> str:
        return (f'd_src={self.d_src}, d_dst={self.d_dst}, rank={self.rank}, '
                f'n_rel={self.n_rel}, nonneg={self.nonneg}')


# ============================================================================
# L3：非对称主体贡献（§2.4）
# ============================================================================


class AsymmetricContribution(nn.Module):
    r"""算 :math:`(\alpha_{i|j},\alpha_{j|i})`，两者相加为 1。

    .. math::

        [\alpha_{i|j},\alpha_{j|i}]=\mathrm{softmax}_2\big(\mathrm{MLP}([h_i;h_j;e_{ij}])
        +b_{r_i\to r_j}\big),\qquad
        s_{ij}=\alpha_{i|j}s^{i\to j}+\alpha_{j|i}s^{j\to i}

    1. **softmax over 2 而不是两个独立 sigmoid**。:math:`\alpha_{i|j}+\alpha_{j|i}=1`
       是「主导权是零和的」这条语义本身。
    2. **末层权重清零** ⇒ 初始 :math:`\alpha=(0.5,0.5)`，训练从「对称」出发，
       非对称是**学出来的**。
    3. **类型偏置 :math:`b_{r_i\to r_j}` 建在节点类型对上**，同时给一份关系类型偏置。
    """

    def __init__(self, d_src: int, d_dst: int, d_edge: int = 0, hidden: int = 64,
                 n_node_types: int = 1, n_rel: int = 1):
        super().__init__()
        assert d_src > 0 and d_dst > 0, (d_src, d_dst)
        assert n_node_types >= 1 and n_rel >= 1, (n_node_types, n_rel)
        self.d_src, self.d_dst, self.d_edge = int(d_src), int(d_dst), int(d_edge)
        self.n_node_types, self.n_rel = int(n_node_types), int(n_rel)
        self.trunk = _mlp(d_src + d_dst + d_edge, hidden, 2, bias_init=0.0)
        self.type_bias = (nn.Parameter(torch.zeros(n_node_types, n_node_types, 2))
                          if n_node_types > 1 else None)
        self.rel_bias = (nn.Parameter(torch.zeros(n_rel, 2))
                         if n_rel > 1 else None)

    def forward(self, h_s: Tensor, h_d: Tensor, e_ij: Optional[Tensor] = None,
                rel: Optional[Tensor] = None, r_s: Optional[Tensor] = None,
                r_d: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""-> ``(alpha_i|j [E], alpha_j|i [E])``。"""
        E = h_s.size(0)
        assert h_d.size(0) == E, f'两端边数不一致：{E} vs {h_d.size(0)}'
        if self.d_edge:
            assert e_ij is not None and e_ij.size(-1) == self.d_edge, \
                f'这层建了 d_edge={self.d_edge}，收到的 e_ij={None if e_ij is None else tuple(e_ij.shape)}'
            e = e_ij.to(h_s.dtype)
        else:
            e = h_s.new_zeros(E, 0)
        logits = self.trunk(torch.cat([h_s, h_d, e], dim=-1))
        logits = logits + self._bias(E, h_s, rel, r_s, r_d)
        a = torch.softmax(logits, dim=-1)
        return a[:, 0], a[:, 1]

    def _bias(self, E: int, h_s: Tensor, rel: Optional[Tensor],
              r_s: Optional[Tensor], r_d: Optional[Tensor]) -> Tensor:
        """``[E,2]`` 的偏置和；同构单关系图（两个偏置都未分配）时直接全零。"""
        if self.rel_bias is None and self.type_bias is None:
            return h_s.new_zeros(E, 2)
        out = h_s.new_zeros(E, 2)
        if self.rel_bias is not None:
            if rel is None:
                raise ValueError(
                    f'这层建了 n_rel={self.n_rel} 的关系偏置，但前向没拿到 rel；'
                    f'请在 batch 里给 rel（见 utils.graph.unpack_batch 的契约）')
            out = out + self.rel_bias[self._idx(rel, self.n_rel, E, 'rel')]
        if self.type_bias is not None:
            if r_s is None:
                raise ValueError(
                    f'这层建了 n_node_types={self.n_node_types} 的类型偏置，'
                    f'但前向没拿到 node_type；请在 batch 里给 node_type')
            i = self._idx(r_s, self.n_node_types, E, 'node_type(src)')
            j = self._idx(r_d if r_d is not None else r_s,
                          self.n_node_types, E, 'node_type(dst)')
            out = out + self.type_bias[i, j]
        return out

    @staticmethod
    def _idx(t: Tensor, n: int, E: int, what: str) -> Tensor:
        t = t.long()
        assert t.numel() == E, f'{what} 的长度 {t.numel()} != 边数 {E}'
        assert int(t.max()) < n, f'{what} 的 id {int(t.max())} 超出范围 {n}'
        return t

    def extra_repr(self) -> str:
        return (f'd_src={self.d_src}, d_dst={self.d_dst}, d_edge={self.d_edge}, '
                f'n_node_types={self.n_node_types}, n_rel={self.n_rel}')


# ============================================================================
# DIALayer：三层拼装（§2.2–§2.6、§4.4）
# ============================================================================

# 返回/aux 里用的键名。写成常量是为了让 test 与画图代码引用同一个名字。
K_GAMMA_I, K_GAMMA_J = 'gamma_i', 'gamma_j'
K_S_IJ, K_S_JI = 's_ij', 's_ji'
K_ALPHA_I, K_ALPHA_J = 'alpha_i', 'alpha_j'
K_SCORE, K_M_IJ, K_M_JI = 'score', 'm_ij', 'm_ji'


class DIALayer(nn.Module, Constraint, Regularized, Explorable):
    r"""DIA 的一层。``d_src``/``d_dst`` 是**两端各自的维度**（异构时不相等）。

    .. math::

        \tilde h_{i|j}=h_i\odot\gamma_{i|j},\ \tilde h_{j|i}=h_j\odot\gamma_{j|i}
        \quad\text{(L1)}

        s^{i\to j}=\tilde h_{i|j}^\top UV^\top \tilde h_{j|i},\quad
        m_{ij}=\tilde h_{i|j}\odot U(V^\top\tilde h_{j|i})
        \quad\text{(L2)}

        s_{ij}=\alpha_{i|j}s^{i\to j}+\alpha_{j|i}s^{j\to i}
        \quad\text{(L3)}
    """

    #: :class:`modules.base.Explorable` 的家族标签。
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
            why = 'use_pairing=False（W:=I）' if not cfg.use_pairing else \
                'symmetric_pairing=True（反向复用 (V,U)）'
            raise ValueError(
                f'{why} 要求两端同维，收到 d_src={d_src}, d_dst={d_dst}；'
                f'要么打开 use_projection 把两端投到同一个 out_dim，要么关掉这个开关')
        assert cfg.rank <= min(d_src, d_dst), \
            f'rank={cfg.rank} 超过 min(d_src,d_dst)={min(d_src, d_dst)}'

        # L1
        self.dim_attn: Optional[EdgeConditionedDimAttention] = None
        if cfg.use_dim_attention:
            self.dim_attn = EdgeConditionedDimAttention(
                d_src, d_dst, d_edge=cfg.d_edge, hidden=cfg.hidden,
                n_rel=cfg.n_rel, gate_bias_init=cfg.gate_bias_init)
        # L2
        self.pairing_ij: Optional[LowRankNonNegPairing] = None
        self.pairing_ji: Optional[LowRankNonNegPairing] = None
        if cfg.use_pairing:
            kw = dict(rank=cfg.rank, n_rel=cfg.n_rel, nonneg=cfg.nonneg,
                      project=cfg.project)
            self.pairing_ij = LowRankNonNegPairing(d_src, d_dst, **kw)
            if not cfg.symmetric_pairing:
                self.pairing_ji = LowRankNonNegPairing(d_dst, d_src, **kw)
        # L3
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
        r"""``{'sp','orth','gamma'}``（未加权）。"""
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
            h_i: ``[E, d_src]`` 主体
            h_j: ``[E, d_dst]`` 邻居
            e_ij: ``[E, d_edge]`` 边特征，可为 ``None``
            rel: ``[E]`` 关系类型 id
            src/dst: ``[E]`` 两端节点号，**只用于记账**
            want: 要哪些量

        Returns:
            至少含 ``score [E]``；``'message' in want`` 时另含 ``m_ij``、``m_ji``。
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
                o_ji = self.pairing_ij(hs_j, hs_i, rel, want=want)
                s_ji = o_ji['s']
        else:
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


# ============================================================================
# 可解释性出口（§4.8、§五）
# ============================================================================


def dia_layers(model) -> List[DIALayer]:
    """模型里全部 :class:`DIALayer`（含 ``DIAConv``/``EdgeScore`` 内部的那些）。"""
    return iter_impl(model, DIALayer)                    # type: ignore[arg-type]


def pairings(model) -> List[Tuple[str, LowRankNonNegPairing]]:
    """``[(路径, 配对模块)]``，路径形如 ``'model.0.layer.pairing_ij'``。"""
    out: List[Tuple[str, LowRankNonNegPairing]] = []
    for name, m in model.named_modules():
        if isinstance(m, LowRankNonNegPairing):
            out.append((name, m))
    return out


def column_supports(p: LowRankNonNegPairing, eps: float = 1e-3,
                    which: str = 'U') -> List[List[int]]:
    r"""每个维度槽（列）的支撑集：``U`` 的哪几行非零。"""
    M = p.U if which == 'U' else p.V
    with torch.no_grad():
        nz = (M > eps)
        out: List[List[int]] = []
        for r in range(M.size(0)):
            for k in range(M.size(2)):
                out.append(torch.nonzero(nz[r, :, k]).flatten().tolist())
        return out


def supports_disjoint(p: LowRankNonNegPairing, eps: float = 1e-3,
                      which: str = 'U') -> bool:
    """分离条件是否成立：任意两列的支撑集不相交（§2.7 推论的可检验形式）。"""
    sup = column_supports(p, eps, which)
    seen: set = set()
    for s in sup:
        if seen.intersection(s):
            return False
        seen.update(s)
    return True


def pairing_report(model, eps: float = 1e-3, max_pairs: int = 5
                   ) -> Dict[str, Any]:
    """逐配对模块的稀疏度/分离性/top 维度对。"""
    rep: Dict[str, Any] = {}
    for name, p in pairings(model):
        with torch.no_grad():
            W = p.W()
            W = W.unsqueeze(0) if W.dim() == 2 else W
            top: List[Any] = []
            for r in range(W.size(0)):
                flat = W[r].flatten()
                v, i = flat.topk(min(max_pairs, flat.numel()))
                d_dst = W.size(2)
                top.append([{'src': int(x // d_dst), 'dst': int(x % d_dst),
                             'w': float(w)} for w, x in zip(v.tolist(), i.tolist())])
        rep[name] = {**p.stats(eps), 'rank': p.rank, 'n_rel': p.n_rel,
                     'disjoint_U': supports_disjoint(p, eps, 'U'),
                     'disjoint_V': supports_disjoint(p, eps, 'V'),
                     'top_pairs': top}
    return rep


def edge_explanation(layer: DIALayer, h_i: Tensor, h_j: Tensor,
                     e_ij: Optional[Tensor] = None, rel: Optional[Tensor] = None,
                     r_i: Optional[Tensor] = None, r_j: Optional[Tensor] = None,
                     topk: int = 8) -> Dict[str, Any]:
    r"""单条（或一小批）边的完整解释：文档 §4.8 的无绘图版。"""
    was_training = layer.training
    layer.eval()
    with torch.no_grad():
        out = layer(h_i, h_j, e_ij, rel, r_i, r_j,
                    want=('score', 'message', 'matrix', 'factors'))
        aux = dict(layer.aux)
        E = h_i.size(0)
        k = min(topk, h_i.size(-1))
        rep: Dict[str, Any] = {
            'E': int(E),
            's_ij': aux[K_S_IJ].tolist(), 's_ji': aux[K_S_JI].tolist(),
            'alpha_i': aux[K_ALPHA_I].tolist(), 'alpha_j': aux[K_ALPHA_J].tolist(),
            'score': aux[K_SCORE].tolist(),
            'gamma_i_top': [torch.topk(aux[K_GAMMA_I][e], k).indices.tolist()
                            for e in range(E)] if K_GAMMA_I in aux else None,
            'gamma_j_top': [torch.topk(aux[K_GAMMA_J][e], k).indices.tolist()
                            for e in range(E)] if K_GAMMA_J in aux else None,
            'factors_ij': [f.tolist() for f in out['factors_ij']]
            if 'factors_ij' in out else None,
        }
        if 'M_ij' in out:
            M = out['M_ij']
            d_dst = M.size(-1)
            flat = M.flatten(1)
            v, i = flat.topk(min(topk, flat.size(1)), dim=1)
            rep['M_top'] = [[{'src': int(a // d_dst), 'dst': int(a % d_dst),
                              'm': float(w)} for w, a in zip(vv.tolist(), ii.tolist())]
                            for vv, ii in zip(v, i)]
    if was_training:
        layer.train()
    return rep


def score_attribution(head, x: Tensor, edge_index: Tensor,
                      batch: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    r""":math:`s_{ij}` 对 logit 的贡献有多大（把 :math:`s_{ij}` 置零后重算一遍）。"""
    was_training = head.training
    head.eval()
    with torch.no_grad():
        a = head.forward_with(x, edge_index, batch)
        b = head.forward_with(x, edge_index, batch, zero_score=True)
        d = (a - b).abs()
        rep = {'logit_abs_mean': float(a.abs().mean()),
               'delta_l1_mean': float(d.mean()),
               'delta_l1_max': float(d.max()),
               'argmax_flip_frac': float((a.argmax(-1) != b.argmax(-1)).float().mean())}
    if was_training:
        head.train()
    return rep


def dump_pairings(path: str, model, eps: float = 1e-3) -> str:
    """把学到的 :math:`U,V,W` 存成 ``.npz``。"""
    arrays: Dict[str, np.ndarray] = {}
    for name, p in pairings(model):
        key = name.replace('.', '_')
        with torch.no_grad():
            arrays[f'{key}__U'] = p.U.detach().cpu().numpy()
            arrays[f'{key}__V'] = p.V.detach().cpu().numpy()
            arrays[f'{key}__W'] = p.W().detach().cpu().numpy()
        arrays[f'{key}__meta'] = np.array([p.rank, p.n_rel, p.d_src, p.d_dst, eps])
    np.savez_compressed(path, **arrays)
    return path
