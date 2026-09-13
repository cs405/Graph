r"""DIA 的配置：三层结构各自的开关 + 消融组合的单一事实源。

风格与 :class:`modules.oca.OCAConfig` 对齐（同一套 ``cfg/models/*.yaml`` 注入路径、
同一套「开关关掉就不分配参数」的约定），但**不共用**任何字段：OCA 的门控是节点级
（:math:`\lambda_i/\alpha_i/\tau_i`），DIA 的量是边级（:math:`\gamma_{i|j}`、
:math:`M_{ij}`、:math:`\alpha_{i|j}`），混在一个 dataclass 里会让消融表读不懂。

三条写死在这里的主张（改任何一条都要重跑 ``test/test_dia_pairing.py``）：

1. ``use_pairing=False`` 时配对矩阵取 :math:`W=I`（逐维内积），**不是**「没有消息」——
   消融要的是「去掉低秩配对」这一件事，其它两层保持原样；
2. ``symmetric_pairing=True`` 时反向复用 :math:`(V,U)`，于是
   :math:`s^{i\to j}=s^{j\to i}`（技术文档 §2.4 明确要求**不**退化到对称，
   这个开关只是用来量「非对称到底值多少分」）；
3. ``use_projection=False`` 时不建投影分支，:math:`U` 的行**就是原始特征维度**——
   可解释性与可识别性主张只在这个模式下成立（§七 局限 1）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ['DIAConfig']


@dataclass
class DIAConfig:
    """DIA 层/头的配置。``cfg/models/*.yaml`` 的 ``dia:`` 块 = 本结构的一组覆盖。"""

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
