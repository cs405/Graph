r"""第一层：边条件维度注意力（技术文档 §2.2）。

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
2. **关系条件用加性偏置，不用逐关系一套 MLP**。文档下标写的是 :math:`\mathrm{MLP}_{r,t}`
   （按关系 r 与节点类型 t 各一套）。真按关系建 MLP，参数量随 R 线性涨，
   R 大时（DBLP 有 4 类节点、ACM 3 类）主干根本喂不动数据。这里折中成
   主干共享 + 逐关系的加性偏置 ``rel_bias[r]``，R=1 时偏置就是一条全零向量。
3. **sigmoid 而不是 softmax**。softmax 会把「这条边整体不重要」表达不出来
   （它只分配相对权重），而 §2.2 要的是**逐维筛选**：某一维在这条边上完全没用，
   就该是 0，而不是「比别的维度小一点」。

初始化：末层 ``Linear.weight`` 置零、``bias`` 置 ``gate_bias_init``，于是初始
:math:`\gamma\equiv\sigma(b_0)`。取 ``gate_bias_init=0`` 时 :math:`\gamma=0.5`
（半开），取大正值时接近 1（不筛选，等价于关掉这一层）—— 消融表里
``use_dim_attention=False`` 与一个足够大的 ``gate_bias_init`` 在**初值下**给出完全
相同的前向（``rel_bias`` 零初始化，此时不参与），
``test/test_dia_decoupling.py`` 钉住这条。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ['EdgeConditionedDimAttention']


def _mlp(in_dim: int, hidden: int, out_dim: int,
         bias_init: float = 0.0) -> nn.Sequential:
    """``Linear -> ReLU -> Linear``，末层权重清零（理由见模块 docstring 的「初始化」）。"""
    head = nn.Linear(hidden, out_dim)
    nn.init.zeros_(head.weight)
    nn.init.constant_(head.bias, bias_init)
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), head)


class EdgeConditionedDimAttention(nn.Module):
    r"""算 :math:`\gamma_{i|j}` 与 :math:`\gamma_{j|i}`（都是逐维、都在 (0,1)）。"""

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
        # 它永远拿不到梯度（没有 rel 可查），白白占一行参数与一份优化器状态 ——
        # 与 OCAConfig「开关关掉就不分配参数」同一套约定。
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
            # 建了逐关系偏置却拿不到关系 id：cfg 与数据对不上。不报错的话这一路
            # 信息会静默不参与（前向照跑，只是每层都用同一套偏置）。
            raise ValueError(
                f'这层建了 n_rel={self.n_rel} 的关系偏置，但前向没拿到 rel；'
                f'请在 batch 里给 rel（见 modules/dia/edges.py 的契约）')
        rel = rel.long()
        assert int(rel.max()) < self.n_rel, \
            f'关系类型 id {int(rel.max())} 超出 n_rel={self.n_rel}'
        return self.rel_bias[rel]

    def extra_repr(self) -> str:
        return (f'd_src={self.d_src}, d_dst={self.d_dst}, d_edge={self.d_edge}, '
                f'n_rel={self.n_rel}')
