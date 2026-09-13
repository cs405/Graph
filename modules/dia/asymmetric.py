r"""第三层：非对称主体贡献（技术文档 §2.4）。

.. math::

    [\alpha_{i|j},\alpha_{j|i}]=\mathrm{softmax}_2\big(\mathrm{MLP}([h_i;h_j;e_{ij}])
    +b_{r_i\to r_j}\big),\qquad
    s_{ij}=\alpha_{i|j}s^{i\to j}+\alpha_{j|i}s^{j\to i}

这一层解决的是「这条边由谁主导」。文档给的例子是人-动物共现：判断「这是一篇
讲动物行为的论文」时主体是人（作者视角），判断「这个物种出现在哪些文献里」时主体
是动物 —— 同一条边，两个方向，主导方不同。

实现取舍：

1. **softmax over 2 而不是两个独立 sigmoid**。:math:`\alpha_{i|j}+\alpha_{j|i}=1`
   是「主导权是零和的」这条语义本身；两个独立 sigmoid 会给出 (0.9, 0.9) 这种
   「双方都主导」的读数，解释性上说不通。
2. **末层权重清零** ⇒ 初始 :math:`\alpha=(0.5,0.5)`，于是初始分数恰好是对称平均
   :math:`(s^{i\to j}+s^{j\to i})/2`。这不是巧合而是刻意的：训练从「对称」出发，
   非对称是**学出来的**，于是消融表里 ``use_asymmetric=False``（恒取 0.5）与
   训练 0 轮的完整模型严格相等，而训练后的差值就是「非对称学到的那部分」。
3. **类型偏置 :math:`b_{r_i\to r_j}` 建在节点类型对上**，同时给一份关系类型偏置。
   同构图（``n_node_types=1``）下它就退化成 2 个标量，零初始化 = 不参与。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from modules.dia.attention import _mlp

__all__ = ['AsymmetricContribution']


class AsymmetricContribution(nn.Module):
    r"""算 :math:`(\alpha_{i|j},\alpha_{j|i})`，两者相加为 1。"""

    def __init__(self, d_src: int, d_dst: int, d_edge: int = 0, hidden: int = 64,
                 n_node_types: int = 1, n_rel: int = 1):
        super().__init__()
        assert d_src > 0 and d_dst > 0, (d_src, d_dst)
        assert n_node_types >= 1 and n_rel >= 1, (n_node_types, n_rel)
        self.d_src, self.d_dst, self.d_edge = int(d_src), int(d_dst), int(d_edge)
        self.n_node_types, self.n_rel = int(n_node_types), int(n_rel)
        self.trunk = _mlp(d_src + d_dst + d_edge, hidden, 2, bias_init=0.0)
        # 类型偏置 b_{r_i -> r_j}：[T, T, 2]，零初始化。同构图（T=1）下它只是 2 个
        # 标量、且永远拿不到梯度（没有类型可查），故不分配 —— 与「开关关掉就不
        # 分配参数」同一套约定；关系偏置同理。
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
        logits = self.trunk(torch.cat([h_s, h_d, e], dim=-1))     # [E,2]
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
                    f'请在 batch 里给 rel（见 modules/dia/edges.py 的契约）')
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
