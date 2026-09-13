r"""第二层：低秩非负维度配对（技术文档 §2.3、§2.7）。

配对矩阵 :math:`W = U V^\top`，:math:`U\in\mathbb R_{\ge0}^{d_i\times k}`、
:math:`V\in\mathbb R_{\ge0}^{d_j\times k}`，**按关系类型**各持有一份
（:math:`U` 的行/列是「维度槽」，不是逐边参数 —— 逐边的东西是 §2.2 的
:math:`\gamma_{i|j}`；把 :math:`U,V` 也做成逐边的，可识别性定理的前提就不成立了）。

两个实现上的决定，都与文档 §4.2 的参考代码不同，理由写在下面：

1. **不物化 :math:`M`**。文档里 ``M = h_i.unsqueeze(-1) * W.unsqueeze(0) *
   h_j.unsqueeze(-2)`` 是 :math:`[E,d_i,d_j]`：E=1e5、d=64 时一层就要 1.6 GB。
   而分数与消息都能因子化，
   :math:`s = \tilde h_i^\top U V^\top \tilde h_j = (U^\top\tilde h_i)\cdot(V^\top\tilde h_j)`、
   :math:`m = \tilde h_i\odot U(V^\top\tilde h_j)`，
   中间量只有 :math:`[E,k]`，代价 :math:`O(E\,d\,k)`。``M`` 只在
   ``want='matrix'``（对拍、画图）时才算。
2. **先算全部关系类型再 gather，不先 ``U[rel]``**。后者要物化 :math:`[E,d,k]`，
   比 :math:`M` 好不了多少；前者只多算 :math:`R` 倍、中间量是 :math:`[E,R,k]`。

可识别性（§2.7）在代码里对应三件事，缺一件定理就不成立：
``nonneg``（:math:`U,V\ge0`，由 :meth:`project_parameters` 在 ``optimizer.step()``
之后强制）、``cfg.loss.lambda_sp``（列稀疏）、``cfg.loss.lambda_orth``（列间分离）。
非负 + 列正交只有一种解：**各列支撑集互不相交** —— 这正是「哪个维度槽对应哪一组
原始维度」能被恢复出来的原因。
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from modules.base import Constraint, Regularized

__all__ = ['LowRankNonNegPairing']


class LowRankNonNegPairing(nn.Module, Constraint, Regularized):
    r"""一个方向（``src -> dst``）的低秩非负配对。反向由 :class:`DIALayer` 另建一份。"""

    def __init__(self, d_src: int, d_dst: int, rank: int = 8, n_rel: int = 1,
                 nonneg: bool = True, project: str = 'clamp',
                 init_scale: float = 0.1, sparse_penalty: bool = True,
                 orth_penalty: bool = True):
        super().__init__()
        assert d_src > 0 and d_dst > 0, f'维度必须为正：({d_src}, {d_dst})'
        assert n_rel >= 1, f'n_rel 至少 1（收到 {n_rel}）'
        if rank > min(d_src, d_dst):
            # 秩超过 min(d_src,d_dst) 时 UV^T 能表示任意矩阵，"低秩"这个前提没了，
            # 可识别性定理的结论也就不再是「唯一到置换与缩放」而是「什么都不保证」。
            raise ValueError(
                f'rank={rank} 超过 min(d_src, d_dst)={min(d_src, d_dst)}：'
                f'那就不是低秩分解了（改成 rank<= {min(d_src, d_dst)}）')
        self.d_src, self.d_dst, self.rank, self.n_rel = \
            int(d_src), int(d_dst), int(rank), int(n_rel)
        self.nonneg, self.project = bool(nonneg), project
        self.sparse_penalty, self.orth_penalty = sparse_penalty, orth_penalty
        # torch.rand >= 0：初值本身就在可行域里，第一次投影是 no-op（可测试）
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
        r"""``{'sp': ||U||_1+||V||_1, 'orth': ||U^\top U-I||_F^2+||V^\top V-I||_F^2}``。

        返回未加权项，权重在 ``cfg.loss.lambda_*``（算子不该知道自己占多大比例）。
        """
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
        W = self.U @ self.V.transpose(1, 2)                # [R, d_src, d_dst]
        if rel is None:
            return W[0] if self.n_rel == 1 else W
        return W[int(rel)]

    def _project_bank(self, h: Tensor, bank: Tensor, rel: Tensor) -> Tensor:
        r"""``h [E,d]`` 与 ``bank [R,d,k]`` -> ``[E,k]``（按 ``rel`` 取用那一套因子）。"""
        all_r = torch.einsum('ed,rdk->erk', h, bank)       # [E,R,k]
        idx = rel.view(-1, 1, 1).expand(-1, 1, bank.size(2))
        return all_r.gather(1, idx).squeeze(1)

    def _back_bank(self, c: Tensor, bank: Tensor, rel: Tensor) -> Tensor:
        r"""``c [E,k]`` 与 ``bank [R,d,k]`` -> ``[E,d]``（:math:`U(V^\top\tilde h)` 的后半）。"""
        all_r = torch.einsum('ek,rdk->erd', c, bank)       # [E,R,d]
        idx = rel.view(-1, 1, 1).expand(-1, 1, bank.size(1))
        return all_r.gather(1, idx).squeeze(1)

    def matrix(self, h_s: Tensor, h_d: Tensor,
               rel: Optional[Tensor] = None) -> Tensor:
        r""":math:`M_{ij}=\mathrm{diag}(\tilde h_i)UV^\top\mathrm{diag}(\tilde h_j)`。

        形状 ``[E, d_src, d_dst]`` —— 只给对拍与画图用，训练路径不要碰（见模块 docstring）。
        """
        rel = self._rel(h_s, rel)
        W = self.U[rel] @ self.V[rel].transpose(1, 2)       # [E, d_src, d_dst]
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

        Args:
            h_s: ``[E, d_src]`` 筛选后的源端特征 :math:`\tilde h_{i|j}`
            h_d: ``[E, d_dst]`` 筛选后的邻端特征 :math:`\tilde h_{j|i}`
            rel: ``[E]`` 关系类型 id；``None`` = 单关系
            want: 要哪些量。``'score'``（标量边强度 :math:`s`）、
                ``'message'``（向量消息 :math:`m\in\mathbb R^{d_src}`）、
                ``'matrix'``（:math:`M`，贵）。

        Returns:
            ``{'s': [E], 'm': [E,d_src], 'factors': ([E,k], [E,k])}`` 的子集；
            ``factors`` 恒返回，它是 :math:`s=(U^\top\tilde h_i)\cdot(V^\top\tilde h_j)`
            的两个因子，也是「哪些维度槽在这条边上被点亮」的直接读数。
        """
        want = set(want)
        rel = self._rel(h_s, rel)
        a = self._project_bank(h_s, self.U, rel)            # [E,k]
        b = self._project_bank(h_d, self.V, rel)            # [E,k]
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
        """配对矩阵的稀疏度/秩诊断（``explain()`` 与 ``pairing_report`` 共用）。"""
        with torch.no_grad():
            U, V = self.U, self.V
            W = U @ V.transpose(1, 2)                       # [R, d_src, d_dst]
            # 可识别性看的是**列**（维度槽）：每列的非零行数 = 这个槽管着多少原始维度，
            # 分离条件要求两列支撑集不相交 —— 所以这里报列的均值与最大值，不报行的。
            col_nnz = (U > eps).to(U.dtype).sum(1)          # [R, k]
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
