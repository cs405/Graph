r"""任务损失：:class:`Criterion` 抽象 + 注册表（对应 ultralytics 的 ``utils/loss.py``）。

上一版训练循环里写死了 ``F.cross_entropy(logits[train_mask], y[train_mask])``，
于是「换一个任务」＝「改 trainer」：边级监督的 ``mask`` 索引的是边而不是节点，
单 logit 的二分类要走阈值而不是 ``argmax``，DIA 的稀疏/正交惩罚项也得有个地方加。
这三件事全塞进 trainer 就是第二次耦合，所以拆出来：

* trainer 只调 ``crit.train_loss(...)`` / ``crit.subset_loss(...)`` / ``crit.predict(...)``，
  不知道损失长什么样；
* **惩罚项的权重在 ``cfg.loss``**，不在算子里：算子只报未加权的项
  （:meth:`modules.base.Regularized.penalties`），同一个 DIA 层因此能在不同任务里
  用不同的 :math:`\lambda`；
* ``loss_type='auto'`` 按数据集的 ``num_classes`` 选：>1 走 CE，==1 走 BCE。
  不按 ``supervision`` 选是因为「边级 + 多关系」仍然是多分类，层级只决定 mask 索引什么。

本文件只 import :mod:`modules.base`（协议层），**不 import 任何具体算子** ——
判据与 ``model_builder`` 一致。
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import TrainConfig
from dataset.base import GraphBundle
from modules.base import collect_penalties

__all__ = ['Criterion', 'CrossEntropyCriterion', 'BinaryCriterion',
           'LOSS_REGISTRY', 'build_criterion', 'class_weights',
           'weighted_penalties', 'describe', 'PENALTY_WEIGHTS']

_log = logging.getLogger('oca')

#: 算子报上来的惩罚项名 -> ``cfg.loss`` 里的权重字段名。
#: 新算子要加一项惩罚，就在 :data:`modules.base.SPEC_BLOCKS` 旁边登记一个权重字段，
#: 两边都改一处，不会出现「算子报了项但没人加权」的静默丢弃。
PENALTY_WEIGHTS: Dict[str, str] = {
    'sp': 'lambda_sp',
    'orth': 'lambda_orth',
    'gamma': 'lambda_gamma',
}


def class_weights(ds: GraphBundle, device: torch.device,
                  power: float = 1.5) -> Optional[Tensor]:
    r"""部分平衡权重：:math:`w_c \propto 1/n_c^{1.5}`（而不是纯逆频率）。

    纯逆频率下稀分类的单样本梯度比多类大两个量级，val 曲线会抖到不可用；
    1.5 次幂是「跟一下不均衡」与「别被稀有类绑架」之间的一个固定选择，
    不是调参旋钮 —— 改动时必须在论文里写明。
    """
    cnt = torch.bincount(ds.y[ds.train_mask], minlength=ds.num_classes).float()
    cnt = cnt.clamp(min=1.0)
    w = cnt.sum() / (cnt ** power)
    return (w / w.mean()).to(device)


def weighted_penalties(model: nn.Module, cfg: TrainConfig) -> Dict[str, Tensor]:
    """把算子报的未加权项按 ``cfg.loss.lambda_*`` 加权；权重为 0 的项**不算**。

    不算而不是乘 0：``orth`` 项要做一次 ``U^T U``，全 0 权重还去算它，
    在大 rank 下是白烧时间；而且乘 0 之后 ``loss.item()`` 里看不出这项到底开没开。
    """
    raw = collect_penalties(model)
    out: Dict[str, Tensor] = {}
    for key, value in raw.items():
        field_name = PENALTY_WEIGHTS.get(key)
        if field_name is None:
            _log.warning('算子报了未登记的惩罚项 %r（不在 %s 里），已忽略',
                         key, sorted(PENALTY_WEIGHTS))
            continue
        lam = float(getattr(cfg.loss, field_name, 0.0))
        if lam > 0.0:
            out[key] = lam * value
    return out


class Criterion(abc.ABC):
    """「怎么把 logits 与标签变成一个标量」+「怎么把 logits 变成预测」。

    ``mask`` 索引的是**被监督的单元**：节点级任务里是节点，边级任务里是边。
    这个语义由 ``ds.supervision`` 决定，Criterion 本身不区分 —— 所以节点分类与
    关系分类可以共用同一个 CE 实现。
    """

    #: 报告用；真正的层级来自 ``ds.supervision``
    name: str = 'criterion'

    def __init__(self, cfg: TrainConfig, ds: GraphBundle,
                 device: Optional[torch.device] = None):
        self.cfg, self.ds = cfg, ds
        self.device = device or ds.x.device
        self.num_classes = int(ds.num_classes)
        self.last_penalty: Dict[str, float] = {}      # 逐轮记录，供日志/结果表

    @property
    def metric_classes(self) -> int:
        """算 ``prf1`` 时传的类别数（混淆矩阵必须是方阵，见 :func:`metrics.scoring.prf1`）。"""
        return self.num_classes

    @abc.abstractmethod
    def raw_loss(self, logits: Tensor, mask: Tensor) -> Tensor:
        """只在 ``mask`` 选中的单元上算任务损失（不含任何惩罚项）。"""

    def predict(self, logits: Tensor) -> Tensor:
        return logits.argmax(-1)

    # ---- trainer 用的三个入口 ----------------------------------------------
    def train_loss(self, model: nn.Module, ds: GraphBundle, logits: Tensor) -> Tensor:
        r"""训练损失 = 任务损失（train mask 上）+ :math:`\sum_k \lambda_k P_k`。

        把非 train 单元的预测也塞进损失，等于把手工标注的 val/test 标签喂进训练 ——
        transductive 全图前向时这一步最容易写错。
        """
        loss = self.raw_loss(logits, ds.train_mask)
        extra = weighted_penalties(model, self.cfg)
        self.last_penalty = {k: float(v.detach()) for k, v in extra.items()}
        if extra:
            loss = loss + sum(extra.values())
        return loss

    def subset_loss(self, ds: GraphBundle, logits: Tensor, mask: Tensor) -> Tensor:
        """某个子集上的**纯任务损失**（``best_metric=val_loss`` 用它选模型）。

        不含惩罚项：惩罚项是训练手段不是评价指标，混进去会让「调大
        :math:`\lambda_{sp}`」看起来像是模型变差了，而它其实只是把损失尺度抬高了。
        """
        return self.raw_loss(logits, mask)

    def __repr__(self) -> str:
        return f'{type(self).__name__}(nc={self.num_classes})'


class CrossEntropyCriterion(Criterion):
    """多分类交叉熵（``cfg.optimization.class_weight=True`` 时按类加权）。"""

    name = 'ce'

    def __init__(self, cfg: TrainConfig, ds: GraphBundle,
                 device: Optional[torch.device] = None):
        super().__init__(cfg, ds, device)
        self.weight = (class_weights(ds, self.device)
                       if cfg.optimization.class_weight else None)

    def raw_loss(self, logits: Tensor, mask: Tensor) -> Tensor:
        return F.cross_entropy(logits[mask], self.ds.y[mask], weight=self.weight)


class BinaryCriterion(Criterion):
    """单 logit 的二分类：``BCEWithLogits`` + 阈值判决。

    ``num_classes`` 在数据集里记的是 **logit 数**（1），而 ``prf1`` 的混淆矩阵需要
    真实的类别数（2），故 :attr:`metric_classes` 单独给 —— 直接传 1 会让
    ``index_put_`` 在下标 1 上越界。
    """

    name = 'bce'

    def __init__(self, cfg: TrainConfig, ds: GraphBundle,
                 device: Optional[torch.device] = None):
        super().__init__(cfg, ds, device)
        assert self.num_classes <= 1, (
            f'BCE 只适用于单 logit（num_classes=1），当前 nc={self.num_classes}；'
            f'多分类请用 loss_type=ce')
        self.threshold = float(cfg.task.threshold)
        self.pos_weight = self._pos_weight()

    def _pos_weight(self) -> Optional[Tensor]:
        if self.cfg.loss.pos_weight > 0.0:
            return torch.tensor([float(self.cfg.loss.pos_weight)],
                                device=self.device)
        if not self.cfg.optimization.class_weight:
            return None
        y = self.ds.y[self.ds.train_mask].float()
        n_pos = float(y.sum())
        n_neg = float(y.numel()) - n_pos
        if n_pos <= 0 or n_neg <= 0:
            return None
        return torch.tensor([n_neg / n_pos], device=self.device)

    @property
    def metric_classes(self) -> int:
        return max(2, self.num_classes)

    def raw_loss(self, logits: Tensor, mask: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            logits[mask].squeeze(-1), self.ds.y[mask].float(),
            pos_weight=self.pos_weight)

    def predict(self, logits: Tensor) -> Tensor:
        return (torch.sigmoid(logits).squeeze(-1) > self.threshold).long()


LOSS_REGISTRY: Dict[str, Type[Criterion]] = {
    'ce': CrossEntropyCriterion,
    'bce': BinaryCriterion,
}


def build_criterion(cfg: TrainConfig, ds: GraphBundle,
                    device: Optional[torch.device] = None) -> Criterion:
    """按 ``cfg.loss.loss_type`` 造损失；``'auto'`` 时按 ``num_classes`` 选。"""
    name = str(cfg.loss.loss_type).lower()
    if name == 'auto':
        name = 'bce' if int(ds.num_classes) <= 1 else 'ce'
    if name not in LOSS_REGISTRY:
        raise KeyError(f'未知 loss_type {name!r}，可选：{sorted(LOSS_REGISTRY)} 或 auto')
    return LOSS_REGISTRY[name](cfg, ds, device)


def describe(crit: Criterion) -> Dict[str, Any]:
    """落进 ``results.json`` 的一行：损失类型 + 生效的惩罚权重。"""
    lam = {k: float(getattr(crit.cfg.loss, v)) for k, v in PENALTY_WEIGHTS.items()}
    return {'loss_type': crit.name, 'num_classes': crit.num_classes,
            'metric_classes': crit.metric_classes,
            'lambda': {k: v for k, v in lam.items() if v > 0.0}}
