r"""算子与框架之间的**契约**：注册表 + 四条协议（对应 ultralytics 的
``nn/modules/__init__.py`` 那一层，但把「谁认识谁」倒了过来）。

解耦的判据只有一句：**训练框架不许 import 任何具体算子**。
框架（``model_builder`` / ``training`` / ``tasks``）只认下面这四件事，算子爱怎么实现
怎么实现；反过来算子也不知道框架长什么样：

| 协议 | 谁实现 | 框架什么时候调 | 不调会怎样 |
| :-- | :-- | :-- | :-- |
| :class:`GraphOp`    | 能被 ``cfg/models/*.yaml`` 引用的结构层 | 构建（宽度缩放/超参注入）与前向 | 结构表里写不了 |
| :class:`Constraint` | 权重空间有约束的算子（DIA 的非负 U/V） | ``optimizer.step()`` **之后** | 约束静默失效 |
| :class:`Regularized`| 有附加损失项的算子（稀疏/正交/门控 L1） | 每次算 loss 时 | 可识别性没有保证 |
| :class:`Explorable` | 有可解释量的算子（OCA 的 λ/α/τ、DIA 的 γ/M/α） | 诊断、``*.npz`` 导出 | 消融表报不出数 |

三条工程约定（``GraphOp`` 的类属性，builder 靠它们分派，不再维护
``BASE_MODULES``/``DERIVED_MODULES``/``INJECT`` 这种「builder 认识算子」的表）：

1. ``scale_out``：``args[0]`` 是输出宽度、要按 ``width_multiple`` 缩放；
2. ``derived_out``：输出宽度由输入推导（``Concat``），``args`` 里没有通道数；
3. ``takes_batch``：前向还要吃 ``batch``（边级任务的关系类型/边特征）。
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch.nn as nn
from torch import Tensor

__all__ = ['MODULES', 'register_module', 'GraphOp', 'Constraint', 'Regularized',
           'Explorable', 'iter_impl', 'collect_constraints', 'apply_constraints',
           'collect_regularizers', 'collect_penalties', 'collect_aux',
           'explainable_layers', 'explain_report', 'SPEC_BLOCKS']

# YAML 里可用的模块名 -> 构造 callable。**由算子自己登记**（``@register_module()``），
# builder 只读这张表：新增一个算子不需要改 builder 一行代码。
MODULES: Dict[str, Callable[..., nn.Module]] = {}


def register_module(name: Optional[str] = None) -> Callable[[type], type]:
    """把结构层登记进 :data:`MODULES`；``name`` 缺省用类名。

    重名当场报错而不是后者悄悄覆盖前者：``GATConv`` 这种「注册名 ≠ 类名」的映射
    一旦撞车，结构表里那行到底建了哪个类就无从查起（``m.type`` 记的是类名，
    读日志的人会以为自己在跑 GAT 基线）。
    """

    def deco(cls: type) -> type:
        key = name or cls.__name__
        old = MODULES.get(key)
        if old is not None and old is not cls:
            raise KeyError(f'模块名 {key!r} 重复注册：{old} 与 {cls} 撞车')
        cls.op_name = key                       # type: ignore[attr-defined]
        MODULES[key] = cls
        return cls

    return deco


# ---------------------------------------------------------------------------
# 协议一：结构层
# ---------------------------------------------------------------------------

class GraphOp(nn.Module):
    """能被结构表引用的层的公共约定。

    ``out_dim`` / ``n_parameters`` 由 :func:`modules.convs._tag` 统一盖上去
    （多输入层与 ``Repeat`` 容器也要有，所以不写成 ``__init__`` 里的赋值）。
    """

    scale_out: bool = True          # args[0] 是输出宽度，按 width_multiple 缩放
    derived_out: bool = False       # 输出宽度由输入推导（Concat）
    takes_batch: bool = False       # forward 还要吃 batch（边级任务）
    # 输出的粒度：'node' = [N, c]，'edge' = [E, c]。结构表把它接错了不会报错（
    # 两边都是二维张量），只会静默错位，故由 builder 报给 ``net.level`` 让上层校验。
    level: str = 'node'
    # 构造参数名 -> 配置来源名；来源可以是标量（'dropout'）或结构表的算子块（'oca'/'dia'）
    inject: Dict[str, str] = {}
    out_dim: int = 0
    n_parameters: int = 0

    def forward(self, x: Any, edge_index: Optional[Tensor] = None,
                **kw: Any) -> Tensor:
        raise NotImplementedError(f'{type(self).__name__} 没实现 forward')


# ---------------------------------------------------------------------------
# 协议二/三/四：约束、附加损失、可解释量
# ---------------------------------------------------------------------------

class Constraint(abc.ABC):
    """有权重空间约束的算子。投影发生在 ``optimizer.step()`` 之后（不在 forward 里），
    因为投影是不可导的：放进前向等于每步都把梯度改掉，收敛性质说不清。
    """

    @abc.abstractmethod
    def project_parameters(self) -> None:
        """把参数投影回可行域（如 :math:`U\\leftarrow\\max(U,0)`）。幂等。"""


class Regularized(abc.ABC):
    """有附加损失项的算子。返回**未加权**的各项，权重由 ``cfg.loss`` 决定 ——
    算子不该知道自己在整个训练里占多大比例。
    """

    @abc.abstractmethod
    def penalties(self) -> Dict[str, Tensor]:
        """``{'sp': tensor, 'orth': tensor, ...}``（标量、在同一 device 上）。"""


class Explorable(abc.ABC):
    """有可解释量的算子：``aux`` 是**每次 forward 重写**的字典。

    每次重写而不是增量更新：读到上一次（可能是另一套开关/另一个后端）的残留，
    是这类「顺手存个中间量」最容易踩的坑。
    """

    aux: Dict[str, Tensor]
    #: 家族标签（``'oca'`` / ``'dia'``）。框架侧需要「只挑某一个算子的可解释层」时
    #: 比的是这个字符串，而不是 import 那个算子（否则解耦就白做了）。
    family: str = ''

    @abc.abstractmethod
    def explain(self) -> Dict[str, Any]:
        """给人看的一行摘要（标量/小张量），诊断与导出都从这里取。"""


# ---------------------------------------------------------------------------
# 发现：框架侧只调这四个函数，不 import 任何算子
# ---------------------------------------------------------------------------

def iter_impl(model: nn.Module, kind: type) -> List[nn.Module]:
    """模型里实现了 ``kind`` 协议的子模块，按 ``modules()`` 的遍历序（= 构建序）。"""
    return [m for m in model.modules() if isinstance(m, kind)]


def collect_constraints(model: nn.Module) -> List[nn.Module]:
    return iter_impl(model, Constraint)


def apply_constraints(model: nn.Module) -> int:
    """对全模型做一次投影，返回被投影的模块数（0 = 这个模型没有约束，正常）。"""
    cs = collect_constraints(model)
    for m in cs:
        m.project_parameters()                     # type: ignore[attr-defined]
    return len(cs)


def collect_regularizers(model: nn.Module) -> List[nn.Module]:
    return iter_impl(model, Regularized)


def collect_penalties(model: nn.Module) -> Dict[str, Tensor]:
    """把所有 :class:`Regularized` 的同名项**相加**（不同算子的 'sp' 就该合并成一项）。

    返回空 dict 表示这个模型没有附加损失 —— 调用方要能接受，不能假设有 'sp'。
    """
    out: Dict[str, Tensor] = {}
    for m in collect_regularizers(model):
        for k, v in m.penalties().items():         # type: ignore[attr-defined]
            out[k] = v if k not in out else out[k] + v
    return out


def collect_aux(model: nn.Module) -> List[Dict[str, Tensor]]:
    """逐层的 ``aux``（顺序 = 层序），诊断与 ``*.npz`` 导出共用这一条取数路径。"""
    return [dict(m.aux) for m in iter_impl(model, Explorable)
            if getattr(m, 'aux', None)]


def explainable_layers(model: nn.Module, family: Optional[str] = None) -> nn.ModuleList:
    """模型里全部可解释层（:class:`Explorable`），可用 ``family`` 再筛一道。

    它是 ``GraphSequential.oca_layers`` 的**去算法版**：诊断代码拿这个就够了，
    不需要知道层是 OCA 还是 DIA（``family='oca'`` 也只是比一个字符串）。
    """
    ms = iter_impl(model, Explorable)
    if family is not None:
        ms = [m for m in ms if getattr(m, 'family', '') == family]
    return nn.ModuleList(ms)


def explain_report(model: nn.Module) -> Dict[str, Any]:
    """``{层号: explain()}``，供 ``results.json`` 直接落盘。"""
    out: Dict[str, Any] = {}
    for i, m in enumerate(iter_impl(model, Explorable)):
        try:
            out[f'layer{i}'] = m.explain()          # type: ignore[attr-defined]
        except Exception as e:                      # 解释量不该让训练失败
            out[f'layer{i}'] = {'error': f'{type(e).__name__}: {e}'}
    return out


# 结构表里可以出现的「算子超参块」名。``model_builder`` 与 ``config`` 都读它：
# 加一个新算子（自带一份超参块）时只改这一处，两边不会漂。
SPEC_BLOCKS: Tuple[str, ...] = ('oca', 'dia')
