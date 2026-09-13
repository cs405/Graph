r"""训练侧的扩展点：:class:`TrainHooks`（对应 ultralytics ``engine/trainer.py`` 里
那一堆 ``self.add_callback(...)``，但这里用**一个显式参数**而不是全局回调表）。

``trainer.fit`` 只干一件事：全图前向 → 损失 → 反传 → 更新 → 评估 → 选最优。
它不该知道：

* 损失是 CE 还是 BCE、要不要加稀疏/正交惩罚（→ :class:`training.losses.Criterion`）；
* 训练完要诊断什么（OCA 报 λ 与局部同质性的相关，DIA 报配对矩阵的支撑集）；
* 要往 ``run_dir`` 里落哪些 ``*.npz``（``gates.npz`` / ``pairings.npz``）；
* ``optimizer.step()`` 之后要不要把参数投影回可行域。

这四件事由调用方（``tasks/*``）打包成 ``TrainHooks`` 传进来。**默认值是协议驱动的**：
``post_step`` 缺省就是 :func:`modules.base.apply_constraints` —— 它只问「你实现了
:class:`~modules.base.Constraint` 吗」，不问「你是 DIA 吗」，所以基线模型上它是
一次空遍历，DIA 上它自动生效，两边都不用改 trainer。

不选全局回调表的原因：回调表的注册顺序与生效范围是隐式的，消融跑两轮之间
忘清一次就会串味；显式传参则每个 ``RunResult`` 都能追溯到当时那份 hooks。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional

from torch import nn

from config import TrainConfig
from dataset.base import GraphBundle
from modules.base import apply_constraints
from training.losses import Criterion, build_criterion

__all__ = ['TrainHooks', 'DiagFn', 'DumpFn']

#: ``(model, ds) -> {标量或列表}``，进 ``RunResult.diagnostics`` 与 ``results.json``
DiagFn = Callable[[nn.Module, GraphBundle], Dict[str, Any]]
#: ``(path, model, ds) -> path``，落盘画图用的数据（与 ``training.diagnostics.dump_gates`` 同签名）
DumpFn = Callable[[str, nn.Module, GraphBundle], str]


@dataclass
class TrainHooks:
    """一次训练需要的全部「任务/算子特定」行为。"""

    criterion: Criterion
    #: 任务名（``tasks.TASK_MAP`` 的键）：写进 ``results.json``，事后才知道这跑的是哪个任务
    task: str = ''
    #: 训练结束后算一次（在装回 val 最优权重之后）
    diagnostics: Optional[DiagFn] = None
    #: 名字 -> 落盘函数；``run_experiment`` 只遍历**已启用**的项，
    #: 「启用与否」由 tasks 决定（``cfg.dump_gates`` 且模型里真有那一族的可解释层）
    dumps: Dict[str, DumpFn] = field(default_factory=dict)
    #: ``optimizer.step()`` 之后调；返回被投影的模块数（0 也正常）
    post_step: Callable[[nn.Module], Any] = apply_constraints

    @classmethod
    def default(cls, cfg: TrainConfig, ds: GraphBundle,
                device: Optional[Any] = None,
                diagnostics: Optional[DiagFn] = None,
                dumps: Optional[Dict[str, DumpFn]] = None,
                task: str = '') -> 'TrainHooks':
        """损失按 ``cfg.loss`` 自动选，诊断/落盘由调用方给（缺省就是没有）。"""
        return cls(criterion=build_criterion(cfg, ds, device), task=task,
                   diagnostics=diagnostics, dumps=dict(dumps or {}))

    def with_task(self, name: str) -> 'TrainHooks':
        """盖个任务名（``tasks.build_hooks`` 用）。``replace`` 而不是就地改：
        多种子共用同一份 hooks 时，就地改会把这个种子的修改泄漏给下一个。"""
        return replace(self, task=name)

    def describe(self) -> Dict[str, Any]:
        from training.losses import describe
        out = describe(self.criterion)
        out['task'] = self.task
        out['dumps'] = sorted(self.dumps)
        out['diagnostics'] = getattr(self.diagnostics, '__name__', None)
        return out
