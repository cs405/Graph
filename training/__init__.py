"""训练层：训练循环、损失、运行目录、checkpoint、诊断、消融表。

配置树在根目录 :mod:`config`，网络结构在 ``cfg/models/*.yaml``（由
:mod:`model_builder` 展开）—— 这里**不再放任何配置定义**，避免同一个概念有
两个 import 路径。

三层的分工（新增算子/任务时照这个分）：

* :mod:`training.trainer` —— 循环本身，任务无关、算子无关；
* :mod:`training.losses` / :mod:`training.base` —— 损失与扩展点的**抽象**，
  只认 :mod:`modules.base` 的协议；
* :mod:`tasks` —— 具体任务（节点/边）把上面两样组装成 ``TrainHooks``。
"""

from training.base import DiagFn, DumpFn, TrainHooks
from training.checkpoints import (CheckpointManager, atomic_save,
                                  checkpoint_name, load_checkpoint, save_model)
from training.diagnostics import collect_gates, dump_gates, gate_report
from training.losses import (LOSS_REGISTRY, BinaryCriterion, Criterion,
                             CrossEntropyCriterion, build_criterion,
                             class_weights, weighted_penalties)
from training.runs import has_checkpoint, list_run_dirs, prepare_run_dir, \
    save_config_used
from training.sweep import ABLATIONS, apply_variant, format_table, run_ablation
from training.trainer import (RunResult, build, evaluate, fit, load_data,
                              run_experiment, run_once, run_seeds)

__all__ = ['RunResult', 'build', 'load_data', 'evaluate', 'fit', 'run_once',
           'run_seeds', 'run_experiment', 'collect_gates', 'gate_report',
           'dump_gates', 'ABLATIONS', 'apply_variant', 'run_ablation',
           'format_table', 'prepare_run_dir', 'list_run_dirs', 'has_checkpoint',
           'save_config_used', 'CheckpointManager', 'checkpoint_name',
           'atomic_save', 'save_model', 'load_checkpoint',
           'TrainHooks', 'DiagFn', 'DumpFn', 'Criterion', 'LOSS_REGISTRY',
           'CrossEntropyCriterion', 'BinaryCriterion', 'build_criterion',
           'class_weights', 'weighted_penalties']
