"""全图训练循环（任务无关）。

配置来自根目录 :mod:`config`（嵌套 ``TrainConfig``），结构来自
``cfg/models/*.yaml``（经 :mod:`model_builder`）。本文件只管「怎么训、产出落哪儿」。

协议（与文献对齐，写死在这里以免每次实验重新决定）：

* 每个 epoch 在全图上做一次 forward（transductive），dropout 开；
* 用 **验证集指标** 选最佳 epoch，**test 只在该 epoch 上报一次** ——
  反过来（报 test 上的最高点）就是把测试集当验证集用，异配数据集上能虚高 1-2 点；
* 早停：验证指标连续 ``patience`` 轮无提升即停，最后把最佳权重装回来；
* ``best_metric`` 只允许 ``val_*`` 开头的键，写成 ``test_acc`` 会被当场拒绝（见
  :func:`_metric_key`）—— 这不是防御式编程，是这类项目最常见的「不小心看了测试集」。

**本文件不认识任务，也不认识算子。** 任务特定的三件事（损失/诊断/落盘）全部
装在 :class:`training.base.TrainHooks` 里，由 :mod:`tasks` 按 ``cfg.task`` 分派；
算子特定的两件事（附加损失、参数投影）走 :mod:`modules.base` 的协议。
所以上一版的 ``F.cross_entropy(logits[train_mask], ...)``、``_class_weights``、
``_diag_fn``（靠 ``getattr(model, 'oca_layers')`` 猜算子）已经不在这里了。
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn

from config import TrainConfig
from dataset import GraphBundle, load_dataset
from devices import resolve_device
from metrics.scoring import prf1, summarize
from model_builder import build_model
from training.base import TrainHooks
from training.checkpoints import (CheckpointManager, capture_rng_state,
                                  save_model)
from training.runs import prepare_run_dir, save_config_used, save_json
from utils.seed import count_parameters, set_seed

__all__ = ['RunResult', 'build', 'load_data', 'evaluate', 'fit', 'run_once',
           'run_seeds', 'run_experiment']

_log = logging.getLogger('oca')


@dataclass
class RunResult:
    test: Dict[str, float]
    val: Dict[str, float]
    best_epoch: int
    epochs_ran: int
    n_params: int
    history: List[Dict[str, float]] = field(default_factory=list)
    diagnostics: Dict[str, float] = field(default_factory=dict)
    # 装回 val 最优权重后的模型本身：``hooks.dumps`` 与 ``.val()`` 都需要它。
    # 不参与比较/序列化，仅方便调用方不必自己保存。
    model: Optional[nn.Module] = None
    # 本次跑用的 hooks：``run_experiment`` 要从这里拿落盘函数（它自己没有模型，
    # 而「该不该落 gates.npz」得看模型里有没有那一族的层）。
    hooks: Optional[TrainHooks] = None

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)


def build(cfg: TrainConfig, ds: GraphBundle, device: torch.device,
          verbose: bool = False) -> nn.Module:
    """按 ``cfg.model`` 从结构表搭模型（OCA、DIA 与基线走同一条路，差别只在 YAML）。

    搭完立即校验输出层级：``net.level`` 必须等于 ``ds.supervision``（见
    :meth:`tasks.Task.check_model`）—— 接错了不报错，只会静默错位。
    """
    from tasks import assert_level
    net = build_model(cfg.model.spec, ds.num_features, nc=ds.num_classes,
                      scale=cfg.model.scale, oca=cfg.model.oca,
                      dropout=cfg.model.dropout, device=device,
                      verbose=verbose,
                      spec_overrides=cfg.spec_overrides())
    assert_level(net, ds, cfg)
    return net


def load_data(cfg: TrainConfig, seed: Optional[int] = None) -> GraphBundle:
    """按 ``cfg.dataset`` 装载。``seed`` 决定划分 —— 多种子时要显式传当前种子。

    节点/边两种数据集都走这一个入口，区别只在 ``supervision`` 字段；
    ``cfg.task.edge_neg_ratio`` 只对关系图有意义，普通数据集忽略它。
    """
    return load_dataset(cfg.dataset.name, root=cfg.dataset.root,
                        split_idx=cfg.dataset.split_idx,
                        seed=cfg.seed if seed is None else seed,
                        train_per_class=cfg.dataset.train_per_class,
                        homophily=cfg.dataset.homophily,
                        n_per_class=cfg.dataset.n_per_class,
                        fallback_to_synth=cfg.dataset.fallback_to_synth,
                        edge_neg_ratio=cfg.task.edge_neg_ratio)


def _metric_key(name: str) -> str:
    """``'val_acc' -> 'acc'``，并拒绝任何 ``test_*``（拿 test 选模型 = 数据泄漏）。"""
    if name.startswith('test'):
        raise ValueError(
            f'best_metric={name!r} 是用测试集选模型 —— 这是数据泄漏，改成 val_*')
    if not name.startswith('val'):
        raise ValueError(f'best_metric={name!r} 既不是 val_* 也不是 test_*，'
                         f'无法判断是哪个集合')
    return name.split('_', 1)[1] if '_' in name else 'acc'


def _optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    name = str(cfg.optimizer).lower()
    common: Dict[str, Any] = {'weight_decay': cfg.weight_decay}
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, **common)
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, **common)
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=cfg.lr,
                               momentum=cfg.momentum, **common)
    raise KeyError(f'未知优化器 {name!r}，可选：adam | adamw | sgd')


@torch.no_grad()
def evaluate(model: nn.Module, ds: GraphBundle, mask: Tensor,
             loss_fn=None, crit=None) -> Dict[str, float]:
    """在 ``mask`` 选中的单元上报 ``acc/precision/recall/f1/loss``。

    ``crit`` 给了就用它的判决口径与类别数：单 logit 的二分类必须走阈值，
    ``argmax`` 在 ``[N,1]`` 上恒返回 0（指标不会报错，只会全是 0）。没给则退回
    多分类默认 —— ``model.val()`` 这种「只想要个数」的调用不必先造 criterion。
    ``loss_fn`` 是旧口径 ``(logits, targets) -> Tensor``，保留给外部脚本。
    """
    model.eval()
    logits = model(**ds.forward_kwargs())
    if crit is not None:
        pred, nc = crit.predict(logits), crit.metric_classes
        if loss_fn is None:
            def loss_fn(lg: Tensor, _y: Tensor) -> Tensor:
                return crit.subset_loss(ds, lg, mask)
    else:
        pred, nc = logits.argmax(-1), ds.num_classes
    loss = float('nan') if loss_fn is None else float(loss_fn(logits, ds.y))
    out = prf1(pred, ds.y, mask, nc)
    out['loss'] = loss
    return out


def fit(model: nn.Module, ds: GraphBundle, cfg: TrainConfig,
        device: Optional[torch.device] = None, verbose: bool = False,
        diag_fn=None, run_dir=None,
        hooks: Optional[TrainHooks] = None) -> RunResult:
    """训练一个模型；给了 ``run_dir`` 就顺带写 ``weights/best.pt`` 与滚动 checkpoint。

    ``hooks`` 缺省时按 ``cfg.task`` 现场解析（:func:`tasks.build_hooks`），所以
    ``fit(m, ds, cfg)`` 这种老写法照旧是「节点分类 + CE + 门控诊断」。
    ``diag_fn`` 也是旧参数：显式给了就覆盖 ``hooks.diagnostics``。
    """
    device = device or ds.x.device
    model = model.to(device)
    if hooks is None:
        from tasks import build_hooks
        hooks = build_hooks(cfg, ds, model, device)
    if diag_fn is not None:
        hooks = replace(hooks, diagnostics=diag_fn)
    crit = hooks.criterion
    opt = _optimizer(cfg, model)
    key = _metric_key(cfg.best_metric)
    minimize = key == 'loss'                    # 只有 loss 是越小越好

    ckpts = None
    if run_dir is not None and cfg.ckpt_interval > 0:
        ckpts = CheckpointManager(run_dir / 'checkpoints',
                                 max_keep=cfg.max_ckpt)

    best_val, best_state, best_ep = (float('inf') if minimize else -1.0), None, 0
    bad = 0
    history: List[Dict[str, float]] = []
    for ep in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        # transductive：全图 forward，但 loss 只取 train 单元。把非 train 单元的
        # 预测也塞进损失等于把手工标注的 val/test 标签喂进训练。
        logits = model(**ds.forward_kwargs())
        loss = crit.train_loss(model, ds, logits)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        # 投影必须在 step **之后**：DIA 的非负 U/V 靠它维持（:class:`modules.base.Constraint`），
        # 而没有约束的模型上它是一次空遍历 —— 两边都不用改这里。
        hooks.post_step(model)

        tr = evaluate(model, ds, ds.train_mask, crit=crit)
        va = evaluate(model, ds, ds.val_mask, crit=crit)
        row = {'epoch': ep, 'loss': loss.item(), 'train_acc': tr['acc'],
               'val_acc': va['acc'], 'val_f1': va['f1'], 'val_loss': va['loss']}
        # 惩罚项逐轮记下：「λ 调大了到底有没有把 U 压稀疏」只能从这条曲线看
        row.update({f'pen_{k}': v for k, v in crit.last_penalty.items()})
        history.append(row)
        score = va[key]
        if verbose and (ep % max(cfg.log_interval, 1) == 0 or ep == 1):
            _log.info('ep%-5d loss=%.4f train=%.4f val=%.4f%s%s',
                      ep, loss.item(), tr['acc'], va['acc'],
                      f'  [{cfg.best_metric}={score:.4f}]'
                      if key not in ('acc',) else '',
                      ''.join(f'  {k}={v:.4f}'
                              for k, v in crit.last_penalty.items()))
        better = score < best_val if minimize else score > best_val
        if better:
            best_val, best_ep, bad = score, ep, 0
            best_state = copy.deepcopy(model.state_dict())
            if run_dir is not None and cfg.save:
                save_model(model, run_dir / 'weights' / 'best.pt',
                           meta={'epoch': ep, cfg.best_metric: float(score),
                                 'config': cfg.to_dict(),
                                 'spec': cfg.model.spec, 'scale': cfg.model.scale,
                                 'nc': ds.num_classes,
                                 'in_dim': ds.num_features,
                                 'supervision': ds.supervision})
        else:
            bad += 1
            if bad >= cfg.patience:
                break
        if ckpts is not None and ep % cfg.ckpt_interval == 0:
            ckpts.save({'model': model.state_dict(), 'epoch': ep,
                        'optimizer': opt.state_dict(),
                        'rng': capture_rng_state()}, ep, 0)

    assert best_state is not None, (
        f'{cfg.best_metric} 一个有限值的提升都没有（lr/epochs 不匹配？'
        f'或该指标在 val 上恒为 NaN）')
    model.load_state_dict(best_state)            # 装回 val 最优权重
    te = evaluate(model, ds, ds.test_mask, crit=crit)
    va = evaluate(model, ds, ds.val_mask, crit=crit)
    res = RunResult(test=te, val=va, best_epoch=best_ep, epochs_ran=len(history),
                    n_params=count_parameters(model), history=history, model=model,
                    hooks=hooks)
    if hooks.diagnostics is not None:
        res.diagnostics = hooks.diagnostics(model, ds)
    return res


def run_once(cfg: TrainConfig, ds: Optional[GraphBundle] = None,
             verbose: bool = False, with_diag: bool = True,
             run_dir=None, hooks: Optional[TrainHooks] = None) -> RunResult:
    """种子在函数内部落地：同一 cfg 两次调用结果必须一致。"""
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    if ds is None:
        ds = load_data(cfg).to(device)
    ds = ds.to(device)
    model = build(cfg, ds, device)
    if hooks is None:
        from tasks import build_hooks
        hooks = build_hooks(cfg, ds, model, device)
    if not with_diag:
        # replace 而不是就地改：多种子共用同一份 hooks，就地改会把第一个种子的
        # 「不要诊断」泄漏给后面几个。
        hooks = replace(hooks, diagnostics=None)
    return fit(model, ds, cfg, device, verbose=verbose, run_dir=run_dir,
               hooks=hooks)


def run_seeds(cfg: TrainConfig, seeds: Optional[List[int]] = None,
              ds: Optional[GraphBundle] = None, verbose: bool = False,
              run_dir=None, hooks: Optional[TrainHooks] = None) -> Dict[str, Any]:
    """多种子：报 mean ± std。单次跑分在异配数据集上没有意义（std 常 1-3 点）。

    ``run_dir`` 只给**第一个种子**用 —— 否则同一目录下后一个种子会覆盖前一个的
    ``best.pt``，而 ``results.json`` 里写的是三个种子的平均，读的人无从察觉。
    """
    seeds = [int(s) for s in (seeds if seeds is not None else cfg.seeds)]
    rows = [run_once(cfg.patched(seed=s), ds, verbose=verbose,
                     run_dir=(run_dir if k == 0 else None), hooks=hooks)
            for k, s in enumerate(seeds)]
    acc = summarize([r.test['acc'] for r in rows])
    f1 = summarize([r.test['f1'] for r in rows])
    return {'acc': acc, 'f1': f1, 'runs': rows, 'seeds': seeds,
            'best_epoch': summarize([float(r.best_epoch) for r in rows])['mean']}


def run_experiment(cfg: TrainConfig, ds: Optional[GraphBundle] = None,
                   verbose: bool = False) -> Dict[str, Any]:
    """一次完整实验：建目录 → 快照配置 → 训练（可能多种子）→ 落盘产物。

    产物（``cfg.run.save=False`` 时一个都不写）：

    * ``config_used.yaml`` —— 实际生效的配置（追溯以它为准，不以 YAML 默认值为准）
    * ``logs/train.log``   —— 同一个日志文件里还留有逐轮曲线
    * ``weights/best.pt``  —— 第一个种子、val 最优 epoch 的权重（含重建元信息）
    * ``results.json``     —— test/val 指标、逐种子明细、诊断摘要
    * ``<dump>.npz``       —— 任务声明的导出数据（节点分类是 ``gates.npz``，
      边级任务是 ``pairings.npz``）；哪些该写由 ``hooks.dumps`` 说了算，
      本函数不认得它们的内容
    """
    from log_setup import add_file_logging, setup_logging

    setup_logging()
    device = resolve_device(cfg.device)
    run_dir = None
    if cfg.save:
        run_dir, _ = prepare_run_dir(cfg.project, cfg.run_name, 'train')
        log_path = add_file_logging(run_dir)
        save_config_used(run_dir, cfg.to_dict())
    if ds is None:
        ds = load_data(cfg).to(device)
    _log.info('%s on %s：N=%d E=%d C=%d F=%d split=%s device=%s',
              cfg.model.spec, ds.name, ds.num_nodes, ds.edge_index.size(1),
              ds.num_classes, ds.num_features,
              ds.meta.get('split_source', '?'), device)
    # vary_split=False => 所有种子共用同一份划分（差异只来自初始化/dropout）
    out = run_seeds(cfg, ds=(None if cfg.vary_split else ds), verbose=verbose,
                    run_dir=run_dir)
    r0 = out['runs'][0]
    paths: Dict[str, str] = {}
    if run_dir is not None:
        paths['config_used'] = str(run_dir / 'config_used.yaml')
        paths['log'] = str(log_path)
        paths['best'] = str(run_dir / 'weights' / 'best.pt')
        # 落盘项由 hooks 给（已启用的那几个）；放 run_dir 根而不是 weights/：
        # 它们不是权重，是画图用的导出数据。
        for name, fn in (r0.hooks.dumps.items() if r0.hooks else {}):
            paths[name] = str(fn(str(run_dir / f'{name}.npz'), r0.model, ds))
        payload = {'acc': out['acc'], 'f1': out['f1'],
                   'best_epoch': out['best_epoch'], 'seeds': out['seeds'],
                   'task': (r0.hooks.task if r0.hooks is not None else ''),
                   'dataset': {'name': ds.name, 'source': ds.source,
                               'num_nodes': ds.num_nodes,
                               'num_edges': ds.num_edges,
                               'supervision': ds.supervision,
                               'num_classes': ds.num_classes,
                               'num_features': ds.num_features},
                   'params': [r.n_params for r in out['runs']],
                   'loss': (r0.hooks.describe() if r0.hooks is not None else {}),
                   'diagnostics': {k: (list(map(float, v))
                                       if isinstance(v, (list, tuple)) else
                                       float(v))
                                   for k, v in r0.diagnostics.items()},
                   'paths': paths}
        paths['results'] = str(save_json(run_dir / 'results.json', payload))
    out.update({'cfg': cfg, 'run_dir': run_dir, 'paths': paths, 'dataset': ds})
    _log.info('test acc %.4f ± %.4f   f1 %.4f ± %.4f（n=%d）',
              out['acc']['mean'], out['acc']['std'], out['f1']['mean'],
              out['f1']['std'], out['acc']['n'])
    return out
