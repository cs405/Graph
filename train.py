r"""OCA 训练入口（对应 yolov8 根目录的 ``train.py``）。

    python train.py                                   # 用 cfg/train/default.yaml
    python train.py --dataset cora --seeds 0 1 2      # 多种子，报 mean ± std
    python train.py --model cfg/models/gat.yaml       # 换结构表就是换模型
    python train.py --set optimization.lr=5e-3 model.scale=l
    python train.py --oca T=4 beta=0.3 use_phi=false  # 覆盖 OCAConfig
    python train.py --ablate full no_competition T0   # docs/OCA.md §八 的消融表
    python train.py --dry-run --dataset chameleon     # 只体检数据 + 打印结构表

约定两件事，写死在这里：

* **模型选择只看 val**（见 :func:`training.trainer.fit`），test 只报一次；
* 真实数据没预下载时会直接报错而不是偷偷换合成图 —— 除非显式加 ``--fallback-synth``
  （换成了的话输出里会写明 ``source=synthetic``，因为它不能进论文表格）。

命令行只是三个函数外壳，脚本里直接用它们就行：:func:`data_check`（数据体检）、
:func:`ablate`（消融表）、以及 yolov8 风格的 ``OCA(...).train(...)``；测试聚合跑
``python -m test.run_all``（本项目不依赖 pytest）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if hasattr(sys.stdout, 'reconfigure'):              # Windows 控制台默认 GBK
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (ValueError, OSError):                    # 输出已被重定向/关闭
        pass

from config import TrainConfig, load_train_config, parse_overrides   # noqa: E402
from dataset import REAL_DATASETS, GraphBundle, load_dataset         # noqa: E402
from devices import describe_device                                  # noqa: E402
from log_setup import setup_logging                                  # noqa: E402
from model import OCA                                                # noqa: E402
from training.sweep import format_table, run_ablation                # noqa: E402

REAL_CHOICES = sorted(REAL_DATASETS) + ['synth']


# ---------------------------------------------------------------------------
# 可直接调用的入口
# ---------------------------------------------------------------------------

def _cfg(cfg: Optional[TrainConfig] = None, config: Optional[str] = None,
         **overrides: Any) -> TrainConfig:
    """``cfg`` 给了就用，没给则 ``load_train_config(config, **overrides)``。"""
    if cfg is not None:
        return cfg if not overrides else cfg.patched(**overrides)
    return load_train_config(config, **overrides)


def data_check(cfg: Optional[TrainConfig] = None, config: Optional[str] = None,
               print_report: bool = True, **overrides: Any) -> Dict[str, Any]:
    """只做数据体检（不训练），顺带把结构表逐层打一遍 —— 结构错在这里就能看见。"""
    conf = _cfg(cfg, config, **overrides)
    ds = load_dataset(conf.dataset.name, root=conf.dataset.root,
                      split_idx=conf.dataset.split_idx, seed=conf.seed,
                      train_per_class=conf.dataset.train_per_class,
                      homophily=conf.dataset.homophily,
                      n_per_class=conf.dataset.n_per_class,
                      fallback_to_synth=conf.dataset.fallback_to_synth)
    if print_report:
        print(f'== {ds.name}（source={ds.source}，device={describe_device(conf.device)}）==')
        for k, v in ds.stats().items():
            print(f'  {k:<16} {v:g}')
        for k, v in sorted(ds.meta.items()):
            print(f'  meta.{k:<12} {v}')
        if ds.source != 'real':
            print(f'  [!] 数据源是 {ds.source}：只能说明管线跑通，不可当作论文结果')
        info = OCA(conf.model.spec, scale=conf.model.scale,
                   oca=conf.model.oca or None).info(
            verbose=True, in_dim=ds.num_features, nc=ds.num_classes)
        print(f'== 结构：{info["layers"]} 层 / {info["params_total"]:,} 参数 '
              f'（{Path(conf.model.spec).name}, scale={conf.model.scale}）==')
    return {'stats': ds.stats(), 'source': ds.source, 'bundle': ds}


def ablate(cfg: Optional[TrainConfig] = None, variants: Optional[Sequence[str]] = None,
           ds: Optional[GraphBundle] = None, seeds: Optional[List[int]] = None,
           extra: Optional[Dict[str, Dict[str, Any]]] = None,
           verbose: bool = False, config: Optional[str] = None,
           **overrides: Any) -> List[Dict[str, Any]]:
    """跑 docs §八 的消融/对照表；``variants`` 缺省 = 全部。"""
    conf = _cfg(cfg, config, **overrides)
    if seeds:
        conf = conf.patched(seeds=list(seeds))
    rows = run_ablation(conf, variants=variants, seeds=list(conf.seeds), ds=ds,
                        extra=extra, verbose=verbose)
    print(format_table(rows))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(out: Dict[str, Any]) -> None:
    acc, f1 = out['acc'], out['f1']
    ds: GraphBundle = out['dataset']
    cfg: TrainConfig = out['cfg']
    print(f'== {Path(cfg.model.spec).name}/{cfg.model.scale} on {ds.name} '
          f'(N={ds.num_nodes} E={ds.edge_index.size(1)} C={ds.num_classes} '
          f'F={ds.num_features} split={ds.meta.get("split_source", "?")} '
          f'device={describe_device(cfg.device)}) ==')
    if ds.source != 'real':
        print(f'  [!] source={ds.source}：不可当作论文结果')
    print(f'  acc {acc["mean"]:.4f} ± {acc["std"]:.4f}   '
          f'f1 {f1["mean"]:.4f} ± {f1["std"]:.4f}   n={acc["n"]}   '
          f'[{acc["min"]:.4f}, {acc["max"]:.4f}]')
    r0 = out['runs'][0]
    print(f'  首轮细节：params={r0.n_params}  best_ep={r0.best_epoch}/'
          f'{r0.epochs_ran}  val={r0.val["acc"]:.4f}')
    for k in sorted(r0.diagnostics):
        v = r0.diagnostics[k]
        txt = ('[' + '  '.join(f'{x:.3f}' for x in v) + ']') \
            if isinstance(v, (list, tuple)) else f'{v:g}'
        print(f'    {k:<28} {txt}')
    for k, v in out['paths'].items():
        print(f'    {k:<28} {v}')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='train.py', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=None, metavar='PATH',
                   help='cfg/train/*.yaml，缺省 default.yaml')
    p.add_argument('--model', default=None, metavar='SPEC',
                   help='结构表路径，覆盖 model.spec（换模型就改这里）')
    p.add_argument('--scale', default=None, choices=list('nslmx'))
    p.add_argument('--dataset', default=None, choices=REAL_CHOICES)
    p.add_argument('--root', default=None)
    p.add_argument('--split-idx', type=int, default=None,
                   help='WebKB 10 折官方 split 的第几折')
    p.add_argument('--train-per-class', type=int, default=None,
                   help='仅无官方 split 时生效')
    p.add_argument('--fallback-synth', action='store_true',
                   help='真实数据装载失败时退回合成图（结果不可与真实数据混报）')
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--vary-split', action='store_true',
                   help='每个种子重新划分（无官方 split 时才建议开）')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--wd', type=float, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--optimizer', default=None, choices=['adam', 'adamw', 'sgd'])
    p.add_argument('--grad-clip', type=float, default=None)
    p.add_argument('--class-weight', action='store_true')
    p.add_argument('--device', default=None, choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--name', default=None, metavar='RUN_NAME',
                   help='runs/ 下的目录名（缺省取 run.run_name）')
    p.add_argument('--no-save', dest='save', action='store_false', default=None,
                   help='不建 runs/ 目录（冒烟测试用）')
    p.add_argument('--ckpt-interval', type=int, default=None,
                   help='>0 时每 N 个 epoch 存一个滚动 checkpoint')
    p.add_argument('--oca', nargs='*', default=[], metavar='K=V',
                   help='覆盖 OCAConfig，如 --oca T=4 beta=0.3 use_phi=false')
    p.add_argument('--set', dest='sets', nargs='*', default=[], metavar='K=V',
                   help='覆盖任意配置项，如 optimization.lr=5e-3 model.scale=l')
    p.add_argument('--ablate', nargs='*', default=None, metavar='VARIANT',
                   help='跑消融表；不带名字 = 全部变体')
    p.add_argument('--dry-run', action='store_true',
                   help='只体检数据 + 打印结构表，不训练')
    p.add_argument('--verbose', action='store_true')
    return p


def _overrides(a: argparse.Namespace) -> Dict[str, Any]:
    """把命令行参数拍成 ``TrainConfig`` 的覆盖项（命令行 > YAML 文件 > dataclass 默认）。"""
    over = parse_overrides(a.sets)
    pairs: Dict[str, Any] = {
        'dataset.name': a.dataset, 'dataset.root': a.root,
        'dataset.split_idx': a.split_idx,
        'dataset.train_per_class': a.train_per_class,
        'dataset.fallback_to_synth': a.fallback_synth or None,
        'model.spec': a.model, 'model.scale': a.scale, 'model.dropout': a.dropout,
        'run.seeds': a.seeds, 'run.vary_split': a.vary_split or None,
        'run.device': a.device, 'run.run_name': a.name, 'run.save': a.save,
        'run.ckpt_interval': a.ckpt_interval,
        'optimization.epochs': a.epochs, 'optimization.patience': a.patience,
        'optimization.lr': a.lr, 'optimization.weight_decay': a.wd,
        'optimization.optimizer': a.optimizer,
        'optimization.grad_clip': a.grad_clip,
        'optimization.class_weight': a.class_weight or None}
    over.update({k: v for k, v in pairs.items() if v is not None})
    if a.oca:
        over['model.oca'] = parse_overrides(a.oca)
    return over


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    setup_logging()
    over = _overrides(a)
    conf = load_train_config(a.config, **over)
    if a.dry_run:
        data_check(conf)
        return 0
    if a.ablate is not None:
        ablate(conf, variants=a.ablate or None, verbose=a.verbose)
        return 0
    model = OCA(conf.model.spec, config=a.config,
                **{k: v for k, v in over.items() if k != 'model.spec'})
    out = model.train(verbose=a.verbose)
    _print_summary(out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
