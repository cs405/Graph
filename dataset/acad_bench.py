r"""学术图上的外部基线表：``mlp`` / ``floor+`` / ``gcn`` / ``gat`` / ``oca`` 同表对账。

为什么要有这个文件：§9.12 的地板与地板+ 用的是诊断脚本自己那套拟合配方
（``hetero_probe._fit_eval``：Adam lr 3e-3、wd 1e-5、无 dropout），而仓库的正式训练路径
（``train.py`` → :func:`training.trainer.fit`）是另一套（lr 1e-2、wd 5e-4、dropout 0.5）。
同一个 chameleon、同一份 ``row_normalize`` 特征，MLP 地板在这两套协议下分别是
**0.4842（诊断档）与 0.3858（正式档 scale n；换到文献宽度 scale l 是 0.4401）**。
两套之间差的不仅是 lr/wd/dropout，还有 hidden 128×3 vs 16，所以不能把 0.098 全记在
协议头上 —— 但 0.04–0.10 这个量级已经与要量的增益同阶。所以「GCN 打不打得过一个
加了邻居特征均值的 MLP」这种问题**跨表问就没有答案**，必须把几行放进同一套协议重测。

各行的定义（前两行是「没有可学聚合」的对照组）：

* ``mlp``    —— 只有 $x$，完全不读 ``edge_index``（``cfg/models/mlp.yaml``，2 层 Linear 64）；
* ``floor+`` —— $[\,x \mid A_{\mathrm{norm}}x\,]$ 喂进同一个 MLP，即**不可学的**邻居特征
  均值。聚合式与 §9.12 的 floor+ 完全同源（直接复用 ``acad_probe._spmm``，不重写一遍），
  它是「消息传递白送多少」的替身，也是 route A 真正的对手；
* ``gcn`` / ``gat`` —— 文献基线（``cfg/models/gcn.yaml`` 2 层 GCNConv 64；
  ``gat.yaml`` 2 层 GATConv 64×8 头）；
* ``oca``    —— ``cfg/models/oca.yaml``，留给它和上面几行同协议、同折、同种子对账；
* ``oca_gat`` —— ``cfg/models/oca_gat.yaml``：同骨架但算子退成 GAT（§3 七条件），用来拆
  「算子的贡献」与「多尺度骨架的贡献」。

选点规则与 §7 一致：每个（折, 种子）只在 **val** 上选 epoch，装回最优权重后 test 报一次。
配对单位是（官方折, 种子），$\Delta$ 是配对差 —— 即 §9.5 那个种子漏洞修好之后的口径。

用法::

    python -m dataset.acad_bench --datasets chameleon --models mlp,gcn,gat
    # 正式表（默认已是冻结协议：scale l / 3000 轮 / patience 300）：先按 val 选参再跑。
    # 选参用没报出来的 5/6 折，于是折 0-4 的 test 仍未被看过一眼。
    python -m dataset.acad_bench --tune --tune-folds 5,6 --tune-seeds 0 \
        --datasets chameleon,squirrel,actor,wisconsin,texas,cornell
    python -m dataset.acad_bench --scale n --wd 0.0005      # 回到旧档（§9.13 里的 scale n 表）
    # OCA 行进同一张表（§9.6 的显存墙要先拿 --oca 探）：
    python -m dataset.acad_bench --models mlp,gcn,gat,oca --folds 0 --seeds 0 --oca T=0
    # 只补一行（基准从 --archive 里读，单元不一致会直接拒）：
    python -m dataset.acad_bench --models oca --base gcn --tune --tune-folds 5,6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from config import TrainConfig, load_train_config, parse_overrides
from dataset import load_dataset
from dataset.acad_probe import _spmm
from dataset.base import GraphBundle
from training.trainer import run_once

# 行名 -> (结构表, 是否先把邻居特征均值拼进 x)
MODELS: Dict[str, Tuple[str, bool]] = {
    'mlp': ('cfg/models/mlp.yaml', False),
    'floor+': ('cfg/models/mlp.yaml', True),
    'gcn': ('cfg/models/gcn.yaml', False),
    'gat': ('cfg/models/gat.yaml', False),
    'oca': ('cfg/models/oca.yaml', False),
    # 归因对照：与 `oca` 同一骨架（4 尺度 + PANet），只把算子按 §3 七条件退成 GAT。
    # 于是 `oca` − `oca_gat` = 竞争算子的净贡献，`oca_gat` − `gat` = 骨架的贡献。
    # 少了这一行，「OCA 赢 gat」会被读成「又一个可学的边权」而非竞争项有用。
    'oca_gat': ('cfg/models/oca_gat.yaml', False),
    # 但上面的分解在 chameleon 上不成立：七条件连残差与 LayerNorm 一起关，于是
    # `oca_gat` − `gat` 混进了「4 层无残差的栈自己就训不动」。只撤 λ 的那一行才是
    # 干净对照：`oca` − `oca_nocomp` = 竞争项净贡献，`oca_nocomp` − `gat` = 骨架。
    'oca_nocomp': ('cfg/models/oca_nocomp.yaml', False),
}

# 每个（图, 行）自己的超参网格：**只在 val 上选，全程不看 test**。
#
# 为什么不能一套共享配方到底（实测，scale l）：wd 5e-4 使 squirrel 上的 GAT 塌成
# 平凡解（loss 恒 $\ln 5$、test acc 恒等于某一个类的占比），而同一个 wd 在 wisconsin
# 上却是 MLP 最好的一档（test 0.8013 vs wd=0 的 0.7647）。共享配方必然把某一行做成
# 稻草人；而“基线没调参”是这类表最常被审稿人打的一枪。网格只取两个确实伤人的旋钮。
GRID: List[Tuple[float, float]] = [(lr, wd) for lr in (0.01, 0.005)
                                    for wd in (5e-4, 0.0)]


def _pooled(ds: GraphBundle) -> GraphBundle:
    r"""$[\,x \mid A_{\mathrm{norm}}x\,]$：邻居特征均值（不含自环），按度归一。"""
    n = int(ds.x.size(0))
    deg = _spmm(ds.edge_index, n, torch.ones(n, 1, device=ds.x.device))
    xp = torch.cat([ds.x, _spmm(ds.edge_index, n, ds.x) / deg.clamp(min=1.0)], 1)
    return replace(ds, x=xp, name=f'{ds.name}+pool',
                   meta={**ds.meta, 'pooled': 'nbr-mean x'})


def bench(names: Sequence[str], models: Sequence[str], folds: Sequence[int],
          seeds: Sequence[int], proto: TrainConfig, root: str,
          quiet: bool = False,
          protos: Optional[Dict[Tuple[str, str], TrainConfig]] = None,
          cfgs: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
          base: str = '', archive: str = '',
          ) -> Dict[str, Dict[str, List[float]]]:
    r"""返回 {图: {行名/度量: [逐配对单元的值]}}；单元顺序 = (折, 种子) 双重循环。

    ``protos`` 给定时按（图, 行）取各自调过参的协议（:func:`tune` 的产物），
    否则全表共用 ``proto``。``base`` 是 $\Delta$ 的基准行（默认 = ``models[0]``）；
    它不在本次 ``models`` 里时从 ``archive`` 读（:func:`load_units`）——
    没这个口子，将来加一行（比如 ``oca``）就得把四条基线重跑一遍。

    副作用：每跑完一张图就先把它的表印出去（:func:`report_one`），并按行存盘。
    """
    base = base or models[0]
    dev = torch.device(proto.device) if proto.device != 'auto' else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out: Dict[str, Dict[str, List[float]]] = {}
    for name in names:
        acc: Dict[str, List[float]] = {}
        print('=' * 78)
        print(f'### {name}   配对单位 (fold, seed) = {len(folds) * len(seeds)}   '
              f'device={dev}')

        def put(tag: str, met: str, v: float) -> None:
            acc.setdefault(f'{tag}/{met}', []).append(v)

        for f in folds:
            ds = load_dataset(name, root=root, split_idx=f).to(dev)
            if f == folds[0]:
                s = ds.stats()
                print(f'  N {int(s["num_nodes"]):,}  E {int(s["num_edges"]):,}  '
                      f'F {int(s["num_features"]):,}  C {int(s["num_classes"])}  '
                      f'h_edge {s["edge_homophily"]:.4f}  '
                      f'train/val/test {int(s["train_nodes"]):,}/'
                      f'{int(s["val_nodes"]):,}/{int(s["test_nodes"]):,}  '
                      f'[{ds.meta.get("split_source", "?")}]')
            # 缓存按「要不要拼邻居均值」分流：同一个 pool 标志对应一份 bundle，
            # 省了同一个 fold 里多行的重复预处理
            cache: Dict[bool, GraphBundle] = {}
            for tag in models:
                spec, pool = MODELS[tag]
                if pool not in cache:
                    cache[pool] = _pooled(ds) if pool else ds
                b = cache[pool]
                cfg = (protos or {}).get((name, tag), proto)
                cfg = cfg.patched(**{'model.spec': spec, 'run.device': str(dev)})
                for s_ in seeds:
                    t0 = time.time()
                    r = run_once(cfg.patched(seed=s_), ds=b)
                    put(tag, 'acc', float(r.test['acc']))
                    put(tag, 'f1', float(r.test['f1']))
                    put(tag, 'val', float(r.val['acc']))
                    put(tag, 'ep', float(r.best_epoch))
                    # 停止轮与 best_ep 分开记：前者才能证「没被 epochs 上限截断」，
                    # 后者只能说明「最佳点在哪」。squirrel 的 gcn/gat best_ep 到 1493/1384，
                    # 光看 best_ep 均值分不清「早停于 1800」和「跑满 3000」。
                    put(tag, 'rn', float(r.epochs_ran))
                    put(tag, 'par', float(r.n_params))
                    put(tag, 'sec', time.time() - t0)
                    if not quiet:
                        print(f'  fold{f} seed{s_} {tag:<7} '
                              f'acc {r.test["acc"]:.4f} f1 {r.test["f1"]:.4f} '
                              f'val {r.val["acc"]:.4f} ep{r.best_epoch}/{r.epochs_ran} '
                              f'{time.time() - t0:.1f}s', flush=True)
        out[name] = acc
        rowcfg = {t: (cfgs or {}).get((name, t),
                                     (proto.lr, proto.weight_decay))
                  for t in models}
        extra = ({} if base in models else load_units(archive, name, base, folds, seeds))
        report_one(name, acc, models, folds, seeds, base=base,
                   cfgs=rowcfg, extra=extra)
        if archive:
            dump_units(archive, name, acc, models, folds, seeds, proto, rowcfg)
    return out


def _doc_path(archive: str, name: str) -> str:
    return os.path.join(archive, f'{name}.json')


def dump_units(archive: str, name: str, acc: Dict[str, List[float]],
               models: Sequence[str], folds: Sequence[int],
               seeds: Sequence[int], proto: TrainConfig,
               cfgs: Dict[str, Tuple[float, float]]) -> str:
    r"""把逐配对单元存盘，**每行自带自己的协议指纹**。

    为什么存 units而不是均值：$\Delta$ 必须是配对差（§9.5 那个种子漏洞的教训），
    而均值里配不了对。为什么指纹按行而不按文件：一行一行地补跑时，文件级的
    meta 会被最后一次写入覆掉，于是基线行看起来用了 OCA 行的 epochs/scale。
    """
    p = _doc_path(archive, name)
    os.makedirs(archive, exist_ok=True)
    doc: Dict[str, object] = {'graph': name, 'rows': {}}
    if os.path.exists(p):
        with open(p, encoding='utf-8') as fh:
            doc = json.load(fh)
    rows: Dict[str, object] = doc.setdefault('rows', {})
    for tag in models:
        units = {k.split('/', 1)[1]: v for k, v in acc.items()
                 if k.startswith(f'{tag}/')}
        if 'acc' not in units:
            continue
        lr, wd = cfgs.get(tag, (proto.lr, proto.weight_decay))
        rows[tag] = {'folds': list(folds), 'seeds': list(seeds),
                     'spec': MODELS[tag][0], 'scale': proto.scale,
                     'epochs': proto.epochs, 'patience': proto.patience,
                     'dropout': proto.dropout, 'lr': lr, 'wd': wd,
                     'oca': dict(proto.oca), 'units': units}
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    return p


def load_units(archive: str, name: str, base: str, folds: Sequence[int],
               seeds: Sequence[int]) -> Dict[str, List[float]]:
    """从存档取已跑过的基准行；**单一对不上就拒**，因为配对差断了就没有 $\\Delta$。"""
    p = _doc_path(archive, name) if archive else '(未给 --archive)'
    if not archive or not os.path.exists(p):
        raise SystemExit(f'基准行 {base!r} 不在本次 --models 里，而存档 {p} 读不到：'
                         f'要么把 {base} 一起跑（--models {base},…），'
                         f'要么检查 --archive 路径')
    with open(p, encoding='utf-8') as fh:
        row = json.load(fh).get('rows', {}).get(base)
    if row is None:
        raise SystemExit(f'存档 {p} 里没有 {base} 行（已有：'
                         f'{sorted(json.load(open(p, encoding="utf-8"))["rows"])}）')
    if row['folds'] != list(folds) or row['seeds'] != list(seeds):
        raise SystemExit(f'存档 {p} 里 {base} 的单位是 folds {row["folds"]} × seeds '
                         f'{row["seeds"]}，与本次 {list(folds)} × {list(seeds)} 不一致：'
                         f'配对差断了，不能减')
    print(f'  [i] Δ 基准行 {base} 取自存档 {p}'
          f'（lr {row["lr"]:g} wd {row["wd"]:g} scale {row["scale"]}）', flush=True)
    return {f'{base}/{k}': v for k, v in row['units'].items()}


def _ms(v: Sequence[float]) -> Tuple[float, float]:
    t = torch.tensor(list(v), dtype=torch.float64)
    return float(t.mean()), float(t.std(unbiased=False))


def report_one(name: str, acc: Dict[str, List[float]], models: Sequence[str],
               folds: Sequence[int], seeds: Sequence[int],
               base: str = 'mlp',
               cfgs: Optional[Dict[str, Tuple[float, float]]] = None,
               extra: Optional[Dict[str, List[float]]] = None) -> None:
    """一张图的表。每张图跑完就印（不等全部图完），否则长任务中段看不到读数。

    ``extra`` 是同一张图上**本次没跑**的行（只可能是从存档读回来的基准行），
    它们也会出现在表里，但尾部标 `[存档]` —— 不标就会被读成本次跑的。
    """
    extra = extra or {}
    pool = {**extra, **acc}
    n = len(folds) * len(seeds)
    print('\n' + '=' * 78)
    print(f'### {name}  （{n} 个配对单元；± 是配对间总体 std，不是标准误）')
    shown = list(models)
    if base not in shown and f'{base}/acc' in pool:
        shown.insert(0, base)
    if f'{base}/acc' not in pool:
        print(f'  [!] 基准行 {base} 既不在本次 --models 里也不在存档里，Δ 列无意义')
    for tag in shown:
        src = acc if f'{tag}/acc' in acc else extra
        if f'{tag}/acc' not in src:
            continue
        am, asd = _ms(src[f'{tag}/acc'])
        fm, fsd = _ms(src[f'{tag}/f1'])
        vm, _ = _ms(src[f'{tag}/val'])
        em, _ = _ms(src[f'{tag}/ep'])
        pm, _ = _ms(src[f'{tag}/par'])
        sm, _ = _ms(src[f'{tag}/sec'])
        line = (f'  {tag:<8} acc {am:.4f}±{asd:.4f}  f1 {fm:.4f}±{fsd:.4f}  '
                f'val {vm:.4f}  best_ep {em:5.1f}')
        if f'{tag}/rn' in src:
            line += f'  stop_ep {_ms(src[f"{tag}/rn"])[0]:5.1f}'
        line += (f'  {pm:9,.0f} par  {sm:5.1f}s/run')
        if cfgs and tag in cfgs:
            line += f'  lr {cfgs[tag][0]:g} wd {cfgs[tag][1]:g}'
        if src is not acc:
            line += '  [存档]'
        if f'{base}/acc' in pool and tag != base:
            b = pool[f'{base}/acc']
            if len(b) != len(src[f'{tag}/acc']):
                print(f'{line}   [!] 单元数 {len(src[f"{tag}/acc"])} vs 基准 {len(b)}，Δ 不算')
                continue
            d = [a - c for a, c in zip(src[f'{tag}/acc'], b)]
            dm, dsd = _ms(d)
            line += f'   Δacc(vs {base}) {dm:+.4f}±{dsd:.4f}'
        print(line, flush=True)
    # 逐折拆开：折间差异比配对 std 更能说明「这行数字稳不稳」
    print('  逐折 acc：' + '  '.join(
        f'f{f}[' + '/'.join(f'{_ms(pool[f"{tag}/acc"][f * len(seeds):(f + 1) * len(seeds)])[0]:.4f}'
                             for tag in shown if f'{tag}/acc' in pool) + ']'
        for f in range(len(folds))), flush=True)


def report(res: Dict[str, Dict[str, List[float]]], models: Sequence[str],
           folds: Sequence[int], seeds: Sequence[int], base: str = 'mlp') -> None:
    for name, acc in res.items():
        report_one(name, acc, models, folds, seeds, base)


def tune(names: Sequence[str], models: Sequence[str], folds: Sequence[int],
         seeds: Sequence[int], proto: TrainConfig, root: str,
         grid: Sequence[Tuple[float, float]] = GRID,
         ) -> Tuple[Dict[Tuple[str, str], TrainConfig],
                    Dict[Tuple[str, str], Tuple[float, float]]]:
    r"""按 **val_acc** 给每个（图, 行）在 `grid` 上挑 $(lr, wd)$。

    test 集在这一步一次都不读（:func:`run_once` 的返回值里有 test 指标，这里只取
    ``r.val``）；挑完才交给 :func:`bench` 去跑正式表。选参只用了 `folds` 那几折，
    所以正式表里那些折的 test 数字带一点「选参折偏乐观」，其余折干净 —— 与文献里
    「在验证集上调参」同病，写进表注。
    """
    dev = torch.device(proto.device) if proto.device != 'auto' else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scores: Dict[Tuple[str, str], Dict[Tuple[float, float], List[float]]] = {}
    for name in names:
        for f in folds:
            ds = load_dataset(name, root=root, split_idx=f).to(dev)
            cache: Dict[bool, GraphBundle] = {}
            for tag in models:
                spec, pool = MODELS[tag]
                if pool not in cache:
                    cache[pool] = _pooled(ds) if pool else ds
                b = cache[pool]
                for lr, wd in grid:
                    cfg = proto.patched(**{'model.spec': spec,
                                           'run.device': str(dev),
                                           'optimization.lr': lr,
                                           'optimization.weight_decay': wd})
                    vs = [float(run_once(cfg.patched(seed=s_), ds=b).val['acc'])
                          for s_ in seeds]
                    scores.setdefault((name, tag), {}).setdefault((lr, wd), []).extend(vs)
                    print(f'  [{name} {tag}] lr {lr:g} wd {wd:g}  '
                          f'val ' + ' '.join(f'{v:.4f}' for v in vs), flush=True)
    protos: Dict[Tuple[str, str], TrainConfig] = {}
    cfgs: Dict[Tuple[str, str], Tuple[float, float]] = {}
    print('\n### 选参结果（按 val 降序，括号里是各档的 val 均值）')
    for (name, tag), per in scores.items():
        ranked = sorted(per.items(), key=lambda kv: -_ms(kv[1])[0])
        (lr, wd), vs = ranked[0]
        protos[(name, tag)] = proto.patched(**{
            'model.spec': MODELS[tag][0], 'run.device': str(dev),
            'optimization.lr': lr, 'optimization.weight_decay': wd})
        cfgs[(name, tag)] = (lr, wd)
        print(f'  {name:<10} {tag:<7} -> lr {lr:g} wd {wd:g}（val {_ms(vs)[0]:.4f}）   '
              + '  '.join(f'[{k[0]:g}/{k[1]:g}] {_ms(v)[0]:.4f}' for k, v in ranked),
              flush=True)
    return protos, cfgs


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description='学术图外部基线表（同一协议、配对折×种子）')
    p.add_argument('--datasets', default='chameleon,squirrel,actor')
    p.add_argument('--models', default='mlp,floor+,gcn,gat',
                   help=f'逗号分隔，可选 {sorted(MODELS)}；第一行是 Δ 的基准')
    p.add_argument('--folds', default='0,1,2,3,4')
    p.add_argument('--seeds', default='0,1,2')
    # 默认值 = 冻结的基线协议。轮数不是随手定的：400/1500 轮时 gcn、floor+ 的
    # best_ep 顶在上限（欠训练的稻草人），3000/patience 300 下四行都不顶。
    p.add_argument('--epochs', type=int, default=3000)
    p.add_argument('--patience', type=int, default=300)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--dropout', type=float, default=0.5)
    # scale l 才是文献宽度（hidden 64）：scale n 把 width_multiple 压到 0.25，
    # 基线的 hidden 只剩 16，与 OCA 同 scale 下 169k 参数根本不同量级。
    p.add_argument('--scale', default='l')
    # `oca` 行专用的算子级旋钮（只对 ``cfg/models/*.yaml`` 里的 OCAConv 生效，
    # 其余行会静默忽略）。为什么入口里必须有它：§9.6 补测实测六张图里只有
    # squirrel 装不下 —— scale $l$、$T{=}2$、4 层 backbone 峰值 +8,983 MiB > 卡的
    # 8,151 MiB（溢到共享内存，7.73 s/step），而 `detach_iterations=True` 把它压到
    # +7,084 MiB / 1.20 s/step。其余五张图同一档 $\le$ 1.6 GiB，无需折衷。
    p.add_argument('--oca', nargs='*', default=[], metavar='K=V',
                   help='覆盖 OCAConfig，如 --oca T=0 heads=8（仅对 oca 行有效）')
    p.add_argument('--tune', action='store_true',
                   help=f'先按 val 在网格 {GRID} 上给每行挑 (lr, wd)，再跑正式表')
    p.add_argument('--tune-folds', default='0')
    p.add_argument('--tune-seeds', default='0,1,2')
    p.add_argument('--device', default='auto')
    p.add_argument('--root', default='data')
    p.add_argument('--base', default='',
                   help=f'Δ 的基准行，默认 = --models 的第一行；不在本次 --models 里时从 --archive 读')
    p.add_argument('--archive', default='runs/acad_bench',
                   help='逐配对单元存盘目录（每图一个 json，按行带协议指纹）；空串 = 不存')
    p.add_argument('--quiet', action='store_true', help='不打逐 run 的行')
    a = p.parse_args(argv)

    models = [m.strip() for m in a.models.split(',') if m.strip()]
    bad = [m for m in models if m not in MODELS]
    if bad:
        raise SystemExit(f'未知行 {bad}，可选：{sorted(MODELS)}')
    folds = [int(v) for v in a.folds.split(',') if v.strip()]
    seeds = [int(v) for v in a.seeds.split(',') if v.strip()]
    names = [s.strip() for s in a.datasets.split(',') if s.strip()]

    proto = load_train_config(
        **{'run.device': a.device, 'run.save': False, 'run.dump_gates': False,
           'dataset.name': names[0], 'dataset.root': a.root,
           'model.scale': a.scale, 'model.dropout': a.dropout,
           'optimization.epochs': a.epochs, 'optimization.patience': a.patience,
           'optimization.lr': a.lr, 'optimization.weight_decay': a.wd})
    assert isinstance(proto, TrainConfig)
    if a.oca:
        if 'oca' not in models:
            raise SystemExit(f'--oca 只在 `oca` 行上有意义，但 --models 是 {models}')
        proto = proto.with_oca(**parse_overrides(a.oca))
    print(f'协议：lr {a.lr}  wd {a.wd}  dropout {a.dropout}  epochs {a.epochs}'
          f'（patience {a.patience}，val_acc 选点）  scale {a.scale}  '
          f'特征 row_normalize（不做 §9.12 的公平档 z-score）'
          + (f'  oca {dict(proto.oca)}' if proto.oca else ''))
    protos = cfgs = None
    if a.tune:
        tf = [int(v) for v in a.tune_folds.split(',') if v.strip()]
        ts = [int(v) for v in a.tune_seeds.split(',') if v.strip()]
        print(f'选参：折 {tf} × 种子 {ts} × 网格 {GRID}（只看 val）')
        protos, cfgs = tune(names, models, tf, ts, proto, a.root)
    # 表在 bench 里逐图印（report_one）；这里不再汇总一次，否则同一张表会写进日志两遍。
    bench(names, models, folds, seeds, proto, a.root, a.quiet,
          protos=protos, cfgs=cfgs, base=a.base, archive=a.archive)
    return 0


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
