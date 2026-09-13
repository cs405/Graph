r"""把 §9.11 的「裸分 vs 地板」判据拿到标准异配基准上重测（诊断脚本，不参与训练）。

§9.11 归纳出一条能提前否决的判据：**一列组级统计量能不能涨，先看它的裸分越不越过
地板**。但那条判据是从 SOY/CORN 一张图上归纳出来的，而那张图有两个巧合同时成立：
地板异常高（属性 one-hot 单用就有 0.805 AUC，§9.7），图结构异常弱（邻居池化增益
$\le$ 0.001）。在这两个条件下「裸分越不过地板」可能是规律，也可能只是环境。

本文件把它放回反面条件下重测：地板弱（MLP 在词袋特征上只有五六成）、图结构强（$h_{edge}
\approx 0.22$ 的三类标准异配图，邻居标签是这些图的主要信号源）。同一个统计量、同一条
判据、同一套拟合配方。结论的读法只有一句：

* 若这里裸分**越过**了地板 $\Rightarrow$ 判据是数据形态的函数，不是规律，route A 的先验
  否决要撤回，改为「先量裸分」这个流程性约束；
* 若裸分**仍越不过**地板 $\Rightarrow$ 判据扛住了一次真正的压力测试，它可以用来省实验。

用法::

    python -m dataset.acad_probe                                  # 五张图全跑
    python -m dataset.acad_probe --datasets chameleon --folds 0,1,2 --seeds 0,1,2
    python -m dataset.acad_probe --datasets 'synth:h=0.85:sep=1.5' --folds 0,1,2

同质的那一头（Cora / CiteSeer / PubMed）在本地拿不到：Planetoid 的源站在
``github.com``，而那台机器上 ``github.com:443`` 不通（``raw.githubusercontent.com``
通，所以 geom-gcn 那批能下）。因此同质对照退化成上面的 ``synth:h=...`` ——
它按类中心 + 高斯噪声造特征，**准确率不能当论文卖点**，只能用来填判据曲线上
「$h$ 高」那一侧的形状。

除 ``synth`` 之外一律用数据集自带的官方 split（geom-gcn 10 折），折号在 ``meta`` 里。
配对单位是 (折, 种子)：地板与每一行候选在同一个单元上重建网络、同一份 init，所以
$\Delta$ 是配对差而不是两个均值的差 —— 这是 §9.5 那个种子漏洞修好之后的口径。
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from dataset.hetero_probe import _dsp, _fit_eval, _mlp, _ms, _pm, macro_f1
from metrics.homophily import edge_homophily

ACAD = ['chameleon', 'squirrel', 'actor', 'texas', 'cornell', 'wisconsin']
TAUS = (0., 1., 5., 20.)


def _z(m: torch.Tensor) -> torch.Tensor:
    """逐列零均值单位方差（用全部行：特征与邻居标签率都是直推式可见的）。"""
    return (m - m.mean(0)) / m.std(0).clamp(min=1e-9)


# --------------------------------------------------------------- features ----

def _spmm(ei: torch.Tensor, n: int, m: torch.Tensor) -> torch.Tensor:
    """邻域求和 $A m$。必须走稀疏：squirrel 的 $|E|=3.97\\times10^5$、$F=2089$，
    先 gather 成 $3.97\\times10^5\\times2089$ 的稠密块是 3.3 GB。"""
    ones = torch.ones(ei.size(1), dtype=m.dtype, device=m.device)
    A = torch.sparse_coo_tensor(ei, ones, (n, n)).coalesce()
    return torch.sparse.mm(A, m)


def _rate(ei, n, y, C, tr, tau) -> Tuple[torch.Tensor, torch.Tensor]:
    """邻居里的标签率（收缩版），只看 train 邻居 —— 与 §9.11 的 $r=(a+\\tau p)/(n+\\tau)$
    同式，只是把「属性格元」换成「图上的 1 跳邻域」。

    返回 $(r, n_{nbr})$：$r$ 是 $[N,C]$ 且每行和为 1，$n_{nbr}$ 是 $[N,1]$ 的 train 邻居数。
    """
    onehot = torch.zeros(n, C, dtype=torch.float32, device=y.device)
    onehot[tr] = nn.functional.one_hot(y[tr], C).float()
    p = onehot[tr].mean(0)
    num, cnt = _spmm(ei, n, onehot), _spmm(ei, n, tr.float().unsqueeze(1))
    den = (cnt + tau).clamp(min=1e-9)
    return torch.where(cnt + tau > 0, (num + tau * p) / den, p.expand_as(num)), cnt


# ------------------------------------------------------------------ rows -----

def _acc_f1(p: torch.Tensor, y: torch.Tensor, C: int) -> Tuple[float, float]:
    yh = p.argmax(-1).cpu()
    yc = y.cpu()
    return (float((yh == yc).float().mean()), macro_f1(yc, yh, C))


def _fit(x, y, masks, C, dev, epochs, seed, din=None) -> Dict[str, float]:
    """一次拟合。`masks` = (tr, va, te)，val 早停，绝不用 test 选点。"""
    tr, va, te = masks
    d = din if din is not None else x.size(1)
    r = _fit_eval(x[tr], y[tr], x[va], y[va], x[te], y[te],
                  lambda d=d: _mlp(d, 128, 3, C), dev, epochs, seed=seed)
    return {k: v for k, v in r.items() if k in ('acc', 'f1')}


def probe(name: str, folds: Sequence[int], seeds: Sequence[int], epochs: int,
          root: str, homophily: float, sep: float, scale: bool = True,
          feat_z: bool = False) -> Dict[str, Dict[str, List[float]]]:
    """跑一张图，返回 {行名: {metric: [逐配对单元的值]}}（行名含 'floor' 的那行是基准）。"""
    from dataset import load_dataset
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    synth = name.startswith('synth')
    acc: Dict[str, Dict[str, List[float]]] = {}

    def put(tag: str, metric: str, v: float) -> None:
        acc.setdefault(tag, {}).setdefault(metric, []).append(v)

    print('=' * 78)
    print(f'### {name}' + (f'  (synthetic: h 目标 {homophily}, sep {sep})' if synth
                           else '') + f'   配对单位 (fold, seed) = {len(folds) * len(seeds)}')
    for f in folds:
        kw: Dict[str, object] = {'root': root}
        if synth:                                     # 合成图没有官方折：换折 = 重抽一张
            b = load_dataset('synth', seed=f, homophily=homophily, **kw)
        else:
            b = load_dataset(name, split_idx=f, **kw)
        x = b.x.to(dev)
        if feat_z:              # 公平档：x 也逐列单位方差，两个块同尺度（否则 rate 列要么
            x = _z(x)           # 太响要么太哑，而“响/哑”本身可以决定符号 —— 见 §9.12 读数 4）
        y = b.y.long().to(dev)
        C = b.num_classes
        ei = b.edge_index.to(dev)
        n = int(x.size(0))
        tr, va, te = (b.train_mask.bool().to(dev), b.val_mask.bool().to(dev),
                      b.test_mask.bool().to(dev))
        if f == folds[0]:
            deg = _spmm(ei, n, torch.ones(n, 1, device=dev))
            print(f'  N {n:,}  E {int(ei.size(1)):,}  F {int(x.size(1)):,}  C {C}  '
                  f'h_edge {edge_homophily(ei, y):.4f}  '
                  f'deg mean {float(deg.mean()):.1f} max {int(deg.max())}  '
                  f'零邻居行 {int((deg.view(-1) == 0).sum()):,}  '
                  f'train/val/test {int(tr.sum()):,}/{int(va.sum()):,}/{int(te.sum()):,}'
                  f'  [{b.meta.get("split_source", "random")}]')
        degv = _spmm(ei, n, torch.ones(n, 1, device=dev))

        # ---- 地板：只有 x ----
        for s in seeds:
            r = _fit(x, y, (tr, va, te), C, dev, epochs, s)
            put('floor: x', 'acc', r['acc'])
            put('floor: x', 'f1', r['f1'])
        # ---- 地板+：x 加邻居特征均值（消息传递不可学的替身，§9.7 那一行）----
        xpool = torch.cat([x, _spmm(ei, n, x) / degv.clamp(min=1.)], 1)
        for s in seeds:
            r = _fit(xpool, y, (tr, va, te), C, dev, epochs, s)
            put('floor+: x + nbr-mean x', 'acc', r['acc'])
            put('floor+: x + nbr-mean x', 'f1', r['f1'])

        # ---- 裸分：邻居标签率当预测，不进任何模型（与种子无关，逐折一份）----
        if f == folds[0]:
            _, c0 = _rate(ei, n, y, C, tr, 0.)
            print(f'  test 行的 train 邻居数：mean {float(c0[te].mean()):.1f}  '
                  f'中位 {float(c0[te].median()):.0f}  为 0 的行 '
                  f'{int((c0[te] == 0).sum())}   ← 裸分有多噪，看这一行')
        for tau in TAUS:
            r, cnt = _rate(ei, n, y, C, tr, tau)
            a, f1 = _acc_f1(r[te], y[te], C)
            put(f'raw: rate tau={tau:g} argmax', 'acc', a)
            put(f'raw: rate tau={tau:g} argmax', 'f1', f1)
            a, f1 = _acc_f1(-r[te], y[te], C)          # 反向：异配图上「最不可能的类」
            put(f'raw: rate tau={tau:g} argmin', 'acc', a)
            put(f'raw: rate tau={tau:g} argmin', 'f1', f1)
        # ---- 泄漏上界：邻居标签不看 train mask（忘了屏蔽会虚高多少）----
        rall, _ = _rate(ei, n, y, C, torch.ones_like(tr), 0.)
        a, f1 = _acc_f1(rall[te], y[te], C)
        put('raw: rate tau=0 用全集标签（泄漏）', 'acc', a)
        put('raw: rate tau=0 用全集标签（泄漏）', 'f1', f1)

        # ---- 喂进模型：与地板同宽度差，逐种子配对 ----
        # 尺度必须先对齐：x 过了 row_normalize，一列的典型 std 只有 xsd 这么小，而 rate
        # 列在 [0,1] 上、std 比它大一个量级 —— 不对齐的话，加列同时改了输入分布，涨的
        # 归因就分不清是信息还是尺度。xsd 用全体行算（与模型无关，对所有行一致）。
        g = torch.Generator(device=dev).manual_seed(
            100003 * (f + 1) + sum(ord(c) for c in name))
        xsd = float(x.std(0).mean()) if scale else 1.0
        for tau in TAUS:
            r, _ = _rate(ei, n, y, C, tr, tau)
            xfed = torch.cat([x, _z(r) * xsd], 1)
            for s in seeds:
                res = _fit(xfed, y, (tr, va, te), C, dev, epochs, s, din=x.size(1) + C)
                put(f'fed: x + rate(scaled) tau={tau:g}', 'acc', res['acc'])
                put(f'fed: x + rate(scaled) tau={tau:g}', 'f1', res['f1'])
        # ---- 三条无信息带（都是 C 列、都对齐尺度）----
        # 1) 行置换：保边缘分布，破掉 rate 与 y 的对应；
        # 2) iid 噪声：只保列尺度；
        # 3) **x 的固定随机投影**：同宽度、信息量是 x 的子集（地板本来就有）——
        #    这一条专门量「加宽本身」值多少，1) 2) 量不到（它们至少引入了新维度）。
        r5, _ = _rate(ei, n, y, C, tr, 5.)
        perm = torch.randperm(n, generator=g, device=dev)
        P = torch.randn(int(x.size(1)), C, generator=g, device=dev) / math.sqrt(x.size(1))
        ctrls = [('ctl: x + rate(scaled) 行置换', torch.cat([x, _z(r5[perm]) * xsd], 1)),
                 ('ctl: x + iid 噪声(scaled)',
                  torch.cat([x, torch.randn(n, C, generator=g, device=dev) * xsd], 1)),
                 ('ctl: x + 随机投影(x)', torch.cat([x, _z(x @ P) * xsd], 1))]
        for tag, xx in ctrls:
            for s in seeds:
                res = _fit(xx, y, (tr, va, te), C, dev, epochs, s, din=x.size(1) + C)
                put(tag, 'acc', res['acc'])
                put(tag, 'f1', res['f1'])
    return acc


def report(acc: Dict[str, Dict[str, List[float]]], folds: Sequence[int]) -> None:
    """打印：每行 mean±std，候选行另报与地板的配对差。"""
    order = ['floor: x', 'floor+: x + nbr-mean x'] + \
            [f'raw: rate tau={t:g} {d}' for t in TAUS for d in ('argmax', 'argmin')] + \
            ['raw: rate tau=0 用全集标签（泄漏）'] + \
            [f'fed: x + rate(scaled) tau={t:g}' for t in TAUS] + \
            ['ctl: x + rate(scaled) 行置换', 'ctl: x + iid 噪声(scaled)',
             'ctl: x + 随机投影(x)']
    base = acc['floor: x']
    for tag in order:
        if tag not in acc:
            continue
        line = f'  {tag:<34}'
        model = not tag.startswith('raw')
        for mt in ('acc', 'f1'):
            v = acc[tag][mt]
            line += f'  {mt} {_pm(v)}'
            if model and tag != 'floor: x':
                line += f'  Δ {_dsp([a - b for a, b in zip(v, base[mt])])}'
        print(line + ('' if model else '   （无模型，逐折确定值）'))

    # 判据读数：最好的裸分 vs 地板
    raws = [(max(acc[t]['acc']), max(acc[t]['f1']), t) for t in acc
            if t.startswith('raw') and '泄漏' not in t]
    ra, rf = max(r[0] for r in raws), max(r[1] for r in raws)
    top = max(raws, key=lambda r: r[0])[2]
    ba, bf = _ms(base['acc'])[0], _ms(base['f1'])[0]
    print(f'\n  判据：裸分峰值 acc {ra:.4f} / macro-F1 {rf:.4f}（来自 {top}）'
          f'   地板 acc {ba:.4f} / F1 {bf:.4f}')
    print(f'        => acc {"越过" if ra > ba else "没越过"}地板（{ra - ba:+.4f}），'
          f'macro-F1 {"越过" if rf > bf else "没越过"}地板（{rf - bf:+.4f}）；'
          f'地板自身的配对间 std acc {_ms(base["acc"])[1]:.4f} / F1 {_ms(base["f1"])[1]:.4f}，'
          f'这就是这个比较的分辨率（折数 {len(folds)}）')


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='学术图上的裸分 vs 地板判据检验')
    p.add_argument('--datasets', default='chameleon,squirrel,actor,wisconsin',
                   help=f'逗号分隔，可选 {ACAD} 或 synth:h=0.85:sep=1.5')
    p.add_argument('--folds', default='0,1,2', help='geom-gcn 官方折号，逗号分隔')
    p.add_argument('--seeds', default='0,1', help='拟合种子，逗号分隔')
    p.add_argument('--epochs', type=int, default=300,
                   help='只影响模型行；_fit_eval 按 val 选点，加大不会过拟合到 test。'
                        '实测 15 轮时地板退化成多数类预测（wisconsin 上所有行 Δ 全为 0）')
    p.add_argument('--root', default='data')
    p.add_argument('--no-scale', dest='scale', action='store_false', default=True,
                   help='rate/噪声列不再乘 x 的列 std，而是保持单位 std（“太响”那一档）')
    p.add_argument('--feat-z', action='store_true',
                   help='把 x 也逐列单位方差 —— 与 --no-scale 合用就是两个块同尺度的公平档，'
                        '单独用等价于公平档（此时 x 的列 std 已是 1）')
    a = p.parse_args(argv)
    folds = [int(v) for v in a.folds.split(',') if v.strip()]
    seeds = [int(v) for v in a.seeds.split(',') if v.strip()]
    for spec in [s.strip() for s in a.datasets.split(',') if s.strip()]:
        h, sep = 0.3, 1.5
        name = spec
        if spec.startswith('synth'):
            parts = spec.split(':')
            name = 'synth'
            for kv in parts[1:]:
                k, _, v = kv.partition('=')
                if k == 'h':
                    h = float(v)
                elif k == 'sep':
                    sep = float(v)
        try:
            acc = probe(name, folds, seeds, a.epochs, a.root, h, sep, a.scale,
                        a.feat_z)
        except Exception as e:
            print(f'  !! {spec} 跑不起来：{type(e).__name__}: {str(e)[:180]}')
            continue
        report(acc, folds)
        print()
    return 0


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')          # 压掉 sparse 不变量检查与 jit 的噪声
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
