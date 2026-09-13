"""SOY / CORN 保险异构图的数据体检（诊断脚本，不参与训练）。

`docs/OCA_algorithm.md` §9.5–§9.7 的每个数字都由本文件产出，用途是**在写 hetero
模型之前**先回答三个能左右研究方向的问题：

1. ``structure`` —— 图的真实形状：谁是 hub、record 的出度、竞争场规模、标签分布、
   按年划分留下什么、两个标签之间是不是嵌套关系；
2. ``cost`` —— 全图全批在本地这张卡上到底装不装得下（单层、逐 $T$ 实测时间与峰值显存）；
3. ``hub`` —— 算法侧的裁决依据（§10 第 2 行）：拿**真实度数分布**去量温度统计量
   分母里那个 $\\sqrt{\\deg}$ 到底在多大程度上把 hub 的竞争场糊成均值；
4. ``floors`` —— 地板值：把「属性当 one-hot 特征」和「邻居信息池化」（= 消息传递在做的
   事，只是不可学）喂给一个 MLP，能拿到多少 AUC / acc。若地板已等于生产方报告的 GNN
   基线，则「图结构有用」这条主张在本数据形态下不成立。
5. ``edges`` —— route B 的 go/no-go：候选造边（特征空间 kNN / 属性交叉格元 / 玉米与
   大豆的同格元耦合 / 月份时域链）到底带不带 12 列气象之外的标签信息，以及能不能推动
   §9.7 的地板。判定必须带 `--seeds`（多于一元时逐行报与地板按种子配对的差与 std）。
6. ``drift`` —— route B 判 no-go 之后新主线（②）的前置量：组级标签先验（查表）随年份
   漂多少。它是 §9.9 那条「目标编码反而最差」的解释，本项把它变成可报的数字。
7. ``te`` —— 拟合通路的最小可证伪对照：§9.9 剩下的唯一硬事实是一张单练值 0.7991 的
   格元标签率表喂进 MLP 只换来零边际贡献，本项只改「怎么喂」（按计数收缩 / 单给一列
   计数），不改模型，检验这句归因成不成立。

用法::

    python -m dataset.hetero_probe --dataset corn --what structure,floors
    python -m dataset.hetero_probe --dataset both --what cost
    python -m dataset.hetero_probe --dataset corn --what hub --steps 200
    python -m dataset.hetero_probe --dataset corn --what edges --epochs 80 --seeds 0,1,2
    python -m dataset.hetero_probe --dataset both --what drift
    python -m dataset.hetero_probe --dataset both --what te --seeds 0,1,2

协议固定为 train = 2008–2018、**val = 2019（从 train 里显式抠出，因为数据没给
``val_mask``）**、test = 2020–2022；模型只选 val 不选 test（与设计稿 §七 一致）。

注意：AUC / PR-AUC / macro-F1 在这里是**本地副本**。它们的正式归属是 ``metrics/``
（算法文档 §9.3 缺口 3），等真正开始训练时再挪过去并补用例，避免诊断脚本和训练管线
长期各持一份实现。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

ATTRS: List[Tuple[str, int]] = [('state', 7), ('year', 15), ('stage', 7),
                                ('month', 12), ('plan', 7), ('coverage', 2)]
VAL_YEAR_ID = 11          # graph_meta.node_index_maps.year["2019"]
FEAT_NAMES = ('prcp_mean prcp_max prcp_cv tmax_mean dtr tmin_min tmax_max '
              'srad_mean gdd_sum heat_days frost_days quantity').split()


# ---------------------------------------------------------------- metrics ----

def _avg_ranks(p: torch.Tensor) -> torch.Tensor:
    n = p.numel()
    srt = torch.argsort(p)
    _, counts = torch.unique_consecutive(p[srt], return_counts=True)
    ends = counts.cumsum(0)
    starts = ends - counts
    mean = (starts.double() + ends.double() - 1) / 2
    r = torch.empty(n, dtype=torch.float64, device=p.device)
    r[srt] = mean.repeat_interleave(counts)          # 并列取平均秩
    return r


def roc_auc(y: torch.Tensor, p: torch.Tensor) -> float:
    y = y.long()
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float('nan')
    r = _avg_ranks(p)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def pr_auc(y: torch.Tensor, p: torch.Tensor) -> float:
    y = y.double()
    ys = y[torch.argsort(p, descending=True)]
    tp = ys.cumsum(0)
    k = torch.arange(1, ys.numel() + 1, dtype=torch.float64, device=ys.device)
    prec, rec = tp / k, tp / ys.sum()
    return float(((rec - torch.cat([rec.new_zeros(1), rec[:-1]])) * prec).sum())


def macro_f1(y: torch.Tensor, yh: torch.Tensor, nc: int) -> float:
    out = []
    for c in range(nc):
        tp = int(((y == c) & (yh == c)).sum())
        fp = int(((y != c) & (yh == c)).sum())
        fn = int(((y == c) & (yh != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(out) / nc


# ------------------------------------------------------------------ data -----

def load(name: str):
    from torch_geometric.data import HeteroData
    path = {'corn': 'data/CORN/corn_hetero.pt',
            'soy': 'data/SOY/soy_hetero.pt'}[name]
    # 生产方的 .pt 含自定义类，weights_only=False 是硬要求（PyTorch 2.6+ 默认会拒）
    d = torch.load(path, map_location='cpu', weights_only=False)
    assert isinstance(d, HeteroData), f'{path} 不是 HeteroData：{type(d).__name__}'
    return d, path


def attr_group_ids(d) -> Dict[str, torch.Tensor]:
    """每类属性：record -> 该属性的节点编号（record 在每条关系上出度恒为 1）。"""
    out = {}
    for a, _ in ATTRS:
        et = [e for e in d.edge_types if e[0] == 'record' and e[2] == a]
        assert len(et) == 1, f'{a}: 找到 {len(et)} 条 record->{a} 关系'
        ei = d[et[0]].edge_index
        assert int(torch.bincount(ei[0], minlength=d['record'].num_nodes).max()) == 1, \
            f'{a}: record 出度不再恒为 1，体检逻辑要重写'
        out[a] = ei[1]
    return out


# ------------------------------------------------------------ 1. structure ----

def probe_structure(name: str) -> None:
    d, path = load(name)
    print('=' * 72)
    print(f'### {name}  ({path})')
    print(f'node_types: {d.node_types}')
    print(f'edge_types: {len(d.edge_types)} -> '
          f'{[f"{a}|{r}|{b}" for a, r, b in d.edge_types]}')

    total = 0
    print('\n-- 边与竞争场（attr 侧中心的字段规模 = 它挂了多少 record）--')
    for a, r, b in d.edge_types:
        ei = d[a, r, b].edge_index
        total += ei.size(1)
        rev = torch.bincount(ei[1], minlength=d[b].num_nodes)
        fwd = torch.bincount(ei[0], minlength=d[a].num_nodes)
        print(f'  {a}|{r}|{b:<9} E={ei.size(1):<7} outdeg[{a}] max={int(fwd.max()):<6}'
              f' field[{b}] mean={rev.float().mean():9.1f} max={int(rev.max()):<7}')
    print(f'  SUM |E_r| = {total:,}')

    rec, x = d['record'], d['record'].x.float()
    print('\n-- record 特征（NaN/Inf 必须为 0；|z|max 揭示重尾列）--')
    print(f'  x {tuple(x.shape)}  nan={int(torch.isnan(x).sum())}  inf={int(torch.isinf(x).sum())}')
    mu, sd = x.mean(0), x.std(0)
    zmax = ((x - mu) / sd.clamp(min=1e-9)).abs().max(0).values
    for j, nm in enumerate(FEAT_NAMES):
        print(f'  {nm:<11} min {float(x[:, j].min()):8.3f} mean {float(mu[j]):7.3f} '
              f'std {float(sd[j]):6.3f} max {float(x[:, j].max()):8.3f} '
              f'|z|max {float(zmax[j]):6.2f}')
    if float(zmax.max()) > 10:
        print(f'  !! {FEAT_NAMES[int(zmax.argmax())]} 的极值到 {float(zmax.max()):.1f} sigma'
              '：门控吃原始 h_i（算法文档 §2.6），这一列会主导 lambda/alpha/tau')

    print('\n-- split（val_mask 缺失，按年显式抠 2019）--')
    gid = attr_group_ids(d)
    tr, te = rec.train_mask.bool(), rec.test_mask.bool()
    va = gid['year'] == VAL_YEAR_ID
    va_mask = getattr(rec, 'val_mask', None)      # 本数据集里这个属性压根不存在
    print(f'  train_mask {int(tr.sum()):,}  val_mask '
          f'{"ABSENT" if va_mask is None else int(va_mask.sum())}  '
          f'test_mask {int(te.sum()):,}')
    print(f'  按年: train 覆盖 year id {sorted(set(gid["year"][tr].tolist()))}')
    print(f'        test 覆盖 year id {sorted(set(gid["year"][te].tolist()))}')
    print(f'  2019 共 {int(va.sum()):,} 条，其中 {int((va & tr).sum()):,} 条本在 train 里'
          f'（=> 抠出后 train {int((tr & ~va).sum()):,}）')

    print('\n-- 标签分布与嵌套关系 --')
    y_cls, lev, y_reg = rec.y_cls.long(), rec.loss_level.long(), rec.y_reg.double()
    print(f'  y_cls   {torch.bincount(y_cls).tolist()}  正类率 {float(y_cls.float().mean()):.4f}')
    print(f'  loss_lvl{torch.bincount(lev).tolist()}  （三分类，完全均衡 => 无需重加权）')
    both = int(((y_cls == 1) & (lev != 2)).sum())
    print(f'  y_cls=1 但 loss_level!=2 的条数 = {both}'
          + ('  => y_cls 是 loss_level==2 的子集，T1/T2 是嵌套任务' if both == 0 else ''))
    print(f'  y_reg min/mean/max = {float(y_reg.min()):.4f}/{float(y_reg.mean()):.4f}/'
          f'{float(y_reg.max()):.4f}，corr(y_reg, loss_level)='
          f'{float(torch.corrcoef(torch.stack([y_reg, lev.double()]))[0, 1]):.4f}')

    print('\n-- 同组内标签一致率（2-hop；贴随机 => 结构不带标签信息）--')
    p_pos = float(y_cls.float().mean())
    chance_c = p_pos ** 2 + (1 - p_pos) ** 2
    for a, ng in ATTRS:
        ag_c, ag_l, npair = _group_agreement(gid[a], y_cls, lev, ng)
        print(f'  {a:<9} y_cls {ag_c:.4f} (chance {chance_c:.4f}, excess {ag_c - chance_c:+.4f})'
              f'   loss_level {ag_l:.4f} (chance 0.3333, excess {ag_l - 1 / 3:+.4f})'
              f'   pairs {npair:,}')

    print('\n-- 各属性组间 high_loss 率的极差（谁在单独决定标签）--')
    tr2 = tr & ~va
    for a, ng in ATTRS:
        rates = []
        for g in range(ng):
            m = tr2 & (gid[a] == g)
            if int(m.sum()) >= 200:
                rates.append((float(y_cls[m].float().mean()) * 100, int(m.sum()), g))
        lo, hi = min(rates), max(rates)
        print(f'  {a:<9} spread {hi[0] - lo[0]:5.1f} pp  '
              f'low: id{lo[2]}={lo[0]:.1f}%(n={lo[1]:,})  high: id{hi[2]}={hi[0]:.1f}%(n={hi[1]:,})')
        small = [g for g in range(ng) if 0 < int((tr2 & (gid[a] == g)).sum()) < 500]
        if small:
            print(f'  {"":<9} 小样本组（n<500，百分比无意义）: '
                  + ', '.join(f'id{g}=n{int((tr2 & (gid[a] == g)).sum())}' for g in small))


def _group_agreement(g, y_cls, lev, ng):
    agree_c = tot_c = agree_l = tot_l = 0.0
    for k in range(ng):
        m = g == k
        n = int(m.sum())
        if n < 2:
            continue
        cc = torch.bincount(y_cls[m]).double()
        agree_c += float((cc ** 2).sum() - n)
        cl = torch.bincount(lev[m]).double()
        agree_l += float((cl ** 2).sum() - n)
        tot_c += n * n - n
        tot_l += n * n - n
    return (agree_c / tot_c, agree_l / tot_l, int(tot_c / 2))


# ------------------------------------------------------------------ 2. cost ----

def flatten_graph(d, ets):
    """把若干关系拼成一张 flatten 的同构图（record 占 0..n_rec-1）。

    节点类型顺序固定为 ``record`` 先、其余按 ``d.node_types``，所以多个调用者
    拿到的下标可互相比对。返回 $(N,\\text{{offset}},\\hat{\\mathcal E})$，后者已对称化 +
    去重 + 补自环（与 ``OCALayer`` 自己做的增广一致，方便直接报 $\\hat E$）。
    """
    from torch_geometric.utils import add_self_loops, coalesce, to_undirected
    n_rec = d['record'].num_nodes
    offset, run = {'record': 0}, n_rec
    for t in d.node_types:
        if t != 'record':
            offset[t], run = run, run + d[t].num_nodes
    eis = []
    for a, r, b in ets:
        ei = d[a, r, b].edge_index.clone()
        if b != 'record':
            ei[1] += offset[b]
        if a != 'record':
            ei[0] += offset[a]
        eis.append(ei)
    ei = coalesce(to_undirected(torch.cat(eis, 1), num_nodes=run), num_nodes=run)
    return run, offset, add_self_loops(ei, num_nodes=run)[0]


def probe_cost(name: str) -> None:
    print('=' * 72)
    print(f'### cost @ {name}')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch {torch.__version__}  device {dev}', end='')
    if dev.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        print(f'（{p.name}，{p.total_memory / 2 ** 30:.1f} GiB）')
    else:
        print()

    d, _ = load(name)
    n_rec = d['record'].num_nodes
    N = flatten_graph(d, [e for e in d.edge_types if e[2] != 'record'])[0]
    x = torch.zeros(N, 12)
    x[:n_rec] = d['record'].x

    def flatten(ets):
        return flatten_graph(d, ets)[2]        # 已 coalesce + 对称化 + 补自环

    for tag, ets in [('single relation record|in_year|year',
                      [e for e in d.edge_types if e[2] == 'year']),
                     ('single relation record|has_coverage|coverage',
                      [e for e in d.edge_types if e[2] == 'coverage']),
                     ('all 6 attr relations (+state-neighbor)',
                      [e for e in d.edge_types if e[2] != 'record'])]:
        ei = flatten(ets)
        deg = torch.bincount(ei[0], minlength=N)
        print(f'\n[{tag}] N={N:,} E_hat={ei.size(1):,} max_deg={int(deg.max()):,}')
        for T in (0, 1, 2, 4):
            _bench(dev, ei, x, T)
    print('\n读数：单层 T=2 的峰值 × backbone 层数（oca.yaml 是 4 层）才是真需求；'
          '两条关系 E_hat 相同而 max_deg 差 7 倍时耗时差数倍 —— 常数由最大 hub 决定。')


def _bench(dev, ei, x, T, d=16, heads=4, n=5) -> None:
    import time
    from modules.oca import OCAConfig, OCALayer
    lay = OCALayer(x.size(1), OCAConfig(out_dim=d, heads=heads, T=T)).to(dev)
    opt = torch.optim.Adam(lay.parameters(), lr=1e-3)
    ei, x_ = ei.to(dev), x.to(dev)
    for _ in range(2):
        loss = lay(x_, ei).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if dev.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(n):
        loss = lay(x_, ei).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    if dev.type == 'cuda':
        peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
        print(f'  T={T} d={d} H={heads}: {dt * 1e3:7.1f} ms/fwd+bwd  '
              f'peak +{peak:6.0f} MiB  => 4 层约 {peak * 4 / 1024:.1f} GiB')
    else:
        print(f'  T={T} d={d} H={heads}: {dt * 1e3:7.1f} ms/fwd+bwd (CPU)')


# ------------------------------------------------------------------ 3. hub ----

_DEG_BINS = (1, 2, 4, 8, 16, 64, 256, 2048, 1 << 30)


def _field_quantities(lay, x, ei, N):
    """一次 forward 后，把逐中心的度数、两版统计量、温度、聚合锐度捣出来。

    代码口径（算法文档 §2.7）分母是 ``sqrt(deg) * sigma``；对比口径只除 ``sigma``
    （严格 z-score）。若槽位里的 $s$ 近似独立，$s_{i,i}-\\mu_i$ 本身的尺度不随
    $\\deg$ 变（约 $\\sigma\\sqrt{1+1/\\deg}$），所以两版的差应当恰好是
    $1/\\sqrt{\\deg}$ —— 两版都算出来，用实测验这个推断成不成立。
    """
    import torch.nn.functional as F
    from torch_geometric.utils import softmax as group_softmax

    with torch.no_grad():
        out = lay.eval()(x, ei)
        s, t = lay.aux['s'], lay.aux['t']
        src, nb, is_self = t['src'], t['nb'], t['is_self']

        def acc(vals):                       # 每次都新建容器：index_add_ 是 in-place
            return torch.zeros(N, vals.size(1), device=s.device, dtype=s.dtype) \
                .index_add_(0, src, vals)

        deg = acc(nb)[:, 0]                          # [N] 真实邻居槽位数（不含自环）
        cnt = deg.clamp(min=1.0).unsqueeze(-1)       # [N,1] 与 _temperature 的 clamp 一致
        s_i = acc(s * (1.0 - nb))                    # [N,H] 中心槽位的分数
        mu = acc(s * nb) / cnt
        var = acc(s * s * nb) / cnt - mu * mu
        # 与 _temperature 同式的单遍方差之外，再算一个两遍的绝对偏差 mad：
        # var = E[s^2] - E[s]^2 在 float32 下会抵消（|s| 大而场内分散小时会算出 0），
        # 分不清「真零方差」还是「抵消误差」，mad 不受这个影响（正态下 sigma ~ 1.2533*mad）。
        mad = acc((s - mu[src]).abs() * nb)[:, 0] / cnt[:, 0]     # [N]
        sig = var.clamp(min=1e-8).sqrt()
        stat_code = (s_i - mu) / (sig * cnt.sqrt() + 1e-6)     # 现行代码
        stat_z = (s_i - mu) / (sig + 1e-6)                     # 去掉 sqrt(deg)
        tau0 = lay.cfg.tau_min
        # 必须与 modules/oca.py 的 tau = softplus(...) + tau_min 同式。少了这个下限，
        # tau 会在 fp32 下溢到 0，下面 sharp() 的 s/tau 先出 inf 再 inf-inf 出 NaN
        # —— 第一版就是这么把「自己度量的 NaN」误当成模型失效的。
        tau = F.softplus(lay.w_tau(torch.cat([x, stat_code.mean(-1, True)], -1)))[:, 0] + tau0
        tau_alt = F.softplus(lay.w_tau(torch.cat([x, stat_z.mean(-1, True)], -1)))[:, 0] + tau0

        def sharp(tau_v):
            """分组 softmax 的参与率 $1/\\sum_j p_j^2$ = 有效平均了几个邻居。"""
            m = ~is_self
            p = group_softmax(s[m] / tau_v[src[m]].unsqueeze(-1), src[m], num_nodes=N)
            sq = torch.zeros(N, p.size(1), device=p.device, dtype=p.dtype) \
                .index_add_(0, src[m], p * p)
            return (1.0 / sq.clamp(min=1e-12)).mean(-1)         # [N]

        return dict(deg=deg, code=stat_code.mean(-1), z=stat_z.mean(-1), sig=sig[:, 0],
                    mad=mad, tau=tau, tau_alt=tau_alt, pr=sharp(tau), pr_alt=sharp(tau_alt),
                    pinned=int((var <= 1e-8).any(-1).sum()),
                    bad=dict(out=int((~torch.isfinite(out)).any(-1).sum()),
                             s=int((~torch.isfinite(s)).sum()),
                             tau=int((~torch.isfinite(tau)).sum()),
                             stat=int((~torch.isfinite(stat_code)).sum())))


def _pct(v, f):
    n = v.numel()
    return float(v.kthvalue(max(1, min(n, int(f * n)))).values)


def _bad_report(q, N):
    """温度与场内标准差的量级清点。这里曾经断言过「A 形态读出 NaN」，实测**不成立**
    （非有限计数全为 0，模型里 softplus + tau_min 的下限是有效的）。留着这段是因为
    该看的不是 NaN 而是这两个量级：tau 跨了几个数量级；sig 有没有贴住 var.clamp 的
    下界 —— 贴住则 stat 是被 1e-4 硬除出来的数，不是信号。"""
    b, tq, sg, md = q['bad'], q['tau'], q['sig'], q['mad']
    live = int((q['deg'] >= 1).sum())
    print(f'  非有限值：out {b["out"]:,}/{N:,} 行（逐节点）  s {b["s"]:,}  '
          f'tau {b["tau"]:,}  stat {b["stat"]:,}')
    print(f'  tau p1/p50/p99/max = {_pct(tq, .01):.4f}/{_pct(tq, .5):.4f}/'
          f'{_pct(tq, .99):.4f}/{float(tq.max()):.4f}（tau_min 下限 1e-3）   '
          f'tau < 0.05 的中心 {int((tq < 0.05).sum()):,}  tau > 5 的中心 {int((tq > 5).sum()):,}')
    print(f'  场内标准差 sig p1/p50/p99 = {_pct(sg, .01):.3g}/{_pct(sg, .5):.3g}/'
          f'{_pct(sg, .99):.3g}   方差贴 clamp 下界的中心 {q["pinned"]:,}/{live:,}')
    print(f'  两遍算法的场内绝对偏差 mad p1/p50/p99 = {_pct(md, .01):.3g}/{_pct(md, .5):.3g}/'
          f'{_pct(md, .99):.3g}（正态下 sigma 约 1.2533*mad，拿它与上行比：'
          f'若 mad 不小而 sig 贴 0，则是单遍方差抵消出的 0，不是真零方差）')


def _hub_table(q, base, tag):
    """按度数分桶。若 $\\sqrt{\\deg}$ 真在起作用，|stat| 列会跟着 deg 塌，
    而两版之比应当贴住 $\\sqrt{\\deg}$（最后一列是校核，不是推论）。"""
    deg = q['deg']
    print(f'\n  [{tag}]')
    print(f'  {"deg":<12}{"n":>10}{"均deg":>10}{"|stat|代码":>13}'
          f'{"|stat|严格":>13}{"|z|/|代码|":>12}{"sqrt(deg)":>11}'
          f'{"tau":>9}{"tau无sqrt":>11}{"PR/deg":>9}{"PR/deg无":>11}')
    for lo, hi in zip(_DEG_BINS, _DEG_BINS[1:]):
        sel = base & (deg >= lo) & (deg < hi)
        n = int(sel.sum())
        if not n:
            continue
        lab = f'[{lo},{hi})' if hi < _DEG_BINS[-1] else f'>={lo}'
        dd = deg[sel]
        ratio = (q['z'][sel].abs() / q['code'][sel].abs().clamp(min=1e-12)).median()
        print(f'  {lab:<12}{n:>10,}{float(dd.mean()):>10,.0f}'
              f'{float(q["code"][sel].abs().mean()):>13.4f}'
              f'{float(q["z"][sel].abs().mean()):>13.4f}'
              f'{float(ratio):>12,.1f}{float(dd.sqrt().mean()):>11,.1f}'
              f'{float(q["tau"][sel].mean()):>9.4f}'
              f'{float(q["tau_alt"][sel].mean()):>11.4f}'
              f'{float((q["pr"][sel] / dd).mean()):>9.4f}'
              f'{float((q["pr_alt"][sel] / dd).mean()):>11.4f}')


def _tau_contrib(lay, q, x, sel):
    """$\\tau$ 的先验激活里，stat 那一列占多少 vs 12 列特征占多少（同一尺度才能比）。"""
    w = lay.w_tau.weight.detach()[0]
    col = float(w[-1])
    xc = x[sel] @ w[:-1]
    print(f'  w_tau 的 stat 列权重 {col:+.4f}；它对 pre-activation 的贡献 std：'
          f'代码口径 {abs(col) * float(q["code"][sel].std()):.4f}，'
          f'去掉 sqrt(deg) 后 {abs(col) * float(q["z"][sel].std()):.4f}')
    print(f'  同一尺度下 12 列 x 的贡献 std {float(xc.std()):.4f}'
          f'（只看 record 侧：attr 节点的 x 整行是 0）')


def _hub_corrs(q, keep):
    """逐中心的 log deg 与两版 stat / 两版 tau 的相关系数（只看指定的那批中心）。"""
    ld = torch.log(q['deg'][keep].clamp(min=1.0).double())
    out = []
    for k in ('code', 'z', 'tau', 'tau_alt'):
        v = q[k][keep].double()
        out.append(f'{k} {float(torch.corrcoef(torch.stack([ld, v]))[0, 1]):+.4f}')
    print('  corr(log deg, ·)  ' + '   '.join(out))


def probe_hub(name: str, T: int = 2, steps: int = 200,
              d: int = 16, heads: int = 4) -> None:
    """§10 第 2 行的裁决依据：温度统计量分母里那个 $\\sqrt{\\deg}$ 到底在干什么。

    三个读数：
      (a) 真实度数分桶，比两版 stat 的幅度与比值 —— 比值贴不贴 $\\sqrt{\\deg}$
          直接说明这个因子有没有在吞信号；
      (b) 两版 stat 各自送进同一个 $w_\\tau$，看 $\\tau$ 与聚合锐度（PR/deg，
          = 有效平均了几个邻居 / 场大小）差多少；
      (c) 短训练后 $w_\\tau$ 的 stat 列权重怎么动 —— 若它在长，说明模型在付费
          补偿这个因子；若不涨，hub 的温度就是死的。

    **必须同时报两种特征情形**，否则会得到一个假的裁决依据：当前数据里 attr
    节点无特征（§9.3-2），record 中心的邻居之间只剩关系自身那一项差异，场内方差
    因此可能贴到 var.clamp 的下界，于是 stat 是被小分母除出来的大数而不是信号
    （下面 sig 那行就是用来验这一点的）。所以另给一份随机 embedding，代理修好之后
    的形状。

    注意：这是**单层、只有一个线性头的代理**，结论只能用到“统计量的尺度
    有没有被度数压掉”这一步，不能当精度结论。
    """
    from modules.oca import OCAConfig, OCALayer

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)          # 建层前固定：否则 |stat|/tau 的绝对读数每次跑都换
    print('=' * 72)
    print(f'### hub 诊断 @ {name}（单层 d={d} H={heads} T={T}，device {dev}）')
    dgr, _ = load(name)
    n_rec = dgr['record'].num_nodes
    ets = [e for e in dgr.edge_types if e[2] != 'record']
    N, _, ei = flatten_graph(dgr, ets)
    x_rec = dgr['record'].x.float()
    n_attr = N - n_rec
    g = torch.Generator().manual_seed(0)        # 固定 => 跨进程可复现
    xs = {}
    for tag, attr_x in [('A. attr 无特征（当前数据形态）', torch.zeros(n_attr, 12)),
                        ('B. attr 随机 embedding（§9.3-2 修好后的代理）',
                         torch.randn(n_attr, 12, generator=g))]:
        xx = torch.zeros(N, 12)
        xx[:n_rec], xx[n_rec:] = x_rec, attr_x
        xs[tag] = xx.to(dev)
    ei = ei.to(dev)
    print(f'N={N:,}  E_hat={ei.size(1):,}（record 中心 {n_rec:,} 个、'
          f'attr 中心 {n_attr:,} 个，最大场 {int(torch.bincount(ei[0], minlength=N).max()):,}）')

    lay = OCALayer(12, OCAConfig(out_dim=d, heads=heads, T=T)).to(dev)
    is_rec = torch.zeros(N, dtype=torch.bool, device=dev)
    is_rec[:n_rec] = True
    q0 = _field_quantities(lay, xs[list(xs)[0]], ei, N)
    live = q0['deg'] >= 1                       # 真正有邻居可聚合的中心
    for tag, xx in xs.items():
        print(f'\n-- init（未训练）· {tag} --')
        q = _field_quantities(lay, xx, ei, N)
        _hub_table(q, live & is_rec, 'record 侧（deg 恒为 6）')
        _hub_table(q, live & ~is_rec, 'attr 侧（route A 的二部层归它）')
        _tau_contrib(lay, q, xx, live & is_rec)
        _hub_corrs(q, live & ~is_rec)
        _bad_report(q, N)

    if steps <= 0:
        return
    tag = list(xs)[-1]
    x = xs[tag]
    print(f'\n-- 短训练（{tag}，拿真实 loss_level 训一个线性头，只看梯度往哪走）--')
    rec = dgr['record']
    gid = attr_group_ids(dgr)
    tr = (rec.train_mask.bool() & ~(gid['year'] == VAL_YEAR_ID)).to(dev)
    va = (gid['year'] == VAL_YEAR_ID).to(dev)
    y = rec.loss_level.long().to(dev)
    head = nn.Linear(d, 3).to(dev)
    opt = torch.optim.Adam([{'params': lay.parameters()}, {'params': head.parameters()}],
                           lr=3e-3)
    w0 = float(lay.w_tau.weight.detach()[0, -1])
    best = (float('inf'), None)
    for it in range(steps):
        lay.train()
        opt.zero_grad()
        loss = nn.functional.cross_entropy(head(lay(x, ei)[:n_rec])[tr], y[tr])
        loss.backward()
        opt.step()
        with torch.no_grad():
            lay.eval()
            vl = float(nn.functional.cross_entropy(head(lay(x, ei)[:n_rec])[va], y[va]))
        if vl < best[0]:
            best = (vl, lay.w_tau.weight.detach()[0, -1].item())
        if it == 0 or (it + 1) % 50 == 0:
            print(f'  step {it + 1:>4}  train {float(loss):.4f}  val {vl:.4f}  '
                  f'w_tau[stat] {float(lay.w_tau.weight.detach()[0, -1]):+.4f}')
    print(f'\n-- 训练 {steps} 步后（按 val 选点，val loss {best[0]:.4f}）--')
    lay.w_tau.weight.data[0, -1] = best[1]
    q = _field_quantities(lay, x, ei, N)
    _hub_table(q, live & is_rec, 'record 侧')
    _hub_table(q, live & ~is_rec, 'attr 侧')
    _tau_contrib(lay, q, x, live & is_rec)
    _hub_corrs(q, live & ~is_rec)
    _bad_report(q, N)
    print(f'  w_tau 的 stat 列：init {w0:+.4f} -> 训后 {best[1]:+.4f}'
          f'（倍率 {abs(best[1] / w0) if w0 else float("nan"):.1f}x）')
    print('\n读数：两版 stat 之比若等于 sqrt(deg)，说明这个因子就在算数地吞信号；'
          '此时看 PR/deg 能不能区分（两列差多少）以及训后 w_tau 的 stat 列有没有涨。')


# ---------------------------------------------------------------- 4. floors ----

def probe_floors(name: str, epochs: int = 80) -> None:
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d, _ = load(name)
    rec = d['record']
    x = rec.x.float()
    y_cls, lev = rec.y_cls.long(), rec.loss_level.long()
    tr, te = rec.train_mask.bool(), rec.test_mask.bool()
    gid = attr_group_ids(d)
    stg = gid['stage']
    va = gid['year'] == VAL_YEAR_ID
    tr2 = tr & ~va
    onehot = torch.cat([nn.functional.one_hot(gid[a], ng).float() for a, ng in ATTRS], 1)
    loo = _loo_group_means(x, gid, tr2)

    print('=' * 72)
    print(f'### floors @ {name}（train {int(tr2.sum()):,} / val {int(va.sum()):,} / '
          f'test {int(te.sum()):,}，MLP 128x3、Adam lr 3e-3、batch 8192、val 早停；'
          f'除 stage 复原那行用 3 个种子外均为单种子）')

    blocks = [('x                 (12)', x),
              ('onehot only       (50)', onehot),
              ('x + onehot        (62)', torch.cat([x, onehot], 1)),
              ('x + LOO group-mean(84)', torch.cat([x, loo], 1)),
              ('x + onehot + LOO (134)', torch.cat([x, onehot, loo], 1))]

    r = _fit_eval_many(x[tr2], stg[tr2], x[va], stg[va], x[te], stg[te],
                       lambda: _mlp(12, 128, 3, 7), dev, epochs, seeds=(0, 1, 2))
    maj = float((stg[tr2] == int(torch.bincount(stg[tr2]).argmax())).float().mean())
    print(f'\n-- stage 能否由 12 列气象复原？（3 个种子）acc {r["acc"][0]:.4f}±{r["acc"][1]:.4f}'
          f'（多数类 {maj:.4f}，超出 {(r["acc"][0] - maj) * 100:+.1f} pp） '
          f'macro-F1 {r["f1"][0]:.4f}±{r["f1"][1]:.4f}')
    print('   复原得很差 => stage 带着这 12 列（均值/极值/求和）之外的信息；'
          '但它是源表里已 one-hot 好的类别特征，不是图结构')
    print('   注：本行方差大（`stage=OTHER` 在 test 里只十几条，macro-F1 被它主导），'
          '所以报 mean±std；它只能当定性判据，不能当小数点后的对比')

    print('\n-- 只用 stage 一列类别特征（7 维，无邻接、无气象、无其他属性）--')
    stg_oh = nn.functional.one_hot(gid['stage'], 7).float()
    r = _fit_eval(stg_oh[tr2], y_cls[tr2], stg_oh[va], y_cls[va], stg_oh[te], y_cls[te],
                  lambda: _mlp(7, 128, 3, 1), dev, epochs, cls=True)
    print(f'  T1  AUC {r["auc"]:.4f}  PR-AUC {r["prauc"]:.4f}  '
          f'acc {r["acc"]:.4f}  F1 {r["f1"]:.4f}')
    r = _fit_eval(stg_oh[tr2], lev[tr2], stg_oh[va], lev[va], stg_oh[te], lev[te],
                  lambda: _mlp(7, 128, 3, 3), dev, epochs)
    print(f'  T2  acc {r["acc"]:.4f}  macro-F1 {r["f1"]:.4f}  (val acc {r["val_acc"]:.4f})')

    print('\n-- T1 high_loss（不平衡 25%，pos_weight 校正；看 AUC / PR-AUC）--')
    for nm, xx in blocks[1:]:
        r = _fit_eval(xx[tr2], y_cls[tr2], xx[va], y_cls[va], xx[te], y_cls[te],
                      lambda: _mlp(xx.size(1), 128, 3, 1), dev, epochs, cls=True)
        print(f'  {nm:<24} AUC {r["auc"]:.4f}  PR-AUC {r["prauc"]:.4f}  '
              f'acc {r["acc"]:.4f}  F1 {r["f1"]:.4f}')

    print('\n-- T2 loss_level 三分类（均衡，chance 0.3333）--')
    for nm, xx in blocks:
        r = _fit_eval(xx[tr2], lev[tr2], xx[va], lev[va], xx[te], lev[te],
                      lambda: _mlp(xx.size(1), 128, 3, 3), dev, epochs)
        print(f'  {nm:<24} acc {r["acc"]:.4f}  macro-F1 {r["f1"]:.4f}  '
              f'(val acc {r["val_acc"]:.4f})')
    print('\n-- 重尾列的影响：把 x 整体 clamp 到 ±4σ（`quantity` 的极值就在这一档）--')
    xw = x.clamp(-4, 4)
    for nm, xx in [('x, clamp            (12)', xw),
                   ('x + onehot, clamp   (62)', torch.cat([xw, onehot], 1))]:
        r = _fit_eval(xx[tr2], lev[tr2], xx[va], lev[va], xx[te], lev[te],
                      lambda: _mlp(xx.size(1), 128, 3, 3), dev, epochs)
        print(f'  {nm:<24} acc {r["acc"]:.4f}  macro-F1 {r["f1"]:.4f}  '
              f'(val acc {r["val_acc"]:.4f})')

    print('\n读数：`onehot only` 若已等于生产方 GNN 基线（corn 0.805 / soy 0.894），'
          '而加 LOO 池化不再涨，则 hetero 模型必须越过的是上面最高那一行，不是 chance。')


def _loo_group_means(x, gid, tr2) -> torch.Tensor:
    """每条关系：用 train 记录算的 leave-one-out 特征均值（不可学的消息传递替身）。"""
    out, ntr = [], int(tr2.sum())
    for a, ng in ATTRS:
        g = gid[a]
        s = torch.zeros(ng, x.size(1)).index_add_(0, g[tr2], x[tr2])
        c = torch.zeros(ng).index_add_(0, g[tr2], torch.ones(ntr))
        own = tr2.float().unsqueeze(1) * x
        cnt = (c[g].unsqueeze(1) - tr2.float().unsqueeze(1)).clamp(min=1)
        out.append((s[g] - own) / cnt)
    return torch.cat(out, 1)


def _mlp(din: int, hidden: int, nlayers: int, dout: int) -> List[nn.Module]:
    """nlayers = Linear 层数（最后一个是读出）。"""
    ls, dd = [], din
    for _ in range(nlayers - 1):
        ls += [nn.Linear(dd, hidden), nn.ReLU()]
        dd = hidden
    return ls + [nn.Linear(dd, dout)]


def _ms(v: Sequence[float]) -> Tuple[float, float]:
    m = sum(v) / len(v)
    return m, (sum((t - m) ** 2 for t in v) / len(v)) ** 0.5


def _pm(v: Sequence[float]) -> str:
    """一个种子时只印均值，多个种子时印 mean±std（std 是总体标准差，不是标准误）。"""
    m, s = _ms(v)
    return f'{m:.4f}' if len(v) < 2 else f'{m:.4f}±{s:.4f}'


def _dsp(v: Sequence[float]) -> str:
    """同 _pm，但带正负号 —— 用于「与地板按种子配对」的差。"""
    m, s = _ms(v)
    return f'{m:+.4f}' if len(v) < 2 else f'{m:+.4f}±{s:.4f}'


def _fit_eval_many(xtr, ytr, xva, yva, xte, yte, layers_fn, dev, epochs,
                   seeds: Sequence[int] = (0, 1, 2), cls=False, lr=3e-3):
    """跟 _fit_eval 同一条路径，但跑多个种子，返回 {metric: (mean, std)}。

    只给方差大的行用 —— 实测使用 argmax 的行（T2 的 acc/F1）重启解释器后仍逐位复现，
    而使用连续分数的行（T1 的 AUC/PR-AUC）在 GPU 上重启一次会漂到 0.005（见 §9.9 的注）。

    `layers_fn` 必须是函数：在这里逐种子调用，才会落到各次 _fit_eval 的 seed 之后。
    """
    rs = [_fit_eval(xtr, ytr, xva, yva, xte, yte, layers_fn, dev, epochs,
                    cls=cls, lr=lr, seed=s) for s in seeds]
    out = {}
    for k in rs[0]:
        v = [r[k] for r in rs if r[k] == r[k]]           # 丢弃 nan
        if not v:
            out[k] = (float('nan'), 0.0)
            continue
        mu = sum(v) / len(v)
        out[k] = (mu, (sum((t - mu) ** 2 for t in v) / len(v)) ** 0.5)
    return out


def _fit_eval(xtr, ytr, xva, yva, xte, yte, layers, dev, epochs, cls=False, lr=3e-3,
              seed: int = 0):
    """`layers` 可以是已建好的层列表，也可以是建网函数（推荐后者）。

    传列表时权重是在调用处建的，早于这里的 manual_seed ⇒ 种子管不到 init。坏处不是小数
    点而是配对性：一个进程里只有第一个网络吃到默认 RNG（实测 `torch.initial_seed()` 每次
    重启都不同），之后每个网络都从 manual_seed 之后的同一个状态抽样 —— 于是「同进程内
    比较」恰好对基准行不成立，而候选行彼此反而是同一份 init。传 callable 时建网落在
    seed 之后，同种子同形状必给同一份权重，跨行、跨进程都可配对。
    """
    torch.manual_seed(seed)
    if callable(layers):
        layers = layers()
    xtr, ytr = xtr.to(dev), ytr.to(dev)
    xva, yva = xva.to(dev), yva.to(dev)
    xte, yte = xte.to(dev), yte.to(dev)
    m = nn.Sequential(*layers).to(dev)
    pw = None
    if cls:
        pos = float(ytr.sum())
        pw = torch.tensor([(ytr.numel() - pos) / pos], device=dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-5)
    lossf = ((lambda o, t: nn.functional.binary_cross_entropy_with_logits(
        o.view(-1), t.float(), pos_weight=pw)) if cls
        else nn.functional.cross_entropy)
    perm = torch.randperm(xtr.size(0), device=dev)
    best: Tuple[float, Dict] = (float('inf'), {})
    for _ in range(epochs):
        for i in range(0, perm.numel(), 8192):
            idx = perm[i:i + 8192]
            opt.zero_grad()
            lossf(m(xtr[idx]), ytr[idx]).backward()
            opt.step()
        with torch.no_grad():
            vl = float(lossf(m(xva), yva).mean())
        if vl < best[0]:                       # 只按 val 选点，不碰 test
            best = (vl, {k: v.detach().clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1])
    with torch.no_grad():
        o = m(xte)
        p = (torch.sigmoid(o.view(-1)) if cls else o.softmax(-1)).cpu()
        ov = m(xva)
        va_acc = float((((torch.sigmoid(ov.view(-1)) > .5).long() if cls
                         else ov.argmax(-1)) == yva).float().mean())
    yc = yte.cpu()
    if cls:
        yh = (p > 0.5).long()
        return dict(auc=roc_auc(yc, p), prauc=pr_auc(yc, p), val_acc=va_acc,
                    acc=float((yh == yc).float().mean()), f1=macro_f1(yc, yh, 2))
    yh = p.argmax(-1)
    return dict(acc=float((yh == yc).float().mean()), val_acc=va_acc,
                f1=macro_f1(yc, yh, int(yc.max()) + 1),
                auc=float('nan'), prauc=float('nan'))


# ---------------------------------------------------------------- 5. edges ----

# 邻域聚合张量的列序：x(NF) | 邻居数 1 | y_cls 直方图 2 | loss_level 直方图 3
NF = 12
NCOL = NF + 1 + 2 + 3


def _meta(name: str) -> Dict:
    import pathlib
    p = {'corn': 'data/CORN/graph_meta.json',
         'soy': 'data/SOY/soy_graph_meta.json'}[name]
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))


def _canon(names: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """两个 .pt 各自的 node_index_maps 不保证同序，所以取到同一 key 空间时要重编码。
    全数字的列按数值排（year / month 要在编码上做带 +-band 的算术），否则按字典序。"""
    metas = [_meta(n) for n in names]
    out = {}
    for col in metas[0]['node_index_maps']:
        vals = {v for m in metas for v in m['node_index_maps'][col]}
        try:
            seq = sorted(vals, key=int)
        except ValueError:
            seq = sorted(vals)
        out[col] = {v: i for i, v in enumerate(seq)}
    return out


def _codes(d, meta, canon) -> Dict[str, torch.Tensor]:
    """record -> 各属性的统一编码（经本数据集的 index_map 还原，再查统一表）。"""
    gid = attr_group_ids(d)
    out = {}
    for col, m in meta['node_index_maps'].items():
        inv = torch.full((max(m.values()) + 1,), -1, dtype=torch.long)
        for kk, vv in m.items():
            inv[vv] = canon[col][kk]
        g = inv[gid[col]]
        assert int((g < 0).sum()) == 0, f'{col}: 有节点编号不在 meta 的 index_map 里'
        out[col] = g
    return out


def _cell(codes, cols, npow):
    """把若干属性拼成一个混合进制格元 key（两个数据集用同一 npow 才能对得上）。"""
    key = torch.zeros_like(codes[cols[0]])
    ncell = 1
    for c in cols:
        key = key * npow[c] + codes[c]
        ncell *= npow[c]
    return key, ncell


def _nbr_tensor(x, y_cls, lev):
    def oh(v, k):
        return nn.functional.one_hot(v.long(), k).float()
    return torch.cat([x, torch.ones(x.size(0), 1), oh(y_cls, 2), oh(lev, 3)], 1)


def _cell_S(G, strm, key, axis, ncell, naxis):
    """源侧按 (axis, cell) 压一张求和表，只统计 train 记录。"""
    lin = (axis * ncell + key)[strm]
    return (torch.zeros(naxis * ncell, G.size(1)).index_add_(0, lin, G[strm])
            .view(naxis, ncell, G.size(1)))


def _band_gather(S, ckey, caxis, ncell, band):
    """中心的邻域 = {cell 相同、axis 差不超过 band} 那些源侧格子的和（越界丢，不环绕）。"""
    naxis = S.size(0)
    off = torch.arange(-band, band + 1).unsqueeze(0) + caxis.unsqueeze(1)
    ok = (off >= 0) & (off < naxis)
    idx = off.clamp(0, naxis - 1) * ncell + ckey.unsqueeze(1)
    return S.reshape(-1, S.size(2))[idx].mul_(ok.unsqueeze(-1)).sum(1)        # [N, NCOL]


def _knn_nbr(xz, k, group, blk=1024, cand=None):
    """分块精确欧氏 kNN（排除自身）。group 非空 => 只在同组内找。返回 [2, E]。

    cand 非空 => 只在候选行里找邻居。传 train 掩码是必须的：不限制候选时，test 行
    的十近邻多半也是 test 行，而聚合只统计 train（_knn_agg 的 strm），于是 test 的
    有效邻居只剩 3.1 个、train 有 8.9 个 —— 那是构造本身造的 train/test 失配。
    """
    n, src, dst = xz.size(0), [], []
    if group is not None:
        group = group.to(xz.device)              # xz 在卡上时要跟着上卡，不然 masked_fill 报设备错
    if cand is not None:
        cand = cand.to(xz.device)
    sq = (xz * xz).sum(1)
    for i in range(0, n, blk):
        xb = xz[i:i + blk]
        d2 = (xb * xb).sum(1).unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (xb @ xz.t())
        d2[:, i:i + xb.size(0)] = float('inf')
        if cand is not None:
            d2 = d2.masked_fill(~cand.unsqueeze(0), float('inf'))
        if group is not None:
            d2 = d2.masked_fill(group[i:i + xb.size(0)].unsqueeze(1) != group.unsqueeze(0),
                                float('inf'))
        v, j = d2.topk(k, dim=1, largest=False)
        keep = torch.isfinite(v)                  # 同组不足 k 个时丢掉凑数的 inf
        r = torch.arange(i, i + xb.size(0), device=xz.device).unsqueeze(1).expand_as(j)
        src.append(r[keep])
        dst.append(j[keep])
    return torch.stack([torch.cat(src), torch.cat(dst)])


def _knn_agg(nbr, G, strm, n):
    """kNN 邻域不含自身，所以不需要再减 LOO。"""
    m = strm[nbr[1]]
    return torch.zeros(n, G.size(1)).index_add_(0, nbr[0][m], G[nbr[1][m]])


def _edge_row(tag, agg, y_cls, lev, ch_c, ch_l):
    n = agg[:, NF].clamp(min=0)
    ac = agg[:, NF + 1:NF + 3].gather(1, y_cls.view(-1, 1)).squeeze(1)
    al = agg[:, NF + 3:NF + 6].gather(1, lev.view(-1, 1)).squeeze(1)
    tot = n.sum().clamp(min=1)
    pc, pl = float(ac.sum() / tot), float(al.sum() / tot)
    print(f'  {tag:<38} 邻居/中心 {float(n.mean()):7.1f}  零邻居 {int((n == 0).sum()):>6,}  '
          f'y_cls {pc:.4f} ({pc - ch_c:+.4f})  loss_level {pl:.4f} ({pl - ch_l:+.4f})')
    return agg[:, :NF] / n.clamp(min=1).unsqueeze(1)


def _label_rate(agg, y_cls, lev, tr2):
    """邻域的 train 标签率（目标编码）：2 列 = P(y_cls=1) 与 E[loss_level]。

    上面那个「邻域特征均值」只能搬运特征信息，而可学的 attr embedding 走的是另一条
    通路：把格元学成一个查表（组级标签先验）。这条必须单独测，否则“边无信息”这个
    结论只证了一半。无标签泄漏：`agg` 只对 train 记录求和（_cell_S 的 strm=tr2），
    且 train 行已再扣掉自己那一票，所以 val/test 行的邻域里从来不含自己的标签。
    零邻居（2.6 万行）填全局 train 先验，否则“率 0”与“无数据”会被混为一谈。
    """
    n = agg[:, NF].clamp(min=0)
    d = n.clamp(min=1)                                    # 零邻居行下面会被先验覆盖，先别除出 nan
    pc, pl = float(y_cls[tr2].float().mean()), float(lev[tr2].float().mean())
    rc = agg[:, NF + 1].clamp(min=0) / d                 # y_cls=1 的个数 / 邻居数
    ml = (agg[:, NF + 3:NF + 6] * torch.arange(3.)).sum(1) / d
    return torch.stack([torch.where(n > 0, rc, torch.full_like(rc, pc)),
                        torch.where(n > 0, ml, torch.full_like(ml, pl))], 1)


def probe_edges(name: str, epochs: int = 80, k: int = 10, band: int = 1,
                seeds: Sequence[int] = (0,)) -> None:
    """route B 的 go/no-go：候选邻域构造能不能造出 x 之外的标签信息。

    两条判据缺一不可：（1）邻域标签一致率的 excess 要明显高于同一中心集下的
    单一属性分组（下面参照段的第一行就是 §9.5 里最强的 `stage`）；（2）把「邻域特征
    均值」（消息传递的不可学替身）拼到 x + onehot 上，T1/T2 要越过 §9.7 的地板。
    第二条才是硬的，因为 §9.7 已测出现有分组上的池化增益只有 +0.001 AUC；而判它的
    尺子是最后那段「无信息对照」：同样加 12 列但不携带信息，看地板会掉多少。

    注意源表里没 county 列、id 逐行唯一（见 §9.2），所以下面没有「县级邻接」这一
    候选；能试只剩三条：特征空间 kNN、属性交叉格元、另一作物同格元的统计量。

    两个构造口径上的坑（都是上一版实测踩到的，已修）：一是格元邻域不能拿
    `year+-1` 当带宽 —— 表源只到 2018，那会让 100% 的 test 行邻域为空、特征整列塌 0，
    测出来的是“特征没了”而不是“信息没了”（现在格元行均按全窗口池化，train 行 LOO）；
    二是 kNN 必须把候选限制在 train 行里，否则 test 行只剩三成果邻域。

    第三个坑在本函数的尺子上：早先版本把 `_mlp(...)` 写在 `_fit_eval` 的实参里，而它
    在 `manual_seed` 之前就被求值 ⇒ 进程里第一个建的网络（就是地板行）用的是进程随机的
    默认 RNG，候选行彼此反而抽自同一份状态 —— “同进程比较”恰好不配对，而 corn 地板实测
    跳了七次 0.0074 的极差就来自这里。现在建网进了 seed 之后，且 `seeds` 多于一元时
    逐行报“与地板按种子配对”的差与 std。
    """
    other = 'soy' if name == 'corn' else 'corn'
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    canon = _canon([name, other])
    npow = {c: len(v) for c, v in canon.items()}
    d, _ = load(name)
    o, _ = load(other)
    rec, orec = d['record'], o['record']
    x, y_cls, lev = rec.x.float(), rec.y_cls.long(), rec.loss_level.long()
    ox, oc, ol = orec.x.float(), orec.y_cls.long(), orec.loss_level.long()
    gid, ogid = attr_group_ids(d), attr_group_ids(o)
    tr, te = rec.train_mask.bool(), rec.test_mask.bool()
    va, ova = gid['year'] == VAL_YEAR_ID, ogid['year'] == VAL_YEAR_ID
    tr2, otr2 = tr & ~va, orec.train_mask.bool() & ~ova
    onehot = torch.cat([nn.functional.one_hot(gid[a], ng).float() for a, ng in ATTRS], 1)
    codes = _codes(d, _meta(name), canon)
    ocodes = _codes(o, _meta(other), canon)
    G, OG = _nbr_tensor(x, y_cls, lev), _nbr_tensor(ox, oc, ol)
    ch_c = float(((torch.bincount(y_cls).float() / y_cls.numel()) ** 2).sum())
    ch_l = float(((torch.bincount(lev).float() / lev.numel()) ** 2).sum())

    print('=' * 72)
    print(f'### edges @ {name}（候选邻域构造的 go/no-go，device {dev}，'
          f'{len(seeds)} 个种子）')
    print(f'对侧作物 = {other}；邻域只统计 train 记录（剔 2019），窗口越界不环绕；'
          f'格元 key 走统一编码（两个 .pt 的 index_map 不保证同序）')
    print(f'chance：y_cls {ch_c:.4f}  loss_level {ch_l:.4f}（参照线 = 下面第一段的'
          f'单一属性行，与本节同一中心集；§9.5 那张表中的是 train-only 中心）')

    floor: Dict[str, List[float]] = {}

    def fits(tag, extra):
        """每个种子重跑一遍，并按种子与地板行配对相减（同 seed ⇒ 同一份 init、同一批
        数据顺序），所以括号里的 ± 是配对差的标准差，不是两次单跑相减那种巧合。"""
        xx = torch.cat([x, onehot] + ([extra] if extra is not None else []), 1)
        r1 = [_fit_eval(xx[tr2], y_cls[tr2], xx[va], y_cls[va], xx[te], y_cls[te],
                        lambda: _mlp(xx.size(1), 128, 3, 1), dev, epochs,
                        cls=True, seed=s) for s in seeds]
        r2 = [_fit_eval(xx[tr2], lev[tr2], xx[va], lev[va], xx[te], lev[te],
                        lambda: _mlp(xx.size(1), 128, 3, 3), dev, epochs, seed=s)
                for s in seeds]

        def dsp(v):
            m, s = _ms(v)
            return f'{m:+.4f}' if len(v) < 2 else f'{m:+.4f}±{s:.4f}'

        cur = dict(auc=[r['auc'] for r in r1], acc=[r['acc'] for r in r2])
        if not floor:
            floor.update(cur)                      # 第一次调用就是地板行
        d1 = [a - b for a, b in zip(cur['auc'], floor['auc'])]
        d2 = [a - b for a, b in zip(cur['acc'], floor['acc'])]
        print(f'    {tag:<38} 维度 {xx.size(1):>4}  T1 AUC {_pm(cur["auc"])} '
              f'({dsp(d1)})  PR {_pm([r["prauc"] for r in r1])}  '
              f'T2 acc {_pm(cur["acc"])} ({dsp(d2)})  F1 {_pm([r["f1"] for r in r2])}')

    print('\n-- 地板（与 §9.7 的 x+onehot 行同口径）--')
    fits('x + onehot', None)
    extras, aggs = {}, {}
    gen = torch.Generator().manual_seed(0)

    print('\n-- 参照：单一属性分组（§9.5 的六条关系，但不带年份窗）--')
    for col in ('stage', 'state', 'year', 'month', 'plan', 'coverage'):
        g = codes[col]
        agg = (torch.zeros(int(g.max()) + 1, G.size(1)).index_add_(0, g[tr2], G[tr2])[g]
               - tr2.float().unsqueeze(1) * G)            # LOO：train 行抠掉自己
        _edge_row(f'self: {col}', agg, y_cls, lev, ch_c, ch_l)

    print('\n-- 特征空间 kNN（属性 one-hot 无法表达的构造）--')
    xz = ((x - x.mean(0)) / x.std(0).clamp(min=1e-6)).clamp(-4, 4).to(dev)
    for tag, grp in [(f'kNN(x, k={k}) 全局', None), (f'kNN(x, k={k}) 同 state', codes['state'])]:
        nbr = _knn_nbr(xz, k, grp, cand=tr2).cpu()
        agg = _knn_agg(nbr, G, tr2, x.size(0))
        extra = _edge_row(tag, agg, y_cls, lev, ch_c, ch_l)
        extras[tag], aggs[tag] = extra, agg
        fits(tag + ' + 邻域均值', extra)

    print('\n-- 属性交叉格元（仍是分组池化，但比单一 attr 更细）--')
    # 轴 = None 意味着对该轴求和（= 整个 2008-18 窗口池化）。不用 year+-1 的窗口：表源
    # 只含 train 行（年份 <= 2018），而 test 行在 2020-22，+-1 的带宽越界后整列为空，
    # 那 12 列在 100% 的 test 行上恒为 0（实测：max|x|=0，train 上 9.08）—— 它测到的是
    # “特征塌掉”而不是“邻域没信息”。下面三行均改为无轴限制，使 train/val/test 同口径。
    for tag, cols, axis, src in [
            ('self: state x stage, 全窗口', ('state', 'stage'), None, 'self'),
            ('self: state x month x stage, 全窗口', ('state', 'month', 'stage'), None, 'self'),
            (f'{other}: state x month x stage, 全窗口', ('state', 'month', 'stage'), None, 'other'),
            ('self: state x plan x coverage, month+-1', ('state', 'plan', 'coverage'), 'month', 'self')]:
        ck, ncell = _cell(codes, cols, npow)
        ax = axis or 'year'
        sub = torch.zeros_like(G)
        if src == 'self':
            S = _cell_S(G, tr2, ck, codes[ax], ncell, npow[ax])
            sub[tr2] = G[tr2]                      # LOO：自身在 train 里就得从邻域里抠掉
        else:
            sk, _ = _cell(ocodes, cols, npow)
            S = _cell_S(OG, otr2, sk, ocodes[ax], ncell, npow[ax])
        agg = (S.sum(0)[ck] if axis is None else _band_gather(S, ck, codes[ax], ncell, band)) - sub
        extra = _edge_row(tag, agg, y_cls, lev, ch_c, ch_l)
        extras[tag], aggs[tag] = extra, agg
        fits(tag + ' + 邻域均值', extra)

    print('\n-- 标签通路（目标编码）：邻域不提供特征均值而只提供组级标签率 --')
    for tag, agg in sorted(aggs.items()):
        if 'month x stage' not in tag:
            continue
        rate = _label_rate(agg, y_cls, lev, tr2)
        fits(tag + ' 的标签率 (+10 噪声列)',
             torch.cat([rate, torch.randn(x.size(0), NF - 2, generator=gen)], 1))

    print('\n-- 无信息对照：同样加 12 列，只破坏与样本的对应（用来定上面那列“变化”的噪声地板）--')
    fits('iid 噪声 12 列', torch.randn(x.size(0), NF, generator=gen))
    perm = torch.randperm(x.size(0), generator=gen)
    for tag in sorted(extras):
        if tag.startswith('kNN') or 'month x stage' in tag:
            fits(tag + ' 的行置换', extras[tag][perm])

    print('\n读数：一致率高不等于能涨点 —— 只要邻域均值落在地板行的噪声里，这条边就不值得为'
          '它改模型；而“掉了多少”必须与上面两行对照比，同宽度换输入本身就有零点几的抖动。'
          '单种子时括号里的差不可信（init 已配对，但 GPU 归约的非确定性还在），要定 0.01 以下的'
          '结论请带 `--seeds 0,1,2` 重跑，看配对差的 ±std。带对侧作物名前缀的那一行额外注意：'
          '它用的是另一个作物的 train 理赔统计，它本身就是目标函数的近亲，'
          '当特征合理、当“图结构的证据”不合理。')


# ---------------------------------------------------------------- 6. drift ---

def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """格元级的秩相关（并列取平均秩，复用 roc_auc 那套实现）。"""
    ra, rb = _avg_ranks(a), _avg_ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = (float((ra * ra).sum()) * float((rb * rb).sum())) ** 0.5
    return float((ra * rb).sum()) / den if den > 0 else float('nan')


def _lookup(key, num, cnt, glob, rows):
    """格元查表打分。表里没有这个格元（unseen）时退回全局率 —— 否则「没数据」会被
    当成「率为 0」，那是白送的反向信号。"""
    k = key[rows]
    c = cnt[k]
    return torch.where(c > 0, num[k] / c.clamp(min=1), torch.full_like(c, float(glob)))


def probe_drift(name: str, lags: Sequence[int] = (0, 1, 2, 3, 4, 6, 8, 10),
                min_n: int = 30) -> None:
    """组级标签先验随年份漂多少 —— 把 §9.9 第 3 条那句推断变成可报的数。

    §9.9 里全表最差的两行是目标编码（格元的 train 标签率），当时的解释是「组级先验
    不随年份迁移」。那句话目前仍是推断，本节把它拆成三块（实测结论见函数末尾的读数，
    它把那句解释否了）：
    (a) lag 曲线：拿单一年份的表去预测相隔 L 年的那一年，AUC 随 L 怎么掉，且
        forward（用过去的表预测未来）与 backward（用未来的表预测过去）分开——
        只有趋势会让两个方向不对称，纯年份噪声则应当对称。L=0 是目标年自己的表，
        含自身那一票 ⇒ 上偏，只用来看 L>=1 的衰减形状（行内 LOO 在这里是病态的：
        它会把同一个格元里的正例一律排到负例下面，同格元对的 AUC 直接变 0）；
    (b) 窗口表 → 逐年：表换成整个 2008-18（目标年若在窗口内则整年从表里剔掉，
        与 §9.7/§9.9 的协议一致），看它在 2019 / 2020-22 上比在窗口内低多少；
        末尾再给三个 pooled 数：逐年均值 / 跨年混排（= §9.7、§9.9 那种 pooled AUC）/
        oracle 年度重标定后的混排 —— 后两个的差就是「水平漂移伤多少」，而它恰好
        把漂移拆成两部分：格元间的**排序**能不能迁移（Spearman）与整年的**水平**对不对（这里）；
    (c) 漂移拆成两半：格元间的排序（Spearman）与整年的水平（pooled 与逐年均值之差）；
    (d) 漂移的两个来源分开算：unseen 行占比（格元没了）与格元级 Spearman /
        加权 |gap|（格元还在但率变了）。

    表源只用 **train 行**（2019 是 val，不进表），目标年用该年全部行 —— 那些行要么
    根本不在表里（窗口外），要么整年被剔掉（窗口内），所以不存在自拟合。另给一行
    反序对照：把行→格元的映射整体打乱，表还在、规模还在，只有对应关系没了，它必须
    落在 0.5 上下，否则本节所有读数都别信。
    """
    meta = _meta(name)
    other = 'soy' if name == 'corn' else 'corn'
    canon = _canon([name, other])
    npow = {c: len(v) for c, v in canon.items()}
    inv_year = {v: k for k, v in canon['year'].items()}
    d, path = load(name)
    rec = d['record']
    y_cls, lev = rec.y_cls.long(), rec.loss_level.long()
    n = int(y_cls.numel())
    codes = _codes(d, meta, canon)
    yc, nax = codes['year'], npow['year']
    G = _nbr_tensor(rec.x.float(), y_cls, lev)
    rows = {i: (yc == i).nonzero().view(-1) for i in range(nax)}
    tr2 = rec.train_mask.bool() & (yc != VAL_YEAR_ID)
    W = [i for i in range(nax) if i < VAL_YEAR_ID and int((tr2[rows[i]]).sum())]
    tes = [i for i in range(VAL_YEAR_ID + 1, nax) if int(rows[i].numel())]
    allr = torch.arange(n)
    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=gen)

    print('=' * 72)
    print(f'### drift @ {name}（组级标签先验的年份漂移，{path}）')
    print(f'表源 = 2008-2018 的 train 行（共 {int(tr2.sum()):,} 条，逐年）；目标年 = 该年全部行。'
          f'分数 = 格元在表源里的 y_cls 率，unseen 退回表源全局率；Spearman/|gap| 只算两侧'
          f'计数都 >= {min_n} 的格元，其中目标年的率用该年全部行。')

    for tag, cols in [('stage', ('stage',)), ('state', ('state',)),
                      ('state x stage', ('state', 'stage')),
                      ('state x month x stage', ('state', 'month', 'stage')),
                      ('state x plan x coverage', ('state', 'plan', 'coverage'))]:
        key, ncell = _cell(codes, cols, npow)
        S = _cell_S(G, tr2.nonzero().view(-1), key, yc, ncell, nax)
        cnt, pos = S[:, :, NF].double(), S[:, :, NF + 2].double()
        # 目标侧另建一张表：源侧只能统计 train 行（不然就是自拟合），而目标年在 test 时
        # 一行 train 都没有，拿 cnt 做分母会得到全 0 ⇒ 细粒度的 Spearman 在 2020-22 全变
        # nan，而那恰好本节要问的那一年。所以目标侧用该年全部行（用了标签，但只当诊断量，
        # 不进任何打分通路；窗口内的年份本来就全是 train 行，两者逐元相等）。
        SA = _cell_S(G, allr, key, yc, ncell, nax)
        cntA, posA = SA[:, :, NF].double(), SA[:, :, NF + 2].double()
        gp = pos.sum(1) / cnt.sum(1).clamp(min=1)
        wcnt, wpos = cnt[W].sum(0), pos[W].sum(0)
        gwp = float(wpos.sum() / wcnt.sum().clamp(min=1))
        print(f'\n-- {tag}：格元上限 {ncell:,}，窗口里实际出现 {int((wcnt > 0).sum()):,} 个'
              f'（其中计数 >= {min_n} 的 {int((wcnt >= min_n).sum()):,} 个），'
              f'窗口全局正类率 {gwp:.4f} --')

        def tab(src, dst):
            """单一年份的表 -> 目标年的 AUC。"""
            m = rows[dst]
            return roc_auc(y_cls[m], _lookup(key, pos[src], cnt[src], gp[src], m))

        print('  (a) lag 曲线（L=0 = 目标年自己的表，含自身那一票 ⇒ 上偏，只看形状）')
        print('        ' + ' '.join(f'L={L:<6}' for L in lags))
        for nm, sgn in [('forward 过→未', 1), ('backward 未→过', -1)]:
            vs = []
            for L in lags:
                v = [tab(s, s + sgn * L) for s in W if 0 <= s + sgn * L < nax]
                v = [x for x in v if x == x]
                vs.append(sum(v) / len(v) if v else float('nan'))
            print(f'        {nm:<16}' + ' '.join(f'{x:8.4f}' for x in vs))

        print('  (b) 窗口表（目标年在窗口内则整年剔除）→ 逐年')
        print('        year        n  unseen%  spearman   w|gap|      AUC  备注')
        aucs = {}
        for i in W + [VAL_YEAR_ID] + tes:
            m = rows[i]
            if not int(m.numel()):
                continue
            tc, tp = (wcnt - cnt[i], wpos - pos[i]) if i in W else (wcnt, wpos)
            uns = float((tc[key[m]] <= 0).double().mean())
            q = (tc >= min_n) & (cntA[i] >= min_n)
            a, b = tp[q] / tc[q], posA[i][q] / cntA[i][q]
            ok = int(q.sum()) >= 4
            rho = _spearman(a, b) if ok else float('nan')
            wt = cntA[i][q].double()
            gap = (float((wt * (a - b).abs()).sum() / wt.sum().clamp(min=1))
                   if ok else float('nan'))
            auc = roc_auc(y_cls[m], _lookup(key, tp, tc, gwp, m))
            aucs[i] = (auc, uns, rho, gap, int(m.numel()))
            note = ('窗口内(LOO 一年)' if i in W else
                    'val' if i == VAL_YEAR_ID else 'test')
            print(f'      {inv_year[i]:>7} {int(m.numel()):>9,} {uns:>7.1%} {rho:>10.4f}'
                  f' {gap:>9.4f} {auc:>9.4f}  {note}')
        inw = [aucs[i][0] for i in W if i in aucs]
        tev = [aucs[i][0] for i in tes if i in aucs]
        mu_in = sum(inw) / len(inw)
        print(f'        小计：窗口内均值 {mu_in:.4f} -> 2019 {aucs[VAL_YEAR_ID][0]:.4f}'
              f' -> 2020-22 均值 {sum(tev) / len(tev):.4f}，'
              f'漂移 {sum(tev) / len(tev) - mu_in:+.4f} AUC')
        print(f'        反序对照（行→格元映射打乱）：AUC '
              f'{roc_auc(y_cls, _lookup(key[perm], wpos, wcnt, gwp, allr)):.4f}（该贴 0.5）')

        # (c) pooled：逐年均值 vs 跨年混排 vs oracle 重标定。后两者的差 = 水平漂移的代价
        te = torch.cat([rows[i] for i in tes])
        s_te = _lookup(key, wpos, wcnt, gwp, te)
        mix = roc_auc(y_cls[te], s_te)
        fix = s_te.clone()                      # oracle：把每年的均值掰回该年真实正类率
        o = 0
        for i in tes:
            k = int(rows[i].numel())
            seg = slice(o, o + k)
            fix[seg] = s_te[seg] - float(s_te[seg].mean()) \
                + float(y_cls[rows[i]].double().mean())
            o += k
        print(f'  (c) pooled 2020-22：逐年均值 {sum(tev) / len(tev):.4f}｜跨年混排 {mix:.4f}'
              f'｜oracle 年度重标定后 {roc_auc(y_cls[te], fix):.4f}'
              f'（混排与逐年均值的差 = 水平漂移伤 pooled 的量，用了标签 ⇒ 上界）')

    print('\n读数（已实测过的两个分支，2020-22 上落在前一支）：窗口表在 test 年的 AUC 几乎不低于'
          '窗口内（corn 最细粒度漂 -0.0003），test 年的格元级 Spearman 0.87-0.96，forward/backward'
          '对称、unseen 不超过 2.3% —— 所以「组级先验不随年份迁移」不成立，§9.9 那两行目标编码的'
          '落差在拟合通路而不在信息本身（同一张表单独做分数就有 0.799，喂进 MLP 只剩 0.793）。'
          '同时注意本节量的是裸查表：它只说明「这列信息能拿多少分」，不能拿它当地板。')


# -------------------------------------------------------------------- 7. te ----

def _te(agg, y_cls, lev, tr2, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """格元标签率：tau=0 就是裸查表（与 _label_rate 逐元相等），tau>0 按计数向先验收缩。

    收缩式 r = (a + tau * p) / (n + tau)，a = 格元内的命中数、n = 该格元的 train 记录数、
    p = 全局 train 率。它只做一件事：n 小的时候不把 a/n 当证据。零邻居且 tau=0 时退回
    先验。第 0 列沿用 _label_rate 的方向（P(y_cls=0)），这样收缩前后只有 tau 一个变量
    不同；对 MLP 而言方向无所谓（首层权重可翻符号），但裸查表要算 AUC 就得取
    P(y_cls=1) = 1 - r0。注意它与 _label_rate 并非逐元相等：后者的兜底填错了方向
    （见 probe_te 里那行对比），所以 tau=0 只在有邻居的行上相等。
    """
    n = agg[:, NF].clamp(min=0)
    pc = float(y_cls[tr2].float().mean())
    pm = float(lev[tr2].float().mean())
    a0 = agg[:, NF + 1]
    a1 = (agg[:, NF + 3:NCOL] * torch.arange(3.)).sum(1)
    den, ok = (n + tau).clamp(min=1e-9), (n + tau) > 0
    return (torch.where(ok, (a0 + tau * (1 - pc)) / den, torch.full_like(a0, 1 - pc)),
            torch.where(ok, (a1 + tau * pm) / den, torch.full_like(a1, pm)))


def probe_te(name: str, epochs: int = 80, seeds: Sequence[int] = (0,),
             taus: Sequence[float] = (0, 1, 2, 3, 5, 10, 20, 30, 100, 300),
             mtaus: Sequence[float] = (3., 10., 100.)) -> None:
    """只改「怎么喂」能不能把目标编码那 0.016 的倒贴补回来 —— 拟合通路的最小对照。

    §9.9 配对之后剩下的唯一硬事实：一张单练值 0.7991 的格元标签率表，喂进 MLP 得
    0.7996 —— 零边际贡献，且相对地板倒贴 0.0163±0.0080。§9.10 又证了这张表到 test 年
    依然有效（最细粒度 AUC 只掉 0.0003）⇒ 信息没丢。本节点名其三位地检验「丢在拟合
    通路」这句归因，不改模型、不加参数，只改喂进去的那几列。

    假设：小计数格元的 a/n 是噪声，而 MLP 没有任何输入能告诉它「这一格该信几分」——
    率与计数被压进了同一列，模型只能一律照抄或一律忽略。若此说成立，两种改法应各自
    起作用且可叠加：（i）率按计数收缩（经验贝叶斯 / k-smoothing），小格元自动退回全局
    先验；（ii）单给一列 log(count)，让模型自己做可信度门控。

    三段判据，顺序不能颠倒：
      1 裸查表侧收缩是否变好（与模型无关，纯估计量质量）。若不涨，则「小计数是主要
        病因」先倒，下面 MLP 侧怎么改都不该期待；
      2 裸表涨而 MLP 拉不回 >=0 ⇒ 归因要改写：不是估计量差，是 MLP 吃不下这种列
        （要显式门控或专用读出），这比「拟合通路」更具体，也更贵；
      3 只给 log(count) 不给率 —— 必要的分解，否则计数列自己的增益会被记到收缩头上。

    口径与 §9.9 严格对齐，以便直接引用它的无信息带：同样 x + onehot 打底、同样加 12 列
    （不足补 iid 噪声；噪声列在本函数内是同一批抽样，不保证与 §9.9 那次相同）、同样按
    种子与地板配对；格元表按全窗口池化、train 行 LOO。宽度税因此仍然适用。
    """
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    other = 'soy' if name == 'corn' else 'corn'
    canon = _canon([name, other])              # 与 probe_edges 同序同口径，格元 key 才能直比
    npow = {c: len(v) for c, v in canon.items()}
    d, _ = load(name)
    rec = d['record']
    x, y_cls, lev = rec.x.float(), rec.y_cls.long(), rec.loss_level.long()
    gid = attr_group_ids(d)
    tr, te = rec.train_mask.bool(), rec.test_mask.bool()
    va = gid['year'] == VAL_YEAR_ID
    tr2 = tr & ~va
    onehot = torch.cat([nn.functional.one_hot(gid[a], ng).float() for a, ng in ATTRS], 1)
    codes = _codes(d, _meta(name), canon)
    G = _nbr_tensor(x, y_cls, lev)
    gen = torch.Generator().manual_seed(0)

    print('=' * 72)
    print(f'### te @ {name}（目标编码怎么喂：收缩与计数列，{len(seeds)} 个种子，device {dev}）')
    print(f'格元表源 = train 行（剔 val 年）全窗口池化，train 行 LOO；test {int(te.sum()):,} 行。'
          f'下面每一行都是 x + onehot + 12 列，与 §9.9 同宽度，所以涨不涨仍按无信息对照判。')

    floor: Dict[str, List[float]] = {}

    def fits(tag, extra, tbl=''):
        xx = torch.cat([x, onehot] + ([extra] if extra is not None else []), 1)
        r1 = [_fit_eval(xx[tr2], y_cls[tr2], xx[va], y_cls[va], xx[te], y_cls[te],
                        lambda: _mlp(xx.size(1), 128, 3, 1), dev, epochs,
                        cls=True, seed=s) for s in seeds]
        r2 = [_fit_eval(xx[tr2], lev[tr2], xx[va], lev[va], xx[te], lev[te],
                        lambda: _mlp(xx.size(1), 128, 3, 3), dev, epochs, seed=s)
                for s in seeds]
        cur = dict(auc=[r['auc'] for r in r1], acc=[r['acc'] for r in r2])
        if not floor:
            floor.update(cur)                      # 第一次调用就是地板行
        d1 = [a - b for a, b in zip(cur['auc'], floor['auc'])]
        d2 = [a - b for a, b in zip(cur['acc'], floor['acc'])]
        print(f'    {tag:<44} T1 AUC {_pm(cur["auc"])} ({_dsp(d1)})  '
              f'PR {_pm([r["prauc"] for r in r1])}  '
              f'T2 acc {_pm(cur["acc"])} ({_dsp(d2)})  {tbl}')

    def nz(k):
        return torch.randn(x.size(0), k, generator=gen)

    print('\n-- 地板（同 §9.9 第一行）--')
    fits('x + onehot', None)

    cells = {}
    for tag, cols in [('state x stage', ('state', 'stage')),
                      ('state x month x stage', ('state', 'month', 'stage'))]:
        key, ncell = _cell(codes, cols, npow)
        tab = torch.zeros(ncell, NCOL).index_add_(0, key[tr2], G[tr2])
        agg = tab[key] - tr2.float().unsqueeze(1) * G              # LOO：train 行抠掉自己
        cells[tag] = agg
        nn_, n_te = agg[:, NF].clamp(min=0), agg[te, NF].clamp(min=0)
        seen = tab[:, NF][tab[:, NF] > 0]

        def qt(p, _s=seen):
            return float(torch.quantile(_s, p))

        print(f'\n-- {tag}：格元上限 {ncell:,}，窗口内出现 {int(seen.numel()):,} 个；'
              f'每格 train 数 p10 {qt(.1):,.0f} / 中位 {qt(.5):,.0f} / p90 {qt(.9):,.0f}；'
              f'test 行零邻居 {float((n_te == 0).float().mean()):.1%}、'
              f'落在 n<10 格的 {float((n_te < 10).float().mean()):.1%} --')
        # 不拿 assert 抹平差异：_label_rate 的第 0 列是 P(y_cls=0)，可它的零邻居兜底填的是
        # pc = P(y_cls=1) —— 方向错了一格。那 2.6 万行因此拿到的是反向的极端值，而这不是
        # 估计量的问题、是写代码的人的问题，所以单独留一行量它值多少，而不是默默修掉。
        lab = _label_rate(agg, y_cls, lev, tr2)
        raw0 = torch.stack(_te(agg, y_cls, lev, tr2, 0), 1)
        bad = int(((lab - raw0).abs() > 1e-6).any(1).sum())
        print(f'  与 §9.9 的 _label_rate 对比：不一致 {bad:,} 行，'
              f'零邻居 {int((nn_ == 0).sum()):,} 行（两者应当相等；差异全部来自兜底方向）')
        print('  裸查表（不进模型，分数 = 收缩后的 P(y_cls=1)；E[lev] 那列定不了阈值，'
              '只给秩相关）      tau     AUC    rho(E[lev], lev)')
        for t in taus:
            r0, r1 = _te(agg, y_cls, lev, tr2, t)
            print(f'                                          {t:>7} '
                  f'{roc_auc(y_cls[te], 1 - r0[te]):>7.4f}  '
                  f'{_spearman(r1[te], lev[te].float()):>7.4f}')
        print('  喂进 MLP（每行补齐 12 列）')
        logn = torch.log1p(nn_).unsqueeze(1)
        fits(f'{tag}  §9.9 原样（兜底反向）+ 10 噪声', torch.cat([lab, nz(10)], 1),
             tbl='与下行只差兜底方向')
        fits(f'{tag}  率(tau=0) + 10 噪声', torch.cat([raw0, nz(10)], 1),
             tbl=f'裸表 {roc_auc(y_cls[te], 1 - raw0[te, 0]):.4f}')
        fits(f'{tag}  率(tau=0) + logn + 9 噪声', torch.cat([raw0, logn, nz(9)], 1))
        fits(f'{tag}  仅 logn + 11 噪声', torch.cat([logn, nz(11)], 1))
        for t in mtaus:
            s0, s1 = _te(agg, y_cls, lev, tr2, t)
            shr2 = torch.stack([s0, s1], 1)
            fits(f'{tag}  率(tau={t:g}) + 10 噪声', torch.cat([shr2, nz(10)], 1),
                 tbl=f'裸表 {roc_auc(y_cls[te], 1 - s0[te]):.4f}')
            fits(f'{tag}  率(tau={t:g}) + logn + 9 噪声', torch.cat([shr2, logn, nz(9)], 1))

    print('\n-- 无信息对照（判上面所有行用的尺子）--')
    fits('iid 噪声 12 列', nz(12))
    perm = torch.randperm(x.size(0), generator=gen)
    for tag in ('state x stage', 'state x month x stage'):
        r0, r1 = _te(cells[tag], y_cls, lev, tr2, 0)
        fits(f'{tag}  率(tau=0) 的行置换 + 10 噪声',
             torch.cat([torch.stack([r0, r1], 1)[perm], nz(10)], 1))

    print('\n读数：先看裸表那一列随 tau 的形状（它是估计量自己的质量，与模型无关），'
          '再看 MLP 那几行的配对差能否回到无信息带之内。两者要一起看：裸表涨而 MLP 不涨，'
          '说的是「MLP 吃不下这列」；裸表不涨而 MLP 涨，那涨的是宽度而非信息，'
          '必须用行置换那一行否掉。单种子时括号里的差不可信，定 0.01 以下的结论请带 '
          '`--seeds 0,1,2`。')


# ------------------------------------------------------------------ CLI ------

def main(argv: Sequence[str] | None = None) -> int:
    # 控制台默认 GBK，⇒/≤ 一类符号会直接抛 UnicodeEncodeError 把跑了一半的体检干掉
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='SOY/CORN 数据体检（诊断，不训练）')
    ap.add_argument('--dataset', default='both', choices=['corn', 'soy', 'both'])
    ap.add_argument('--what', default='structure,floors',
                    help='逗号分隔：structure / cost / hub / floors / edges / drift / te / all')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--steps', type=int, default=200,
                    help='hub 那段的训练步数，0 = 只报 init 的读数')
    ap.add_argument('--k', type=int, default=10, help='edges 那段 kNN 的 k')
    ap.add_argument('--seeds', default='0',
                    help='edges / te 那段的种子列表（逗号分隔）；多于一个时逐行报与地板配对差的 mean±std')
    a = ap.parse_args(argv)
    seeds = tuple(int(s) for s in a.seeds.split(','))
    steps = {'structure': probe_structure, 'cost': probe_cost,
             'hub': lambda nm: probe_hub(nm, steps=a.steps),
             'floors': lambda nm: probe_floors(nm, a.epochs),
             'edges': lambda nm: probe_edges(nm, a.epochs, a.k, seeds=seeds),
             'drift': probe_drift,
             'te': lambda nm: probe_te(nm, a.epochs, seeds=seeds)}
    what = list(steps) if a.what == 'all' else [w.strip() for w in a.what.split(',')]
    unknown = [w for w in what if w not in steps]
    if unknown:
        ap.error(f'--what 未知项 {unknown}，可选 {list(steps)} 或 all')
    names = ['corn', 'soy'] if a.dataset == 'both' else [a.dataset]
    for w in what:
        for n in names:
            steps[w](n)
    return 0


if __name__ == '__main__':
    warnings.filterwarnings('ignore')      # torch.jit 的 FutureWarning 会污染输出
    raise SystemExit(main())
