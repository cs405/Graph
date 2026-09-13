"""数据层：可控合成图的统计正确性 + split 不泄漏。

合成图是「λ 随局部异配度上升」这一主张的唯一**受控**检验场所：只有在这里
真实同质性是已知的，才能判断指标算得对不对。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                     # noqa: E402

from dataset import (GraphBundle,  # noqa: E402
                     ba_graph, dup_directed_graph, hetero_bundle,
                     load_dataset, random_split)
from dataset.base import row_normalize                             # noqa: E402
from metrics.homophily import edge_homophily, local_homophily      # noqa: E402
from modules.oca import OCAConfig, OCALayer                        # noqa: E402
from test.support import case, run_registered                      # noqa: E402


@case
def test_random_split_is_stratified_and_disjoint():
    y = torch.arange(4).repeat_interleave(25)
    s = random_split(y, 4, train_per_class=5, val_per_class=5, seed=3)
    tr, va, te = s['train'], s['val'], s['test']
    assert int(tr.sum()) == 20 and int(va.sum()) == 20
    for c in range(4):
        m = (y == c)
        assert int((tr & m).sum()) == 5 and int((va & m).sum()) == 5, c
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert (tr | va | te).all(), '剩下的都该进 test'
    # 同种子 => 逐元素一致；换种子 => 确实变了（否则「多种子」是假的）
    s2 = random_split(y, 4, 5, 5, seed=3)
    assert torch.equal(s2['train'], tr)
    assert not torch.equal(random_split(y, 4, 5, 5, seed=4)['train'], tr)
    print('  分层随机划分 OK（互斥、每类定量、同种子可复现）')


@case
def test_split_rejects_too_few_samples():
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    try:
        random_split(y, 2, train_per_class=5, val_per_class=5)
    except ValueError:
        pass
    else:
        raise AssertionError('样本不够时应抛错，不能静默少给')
    print('  样本不足时抛错 OK')


@case
def test_hetero_bundle_homophily_is_actually_controlled():
    """目标 h 与实测 h 必须单调同向，否则「受控实验」是假的。"""
    rows = []
    for h in (0.1, 0.5, 0.9):
        ds = hetero_bundle(num_classes=4, n_per_class=40, avg_deg=8,
                           homophily=h, seed=0)
        meas = edge_homophily(ds.edge_index, ds.y)
        rows.append((h, meas))
        assert abs(meas - h) < 0.25, (h, meas)
        ds.assert_split_disjoint()
    assert rows[0][1] < rows[1][1] < rows[2][1], rows
    print('  合成图同质性可控 OK   ' + '  '.join(
        f'目标{a:.1f}->实测{b:.3f}' for a, b in rows))


@case
def test_synthetic_graphs_are_reproducible():
    r"""同种子必须逐位同一张图 —— 这一条是被实测坑出来的。

    PyG 的 ``barabasi_albert_graph`` 用 ``np.random.choice`` 选挂载点，而
    ``torch.manual_seed`` 管不到它：早期 :func:`dataset.synthetic.ba_graph`
    只种了 torch，导致「同种子」的 BA 图每次运行都不同，所有基于 hub 度数
    的判据都在飘。换种子则必须变，否则「多种子统计」是假的。
    """
    def het(s):
        d = hetero_bundle(num_classes=3, n_per_class=20, avg_deg=4, seed=s)
        return d.x, d.edge_index              # 三个构造器统一返回 (x, ei)

    for make in (lambda s: ba_graph(n=60, m=3, seed=s),
                 lambda s: dup_directed_graph(n=30, seed=s),
                 het):
        a, b, c = make(7), make(7), make(8)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), \
            '同种子得到了不同的图'
        assert not (torch.equal(a[0], c[0]) and torch.equal(a[1], c[1])), \
            '换种子图却没变'
    print('  合成图可复现 OK（BA / 重边图 / 带标签图的特征与边集均逐位一致）')


@case
def test_local_homophily_definition():
    """手推一个小图：同类的 h=1、异类的 h=0、半同半异的 h=0.5。

    孤立节点必须出现在 ``valid=False`` 里，而不是被填 0 当成「完全异配」。
    """
    # 边（双向）：0-1（同为类 0）、0-2（跨类）、3-4（跨类）
    ei = torch.tensor([[0, 1, 0, 2, 3, 4], [1, 0, 2, 0, 4, 3]])
    y = torch.tensor([0, 0, 1, 1, 0])
    h, valid = local_homophily(ei, y, 5)
    assert valid.tolist() == [True] * 5, valid
    # 节点 0 的邻居 {1,2}：1 同类、2 异类 => 0.5；1 的邻居 {0} => 1；2 => 0
    assert abs(h[0] - 0.5) < 1e-12 and h[1] == 1.0 and h[2] == 0.0, h
    assert h[3] == 0.0 and h[4] == 0.0, h
    assert abs(edge_homophily(ei, y) - 2 / 6) < 1e-12, edge_homophily(ei, y)
    # node_mask 只统计指定中心发出的边
    only0 = edge_homophily(ei, y, node_mask=torch.tensor(
        [True, False, False, False, False]))
    assert abs(only0 - 0.5) < 1e-12, only0
    print('  局部/全局同质性定义 OK（含孤立节点被剔除、mask 只作用中心端）')


@case
def test_row_normalize_handles_zero_rows():
    x = torch.tensor([[1.0, 3.0], [0.0, 0.0]])
    out = row_normalize(x)
    assert torch.allclose(out[0], torch.tensor([0.25, 0.75]))
    assert torch.isfinite(out).all() and out[1].abs().sum() == 0.0
    print('  行归一化 OK（全零行给 0 向量，不产 NaN）')


@case
def test_acad_bench_pooling_is_neighbor_mean_no_selfloop():
    r"""``floor+`` 的聚合口径：邻居特征均值、不含自环、按度归一。

    这一条是 §9.13 整张基线表的地基 —— ``floor+`` 与 ``gcn`` 的差值全解释在它身上。
    拿手推的小图钉住四件事：度不等的中心、孤立点给 0 向量而不是 NaN、前 $F$ 列原样保留。
    """
    from dataset.acad_bench import _pooled
    x = torch.tensor([[1.0, 0.0], [3.0, 2.0], [0.0, 4.0], [5.0, 5.0]])
    # 边（双向）：0-1、1-2 ⇒ 节点 1 度 2，节点 3 孤立
    ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    b = lambda v: torch.tensor(v, dtype=torch.bool)
    ds = GraphBundle(name='tiny', x=x, edge_index=ei,
                     y=torch.zeros(4, dtype=torch.long), num_classes=1,
                     train_mask=b([1, 1, 0, 0]), val_mask=b([0, 0, 1, 0]),
                     test_mask=b([0, 0, 0, 1]))
    xp = _pooled(ds).x
    want = torch.tensor([[3.0, 2.0],         # 0 的邻域只有 1
                         [0.5, 2.0],         # 1 的邻域 {0,2} 取均值
                         [3.0, 2.0],         # 2 的邻域只有 1
                         [0.0, 0.0]])        # 孤立点：0 向量，不是 NaN
    assert torch.allclose(xp[:, 2:], want), xp
    assert torch.equal(xp[:, :2], x), '前 $F$ 列应该原样保留 $x$'
    assert torch.isfinite(xp).all()
    print('  floor+ 聚合口径 OK（邻居均值 / 无自环 / 孤立点给 0）')


@case
def test_acad_bench_archive_keeps_paired_units():
    r"""存档的逐配对单元必须与本次的（折, 种子）一致才对得上 —— §9.5 那个配对漏洞的护栏。

    ``--base`` 允许基准行从存档读（加一行不必重跑全表），但配对口径一断，$\Delta$
    就变成两个不同单位集合的均值差 —— 那正是本项目已经栽过一次的地方，所以把
    「对不上就拒」钉成用例：折不同、种子不同、行名不在存档里、路径没给，都得拒。
    """
    import json
    import tempfile
    from config import TrainConfig, load_train_config
    from dataset.acad_bench import dump_units, load_units

    proto = load_train_config(**{'run.save': False})
    assert isinstance(proto, TrainConfig)
    acc = {f'mlp/{m}': list(v) for m, v in
           dict(acc=[0.5, 0.6], f1=[0.4, 0.5], val=[0.55, 0.65], ep=[3.0, 4.0],
                par=[100.0, 100.0], sec=[1.0, 1.0]).items()}

    def rejected(archive, name, base, folds, seeds, why):
        try:
            load_units(archive, name, base, folds, seeds)
        except SystemExit:
            return
        raise AssertionError(f'{why} 本该被拒')

    with tempfile.TemporaryDirectory() as td:
        p = dump_units(td, 'texas', acc, ['mlp'], [0, 0], [0, 1], proto,
                       {'mlp': (0.01, 0.0)})
        assert os.path.exists(p), p
        got = load_units(td, 'texas', 'mlp', [0, 0], [0, 1])
        assert got['mlp/acc'] == [0.5, 0.6], got      # 顺序就是写入顺序
        assert got['mlp/ep'] == [3.0, 4.0], got
        # 协议指纹按行存，不是按文件存（否则补跑会覆掉基线行的出身）
        with open(p, encoding='utf-8') as fh:
            row = json.load(fh)['rows']['mlp']
        assert row['folds'] == [0, 0] and row['seeds'] == [0, 1], row
        assert (row['lr'], row['wd']) == (0.01, 0.0), row
        rejected(td, 'texas', 'mlp', [0], [0, 1], '只有 1 折')
        rejected(td, 'texas', 'mlp', [0, 0], [0], '只有 1 个种子')
        rejected(td, 'texas', 'mlp', [0, 1], [0, 1], '折号不同但个数相同')
        rejected(td, 'texas', 'gcn', [0, 0], [0, 1], '存档里没这一行')
        rejected('', 'texas', 'mlp', [0, 0], [0, 1], '没给 --archive')
        rejected(td, 'chameleon', 'mlp', [0, 0], [0, 1], '另一张图的文件')
    print('  存档 roundtrip OK（units 对齐才给 Δ；折/种子/行名/路径不匹配都拒）')


@case
def test_real_dataset_optional():
    """真实数据集：已预下载则体检，否则 SKIP。

    查的是 chameleon：论文主表已按裁决定为 geom-gcn 六张（§10 第 10 行），而
    Planetoid 那批在本机拿不到（``github.com:443`` 不通），拿它当探针会恒 SKIP。
    不去现场拉取：源站偶尔会挂住（实测卡 90s 后才报错），测试集不能依赖它。
    """
    from torch_geometric.utils import coalesce
    import glob
    key = 'chameleon'
    # 已预下载 = 本地有 processed/data.pt。不写死一层：WikipediaNetwork 落在
    # data/chameleon/geom_gcn/processed，WebKB 落在 data/texas/processed，深度不一样。
    got = glob.glob(os.path.join('data', key, 'processed', 'data.pt')) + \
        glob.glob(os.path.join('data', key, '*', 'processed', 'data.pt'))
    if not got:
        print(f'  [SKIP] {key} 未预下载（先跑一次 train.py --dataset {key}）')
        return
    ds = load_dataset(key, root='data', fallback_to_synth=False)
    s = ds.stats()
    assert s['mask_overlap'] == 0.0 and 0.0 < s['edge_homophily'] < 0.5, s
    assert ds.source == 'real' and ds.num_classes == 5, ds.meta
    assert str(ds.meta['split_source']).startswith('official-10fold'), ds.meta
    # `cfg/models/gat.yaml` 的注释要求「数据已是无向图」（基线不做 symmetrize），
    # 这一条把那个前提从注释变成断言：real.py 里的 to_undirected 不做了就会挂。
    n = ds.num_nodes
    assert torch.equal(coalesce(ds.edge_index, num_nodes=n),
                       coalesce(ds.edge_index.flip(0), num_nodes=n)), \
        'edge_index 不是无向的，GAT/GCN 基线与 OCA 的邻域口径会错位'
    layer = OCALayer(ds.num_features, OCAConfig(out_dim=16, heads=2, T=2)).eval()
    with torch.no_grad():
        out = layer(ds.x, ds.edge_index)      # 全图：截边集会让节点下标越界
    assert out.shape == (ds.num_nodes, 16) and torch.isfinite(out).all()
    print(f'  {key} 体检 OK   N={int(s["num_nodes"])} E={int(s["num_edges"])} '
          f'h={s["edge_homophily"]:.3f} split={ds.meta["split_source"]}（边集已无向）')


if __name__ == '__main__':
    sys.exit(run_registered('== 数据层 =='))
