r"""结构表 → 网络：``model_builder`` 的展开、路由、缩放与报错文案。

本文件是「图的结构按 cfg 搭建」这件事的验收标准。三件事必须钉住：

1. **拓扑**：``oca.yaml`` 的 neck 是哪 6 个 Merge、读出看哪 4 个尺度 —— 写死成断言。
   上一版这里出过事故：bottom-up 的两行下标写成 ``[[4,6]]``/``[[5,7]]``，
   自称「逐行核对过」，实际上把 D1/D0 重复加了一遍，而前向照样能跑、loss 照样降。
   能跑 ≠ 对，所以拓扑只能靠断言，不能靠眼睛。
2. **缩放**：width 改通道、depth 改 repeats、``max_channels`` 封顶。
3. **报错**：写错结构表时必须当场说清是哪一行、为什么，而不是留下一句
   ``TypeError: 'NoneType' object is not subscriptable``。
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                    # noqa: E402

from dataset.synthetic import ba_graph                          # noqa: E402
from model_builder import (MODULES, GraphSequential, Repeat,    # noqa: E402
                           build_model, make_divisible, model_info,
                           parse_model, yaml_model_load)
from modules.oca import OCAConfig, OCALayer                     # noqa: E402
from test.support import case, run_registered                   # noqa: E402

SPECS = ['oca', 'oca1', 'oca_deep', 'oca_gat', 'oca_nocomp', 'gat', 'gcn', 'mlp']
SPEC_PATH = {s: f'cfg/models/{s}.yaml' for s in SPECS}
# 基线结构表的逐行模块类名。注意与 YAML 里的注册名不同：``m.type`` 存的是
# **类名**（同 yolov8），而 YAML 写的是 PyG 风味的注册名（GATConv ↔ GATBlock）。
BASE_TYPE = {'gat': 'GATBlock', 'gcn': 'GCNBlock', 'mlp': 'LinearBlock'}


def _n_rows(spec: str) -> int:
    d = yaml_model_load(SPEC_PATH[spec])
    return len(d['backbone']) + len(d.get('head') or [])


# ---------------------------------------------------------------------------
# 构建与路由
# ---------------------------------------------------------------------------

@case
def test_all_specs_build_forward_and_route():
    """7 张结构表都能展开成网络，且 ``from`` 引用的行必须真的被缓存。

    ``GraphSequential`` 只把 ``save`` 里的行号存进特征缓存；引用一个没缓存的行不会
    报错，只会拿到 ``None``，然后在某个 ``torch.stack`` 里炸出一句跟结构表无关的话。
    所以这里直接校验不变式：每个输入的绝对行号 ∈ save（``-1`` 走运行值，例外）。
    """
    x, ei = ba_graph(n=64, m=2, seed=1, feat_dim=9)
    n, nc = x.size(0), 5
    for spec in SPECS:
        net = build_model(SPEC_PATH[spec], 9, nc=nc, verbose=False).eval()
        assert isinstance(net, GraphSequential), spec
        assert len(net.model) == _n_rows(spec), (
            f'{spec}: 展开成 {len(net.model)} 行，结构表写了 {_n_rows(spec)} 行')
        assert [m.i for m in net.model] == list(range(len(net.model))), spec
        refs = []
        for m in net.model:
            f = m.f if isinstance(m.f, list) else [m.f]
            refs += [int(j) for j in f if int(j) != -1]
        missing = [r for r in refs if r not in net.save]
        assert not missing, f'{spec}: from 引用了未缓存的行 {missing}'
        with torch.no_grad():
            out = net(x, ei)
        assert out.shape == (n, nc), (spec, tuple(out.shape))
        assert torch.isfinite(out).all(), spec
        assert net.num_classes == nc, spec
    print('  7 张结构表 build/forward/路由 OK   ' +
          '  '.join(f'{s}={_n_rows(s)}层' for s in SPECS))


@case
def test_oca_neck_topology_is_pinned():
    r"""把 ``oca.yaml`` 的 neck 拓扑写死（PANet 式双向融合，docs §六）。

    下标读法：backbone 的 0..3 是 4 个尺度 f0..f3（感受野 1..4 跳）。
    top-down 从最深处往下加：``4=f2+f3``、``5=f1+4``、``6=f0+5``；
    bottom-up 再往回：``7=D1+D0=[5,6]``、``8=D2+U1=[4,7]``、``9=f3+8``；
    读出只看 4 个**融合后**的结果 ``[6,7,8,9]``（不看 backbone，那 4 行已被融合吸收）。
    """
    d = yaml_model_load('cfg/models/oca.yaml')
    heads = [row[0] for row in d['head']]
    assert heads == [[2, 3], [1, 4], [0, 5], [5, 6], [4, 7], [3, 8],
                     [6, 7, 8, 9]], heads
    net = build_model(d, 12, nc=4, verbose=False)
    assert [m.f for m in net.model] == [-1, -1, -1, -1] + heads, \
        [m.f for m in net.model]
    assert [m.type for m in net.model] == ['OCAConv'] * 4 + ['Merge'] * 6 + \
        ['Classify']
    assert net.save == set(range(10)), net.save       # 最后一行的输出没人再引用
    assert len(net.oca_layers) == 4
    widths = [m.out_dim for m in net.model]
    assert widths == [16] * 10 + [4], widths           # scale n => 64*0.25
    print(f'  oca.yaml 拓扑钉死 OK   11 行 / save={net.save} / 宽度序列 {widths}')


@case
def test_baseline_specs_are_single_path():
    """基线结构表没有融合：``save`` 应当是空集，``oca_layers`` 应当为空。

    「基线与 OCA 同一条构建路径」是 §八 对照表可信的前提 —— 如果基线也悄悄走了
    Merge/Classify 之外的胶水代码，那两组数字就不是同一个东西训出来的。
    """
    for spec in ('gat', 'gcn', 'mlp'):
        net = build_model(SPEC_PATH[spec], 9, nc=5, verbose=False)
        assert not net.save, (spec, net.save)
        assert len(net.oca_layers) == 0, spec
        assert [m.type for m in net.model] == \
            [BASE_TYPE[spec]] * 2 + ['Classify'], (spec, [m.type for m in net.model])
    print('  gat/gcn/mlp 单链路 OK（无融合、无 OCA 层）')


@case
def test_registry_is_the_only_entrypoint():
    """YAML 里能出现的模块名恰好是 :data:`MODULES` 的键 —— 注册表是唯一入口。

    注册表由算子自己填（``@register_module``），builder 只读：这张集合就是
    「框架当前认识哪些结构层」的全部事实，多一个少一个都是结构性的变动，
    所以写死成断言（新增一个算子 = 改这一行 = 审阅时看得见）。
    """
    assert set(MODULES) == {'OCAConv', 'GATConv', 'GCNConv', 'Linear', 'Merge',
                            'Concat', 'Classify', 'DIAConv', 'EdgeScore'}, \
        sorted(MODULES)
    # 注册表必须与「算子自己登记」一致：不能有人绕过装饰器往字典里塞东西
    from modules.base import MODULES as REGISTRY
    assert REGISTRY is MODULES, 'builder 里的 MODULES 必须是 modules.base 的那一份'
    assert all(getattr(c, 'op_name', None) == k for k, c in MODULES.items()), \
        [k for k, c in MODULES.items() if getattr(c, 'op_name', None) != k]
    used = {row[2] for s in SPECS for row in
            (yaml_model_load(SPEC_PATH[s])['backbone']
             + (yaml_model_load(SPEC_PATH[s]).get('head') or []))}
    assert used <= set(MODULES), used - set(MODULES)
    assert 'Concat' not in used, '内置结构表目下不用 Concat（等宽融合用 Merge）'
    print(f'  模块注册表 OK（{len(MODULES)} 个，结构表实际用到 {sorted(used)}）')


@case
def test_concat_row_derives_its_width():
    """``Concat`` 不写 args：输出宽度 = 各输入宽度之和（builder 自己推）。

    写 ``Concat, [64]`` 是错的（会被当成可缩放的 c2），所以这里既要验证
    ``Concat`` 行能省 args，也要验证下一层拿到的 c1 是拼接后的总宽。
    """
    d = {'nc': 3, 'scales': {'n': [1.0, 1.0, 1024]},
         'backbone': [[-1, 1, 'OCAConv', [64]], [-1, 1, 'OCAConv', [64]]],
         'head': [[[0, 1], 1, 'Concat'], [-1, 1, 'Classify', ['nc']]]}
    net, save = parse_model(d, 12, nc=3, scale='n', verbose=False)
    assert [m.out_dim for m in net.model] == [64, 64, 128, 3], \
        [m.out_dim for m in net.model]
    assert save == [0, 1], save                       # Classify 走 -1，不需缓存第 3 行
    x, ei = ba_graph(n=40, m=2, seed=1, feat_dim=12)
    with torch.no_grad():
        assert net.eval()(x, ei).shape == (x.size(0), 3)
    assert net.yaml['nc'] == 3 and net.yaml['scale'] == 'n'
    print('  Concat 行（省 args）OK   64+64 -> 128 -> Classify(3)')


@case
def test_uneven_merge_is_rejected_at_build():
    """不同宽尺度用 ``Merge`` 必须在构建时就报错，而不是等 forward 时广播失败。

    把行下标写错（比如把 backbone 的第 0 行与融合后的第 8 行相加）恰好会踩到这个，
    而一旦进了 forward，报的是 ``torch.stack`` 的 shape 错，离结构表已经隔了一层。
    """
    d = {'nc': 3, 'scales': {'n': [1.0, 1.0, 1024]},
         'backbone': [[-1, 1, 'OCAConv', [32]], [-1, 1, 'OCAConv', [64]]],
         'head': [[[0, 1], 1, 'Merge', [64]]]}
    try:
        parse_model(d, 12, nc=3, scale='n', verbose=False)
    except AssertionError as e:
        assert '同宽' in str(e) and 'Concat' in str(e), e
    else:
        raise AssertionError('不等宽 Merge 没在构建时报错')
    print('  Merge 宽度护栏 OK（提醒改用 Concat）')


# ---------------------------------------------------------------------------
# 缩放
# ---------------------------------------------------------------------------

@case
def test_compound_scaling_width_and_depth():
    """width 改通道、depth 改 repeats；两者互不干扰。

    ``oca.yaml`` 每行 repeats 都是 1（多尺度必须逐层编号），所以它的 n/s/m/l/x
    **只差宽度** —— 这是刻意的，也要在文档里写明，否则读者会以为 scale 之间
    深度也变了。深度缩放由 ``oca1``/``oca_deep`` 示范。
    """
    want_w = {'n': 16, 's': 32, 'm': 48, 'l': 64, 'x': 80}
    params = {}
    for sc, want in want_w.items():
        net = build_model('cfg/models/oca.yaml', 12, nc=4, scale=sc,
                          verbose=False)
        assert net.model[0].out_dim == want, (sc, net.model[0].out_dim)
        assert net.model[-1].dims == [want] * 4, net.model[-1].dims
        params[sc] = model_info(net, verbose=False)['params_total']
    assert [params[s] for s in want_w] == sorted(params.values()), params
    assert len(build_model('cfg/models/oca.yaml', 12, nc=4, scale='l',
                           verbose=False).oca_layers) == 4, 'oca.yaml 深度不随 scale 变'

    for spec, want_n in (('oca1', 2), ('oca_deep', 4)):
        shallow = build_model(SPEC_PATH[spec], 12, nc=4, scale='n', verbose=False)
        deep = build_model(SPEC_PATH[spec], 12, nc=4, scale='l', verbose=False)
        assert len(shallow.oca_layers) == want_n, (spec, len(shallow.oca_layers))
        assert len(deep.oca_layers) > want_n, (spec, len(deep.oca_layers))
        # 堆叠发生在行内部的 Repeat 里，而行号不变（一个节点只有一个输出编号）
        assert len(deep.model) == len(shallow.model) == _n_rows(spec), spec
        stacked = [m for m in deep.model if isinstance(m, Repeat)]
        assert stacked and all(len(m.m) > 1 for m in stacked), stacked
        assert all(m.out_dim == m.m[-1].out_dim for m in stacked)
    d1 = build_model('cfg/models/oca1.yaml', 12, nc=4, scale='l', verbose=False)
    assert len(d1.oca_layers) == 5, len(d1.oca_layers)   # 1 + round(4*1.0)
    assert len(build_model('cfg/models/oca_deep.yaml', 12, nc=4, scale='l',
                           verbose=False).oca_layers) == 8
    assert not isinstance(build_model('cfg/models/oca1.yaml', 12, nc=4,
                                      scale='n', verbose=False).model[1], Repeat), \
        'scale n 下 round(4*0.33)=1，不该包成 Repeat'
    print(f'  缩放 OK   oca.yaml 宽度 {list(want_w.values())}，'
          f'参数 {params["n"]:,}→{params["x"]:,}；oca1 2→5、oca_deep 4→8')


@case
def test_max_channels_caps_width():
    """``max_channels`` 必须在缩放**之前**封顶，否则大 scale 会突破设计上限。"""
    d = {'nc': 3, 'scales': {'m': [1.0, 0.75, 512]},
         'backbone': [[-1, 1, 'OCAConv', [2048]]]}
    net = build_model(d, 8, nc=3, scale='m', verbose=False)
    assert net.model[0].out_dim == make_divisible(512 * 0.75) == 384, \
        net.model[0].out_dim
    assert net.model[0].layer.cfg.out_dim == 384
    print(f'  max_channels 封顶 OK   2048 -> {net.model[0].out_dim}')


# ---------------------------------------------------------------------------
# 超参注入
# ---------------------------------------------------------------------------

@case
def test_oca_overrides_flow_per_layer():
    """``build_model(oca=...)`` 是**逐键合并**到结构表的 ``oca:`` 块上，再传进每一层。

    优先级：CLI/参数 > 结构表。整块替换会把 ``heads``/``beta`` 一起清掉，
    那样消融表里 ``T4`` 与 ``full`` 的差值就不只是 T 的差值了。
    """
    net = build_model('cfg/models/oca.yaml', 12, nc=4, oca={'T': 8},
                      verbose=False)
    for lay in net.oca_layers:
        assert lay.cfg.T == 8, lay.cfg
        assert lay.cfg.heads == 4, lay.cfg          # 结构表里的 heads 没被冲掉
    base = build_model('cfg/models/oca.yaml', 12, nc=4, verbose=False)
    assert {l.cfg.T for l in base.oca_layers} == {2}

    # 行内 dict：同一网络里每层不同 T（结构侧消融的表达方式）
    d = yaml_model_load('cfg/models/oca.yaml')
    d['backbone'][2][3] = [64, {'T': 1}]
    mixed = build_model(d, 12, nc=4, verbose=False)
    ts = [l.cfg.T for l in mixed.oca_layers]
    assert ts == [2, 2, 1, 2], ts
    assert mixed.oca_layers[2].cfg.heads == 4, '行内 dict 不该丢掉全局 heads'

    drop = build_model('cfg/models/oca.yaml', 12, nc=4, dropout=0.0,
                       verbose=False)
    mods = [m for m in drop.model[-1].modules() if isinstance(m, torch.nn.Dropout)]
    assert mods and all(m.p == 0.0 for m in mods), [m.p for m in mods]
    print(f'  oca 覆盖 / 逐层差异 / dropout 注入 OK   T={ts}')


@case
def test_gat_yaml_matches_gat_equivalent():
    """``oca_gat.yaml`` 的开关组合必须与 :meth:`OCALayer.gat_equivalent` 完全一致。

    两份地方描述同一件事（七条件），迟早会漂移；漂移之后「同骨架的 GAT 对照」
    这一行就悄悄不等了 —— 它是全文最需要「严格」二字的地方。
    """
    net = build_model('cfg/models/oca_gat.yaml', 12, nc=4, verbose=False)
    got = net.oca_layers[0].cfg
    want = OCALayer.gat_equivalent(12, out_dim=16, heads=4).cfg
    assert isinstance(got, OCAConfig) and got == want, \
        {k: (getattr(got, k), getattr(want, k)) for k in got.__dict__
         if getattr(got, k) != getattr(want, k)}
    assert all(l.cfg == got for l in net.oca_layers), '同一结构表每层开关应一致'
    # 与外基线 GAT 的区别必须留在结构表里，而不是被 oca_gat 偷偷继承
    assert len(net.model) == _n_rows('oca_gat') == 11
    print('  oca_gat.yaml ≡ gat_equivalent OK（七条件逐字段相等）')


@case
def test_nocomp_yaml_differs_only_by_lambda():
    """``oca_nocomp.yaml`` 与 ``oca.yaml`` 的差必须**恰好是 use_competition 这一个键**。

    它是 §9.13 里 `oca` − `oca_nocomp` 那个归因差的载体：一旦结构表顺手改了第二个
    开关（或抄漏一行 backbone），差值就不再是「竞争项的净贡献」，而这类漂移正是
    靠人读 yaml 读不出来的。`oca_gat` 那行做不到这件事 —— 七条件把残差/LayerNorm
    也一起关掉了，于是它同时动了两个东西。
    """
    base = build_model('cfg/models/oca.yaml', 12, nc=4, verbose=False)
    net = build_model('cfg/models/oca_nocomp.yaml', 12, nc=4, verbose=False)
    assert len(net.model) == _n_rows('oca_nocomp') == _n_rows('oca') == 11
    for lb, ln in zip(base.oca_layers, net.oca_layers):
        ca, cb = lb.cfg, ln.cfg
        diff = {k for k in ca.__dict__ if getattr(ca, k) != getattr(cb, k)}
        assert diff == {'use_competition'}, (ca, cb, diff)
        assert cb.use_competition is False and ca.use_competition is True
    # 开关要真的落到参数上：w_lambda 不该还存在
    assert any('w_lambda' in k for k in base.state_dict()), 'oca 侧该有 w_lambda'
    assert not any('w_lambda' in k for k in net.state_dict()), 'nocomp 侧不该有 w_lambda'
    print('  oca_nocomp.yaml 与 oca.yaml 只差 use_competition OK（含 w_lambda 缺席）')


# ---------------------------------------------------------------------------
# 报错文案
# ---------------------------------------------------------------------------

@case
def test_structure_errors_point_at_the_row():
    """结构表写错时必须说清是哪一行 —— 否则一半时间花在猜哪行的下标错了。"""
    def expect(err, msg, d, **kw):
        try:
            build_model(d, 8, verbose=False, **kw)
        except err as e:
            assert msg in str(e), (msg, str(e))
            return str(e)
        raise AssertionError(f'应当抛 {err.__name__}（{msg}）：{d}')

    def rows(head_rows, back=1):
        return {'nc': 3, 'scales': {'n': [1.0, 1.0, 1024]},
                'backbone': [[-1, 1, 'OCAConv', [8]]] * back,
                'head': head_rows}

    t1 = expect(KeyError, 'FuseConv', rows([[-1, 1, 'FuseConv', [8]]]))
    assert 'Merge' in t1, t1                                   # 要列出候选名
    expect(ValueError, 'nc', {'nc': None,
                              'backbone': [[-1, 1, 'Classify', ['nc']]]})
    expect(ValueError, '多输入模块', rows([[[0, 0], 2, 'Merge', [8]]]))
    expect(ValueError, '尚未构建', rows([[[6, 7], 1, 'Merge', [8]]]))
    expect(ValueError, '越界', {'nc': 3, 'backbone': [[-3, 1, 'OCAConv', [8]]]})
    expect(ValueError, '[from, repeats, module',
           {'nc': 3, 'backbone': [[-1, 1]]})
    print('  结构表报错文案 OK（未注册模块/缺 nc/多输入 repeats/前向引用/越界/列数）')


@case
def test_scale_and_file_validation():
    """``--scale`` 打错、文件不存在，都要在构建前就停住。"""
    try:
        build_model('cfg/models/oca.yaml', 8, nc=3, scale='xl', verbose=False)
    except KeyError as e:
        assert 'xl' in str(e), e
    else:
        raise AssertionError('未知 scale 没被拒')
    try:
        yaml_model_load('cfg/models/nope.yaml')
    except FileNotFoundError as e:
        assert 'nope' in str(e), e
    else:
        raise AssertionError('缺文件没被拒')
    # 文件名里的 ``-n``/``_s`` 后缀 = scale（yolov8 惯例），但 ``gcn`` 的 n 不算
    assert yaml_model_load('cfg/models/gcn.yaml').get('scale') is None
    for name in SPECS:
        assert Path(SPEC_PATH[name]).is_file(), name
    print('  scale / 文件校验 OK（gcn.yaml 没被误判成 scale=n）')


@case
def test_verbose_table_lists_every_row():
    """``verbose=True`` 的逐层表必须一行不落地列出所有层（这是排错时的主要输出）。"""
    buf = io.StringIO()
    with redirect_stdout(buf):
        net = build_model('cfg/models/oca.yaml', 12, nc=4, verbose=True)
    text = buf.getvalue()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert 'from' in lines[0] and 'params' in lines[0], lines[0]
    assert len(lines) == 1 + len(net.model), (len(lines), len(net.model))
    for i, ln in enumerate(lines[1:]):
        assert ln.split()[0] == str(i), ln
    print(f'  逐层结构表 OK（{len(net.model)} 行 + 表头）')


if __name__ == '__main__':
    sys.exit(run_registered('== 结构表构建 / 缩放 / 报错 =='))
