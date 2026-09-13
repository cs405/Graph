"""训练层端到端：合成图上确实能学、可复现、选模型只用 val。

这一组用例保证「管线可用」，但**不构成论文证据** —— 数据是合成的
（见 :mod:`dataset.synthetic` 的警告）。真实数字必须来自 ``python train.py --dataset ...``。

配置全部走 ``cfg/train/default.yaml`` + 覆盖项（与命令行同一条路），这样
``default.yaml`` 一改，这里的用例就会跟着动 —— 不会出现「测试里的默认值和
真实默认值不是一回事」。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace                                   # noqa: E402

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

from config import TrainConfig, load_train_config                 # noqa: E402
from dataset import GraphBundle, hetero_bundle                    # noqa: E402
from test.support import case, run_registered                        # noqa: E402
from training import (build, collect_gates, dump_gates, fit,            # noqa: E402
                      gate_report, load_checkpoint, run_ablation, run_once)


def small_cfg(**kw) -> TrainConfig:
    r"""配置走 ``cfg/train/default.yaml`` + 覆盖项（与 CLI 同一条路）。

    这里**没有** ``num_layers`` / ``hidden_dim`` 了 —— 层数与宽度由 ``model.spec``
    指向的结构表决定；``run.save=False`` 是硬要求，测试不该往 ``runs/`` 堆产物。
    """
    base = {'model.spec': 'cfg/models/oca.yaml', 'model.scale': 'n',
            'model.dropout': 0.5, 'model.oca': {'T': 2, 'heads': 4},
            'dataset.name': 'synth', 'dataset.train_per_class': 10,
            'optimization.lr': 0.01, 'optimization.weight_decay': 5e-4,
            'optimization.epochs': 60, 'optimization.patience': 30,
            'run.seed': 0, 'run.device': 'cpu', 'run.save': False}
    base.update(kw)
    return load_train_config(**base)


def small_ds(**kw) -> GraphBundle:
    r"""故意造得**不易**：sep 小 + 异配 + 每类只给 8 个训练节点。

    合成图若能让模型随便刷到 1.0，所有消融行全部饱和，用例就只验证了
    「代码不报错过」，看不出开关有没有接上。
    """
    base = dict(num_classes=4, n_per_class=30, avg_deg=6, homophily=0.3,
                feat_dim=16, sep=1.0, train_per_class=8, val_per_class=8,
                seed=0)
    base.update(kw)
    return hetero_bundle(**base)


@case
def test_training_learns_and_selects_by_val():
    ds = small_ds()
    res = run_once(small_cfg(), ds=ds)
    chance = 0.25
    assert res.test['acc'] > chance + 0.1, res.test
    assert res.history[-1]['train_acc'] >= res.history[0]['train_acc']
    best = max(h['val_acc'] for h in res.history)
    assert res.history[res.best_epoch - 1]['val_acc'] == best, \
        '报告的 epoch 必须是 val 最优（不能拿 test 峰值）'
    assert res.val['acc'] >= best - 1e-12
    print(f'  训练可学 OK   test acc={res.test["acc"]:.4f}（chance {chance}）'
          f' f1={res.test["f1"]:.4f}  best_ep={res.best_epoch}/'
          f'{res.epochs_ran}  params={res.n_params}')


@case
def test_same_seed_is_reproducible():
    """同种子两次必须逐位一致，否则「±0.3 个点」的差异毫无意义。"""
    ds = small_ds()
    a = run_once(small_cfg(seed=5), ds=ds).test['acc']
    b = run_once(small_cfg(seed=5), ds=ds).test['acc']
    c = run_once(small_cfg(seed=6), ds=ds).test['acc']
    assert a == b, (a, b)
    print(f'  可复现 OK   seed5: {a:.6f} == {b:.6f}（seed6 为 {c:.6f}）')


@case
def test_diagnostics_shapes_and_gate_stats():
    """collect_gates 的每个量都是 [L, N]；gate_report 能读到 λ 分布与逐层相关性。

    故意在**训练后**的模型上取数：随机初始化下 λ 几乎等于 bias 对应的常数，
    相关性量的是噪声，看不出取数逻辑对不对。
    """
    cfg, ds = small_cfg(), small_ds()
    m = build(cfg, ds, ds.x.device)
    fit(m, ds, cfg)
    L = len(m.oca_layers)                       # 层数来自结构表，不来自配置
    gates = collect_gates(m, ds)
    assert set(gates) >= {'lambda', 'alpha', 'tau', 's_abs'}, gates.keys()
    for k, v in gates.items():
        assert v.shape == (L, ds.num_nodes), (k, tuple(v.shape))
    rep = gate_report(m, ds)
    assert 'lambda_r_by_layer' in rep, rep.keys()
    assert len(rep['lambda_r_by_layer']) == L
    lam = gates['lambda']
    assert 0.0 <= lam.min() and lam.max() <= 1.0, (lam.min(), lam.max())
    print(f'  诊断取数 OK   {L} 层 λ: '
          f'{[round(float(lam[j].mean()), 3) for j in range(L)]}'
          f'  std={float(lam.std()):.3f}')


@case
def test_ablation_switches_actually_change_output():
    r"""开关必须是「活的」。

    不能只看 acc：合成任务一饱和，所有变体都停在 1.0，差异被掩盖。也不能
    比较两个分别 ``build`` 的模型：关掉某个开关会少建一层 Linear，后续随机
    初始化整体错位，输出必然不同 —— 测不出开关本身。正确做法是在**同一份权重**
    上翻转 ``layer.cfg``。
    """
    ds = small_ds()
    cfg = small_cfg(epochs=40, patience=20)
    names = ['full', 'no_competition', 'gat_degenerate']
    rows = run_ablation(cfg, variants=names, seeds=[0, 1], ds=ds)
    assert [r['variant'] for r in rows] == names, rows
    for r in rows:
        assert r['acc'] == r['acc'] and r['acc'] > 0.2, r      # 非 NaN、非塌成 0
    assert rows[2]['params'] < rows[0]['params'], '退化模式应更少参数（无门控/融合）'
    # λ 列只能来自 λ。实现在这里踩过一次：``no_competition`` 根本没有 lambda 键，
    # 如果按后缀猜键，那一行会填上 alpha 的相关性 —— 表里看不出来，但结论是假的。
    assert rows[1]['r_lambda_homo'] is None, rows[1]
    assert rows[1]['lambda_std'] is None, rows[1]
    assert rows[0]['r_lambda_homo'] is not None, rows[0]

    m = build(cfg, ds, ds.x.device).eval()
    lays = m.oca_layers
    originals = [l.cfg for l in lays]
    with torch.no_grad():
        base_out = m(ds.x, ds.edge_index)
        aux_keys = set(lays[0].aux)
        assert 'lambda' in aux_keys and 'alpha' in aux_keys, aux_keys
        for over in ({'use_competition': False}, {'T': 0}):
            for l, o in zip(lays, originals):
                l.cfg = replace(o, **over)
            flipped = m(ds.x, ds.edge_index)
            gap = float((base_out - flipped).abs().max())
            assert gap > 1e-12, (over, gap)
            if 'use_competition' in over:                 # 语义也要对：λ 直接不产
                assert 'lambda' not in lays[0].aux, 'use_competition=False 仍写入 λ'
            for l, o in zip(lays, originals):
                l.cfg = o
    print('  消融开关生效 OK   ' + '  '.join(
        f'{r["variant"]}={r["acc"]:.4f}±{r["acc_std"]:.4f}' for r in rows))


@case
def test_dump_gates_is_node_aligned():
    """npz 里每个数组都按节点对齐：同一下标 = 同一节点。

    画图脚本（λ vs 局部同质性散点）就靠这个对齐关系；一旦某个量按边排序，
    散点会照画不误、看起来也很合理，但它是错的 —— 所以在这里钉住形状。
    """
    import tempfile

    cfg, ds = small_cfg(), small_ds()
    m = build(cfg, ds, ds.x.device)
    fit(m, ds, cfg)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'gates.npz')
        dump_gates(p, m, ds)
        with np.load(p) as z:               # 必须关句柄：Windows 下不关闭就删不掉临时目录
            assert {'lambda', 'alpha', 'tau', 's_abs', 'h_local', 'h_valid',
                    'y', 'pred'} <= set(z.files), z.files
            L, N = len(m.oca_layers), ds.num_nodes
            assert z['lambda'].shape == (L, N), z['lambda'].shape
            assert z['h_local'].shape == (N,) and z['h_valid'].shape == (N,)
            assert np.array_equal(z['y'], ds.y.cpu().numpy()), 'y 与节点错位'
            assert z['pred'].shape == (N,) and z['pred'].max() < ds.num_classes
            assert np.isfinite(z['lambda']).all() and (z['lambda'] >= 0).all()
            print(f'  门控落盘 OK   lambda{tuple(z["lambda"].shape)}、'
                  f'h_local{tuple(z["h_local"].shape)} 均按节点对齐')


@case
def test_best_metric_guard_and_runs_artifacts():
    r"""``runs/<name>/train`` 的产物与「只能用 val 选模型」这两件事一起验。

    产物路径写错了不会报错，只会让人在三个月后找不到当初的权重，所以这里
    把 ``config_used.yaml`` / ``weights/best.pt`` / ``results.json`` / ``gates.npz``
    逐个存在性钉住。结束前必须 :func:`close_file_logging`：Windows 上不关
    日志句柄，临时目录删不掉（同 npz 那个坑）。
    """
    import tempfile
    from pathlib import Path

    from log_setup import close_file_logging
    from training import run_experiment

    # patience 比 epochs 大：这个用例要的是「跑满 8 轮」，不能提前早停
    cfg = small_cfg(epochs=8, patience=50, seeds=[0], ckpt_interval=4)
    assert cfg.save is False                       # 默认不写盘
    with tempfile.TemporaryDirectory() as td:
        conf = cfg.patched(project=td, save=True)
        try:
            out = run_experiment(conf, ds=small_ds())
        finally:
            close_file_logging()
        run_dir = Path(out['run_dir'])
        assert run_dir.parent == Path(td) / 'oca', run_dir
        for key in ('best', 'gates', 'results'):
            assert Path(out['paths'][key]).is_file(), (key, out['paths'])
        assert Path(out['paths']['config_used']).is_file()
        assert (run_dir / 'logs' / 'train.log').is_file()
        ckpt = sorted((run_dir / 'checkpoints').glob('checkpoint_e*.pth'))
        assert [p.name for p in ckpt] == ['checkpoint_e0004c0000.pth',
                                          'checkpoint_e0008c0000.pth'], ckpt
        payload = load_checkpoint(out['paths']['best'], device='cpu')
        assert payload['spec'] == conf.model.spec and payload['scale'] == 'n'
        assert payload['config']['optimization']['epochs'] == 8
        assert payload['in_dim'] and payload['nc']
        # 同一 project 再跑一次 → train2，不覆盖上一次
        out2 = run_experiment(conf, ds=small_ds())
        assert Path(out2['run_dir']).name == 'train2', out2['run_dir']
        close_file_logging()

    m = build(cfg, small_ds(), torch.device('cpu'))
    try:
        fit(m, small_ds(), cfg.patched(best_metric='test_acc', epochs=3))
    except ValueError as e:
        assert 'test_acc' in str(e), e
    else:
        raise AssertionError('best_metric=test_acc 没被拒 —— 数据泄漏防护失效')
    # val_loss 是合法选项，前提是 fit 真的算了它（否则选出来的 best_epoch 永远为空）
    r = fit(build(cfg, small_ds(), torch.device('cpu')), small_ds(),
            cfg.patched(best_metric='val_loss', epochs=5, patience=3))
    assert np.isfinite(r.val['loss']), r.val
    print(f'  runs 产物 / 选择护栏 OK   {run_dir.name}、val_loss 选模 '
          f'best_ep={r.best_epoch}')


if __name__ == '__main__':
    sys.exit(run_registered('== 训练层端到端 =='))
