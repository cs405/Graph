r"""消融与对照表（docs/OCA.md §八）。

原则：**一张表只动一个开关**，且全部通过 :class:`modules.oca.OCAConfig`（算子侧）
或 ``cfg/models/*.yaml``（结构侧）表达，不引入任何「只在脚本里存在」的临时改动 ——
否则审稿人复现不出来。

两类变体用同一个 dict 表达：值里出现 ``'oca'`` 键就是**算子消融**（合并进
``model.oca``），其余键按 :meth:`config.TrainConfig.patched` 的扁平名/``block.field``
覆盖，所以「换结构表」这种**结构消融**（单尺度对照）也能写进同一张表。

`'gat_degenerate'` 这一行是 §三 的七条件全套（连 `out_mode='gat'`、`score_mode='gat'`
都打开），它与外基线 GAT 的差值才是「同骨架、同参数量级」的诚实对照。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from config import TrainConfig
from dataset import GraphBundle
from training.trainer import run_seeds

__all__ = ['ABLATIONS', 'apply_variant', 'run_ablation', 'format_table']

# 变体名 -> 覆盖项。``oca`` 里的键是 OCAConfig 的覆盖项，顶层键是 TrainConfig 的。
ABLATIONS: Dict[str, Dict[str, Any]] = {
    'full': {},
    # ---- 算子消融（同一张结构表）----
    'no_competition': {'oca': {'use_competition': False}},         # \lambda := 0
    'no_proposal': {'oca': {'use_proposal': False}},               # 去掉纵向提案 b
    'no_phi': {'oca': {'use_phi': False}},                         # 去掉自身嗓门 \phi
    'no_center_field': {'oca': {'use_center_field': False}},       # \alpha := 0（纯 B）
    'center_in_field': {'oca': {'use_center_field': True,
                                'alpha_bias_init': 40.0}},         # 纯 C
    'no_temperature': {'oca': {'use_temperature': False}},         # \tau := 1
    'raw_center_stat': {'oca': {'center_stat': 'raw'}},            # §2.6 的替代方案
    'T0': {'oca': {'T': 0}},
    'T1': {'oca': {'T': 1}},
    'T2': {'oca': {'T': 2}},
    'T4': {'oca': {'T': 4}},
    'T8': {'oca': {'T': 8}},
    'detach_iters': {'oca': {'detach_iterations': True}},          # §7 的 stop-gradient
    'gat_degenerate': {'oca': {'T': 0, 'score_mode': 'gat', 'use_phi': False,
                               'use_competition': False,
                               'use_center_field': False,
                               'use_temperature': False,
                               'out_mode': 'gat'}},
    # ---- 结构消融（同一套算子开关）----
    'single_scale': {'model.spec': 'cfg/models/oca1.yaml'},        # 去掉 neck 融合
    'scale_s': {'model.scale': 's'},                               # 只改宽度，看是否白涨
    'scale_l': {'model.scale': 'l'},
}


def apply_variant(cfg: TrainConfig, over: Dict[str, Any]) -> TrainConfig:
    """把一行变体应用到 ``cfg`` 上；``oca`` 是**合并**而非整块替换。"""
    over = dict(over)
    oca = over.pop('oca', None)
    out = cfg.with_oca(**oca) if oca else cfg
    return out.patched(**over) if over else out


def _row(variant: str, out: Dict[str, Any], spec: str = '') -> Dict[str, Any]:
    acc, f1 = out['acc'], out['f1']
    diag = out['runs'][0].diagnostics if out['runs'] else {}
    r = {'variant': variant, 'spec': _short(spec),
         'acc': round(acc['mean'], 4), 'acc_std': round(acc['std'], 4),
         'f1': round(f1['mean'], 4), 'f1_std': round(f1['std'], 4),
         'params': out['runs'][0].n_params if out['runs'] else 0,
         'best_ep': round(out['best_epoch'], 1)}
    # 必须带 ``lambda_`` 前缀：否则 ``no_competition`` 行会把 alpha 的相关性
    # 当成 lambda 报进表里（它根本没有 lambda 键），正好是论文表格里最难发现的那类错。
    lam_r = [v for k, v in diag.items()
             if k.startswith('lambda_') and k.endswith('_r_homo')]
    lam_s = [v for k, v in diag.items()
             if k.startswith('lambda_') and k.endswith('_std')]
    r['lambda_std'] = round(lam_s[0], 4) if lam_s else None
    r['r_lambda_homo'] = round(lam_r[0], 4) if lam_r else None
    return r


def _short(spec: str) -> str:
    """``cfg/models/oca_gat.yaml`` -> ``oca_gat``：表里放全路径会把列宽撑爆。"""
    return str(spec).replace('\\', '/').rsplit('/', 1)[-1].rsplit('.', 1)[0]


def run_ablation(cfg: TrainConfig, variants: Optional[Iterable[str]] = None,
                 seeds: Optional[List[int]] = None,
                 ds: Optional[GraphBundle] = None,
                 extra: Optional[Dict[str, Dict[str, Any]]] = None,
                 verbose: bool = False) -> List[Dict[str, Any]]:
    """跑一组消融，返回可直接 :func:`format_table` 的行列表。"""
    table: Dict[str, Dict[str, Any]] = dict(ABLATIONS)
    table.update(extra or {})
    names = list(variants) if variants else list(table)
    unknown = [v for v in names if v not in table]
    if unknown:
        raise KeyError(f'未知消融变体 {unknown}，可选：{sorted(table)}')
    rows = []
    for v in names:
        sub = apply_variant(cfg, table[v])
        out = run_seeds(sub, list(seeds or cfg.seeds), ds=ds, verbose=verbose)
        rows.append(_row(v, out, spec=sub.model.spec))
        if verbose:
            print(f'  [{v}] acc={rows[-1]["acc"]}±{rows[-1]["acc_std"]}')
    return rows


def format_table(rows: List[Dict[str, Any]],
                 cols: Optional[List[str]] = None) -> str:
    """定宽文本表。列宽按内容算，避免复制进论文时错位。"""
    if not rows:
        return '(空表)'
    cols = cols or list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r.get(c, ''))) for r in rows))
              for c in cols}
    head = '  '.join(str(c).ljust(widths[c]) for c in cols)
    sep = '-' * len(head)
    body = '\n'.join('  '.join(str(r.get(c, '')).ljust(widths[c]) for c in cols)
                     for r in rows)
    return f'{head}\n{sep}\n{body}'
