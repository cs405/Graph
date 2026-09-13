r"""配置树：YAML → 嵌套 dataclass 的解析、别名、覆盖与快照。

``cfg/train/default.yaml`` 是**给人读的**，``config.py`` 的校验是**给机器用的**。
两边必须始终对得上，且 YAML 1.1 的三个坑（``5e-4`` 读成字符串、``off/yes`` 读成
bool、``key: null`` 与「没写这个键」不同）必须被兜住 —— 这三个都是静默错，
只在训练跑起来后才以奇怪的形式暴露。
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from dataclasses import fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml                                                       # noqa: E402

from config import (BLOCK_NAMES, DatasetConfig, ModelConfig,       # noqa: E402
                    OptimizationConfig, RunConfig, TrainConfig,
                    config_to_dict, flat_names, load_train_config,
                    parse_overrides)
from test.support import case, run_registered                     # noqa: E402


def _write_yaml(text: str) -> str:
    """写一份临时 ``cfg/train/*.yaml``，返回路径。"""
    fd, path = tempfile.mkstemp(suffix='.yaml', text=True)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


# ---------------------------------------------------------------------------

@case
def test_default_yaml_agrees_with_dataclass_defaults():
    """``default.yaml`` 与 dataclass 默认值必须一字不差。

    两边各写一份「默认值」是漂移的开始：文档里抄 YAML、代码里抄 dataclass，
    最后 ``config_used.yaml`` 与谁都像。真要改默认值，就同时改两处。
    """
    from_yaml = load_train_config().to_dict()
    from_code = TrainConfig().to_dict()
    diff = {f'{b}.{k}': (v, from_code[b][k])
            for b, blk in from_yaml.items()
            for k, v in blk.items() if from_code[b][k] != v}
    assert not diff, diff
    print(f'  default.yaml ≡ dataclass 默认 OK（{len(BLOCK_NAMES)} 个块）')


@case
def test_flat_aliases_and_resolve():
    cfg = load_train_config()
    assert cfg.lr == cfg.optimization.lr and cfg.spec == cfg.model.spec
    assert cfg.save is cfg.run.save and cfg.train_per_class == 20
    assert cfg.resolve('lr') == ('optimization', 'lr')
    assert cfg.resolve('model.scale') == ('model', 'scale')
    for key, block in flat_names().items():
        assert getattr(cfg, key) == getattr(getattr(cfg, block), key), key
    try:
        cfg.no_such_field
    except AttributeError as e:
        assert 'no_such_field' in str(e), e
    else:
        raise AssertionError('未知字段应当 AttributeError')
    # 只写块名（``dataset='synth'``）要报到字段级，不能只给一长串候选名
    try:
        cfg.patched(dataset='synth')
    except KeyError as e:
        assert '字段级' in str(e) and 'name' in str(e), e
    else:
        raise AssertionError('裸块名应当被拦下来并提示写到字段级')
    # 扁平名必须全局唯一，否则 cfg.lr 到底指哪个块说不清
    names = [f.name for c in (RunConfig, DatasetConfig, ModelConfig,
                              OptimizationConfig) for f in fields(c)]
    assert len(names) == len(set(names)), '字段名跨块重名'
    print(f'  扁平别名 / resolve OK（{len(names)} 个字段全部可达，裸块名报错点名字段）')


@case
def test_patched_is_a_deep_copy_and_with_oca_merges():
    """``patched`` 不能污染原对象；``with_oca`` 是合并而不是整块替换。

    消融表就是「一份 base cfg + N 份覆盖」，只要有一次把 base 改脏了，后面所有
    行都在错误的基础上算，而且错得很安静。
    """
    cfg = load_train_config(**{'model.oca': {'heads': 4, 'T': 2}})
    sub = cfg.patched(**{'optimization.lr': 0.5, 'model.spec': 'cfg/models/gcn.yaml'})
    assert cfg.lr == 0.01 and cfg.model.spec.endswith('oca.yaml')
    sub.model.oca['T'] = 99
    sub.run.seeds.append(7)
    assert cfg.model.oca == {'heads': 4, 'T': 2}, cfg.model.oca
    assert cfg.run.seeds == [0, 1, 2], cfg.run.seeds

    merged = cfg.with_oca(T=8)
    assert merged.model.oca == {'heads': 4, 'T': 8}, merged.model.oca
    assert cfg.model.oca == {'heads': 4, 'T': 2}
    replaced = cfg.patched(**{'model.oca': {'T': 8}})
    assert replaced.model.oca == {'T': 8}, 'patched 是整块替换（与 with_oca 有别）'
    print('  patched/with_oca 语义 OK')


@case
def test_yaml_11_traps_and_type_coercion():
    r"""科学计数法、bool 词、``null`` 与未写 —— 三个 YAML 1.1 的坑都要兜住。"""
    path = _write_yaml("""
run:
  save: off
  log_interval: 40
  seeds: [3, 4]
dataset:
  train_per_class: null      # 显式置空 => 由每类节点数推导
optimization:
  lr: 5e-4                   # YAML 1.1 读成字符串 '5e-4'
  weight_decay: 1.0e-3
  class_weight: yes
""")
    try:
        cfg = load_train_config(path)
    finally:
        os.unlink(path)
    assert cfg.save is False and cfg.class_weight is True
    assert cfg.lr == 0.0005 and isinstance(cfg.lr, float), cfg.lr
    assert cfg.weight_decay == 0.001
    assert cfg.train_per_class is None, '`key: null` 与「没写」必须可区分'
    assert cfg.seeds == [3, 4] and all(isinstance(s, int) for s in cfg.seeds)
    assert cfg.epochs == 400, '没写的键保持默认值'
    print(f'  YAML 1.1 兜底 OK   lr={cfg.lr!r}（来自字符串 5e-4）、'
          f'save={cfg.save}、train_per_class={cfg.train_per_class}')


@case
def test_unknown_keys_warn_and_bad_values_name_the_path():
    """未知键 → 告警并忽略（注释性字段不该炸跑）；类型错 → 说清是哪个键。"""
    path = _write_yaml("""
run:
  run_name: probe
  some_annotated_note: 1
optimization:
  epochs: abc
""")
    cap = _Capture()
    log = logging.getLogger('oca')
    log.addHandler(cap)
    msg = ''
    try:
        load_train_config(path)
    except ValueError as e:
        msg = str(e)
    finally:
        log.removeHandler(cap)
        os.unlink(path)
    assert 'optimization.epochs' in msg and 'abc' in msg, msg
    warn = [m for m in cap.msgs if 'some_annotated_note' in m]
    assert warn and '未知键' in warn[0], cap.msgs
    print(f'  未知键告警 + 类型错定位 OK   {warn[0]} || {msg}')


@case
def test_parse_overrides_number_before_bool():
    r"""``--oca T=1`` 必须是整数 1，不能是 ``True``。

    YAML 1.1 里 ``1`` 也是真值。推断顺序若反过来，只有命令行会中招（YAML 文件里
    ``T: 1`` 由 PyYAML 直接给 int），变成一个「只在 CLI 出现的静默错」。
    """
    got = parse_overrides(['T=1', 'beta=0.3', 'use_phi=false', 'name=gat',
                           'kernel=shifted_cosine', 'center_stat=None'])
    assert got == {'T': 1, 'beta': 0.3, 'use_phi': False, 'name': 'gat',
                   'kernel': 'shifted_cosine', 'center_stat': None}, got
    assert isinstance(got['T'], int) and not isinstance(got['T'], bool)
    assert parse_overrides([]) == {} and parse_overrides(None) == {}
    for bad in ('T', 'no-equals-here'):
        try:
            parse_overrides([bad])
        except ValueError as e:
            assert 'key=value' in str(e), e
        else:
            raise AssertionError(f'{bad!r} 应当报错')
    # 覆盖项最终落到配置树上：CLI 的 --oca 与 --set 走同一套 key
    cfg = load_train_config(**parse_overrides(['optimization.lr=5e-3']))
    assert cfg.lr == 0.005
    print('  parse_overrides 类型推断 OK（数字优先于 bool 词）')


@case
def test_config_snapshot_roundtrip():
    """``config_used.yaml`` 必须能读回来且完全等价 —— 追溯以快照为准。"""
    cfg = load_train_config(**{'model.oca': {'T': 4}, 'seeds': [0, 1]})
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'config_used.yaml')
        cfg.save_yaml(path)
        with open(path, encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        back = TrainConfig.from_dict(raw)
    assert back.to_dict() == cfg.to_dict()
    assert back.model.oca == {'T': 4} and back.seeds == [0, 1]
    assert isinstance(back.run, RunConfig) and \
        isinstance(back.optimization, OptimizationConfig), '嵌套块必须是 dataclass'
    assert config_to_dict(cfg)['run']['run_name'] == 'oca'
    print('  配置快照 roundtrip OK（含嵌套块重建）')


if __name__ == '__main__':
    sys.exit(run_registered('== 配置树 =='))
