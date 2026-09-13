r"""配置树：YAML 读入 → 嵌套 dataclass → 类型强校验 → 快照落盘。

与 yolov8 的 `config.py` 同一套思路，三点不同是刻意的：

* **不用 loguru**（环境里没有），未知键告警走标准 logging；
* 结构（层数/宽度/算子开关）**不在这里**，在 ``cfg/models/*.yaml``；本文件只管
  「怎么训、跑在哪、用哪个结构文件」；
* 提供**扁平别名**：``cfg.lr`` 等价 ``cfg.optimization.lr``，消融表与训练循环里
  不必到处写块名前缀。

优先级（后者覆盖前者）：dataclass 默认 < ``cfg/train/*.yaml`` < ``train(**kwargs)``
/ CLI 覆盖项。每次运行把最终结果快照成 ``runs/<name>/train/config_used.yaml`` ——
事后追溯以快照为准，不以本文件为准。
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, get_args, get_origin, \
    get_type_hints

import yaml

__all__ = ['RunConfig', 'DatasetConfig', 'ModelConfig', 'TaskConfig',
           'LossConfig', 'OptimizationConfig', 'TrainConfig',
           'load_train_config', 'config_to_dict',
           'parse_overrides', 'DEFAULT_TRAIN_YAML', 'BLOCK_NAMES', 'flat_names']

_log = logging.getLogger('oca')

DEFAULT_TRAIN_YAML = 'cfg/train/default.yaml'


# ---------------------------------------------------------------------------
# 嵌套块
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """跑在哪儿、产出落哪儿、怎么挑模型。"""

    run_name: str = 'oca'
    project: str = 'runs'
    device: str = 'auto'            # auto | cpu | cuda | cuda:1
    seed: int = 0
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    vary_split: bool = False        # True = 每个种子重新划分数据
    log_interval: int = 20
    best_metric: str = 'val_acc'    # 只能看 val，看 test 就是数据泄漏
    ckpt_interval: int = 0          # >0 时每 N 个 epoch 存滚动 checkpoint
    max_ckpt: int = 5
    save: bool = True               # False = 不建 runs/ 目录
    dump_gates: bool = True
    dump_pairings: bool = False     # True = 额外存 DIA 学到的 U/V/W（参数级，与门控不同）


@dataclass
class DatasetConfig:
    name: str = 'synth'
    root: str = 'data'
    split_idx: int = 0
    train_per_class: Optional[int] = 20
    homophily: float = 0.3          # 仅合成图
    n_per_class: int = 60           # 仅合成图
    fallback_to_synth: bool = False


@dataclass
class ModelConfig:
    spec: str = 'cfg/models/oca.yaml'
    scale: str = 'n'                # n | s | m | l | x
    dropout: float = 0.5
    oca: Dict[str, Any] = field(default_factory=dict)   # OCAConfig 覆盖项
    dia: Dict[str, Any] = field(default_factory=dict)   # DIAConfig 覆盖项


@dataclass
class TaskConfig:
    """训什么任务。对应 ultralytics 的 ``model.task``（它的 ``task_map`` 靠这个分派）。

    字段名不叫 ``name``：扁平别名要求字段名跳块唯一（见文件末的 _dupes 检查），
    而 ``dataset.name`` 已经占了 ``name``。
    """

    task_type: str = 'auto'         # auto | node_classification | edge_classification
    threshold: float = 0.5          # 单 logit（nc==1）时的判决阈值
    edge_neg_ratio: float = 1.0     # 边任务的负例采样比例（>0 时生效）

    def __post_init__(self):
        assert self.task_type in ('auto', 'node_classification',
                                  'edge_classification'), \
            f'未知 task_type {self.task_type!r}（可选见 tasks/__init__.py 的 TASK_MAP）'
        assert 0.0 < self.threshold < 1.0, f'threshold 必须在 (0,1)：{self.threshold}'
        assert self.edge_neg_ratio > 0.0, f'edge_neg_ratio 必须为正：{self.edge_neg_ratio}'


@dataclass
class LossConfig:
    """任务损失的选择 + 算子附加项的权重。

    权重放在这里而不是算子里：算子只报**未加权**的惩罚项
    （:meth:`modules.base.Regularized.penalties`），占多大比例是训练侧的事 ——
    这样同一个 DIA 层能在不同任务里用不同的 :math:`\lambda`，不必改算子。
    默认全 0：不开就不付计算代价，也不会让 OCA/基线的 loss 多出莫名其妙的项。
    """

    loss_type: str = 'auto'         # auto | ce | bce
    lambda_sp: float = 0.0          # ||U||_1 + ||V||_1（列稀疏）
    lambda_orth: float = 0.0        # ||U^T U - I||_F^2 + ||V^T V - I||_F^2（列分离）
    lambda_gamma: float = 0.0       # gamma 的 L1（逐维筛选要真的筛掉东西）
    pos_weight: float = 0.0         # >0 时给 bce 的正类加权（0 = 不用）

    def __post_init__(self):
        assert self.loss_type in ('auto', 'ce', 'bce'), \
            f'未知 loss_type {self.loss_type!r}'
        for k in ('lambda_sp', 'lambda_orth', 'lambda_gamma'):
            assert getattr(self, k) >= 0.0, f'{k} 不能为负：{getattr(self, k)}'
        assert self.pos_weight >= 0.0, f'pos_weight 不能为负：{self.pos_weight}'


@dataclass
class OptimizationConfig:
    epochs: int = 400
    patience: int = 100
    optimizer: str = 'adam'         # adam | adamw | sgd
    lr: float = 0.01
    weight_decay: float = 5e-4
    momentum: float = 0.9           # 仅 sgd
    grad_clip: float = 0.0
    class_weight: bool = False


# ---------------------------------------------------------------------------
# 顶层
# ---------------------------------------------------------------------------

_BLOCKS = {'run': RunConfig, 'dataset': DatasetConfig, 'model': ModelConfig,
           'task': TaskConfig, 'loss': LossConfig,
           'optimization': OptimizationConfig}


@dataclass
class TrainConfig:
    run: RunConfig = field(default_factory=RunConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimization: OptimizationConfig = field(
        default_factory=OptimizationConfig)

    # ---- 扁平别名：cfg.lr / cfg.epochs / cfg.spec ... -----------------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(name)
        block = _FLAT.get(name)
        if block is None:
            raise AttributeError(
                f'TrainConfig 没有 {name!r}（也不是任何块的字段）')
        return getattr(object.__getattribute__(self, block), name)

    def resolve(self, key: str) -> Tuple[str, str]:
        """把扁平名解析成 ``(块名, 字段名)``；``'lr' -> ('optimization', 'lr')``。"""
        if '.' in key:
            head, tail = key.split('.', 1)
            if head in _BLOCKS:
                return head, tail
            raise KeyError(f'未知配置块 {head!r}，可选 {sorted(_BLOCKS)}')
        block = _FLAT.get(key)
        if block is None:
            if key in _BLOCKS:
                # ``dataset='synth'`` 这类手滑很常见：只写了块名。不单独接住的话，
                # 报出来的是一长串扁平字段名，看不出自己少写了一段。
                raise KeyError(
                    f'{key!r} 是配置块而不是字段，请写到字段级（如 {key}.name）；'
                    f'该块的字段：{sorted(f.name for f in fields(_BLOCKS[key]))}')
            raise KeyError(f'未知配置项 {key!r}，可选：{sorted(_FLAT)}')
        return block, key

    def patched(self, **overrides: Any) -> 'TrainConfig':
        """深拷贝后按扁平名或 ``block.field`` 覆盖（消融表与 CLI 全靠它）。"""
        out = copy.deepcopy(self)
        for k, v in overrides.items():
            block, fname = self.resolve(k)
            sub = getattr(out, block)
            if not hasattr(sub, fname):
                raise KeyError(f'{block} 没有字段 {fname!r}')
            setattr(sub, fname, v)
        return out

    def with_oca(self, **overrides: Any) -> 'TrainConfig':
        """**合并**进 ``model.oca``（而不是整块替换）。"""
        return self.patched(**{'model.oca': {**dict(self.oca), **overrides}})

    def with_dia(self, **overrides: Any) -> 'TrainConfig':
        """**合并**进 ``model.dia``（同 ``with_oca``：消融表靠它而不是 ``patched``）。"""
        return self.patched(**{'model.dia': {**dict(self.dia), **overrides}})

    def spec_overrides(self) -> Dict[str, Dict[str, Any]]:
        """结构表算子块的覆盖项（``{'oca': ..., 'dia': ...}``）。

        只带非空的块：空字典等价于「没覆盖」，但会让逐层打印里多一行
        无信息量的 ``{}``。块名取自 :data:`modules.base.SPEC_BLOCKS`（延迟 import：
        本文件在模块加载期不依赖算子包）。
        """
        from modules.base import SPEC_BLOCKS
        return {b: dict(getattr(self.model, b)) for b in SPEC_BLOCKS
                if getattr(self.model, b, None)}

    # ---- 序列化 ------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return config_to_dict(self)

    def save_yaml(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False,
                           allow_unicode=True)
        return path

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainConfig':
        return _from_dict(cls, d or {})

    @classmethod
    def from_yaml(cls, path) -> 'TrainConfig':
        return cls.from_dict(_load_yaml(path))


_FLAT: Dict[str, str] = {
    fname: block for block, cls in _BLOCKS.items()
    for fname in [f.name for f in fields(cls)]
}
# 扁平别名要求字段名全局唯一，否则 cfg.lr 到底指哪个块说不清 —— 导入时就炸掉，
# 而不是等某次消融悄悄改了另一个块的同名字段。
_counts: Dict[str, int] = {}
for _b, _c in _BLOCKS.items():
    for _f in fields(_c):
        _counts[_f.name] = _counts.get(_f.name, 0) + 1
_dupes = sorted(k for k, n in _counts.items() if n > 1)
if _dupes:
    raise RuntimeError(f'扁平字段名在不同块里重名，无法做别名：{_dupes}')


# ---------------------------------------------------------------------------
# YAML → dataclass
# ---------------------------------------------------------------------------

_TRUE_WORDS = {'true', 'yes', 'on', '1'}
_FALSE_WORDS = {'false', 'no', 'off', '0'}


def _load_yaml(path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'配置文件不存在：{path}')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _from_dict(cls, data: Dict[str, Any], path: str = 'config') -> Any:
    """按 dataclass 的字段类型构造，未知键告警忽略（注释与实验性字段不该炸跑）。"""
    hints = _hints(cls)
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        _log.warning('忽略 %s 里的未知键：%s', path, sorted(unknown))
    kwargs: Dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        tp = hints.get(name, f.type)
        if value is None:
            # 显式写了 ``key: null`` 与「没写这个键」不同：前者是要把 Optional 字段
            # 置空（如 train_per_class: null => 由每类节点数推导），后者走默认值。
            if _is_optional(tp):
                kwargs[name] = None
            continue
        sub = _unwrap(tp)
        if sub is not None and is_dataclass(sub) and isinstance(value, dict):
            kwargs[name] = _from_dict(sub, value, path=f'{path}.{name}')
            continue
        kwargs[name] = _coerce(tp, value, f'{path}.{name}')
    return cls(**kwargs)


def _is_optional(tp: Any) -> bool:
    return get_origin(tp) is Union and type(None) in get_args(tp)


def _hints(cls):
    """解析后的字段类型。

    本文件开了 ``from __future__ import annotations``，``f.type`` 是**字符串**
    （如 ``'RunConfig'``），直接拿它做类型判断会静默失效 —— 嵌套块会被当成
    普通 dict 塑进 dataclass，字段存在但类型错，要到很晚才在训练循环里拆爆。
    """
    return _CACHED_HINTS.setdefault(cls, get_type_hints(cls))


_CACHED_HINTS: Dict[Any, Dict[str, Any]] = {}


def _unwrap(tp):
    """Optional[X] -> X（Python 3.13 下 f.type 可能是字符串，这里只处理真类型）。"""
    if isinstance(tp, str):
        return None
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        return args[0] if len(args) == 1 else None
    return tp


def _coerce(tp: Any, value: Any, where: str) -> Any:
    """把 YAML 标量强转成声明类型。

    PyYAML 按 YAML 1.1 解析，``5e-4`` 会被读成**字符串**（科学计数法必须带小数点
    才认作 float），这里兜住，免得配置里写个权重衰减就报类型错。
    """
    inner = _unwrap(tp)
    origin = get_origin(tp)
    if origin is list and isinstance(value, (list, tuple)):
        args = get_args(tp)
        arg = _unwrap(args[0]) if args else None
        return [_coerce(arg, v, where) for v in value] if arg else list(value)
    if inner is None or inner is Any or isinstance(value, (dict, list)):
        return value
    if inner in (float, int, bool, str):
        return _coerce_scalar(inner, value, where)
    return value


def _coerce_scalar(expected: type, value: Any, where: str):
    try:
        if expected is bool:
            if isinstance(value, bool):
                return value
            word = str(value).strip().lower()
            if word in _TRUE_WORDS:
                return True
            if word in _FALSE_WORDS:
                return False
            raise ValueError(value)
        if isinstance(value, bool):        # bool 是 int 的子类，数值字段要拒绝它
            raise ValueError(value)
        if expected is float:
            return float(value)
        if expected is int:
            f = float(value)
            if not f.is_integer():
                raise ValueError(value)
            return int(f)
        return value if isinstance(value, str) else str(value)
    except (TypeError, ValueError):
        raise ValueError(
            f'{where} 的值 {value!r} 不是 {expected.__name__} 类型') from None


def config_to_dict(cfg: Any) -> Dict[str, Any]:
    """嵌套 dataclass -> 纯 dict（可直接 yaml.safe_dump）。"""
    if not is_dataclass(cfg):
        return cfg
    out: Dict[str, Any] = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        out[f.name] = config_to_dict(v) if is_dataclass(v) else v
    return out


def load_train_config(path=None, **overrides: Any) -> TrainConfig:
    """读 ``cfg/train/*.yaml``（缺省 default.yaml）并应用覆盖项。"""
    cfg = TrainConfig.from_yaml(path or DEFAULT_TRAIN_YAML)
    return cfg.patched(**overrides) if overrides else cfg


BLOCK_NAMES = tuple(_BLOCKS)


def flat_names() -> Dict[str, str]:
    """扁平字段名 -> 所属块名（``{'lr': 'optimization', ...}``）。

    公开这个映射是为了让入口脚本能判断「用户传的 key 到底认不认」，而不必
    ``from config import _FLAT`` 去摸一个下划线开头的变量。
    """
    return dict(_FLAT)


def parse_overrides(pairs: Optional[List[str]]) -> Dict[str, Any]:
    """CLI 的 ``T=4 beta=0.3 use_phi=false`` -> dict。

    bool/int/float 从字符串推出来，避免「传了字符串 'False' 结果是真值」这种坑。
    """
    out: Dict[str, Any] = {}
    for p in pairs or []:
        if '=' not in p:
            raise ValueError(f'覆盖项需要 key=value 形式，收到 {p!r}')
        k, v = p.split('=', 1)
        out[k.strip()] = _scalar(v.strip())
    return out


def _scalar(v: str) -> Any:
    """CLI 值的类型推断：先试数字，再试 bool 词，最后当字符串。

    顺序不能反：否则 ``--oca T=1`` 会先命中 ``'1' -> True``（YAML 1.1 风格），
    把迭代轮数变成布尔量 —— 一个只在命令行才出现的静默错。
    """
    low = v.lower()
    if low in ('none', 'null'):
        return None
    if low in ('true', 'false', 'yes', 'no', 'on', 'off'):
        return low in ('true', 'yes', 'on')
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            continue
    return v
