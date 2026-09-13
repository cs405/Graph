r"""YAML 驱动的结构构建器（对应 yolov8 的 ``model_builder.py``）。

把 ``cfg/models/*.yaml`` 里 ``[from, repeats, module, args]`` 描述的行列表展开成
真正能跑的 :class:`GraphSequential`，并逐层推算输入/输出宽度（按
``[depth_multiple, width_multiple, max_channels]`` 复合缩放）。

与 yolov8 的三处必须不同的地方，都写死在这里：

1. **edge_index 不进特征缓存**：transductive 全图只有一份边集，每层都要用它，
   所以路由时恒传给模块，而不是像图像侧那样只传 x；
2. **repeats 不能用 ``nn.Sequential``**：``Sequential.forward`` 只传一个参数，
   ``m(x, edge_index)`` 会当场 TypeError，故用 :class:`Repeat`；
3. **``from`` 的行号语义**：``-1`` = 上一层输出（第 0 行时 = 网络输入），非负数 =
   绝对行号，小于 -1 的按「相对当前行」解析（``-2`` 在行 5 就是行 3）。

**本文件不认识任何具体算子。** 它只 ``import modules``（触发各算子的
``@register_module``），然后全部靠 :mod:`modules.base` 的两样东西干活：

* :data:`~modules.base.MODULES` —— 注册表，YAML 里的模块名从这里查；
* :class:`~modules.base.GraphOp` 的类属性 —— ``scale_out``（args[0] 是不是要缩放的
  输出宽度）、``derived_out``（宽度由输入推导）、``takes_batch``（前向要不要 ``batch``）、
  ``inject``（构造参数名 -> 配置来源名）、``level``（输出是节点级还是边级）。

所以上一版那三张表（``BASE_MODULES``/``DERIVED_MODULES``/``INJECT``，用**类对象**
做 key）已经删掉：它们意味着「新增一个算子要改 builder」，而 builder 是框架、
算子是插件，插件不该要求框架改代码。
"""

from __future__ import annotations

import ast
import inspect
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import yaml
from torch import Tensor

import modules                                    # noqa: F401  只为触发 @register_module
from modules.base import MODULES, SPEC_BLOCKS, explainable_layers

__all__ = ['MODULES', 'GraphSequential', 'Repeat', 'make_divisible',
           'yaml_model_load', 'parse_model', 'build_model', 'model_info']

_log = logging.getLogger('oca')


def make_divisible(x: float, divisor: int = 8) -> int:
    """向上取整到 ``divisor`` 的倍数（yolov8 惯例：通道对齐到 8，GPU 友好）。"""
    if isinstance(divisor, Tensor):
        divisor = int(divisor.max())
    return int(math.ceil(float(x) / divisor) * divisor)


def yaml_model_load(path: Union[str, Path]) -> Dict[str, Any]:
    """读结构 YAML；文件名带 ``-n``/``_s`` 后缀时顺便推出 scale（同 yolov8）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'模型结构文件不存在：{path}')
    with open(path, 'r', encoding='utf-8') as f:
        d = yaml.safe_load(f) or {}
    if not d.get('backbone'):
        raise ValueError(f'{path} 里没有 backbone，无法构建模型')
    m = re.search(r'[-_]([nslmx])$', path.stem)
    d.setdefault('scale', m.group(1) if m else None)
    d['yaml_file'] = str(path)
    return d


class Repeat(nn.Module):
    """把同一层堆 n 次并继续传 ``edge_index``（``nn.Sequential`` 做不到，见模块 docstring）。"""

    def __init__(self, blocks: Sequence[nn.Module]):
        super().__init__()
        assert blocks, 'Repeat 需要至少一个模块'
        self.m = nn.ModuleList(list(blocks))
        self.out_dim = int(self.m[-1].out_dim)     # type: ignore[attr-defined]

    def forward(self, x: Union[Tensor, List[Tensor]],
                edge_index: Tensor = None,
                batch: Optional[Dict[str, Any]] = None) -> Tensor:
        for blk in self.m:
            x = _call(blk, x, edge_index, batch)
        return x


def _call(m: nn.Module, x: Any, edge_index: Optional[Tensor],
          batch: Optional[Dict[str, Any]]) -> Any:
    """统一的调用口径：只有声明 ``takes_batch`` 的层才收到 ``batch``。

    不是所有层都收 ``batch`` 是因为签名要稳：基线层（GAT/GCN/MLP）的
    ``forward(x, edge_index)`` 是与外基线对齐的口径，多塞一个参数就得跟着改，
    而它们本来也不需要边级侧信息。
    """
    if getattr(m, 'takes_batch', False):
        return m(x, edge_index, batch=batch)
    return m(x, edge_index)


class GraphSequential(nn.Module):
    """按 ``from`` 路由的前向容器（yolov8 ``FeatureSequential`` 的图版）。

    单输入层收到 tensor，多输入层（``Merge``/``Concat``/多尺度 ``Classify``）收到
    list；``edge_index`` 恒常传，缓存里只放节点特征。``batch``（边级侧信息：
    关系类型/边特征/节点类型）只递给声明了 ``takes_batch`` 的层。
    """

    def __init__(self, modules: Sequence[nn.Module],
                 sources: Sequence[int] = ()):
        super().__init__()
        self.model = nn.ModuleList(list(modules))
        self.save = set(int(s) for s in sources)
        self.yaml: Optional[Dict[str, Any]] = None      # 由 build_model 填，便于溯源

    def forward(self, x: Tensor, edge_index: Tensor,
                batch: Optional[Dict[str, Any]] = None) -> Tensor:
        y: List[Optional[Tensor]] = []                  # 按行号存，未缓存的位置是 None
        for m in self.model:
            if m.f != -1:                               # 不是来自上一层
                x = (y[m.f] if isinstance(m.f, int)
                     else [x if j == -1 else y[j] for j in m.f])
            x = _call(m, x, edge_index, batch)
            y.append(x if m.i in self.save else None)
        return x

    @property
    def explainable_layers(self) -> nn.ModuleList:
        """网络里全部可解释层（:class:`modules.base.Explorable`），逐层读 ``aux``。

        这是**去算法**的取法：诊断代码不需要知道层是 OCA 还是 DIA。
        """
        return explainable_layers(self)

    @property
    def oca_layers(self) -> nn.ModuleList:
        """网络里全部 OCA 层（按 ``family`` 标签筛，不 import :mod:`modules.oca`）。

        保留这个名字是为了向后兼容（``test/``、``training/`` 与旧脚本都在用）；
        新代码请用 :attr:`explainable_layers`。
        """
        return explainable_layers(self, family='oca')

    @property
    def level(self) -> str:
        """输出的粒度：``'node'``（``[N,nc]``）或 ``'edge'``（``[E,nc]``）。

        由最后一行的类属性决定。``model.py`` 拿它与数据集的 ``supervision`` 对齐 ——
        两边都是二维张量，接错了不会报错，只会静默错位。
        """
        return getattr(self.model[-1], 'level', 'node')

    @property
    def num_classes(self) -> Optional[int]:
        last = self.model[-1]
        return getattr(last, 'num_classes', None)

    def extra_repr(self) -> str:
        return f'layers={len(self.model)}, cached={sorted(self.save)}'


def _param_names(cls) -> List[str]:
    """``__init__`` 的形参名（去掉 ``self`` 与 ``c1``）；首项恒为输出宽度那一参。"""
    ps = list(inspect.signature(cls).parameters)
    return [p for p in ps if p != 'self'][1:]


def _after_c2(cls) -> List[str]:
    """``c1/c2`` 之后的形参名，用于判断 YAML 的 args 是否已按位置占住了某参。"""
    names = _param_names(cls)
    return names[1:]


def _resolve(name: str) -> Any:
    if name not in MODULES:
        raise KeyError(f'结构表里的模块 {name!r} 未注册，可选：{sorted(MODULES)}')
    return MODULES[name]


def _sub_one(a: Any, nc: Optional[int], where: int) -> Any:
    if isinstance(a, list):
        return _substitute(a, nc, where)
    if not isinstance(a, str):
        return a
    if a == 'nc':
        if nc is None:
            raise ValueError(
                f'第 {where} 行要用 nc，但构建时没拿到类别数 '
                f'（结构表的 ``nc: null`` 需要由数据集注入）')
        return int(nc)
    if a.lower() in ('null', 'none', '~'):
        return None
    try:
        return ast.literal_eval(a)
    except (ValueError, SyntaxError):
        return a                              # 'elu' 这类激活名保持字符串


def _substitute(args: List[Any], nc: Optional[int], where: int) -> List[Any]:
    """把 ``'nc'`` 换成真实类别数，其余字符串按 python 字面量解析（同 yolov8）。"""
    return [_sub_one(a, nc, where) for a in args]


def _scale_of(d: Dict[str, Any], scale: Optional[str],
              verbose: bool) -> Tuple[float, float, float, Optional[str]]:
    scales = d.get('scales') or {}
    if not scales:
        return 1.0, 1.0, float('inf'), scale
    if scale is None:
        scale = d.get('scale') or next(iter(scales))
        if verbose:
            _log.info("未指定 scale，用 %s", scale)
    if scale not in scales:
        raise KeyError(f'scale {scale!r} 不在结构表的 scales 里，'
                       f'可选：{sorted(scales)}')
    depth, width, max_ch = scales[scale]
    return float(depth), float(width), float(max_ch), scale


def parse_model(d: Dict[str, Any], in_dim: int, nc: Optional[int] = None,
                scale: Optional[str] = None, oca: Optional[Dict[str, Any]] = None,
                dropout: float = 0.5, verbose: bool = True,
                spec: Optional[Dict[str, Dict[str, Any]]] = None
                ) -> Tuple[GraphSequential, List[int]]:
    """结构表 -> ``(GraphSequential, save)``。

    Args:
        d: :func:`yaml_model_load` 的结果（含 backbone/head/scales 与各算子块）。
        in_dim: 输入特征维度（图像侧的 ``ch=3`` 在图侧是节点特征数）。
        nc: 类别数，覆盖 ``d['nc']``（``d['nc']`` 通常是 ``null``）。
        scale: ``n|s|m|l|x``；缺省依次取 ``d['scale']`` / scales 表首项。
        oca: 对结构表 ``oca:`` 块的**逐键覆盖**（旧参数，等价于
            ``spec={'oca': ...}``，保留是因为现有调用方与测试在用它）。
        dropout: 注入给吃 dropout 的模块；YAML 行里已按位置写了就不注入。
        verbose: 逐层打印结构表。
        spec: ``{算子块名: 覆盖字典}``（如 ``{'oca': {...}, 'dia': {...}}``）。
            优先级：CLI/参数 > 结构表；块名必须在 :data:`modules.base.SPEC_BLOCKS` 里。

    Returns:
        ``(net, save)``：每个子模块带 ``.i/.f/.type/.np``；save 是需要缓存输出的行号。
    """
    depth, width, max_ch, scale = _scale_of(d, scale, verbose)
    nc = d.get('nc') if nc is None else nc
    overrides: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in (spec or {}).items()
                                           if v}
    if oca:
        overrides['oca'] = {**overrides.get('oca', {}), **oca}
    unknown = set(overrides) - set(SPEC_BLOCKS)
    if unknown:
        raise KeyError(
            f'结构表里没有名为 {sorted(unknown)} 的算子超参块，'
            f'可选：{list(SPEC_BLOCKS)}（新算子先在 modules/base.py 里登记块名）')
    # 每个块的生效值 = 结构表块 ⊕ 覆盖项（逐键合并，不是整块替换：
    # 整块替换会把 heads/beta 一起清掉，消融表里两行的差值就不只是那一个键了）
    spec_vals: Dict[str, Any] = {
        'dropout': dropout,
        **{b: {**(d.get(b) or {}), **(overrides.get(b) or {})} for b in SPEC_BLOCKS},
    }
    rows: List[Any] = list(d.get('backbone') or []) + list(d.get('head') or [])

    if verbose:
        print(f"\n{'':>3}{'from':>14}{'n':>3}{'params':>10}  "
              f"{'module':<10}{'arguments'}")

    ch: List[int] = []            # ch[i] = 第 i 行的输出宽度
    layers: List[nn.Module] = []
    save: List[int] = []

    for i, row in enumerate(rows):
        if len(row) < 3:
            raise ValueError(f'第 {i} 行需要 [from, repeats, module, args]，'
                             f'实际只有 {row}')
        f, n, name = row[0], int(row[1]), row[2]
        args = list(row[3] or []) if len(row) > 3 else []
        cls = _resolve(name)
        args = _substitute(args, nc, i)
        n_ = n = max(round(n * depth), 1) if n > 1 else 1

        c1: Union[int, List[int]]
        if isinstance(f, (list, tuple)):
            c1 = [_c_width(x, i, ch, in_dim) for x in f]
        else:
            c1 = _c_width(f, i, ch, in_dim)

        inject: Dict[str, str] = dict(getattr(cls, 'inject', {}))
        if getattr(cls, 'derived_out', False):
            c2 = sum(c1) if isinstance(c1, list) else int(c1)
            pos: List[Any] = [c1]
            rest: List[Any] = list(args)
        else:
            raw = int(args[0])
            # 类别数不是通道数，不能拿去缩放（yolov8 里靠 `c2 != nc` 判断，
            # 这里靠类属性 scale_out，意思一样但不靠猜）
            if getattr(cls, 'scale_out', True):
                c2 = make_divisible(min(raw, max_ch) * width, 8)
            else:
                c2 = raw
            pos = [c1, c2]
            rest = list(args[1:])

        # 行内 dict（``[64, {'T': 1}]``）= 「在全局算子块上追加覆盖」，不是整块替换：
        # 「同一个网络里每层不同 T」这类消融就靠这个表达。
        spec_srcs = [s for s in inject.values() if s in SPEC_BLOCKS]
        inline: Dict[str, Any] = {}
        if spec_srcs and rest and isinstance(rest[0], dict):
            inline = dict(rest.pop(0))

        kwargs: Dict[str, Any] = {}
        if inject:
            names = _after_c2(cls)
            for key, src in inject.items():
                if src not in spec_vals:
                    raise KeyError(
                        f'{cls.__name__}.inject 指向了未知的配置来源 {src!r}，'
                        f'可选：{sorted(spec_vals)}')
                val: Any = ({**spec_vals[src], **inline} if src in SPEC_BLOCKS
                            else spec_vals[src])
                # YAML 里已按位置写了这个参（或它前面还有位置参数），就以 YAML 为准
                if key in names and names.index(key) >= len(rest):
                    kwargs[key] = val

        m_ = cls(*pos, *rest, **kwargs)
        if n > 1:
            if isinstance(c1, list) or getattr(cls, 'derived_out', False):
                raise ValueError(
                    f'第 {i} 行 {name} 是多输入模块，不能 repeats>1（堆叠只适用于单输入层）')
            # 重复堆叠：第 2 份起 c1 换成 c2（各份独立参数，共享超参）。
            # pos[1:] = [c2]，必须带上，否则 OCAConv(c1, oca=...) 会缺第二个位置参。
            blocks = [m_] + [cls(c2, *pos[1:], *rest, **kwargs)
                             for _ in range(n - 1)]
            m_ = Repeat(blocks)

        m_.i, m_.f, m_.type = i, _abs_f(f, i), cls.__name__
        m_.np = sum(p.numel() for p in m_.parameters())   # type: ignore[attr-defined]
        if verbose:
            print(f"{i:>3}{str(f):>14}{n_:>3}{m_.np:>10}  "
                  f"{cls.__name__:<10}{pos + rest}"
                  + (f'  {kwargs}' if kwargs else ''))
        save.extend(_abs_from(int(x), i)
                    for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        ch.append(c2)

    save = sorted(set(s for s in save if 0 <= s < len(layers)))
    net = GraphSequential(layers, save)
    net.yaml = {**d, 'scale': scale, 'nc': nc, 'in_dim': int(in_dim),
                # 生效的算子超参（结构表 ⊕ 覆盖）也存一份：光看结构表不知道
                # 这张网到底是哪套开关训出来的（消融表的可复现性靠这一行）
                'spec_effective': {b: dict(spec_vals[b]) for b in SPEC_BLOCKS}}
    return net, save


def _c_width(f: int, i: int, ch: List[int], in_dim: int) -> int:
    """第 ``i`` 行的第 ``f`` 号输入的宽度。"""
    if f == -1:
        return ch[-1] if ch else int(in_dim)
    if f < -1:
        f = i + f                              # 相对当前行往前数
        if f < 0:
            raise ValueError(f'第 {i} 行的 from={f} 越界（负得比已建成的层数还多）')
    if f >= len(ch):
        raise ValueError(f'第 {i} 行引用了尚未构建的层 {f}（`from` 只能往回指）')
    return int(ch[f])


def _abs_from(x: int, i: int) -> int:
    """把单个 ``from`` 规范化：``-1`` 保留（= 上一层/运行值），其余换成绝对行号。"""
    return x if x == -1 else (x % i if x < 0 else x)


def _abs_f(f: Union[int, List[int]], i: int) -> Union[int, List[int]]:
    """整行 ``from`` 的规范化（列表逐项，单值直接转）。"""
    if isinstance(f, (list, tuple)):
        return [_abs_from(int(x), i) for x in f]
    return _abs_from(int(f), i)


def build_model(spec: Union[str, Path, Dict[str, Any]], in_dim: int,
                nc: Optional[int] = None, scale: Optional[str] = None,
                oca: Optional[Dict[str, Any]] = None, dropout: float = 0.5,
                device: Union[str, torch.device, None] = None,
                verbose: bool = True,
                spec_overrides: Optional[Dict[str, Dict[str, Any]]] = None
                ) -> GraphSequential:
    """``spec``（结构 YAML 路径或已解析的 dict）→ 可直接 ``net(x, edge_index)`` 的模型。

    ``spec_overrides`` 是算子超参块的覆盖（``{'oca': {...}, 'dia': {...}}``）；
    ``oca=`` 是它的旧写法，两者同时给时逐键合并（``oca=`` 赢）。
    """
    d = spec if isinstance(spec, dict) else yaml_model_load(spec)
    net, _ = parse_model(d, in_dim, nc=nc, scale=scale, oca=oca,
                         dropout=dropout, verbose=verbose,
                         spec=spec_overrides)
    if device is not None:
        net = net.to(device)
    return net


def model_info(net: nn.Module, verbose: bool = True) -> Dict[str, Any]:
    """参数量/层数摘要（对应 yolov8 的 ``model.info()`` 的返回值部分）。"""
    n_p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_g = sum(p.numel() for p in net.parameters())
    types: Dict[str, int] = {}
    for m in net.modules():
        if m is net or isinstance(m, (Repeat,)):
            continue
        key = type(m).__name__
        types[key] = types.get(key, 0) + 1
    info = {'params': n_p, 'params_total': n_g, 'gradients': n_g - n_p,
            'layers': len(getattr(net, 'model', [])), 'module_types': types}
    if verbose:
        print(f'  {info["layers"]} 层, {n_g:,} 参数（可训练 {n_p:,}）')
    return info
