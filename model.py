r"""顶层入口（对应 yolov8 根目录的 ``model.py`` / ``class YOLO``）。

    from model import OCA, DIA

    m = OCA('cfg/models/oca.yaml')                       # 只读结构表，不建参数
    m.train(data='cora', scale='n', epochs=200)          # 构建 + 训练，返回实验摘要
    m.val(data='cora')                                   # 用训好的权重评估
    m.info()                                             # 层数 / 参数量 / 结构表

    m = DIA('cfg/models/dia.yaml').train(data='synth')    # 换算法 = 换一个类（或只换结构表）
    m = OCA('runs/oca/train/weights/best.pt')             # 也可以直接指向权重

与 yolov8 一样采用**懒构建**：``in_dim``（节点特征数）和 ``nc``（类别数）只能来自
数据，所以 ``OCA(...)`` 这一步不产生任何参数；:meth:`build` 或第一次 :meth:`train`
才真正搭网络。别为了「先看看结构」去硬编一个 in_dim —— ``OCA(...).info(in_dim=1433)``
是显式传参，走的是同一条路，不会假装已经看过数据。

:class:`GraphModel` 是唯一的实现，``OCA``/``DIA`` 只是两个改了默认结构表的子类。
**任务不由类名决定**，而由 ``cfg.task.task_type``（缺省 ``auto``，按数据集的
``supervision`` 推）决定，分派表在 :data:`tasks.TASK_MAP` —— 这是 ultralytics
``YOLO`` 与 ``task_map`` 那对关系的简化版：算法（结构表）与任务（监督层级）正交，
上一个版本那句 ``assert task == 'node_classification'`` 把两者硬绑在一起了。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor, nn

from config import (BLOCK_NAMES, TrainConfig, flat_names,
                    load_train_config)
from dataset import GraphBundle
from devices import describe_device, resolve_device
from model_builder import build_model, model_info, yaml_model_load

__all__ = ['GraphModel', 'OCA', 'DIA', 'DEFAULT_SPEC', 'DIA_SPEC']

_log = logging.getLogger('oca')

DEFAULT_SPEC = 'cfg/models/oca.yaml'
DIA_SPEC = 'cfg/models/dia.yaml'
WEIGHT_SUFFIXES = ('.pt', '.pth')


class GraphModel:
    """「一个结构表 + 一份配置」的对象化封装：构建 / 训练 / 评估 / 查询。"""

    #: 子类只改这两个类属性（不重写 ``__init__``）
    default_spec: str = DEFAULT_SPEC
    default_task: str = 'auto'

    def __init__(self, model: Union[str, Path, Dict[str, Any], None] = None,
                 task: Optional[str] = None, **overrides: Any):
        src: Any = self.default_spec if model is None else model
        self.task = str(task or self.default_task)
        self._weights: Optional[Path] = None
        if isinstance(src, (str, Path)) and Path(src).suffix in WEIGHT_SUFFIXES:
            from training.checkpoints import load_checkpoint
            payload = load_checkpoint(src)               # 先只拿元信息
            self._weights = Path(src)
            self._meta = payload
            src = yaml_model_load(payload.get('spec') or self.default_spec)
            base = TrainConfig.from_dict(payload['config']) \
                if isinstance(payload.get('config'), dict) else None
            if task is None and payload.get('config', {}).get('task'):
                # 权重里存了当时的任务：重建时用它，而不是用子类的默认值
                self.task = str(payload['config']['task'].get('task_type')
                                or self.default_task)
        else:
            self._meta = {}
            base = None
        self.yaml: Dict[str, Any] = src if isinstance(src, dict) \
            else yaml_model_load(src)
        kw = dict(overrides)
        self.nc: Optional[int] = _int_or_none(kw.pop('nc', None))
        verbose = kw.pop('verbose', False)
        config_path = kw.pop('config', None)
        self.verbose = bool(verbose)
        if base is not None and config_path is not None:
            raise ValueError('已经加载了带配置的权重，不要再传 config=（会打架）')
        self.cfg: TrainConfig = base if base is not None \
            else load_train_config(config_path)
        # 结构表以构造参数为准：OCA('cfg/models/xxx.yaml') 不该还被 cfg.model.spec 影响
        spec = self.yaml.get('yaml_file')
        self.cfg = self._patch(self.cfg, {'model.spec': spec} if spec else {})
        self.cfg = self._patch(self.cfg, {'task.task_type': self.task})
        self.cfg = self._patch(self.cfg, kw)
        self.net: Optional[nn.Module] = None
        self.run_dir: Optional[Path] = None
        self.results: Optional[Dict[str, Any]] = None
        self._built: Optional[Dict[str, int]] = None

    # ------------------------------------------------------------------ 配置
    @staticmethod
    def _patch(cfg: TrainConfig, kw: Dict[str, Any]) -> TrainConfig:
        """把 ``lr=...`` / ``model.scale=...`` 之类的覆盖项应用到 ``cfg``。

        ``oca=``/``dia=`` 是**合并**（``with_oca``/``with_dia``）而不是整块替换：
        ``train(oca={'T': 4})`` 只想动 T，不想把结构表里的 heads/beta 一起清掉。
        ``task=`` 是 ``task.task_type=`` 的简写（与 ``data=`` 对 ``dataset.name=`` 同理）。
        """
        kw = dict(kw)
        oca = kw.pop('oca', None)
        dia = kw.pop('dia', None)
        task = kw.pop('task', None)
        if 'data' in kw:
            kw.setdefault('dataset.name', str(kw.pop('data')))
        if task is not None:
            kw['task.task_type'] = str(task)
        out = cfg.with_oca(**oca) if oca else cfg
        out = out.with_dia(**dia) if dia else out
        flat = flat_names()
        unknown = [k for k in kw
                   if k.split('.')[0] not in BLOCK_NAMES and k not in flat]
        if unknown:
            raise TypeError(f'未知覆盖项 {sorted(unknown)}；可用键见 '
                            f'config.TrainConfig 的配置块 {sorted(BLOCK_NAMES)}')
        return out.patched(**kw) if kw else out

    # ------------------------------------------------------------------ 构建
    def build(self, in_dim: Optional[int] = None, nc: Optional[int] = None,
              data: Union[str, Path, GraphBundle, None] = None,
              device: Optional[str] = None, verbose: Optional[bool] = None
              ) -> nn.Module:
        """真正建网络（参数在这一刻才分配）。``in_dim/nc`` 缺任一个时从数据推。

        拿到数据时顺手校验输出层级：``net.level`` 必须等于 ``ds.supervision``。
        两者都是二维张量，接错了不会报错，只会让每个指标都静默错位。
        """
        nc = self.nc if nc is None else int(nc)
        probe: Optional[GraphBundle] = None
        if in_dim is None or nc is None:
            probe = data if isinstance(data, GraphBundle) \
                else _load_any(self.cfg, data)
            in_dim = probe.num_features if in_dim is None else int(in_dim)
            nc = probe.num_classes if nc is None else int(nc)
        net = build_model(self.yaml, int(in_dim), nc=int(nc),
                          scale=self.cfg.model.scale, oca=self.cfg.model.oca,
                          dropout=self.cfg.model.dropout,
                          device=resolve_device(device or self.cfg.device),
                          verbose=self.verbose if verbose is None else verbose,
                          spec_overrides=self.cfg.spec_overrides())
        if probe is not None:
            from tasks import assert_level
            self.task = assert_level(net, probe, self.cfg)
        if self._weights is not None:
            from training.checkpoints import load_checkpoint
            load_checkpoint(self._weights, net, device or self.cfg.device)
            _log.info('已加载权重：%s（epoch=%s）', self._weights,
                      self._meta.get('epoch'))
        self.net = net
        self._built = {'in_dim': int(in_dim), 'nc': int(nc)}
        return net

    def _require_net(self, what: str) -> nn.Module:
        if self.net is None:
            raise RuntimeError(
                f'{what} 需要先构建模型：先 .build(in_dim=..., nc=...) 或 .train(...)')
        return self.net

    # ------------------------------------------------------------------ 前向
    def forward(self, x: Tensor, edge_index: Tensor,
                batch: Optional[Dict[str, Any]] = None) -> Tensor:
        return self._require_net('forward')(x, edge_index, batch=batch)

    __call__ = forward

    def forward_ds(self, ds: GraphBundle) -> Tensor:
        """``net(**ds.forward_kwargs())`` 的简写。

        走数据集自己报的入参而不是手写 ``(ds.x, ds.edge_index)``：关系图的 bundle
        还要带 ``batch``（关系类型/边特征/节点类型），少传一个就是静默少一层信息。
        """
        return self._require_net('forward_ds')(**ds.forward_kwargs())

    @torch.no_grad()
    def predict(self, data: Union[str, Path, GraphBundle, None] = None,
                x: Optional[Tensor] = None, edge_index: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                batch: Optional[Dict[str, Any]] = None) -> Dict[str, Tensor]:
        """全图推理：返回 ``logits`` / ``prob`` / ``pred``（``mask`` 给了就只留那些单元）。

        单 logit（``nc==1``）时走 ``cfg.task.threshold`` 而不是 ``argmax`` ——
        ``argmax`` 在 ``[N,1]`` 上恒返回 0，不报错，只把所有单元归成负类。
        """
        if x is None or edge_index is None:
            ds = data if isinstance(data, GraphBundle) else _load_any(self.cfg, data)
            kw = ds.forward_kwargs()
            self.build(in_dim=ds.num_features, nc=ds.num_classes, data=ds)
        else:
            kw = {'x': x, 'edge_index': edge_index, 'batch': batch}
        net = self._require_net('predict').eval()
        logits = net(**kw)
        binary = logits.size(-1) == 1
        prob = torch.sigmoid(logits) if binary else torch.softmax(logits, -1)
        pred = ((prob.squeeze(-1) > float(self.cfg.task.threshold)).long()
                if binary else logits.argmax(-1))
        out = {'logits': logits, 'prob': prob, 'pred': pred}
        if mask is not None:
            out = {k: v[mask] for k, v in out.items()}
        return out

    # ------------------------------------------------------------------ 训练
    def train(self, mode: Union[bool, str, Path, GraphBundle] = True,
              **kwargs: Any) -> Union['GraphModel', Dict[str, Any]]:
        """双签名（同 yolov8）：``train(False)`` 切评估模式；其余启动训练。

        ``train(data='cora', epochs=200, scale='s', oca={'T': 4})`` —— kwargs 里
        任何 ``TrainConfig`` 认识的键都可以写（包括 ``dia={...}`` 与
        ``task='edge_classification'``），未知键直接报错。
        """
        if not isinstance(mode, bool):
            mode, kwargs = True, {'data': mode, **kwargs}
        if not mode:
            if self.net is not None:
                self.net.eval()
            return self
        kw = dict(kwargs)
        verbose = bool(kw.pop('verbose', self.verbose))
        ds = kw.pop('data', None)
        if ds is not None and not isinstance(ds, (str, Path, GraphBundle)):
            raise TypeError(f'data 需要是数据集名/路径/GraphBundle，收到 {type(ds)}')
        cfg = self._patch(self.cfg, kw)
        self.cfg = cfg
        from training.trainer import run_experiment
        out = run_experiment(cfg, ds=ds if isinstance(ds, GraphBundle) else None,
                             verbose=verbose)
        self.results, self.run_dir = out, out.get('run_dir')
        r0 = out['runs'][0]
        self.net = r0.model                        # 第一个种子的最优权重
        if r0.hooks is not None and r0.hooks.task:
            self.task = r0.hooks.task              # 'auto' 已被解析成具体任务名
        self._built = {'in_dim': out['dataset'].num_features,
                       'nc': out['dataset'].num_classes}
        return out

    def val(self, data: Union[str, Path, GraphBundle, None] = None,
            **kwargs: Any) -> Dict[str, float]:
        """在 val/test 上各评一次（模型必须已构建且带权重）。

        判决口径走 :func:`training.losses.build_criterion`，与训练时一致：
        单 logit 的任务用阈值而不是 ``argmax``。
        """
        kw = dict(kwargs)
        cfg = self._patch(self.cfg, kw)
        ds = data if isinstance(data, GraphBundle) else _load_any(cfg, data)
        if self.net is None:
            self.build(in_dim=ds.num_features, nc=ds.num_classes, data=ds)
        net = self._require_net('val')
        from training.losses import build_criterion
        from training.trainer import evaluate
        crit = build_criterion(cfg, ds, resolve_device(cfg.device))
        return {'val': evaluate(net, ds, ds.val_mask, crit=crit),
                'test': evaluate(net, ds, ds.test_mask, crit=crit)}

    # ------------------------------------------------------------------ 查询
    def info(self, verbose: Optional[bool] = None, in_dim: Optional[int] = None,
             nc: Optional[int] = None) -> Dict[str, Any]:
        """返回（按需打印）结构摘要。还没构建时可以传 ``in_dim/nc`` 预演一次。"""
        built = self.net is not None
        verbose = self.verbose if verbose is None else bool(verbose)
        # 逐层表是在**构建时**打印的（与 yolov8 同），所以 verbose 必须透传给 build，
        # 否则 ``--dry-run`` 只能看到一行汇总，看不到「哪一行结构写错了」。
        net = self.net if built else self.build(in_dim=in_dim, nc=nc,
                                                verbose=verbose)
        spec = model_info(net, verbose=verbose)
        spec.update({'spec': self.cfg.model.spec, 'scale': self.cfg.model.scale,
                     'oca': dict(self.cfg.model.oca),
                     'dia': dict(self.cfg.model.dia),
                     'task': self.cfg.task.task_type,
                     'level': getattr(net, 'level', 'node'),
                     'in_dim': in_dim, 'nc': nc,
                     'device': describe_device(self.cfg.device)})
        if not built:
            self.net, self._built = None, None      # 预演不留半个网络
        return spec

    def to(self, device: Union[str, torch.device]) -> 'GraphModel':
        if self.net is not None:
            self.net.to(resolve_device(device) if isinstance(device, str)
                        else device)
        return self

    @property
    def explainable_layers(self) -> nn.ModuleList:
        """全部可解释层（:class:`modules.base.Explorable`），逐层读 ``aux``。

        不区分算子：OCA 的 λ/α/τ 与 DIA 的 γ/M/α 都在这一列表里，靠
        ``layer.family`` 区分。
        """
        return self._require_net('explainable_layers').explainable_layers

    @property
    def oca_layers(self) -> nn.ModuleList:
        """全部 OCA 层（``family='oca'``），供逐层读 ``aux``（λ/α 分析用）。"""
        net = self._require_net('oca_layers')
        return getattr(net, 'oca_layers', nn.ModuleList())

    def __repr__(self) -> str:
        state = ('built' if self.net is not None else
                 'weights' if self._weights is not None else 'config-only')
        return (f'{type(self).__name__}(spec={self.cfg.model.spec}, '
                f'scale={self.cfg.model.scale}, task={self.task}, '
                f'nc={self.nc}, state={state})')


class OCA(GraphModel):
    """默认结构表指向 ``cfg/models/oca.yaml``（节点分类）。

    保留 ``OCA(...)`` 这个写法是为了向后兼容（脚本、``train.py``、``docs/OCA.md``
    里全是它）；新代码写 ``GraphModel('cfg/models/oca.yaml')`` 也一样。
    """

    default_spec = 'cfg/models/oca.yaml'
    default_task = 'node_classification'


class DIA(GraphModel):
    """默认结构表指向 ``cfg/models/dia.yaml``。

    任务**不在这里写死**（``default_task='auto'``）：节点级的 ``dia.yaml`` 配节点
    数据集、边级的 ``dia_rel.yaml`` 配关系图，同一个类都能跑。这正是「算法与任务
    正交」的体现；写死成 ``edge_classification`` 会让「拿 DIA 做节点分类」这个
    消融行无法表达。
    """

    default_spec = DIA_SPEC
    default_task = 'auto'


def _int_or_none(v: Any) -> Optional[int]:
    return None if v is None else int(v)


def _load_any(cfg: TrainConfig, data: Union[str, Path, GraphBundle, None]
              ) -> GraphBundle:
    """``data`` 是名字就按名字装载（其余口径全用 ``cfg.dataset``），是 bundle 就原样返回。"""
    from training.trainer import load_data
    if isinstance(data, GraphBundle):
        return data
    if data is None:
        return load_data(cfg)
    return load_data(cfg.patched(**{'dataset.name': str(data)}))
