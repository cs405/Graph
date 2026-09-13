"""算子与结构层。

分层原则与 yolov8 的 ``nn/modules`` 一致，但**依赖方向反过来**：本包不认识
``model_builder``/``training``，只认识 :mod:`modules.base` 里的契约；框架侧反过来
只认识契约（注册表 + 四条协议），不 import 这里任何一个具体算子。

* :mod:`modules.base` —— 契约本身：``MODULES`` 注册表、``GraphOp``、``Constraint``、
  ``Regularized``、``Explorable``，以及框架侧用的发现函数；
* :mod:`modules.oca` —— 自研算子 OCA（含退化到 GAT 的开关
  :attr:`OCALayer.gat_equivalent`）；
* :mod:`modules.dia` —— 自研算子 DIA（配置 / 三层组件 / 拼装 / 可解释性出口）；
* :mod:`modules.convs` —— 能被 ``cfg/models/*.yaml`` 引用的结构层（算子包装 +
  GAT/GCN/MLP 外基线 + DIAConv）；
* :mod:`modules.neck` —— 多输入融合（``Merge``/``Concat``）；
* :mod:`modules.head` —— 读出层（``Classify`` + ``EdgeScore``）。

这里**不堆拓扑**：拓扑在 ``cfg/models/*.yaml`` 里，由 :mod:`model_builder` 展开。
``@register_module`` 在 import 时登记，所以只要本包被 import 过，注册表就是满的
（``model_builder`` 只 ``import modules``，不点名任何类）。
"""

from modules.base import (MODULES, Constraint, Explorable, GraphOp, Regularized,
                          SPEC_BLOCKS, apply_constraints, collect_aux,
                          collect_penalties, explain_report, explainable_layers,
                          register_module)
from modules.convs import ACTS, DIAConv, GATBlock, GCNBlock, LinearBlock, OCAConv
from modules.dia import (DIAConfig, DIALayer,
                         EdgeConditionedDimAttention,
                         LowRankNonNegPairing)
from modules.head import Classify, EdgeScore
from modules.neck import Concat, Merge
from modules.oca import OCAConfig, OCALayer

__all__ = ['MODULES', 'SPEC_BLOCKS', 'GraphOp', 'Constraint', 'Regularized',
           'Explorable', 'register_module', 'apply_constraints', 'collect_aux',
           'collect_penalties', 'explain_report', 'explainable_layers',
           'OCAConfig', 'OCALayer', 'OCAConv', 'GATBlock', 'GCNBlock',
           'LinearBlock', 'ACTS', 'Merge', 'Concat', 'Classify',
           'DIAConfig', 'DIALayer', 'DIAConv', 'EdgeScore',
           'EdgeConditionedDimAttention', 'LowRankNonNegPairing']
