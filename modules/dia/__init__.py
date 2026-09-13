r"""DIA：维度级可识别注意力（技术文档《维度级可识别注意力机制（DIA）》的实现）。

文件划分照文档附录的 ``dia/`` 提案，一个文件一层（对应 ultralytics 把算子按类别拆进
``nn/modules/{conv,block,head}.py`` 的做法）：

| 文件 | 内容 | 文档 |
| :-- | :-- | :-- |
| ``config.py``     | :class:`DIAConfig`：三层开关 + 约束开关，消融表的单一事实源 | §2.2–§2.6 |
| ``attention.py``  | :class:`EdgeConditionedDimAttention`（L1 逐维筛选 :math:`\gamma`） | §2.2 |
| ``pairing.py``    | :class:`LowRankNonNegPairing`（L2 :math:`W=UV^\top`，非负+稀疏+正交） | §2.3、§2.7 |
| ``asymmetric.py`` | :class:`AsymmetricContribution`（L3 主导权 :math:`\alpha`） | §2.4 |
| ``layer.py``      | :class:`DIALayer`：三层拼装 + 约束/惩罚/解释三个协议的落地点 | §2.2–§2.6 |
| ``conv.py``       | :class:`DIAConv`：节点分类用的消息传递层（``cfg/models`` 可引用） | §2.5 |
| ``head.py``       | :class:`EdgeScore`：边分类读出头（输出 ``[E,nc]``） | §4.5 |
| ``edges.py``      | ``batch`` 契约（rel / edge_attr / node_type）与对称化的职责划分 | §三 |
| ``explain.py``    | 支撑集、单边解释、分数归因、参数导出（无绘图） | §4.8、§五 |

框架侧（``model_builder``/``training``/``tasks``）**不 import 本包**：注册靠
``@register_module``，约束/惩罚/解释靠 :mod:`modules.base` 的三条协议被发现。
"""

from modules.dia.asymmetric import AsymmetricContribution
from modules.dia.attention import EdgeConditionedDimAttention
from modules.dia.config import DIAConfig
from modules.dia.conv import DIAConv
from modules.dia.head import EdgeScore
from modules.dia.layer import DIALayer
from modules.dia.pairing import LowRankNonNegPairing

__all__ = ['DIAConfig', 'DIALayer', 'EdgeConditionedDimAttention',
           'LowRankNonNegPairing', 'AsymmetricContribution', 'DIAConv',
           'EdgeScore']
