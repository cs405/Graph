"""评估指标：分类打分 + 同质性（与算子无关的独立实现）。"""

from metrics.homophily import (edge_homophily, gate_homophily_report,
                               local_homophily, neighbor_label_entropy,
                               pearson, spearman)
from metrics.scoring import accuracy, macro_f1, prf1, summarize

__all__ = ['accuracy', 'macro_f1', 'prf1', 'summarize', 'edge_homophily',
           'local_homophily', 'neighbor_label_entropy', 'pearson', 'spearman',
           'gate_homophily_report']
