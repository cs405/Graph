"""全量测试入口（环境里没有 pytest，这是唯一的聚合跑法）。

    python -m test.run_all                # 全部
    python -m test.run_all oca parity     # 只跑模块名含 'oca'/'parity' 的文件
    python test/test_oca_energy.py        # 单文件也可直跑（各自带 __main__）

实现细节：各用例文件的 ``@case`` 装饰器把函数注册进 :data:`test.support.RESULTS`
这一个全局列表，所以**先 import 全部模块再统一执行**，顺序即文件顺序。
"""

from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test.support import run_registered                     # noqa: E402

# 顺序有意为之：先算子正确性（parity/energy/degeneration/stability），再配置与
# 构建（两者决定「跑的是什么」），最后上层（model/dataset/metrics/training）
# —— 底层挂了就没必要看上面的红。
MODULES = [
    'test_oca_parity',
    'test_oca_energy',
    'test_oca_degeneration',
    'test_oca_stability',
    'test_config',
    'test_model_builder',
    'test_model',
    'test_dataset',
    'test_metrics',
    'test_training',
]


def main(argv: list) -> int:
    picked = [m for m in MODULES if not argv or any(a in m for a in argv)]
    if not picked:
        print(f'没有匹配 {argv} 的测试模块，可选：{MODULES}')
        return 2
    for name in picked:
        importlib.import_module(f'test.{name}')
        print(f'  · 已装载 test/{name}.py')
    print(f'== 全量测试：{len(picked)} 个模块 ==\n')
    return run_registered()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
