"""测试包。

必须是**常规包**（有本文件）：Python 解析 namespace package 的优先级低于常规包，
若缺 ``__init__.py``，``from test.support import ...`` 会命中标准库的 ``test`` 包。
"""
