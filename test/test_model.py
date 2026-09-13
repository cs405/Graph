"""骨架与训练可用性：shape、梯度、门控是否真的能被学。

结构现在全部来自 ``cfg/models/*.yaml``（构建细节由 ``test_model_builder.py`` 覆盖），
本文件只管「搭出来的东西能训」：

* ``test_gradients_finite_and_gates_learnable`` 是 P1-3 的最低门槛：
  若 ``dL/dw_lambda`` 恒为 0，「自适应竞争强度」这个故事当场失效。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                    # noqa: E402
import torch.nn.functional as F                                   # noqa: E402

from dataset.synthetic import ba_graph                            # noqa: E402
from model_builder import build_model                               # noqa: E402
from modules.oca import OCAConfig, OCALayer                         # noqa: E402
from test.support import case, run_registered                      # noqa: E402


@case
def test_cfg_built_net_shapes():
    """多尺度骨架（oca.yaml）：shape、层数、逐层输出宽度。"""
    x, ei = ba_graph(n=80, m=3, seed=1, feat_dim=12)
    n = x.size(0)
    net = build_model('cfg/models/oca.yaml', 12, nc=5, verbose=False)
    assert net(x, ei).shape == (n, 5)
    assert len(net.oca_layers) == 4, '4 个 OCA 层 => aux 取不到各层则 λ 分析无从谈起'
    widths = [m.out_dim for m in net.model]
    assert widths[:4] == [16] * 4, widths           # scale n => width_multiple 0.25
    assert widths[-1] == 5, widths                  # 最后一层是 Classify，输出 nc
    print('  oca.yaml（4 尺度 + 融合 + 读出）OK   '
          f'in_dim=12 → hidden={widths[0]}，宽度序列 {widths}')


@case
def test_gradients_finite_and_gates_learnable():
    """5 步 Adam 后：所有参数梯度有限，且 λ 门控在「默认不竞争」初始化下仍收到非零梯度。"""
    x, ei = ba_graph(n=120, m=3, seed=1, feat_dim=12)
    n = x.size(0)
    y_true = torch.randint(0, 4, (n,))
    model = OCALayer(x.size(1), OCAConfig(out_dim=32, heads=4, T=2))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    first = None
    for _ in range(5):
        opt.zero_grad()
        loss = F.cross_entropy(model(x, ei), y_true)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name
        g = model.w_lambda.weight.grad.abs().sum().item()
        first = g if first is None else first
        opt.step()
    assert first > 0, 'λ 门控梯度恒 0 —— 自适应故事不成立'
    lam = model.aux['lambda']
    assert loss.item() < 5.0
    print(f'  梯度 OK   loss={loss.item():.3f}  dL/dw_λ={first:.3e}  '
          f'λ: mean={lam.mean():.3f} std={lam.std():.3f}')


@case
def test_whole_net_gate_gradients():
    """整网（含融合与读出）也要能把梯度传回每层的 λ/α —— 只测单层会漏掉「融合层把门控梯度吃掉」。"""
    x, ei = ba_graph(n=100, m=3, seed=2, feat_dim=10)
    y = torch.randint(0, 4, (x.size(0),))
    net = build_model('cfg/models/oca.yaml', 10, nc=4, verbose=False)
    net.train()
    loss = F.cross_entropy(net(x, ei), y)
    loss.backward()
    for j, lay in enumerate(net.oca_layers):
        for pname in ('w_lambda', 'w_alpha'):
            gate = getattr(lay, pname, None)          # nn.Linear，不是 Parameter
            if gate is None:                          # 开关关掉时不分配该参数
                continue
            g = gate.weight.grad
            assert g is not None and torch.isfinite(g).all(), (j, pname)
            assert g.abs().sum() > 0, f'第 {j} 层 {pname} 梯度为 0'
    print(f'  整网反传 OK   loss={loss.item():.3f}，'
          f'{len(net.oca_layers)} 层的 λ/α 都收到非零梯度')


@case
def test_baseline_specs_forward_same_contract():
    """基线必须与 OCA 吃同一套 ``(x, edge_index)`` 调用约定，否则对照表不可信。"""
    x, ei = ba_graph(n=90, m=3, seed=1, feat_dim=12)
    n = x.size(0)
    for spec in ('cfg/models/mlp.yaml', 'cfg/models/gcn.yaml',
                 'cfg/models/gat.yaml', 'cfg/models/oca_gat.yaml'):
        net = build_model(spec, 12, nc=5, dropout=0.5, verbose=False).eval()
        with torch.no_grad():
            out = net(x, ei)
        assert out.shape == (n, 5), spec
        assert torch.isfinite(out).all(), spec
    print('  mlp/gcn/gat/oca_gat 结构表 forward OK（同调用约定）')


@case
def test_dropout_changes_output_in_train_mode():
    """``Dropout`` 真在图里生效（读出层接的是 cfg.model.dropout）。"""
    x, ei = ba_graph(n=80, m=3, seed=3, feat_dim=12)
    net = build_model('cfg/models/oca.yaml', 12, nc=5, dropout=0.5,
                      verbose=False)
    torch.manual_seed(0)
    net.train()
    a = net(x, ei)
    net.eval()
    b = net(x, ei)
    assert not torch.allclose(a, b), 'train/eval 输出一致 => dropout 没接上'
    assert torch.isfinite(b).all()
    print('  dropout 生效 OK   train≠eval（eval 输出确定）')


if __name__ == '__main__':
    sys.exit(run_registered('== 骨架 / 梯度 / 基线 =='))
