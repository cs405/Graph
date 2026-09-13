"""临时冒烟脚本：验证 modules 注册表与 DIALayer 三层前向（跑完即删）。"""
import torch

import modules
from modules.dia import DIAConfig, DIALayer

print(sorted(modules.MODULES))
lay = DIALayer(6, 4, DIAConfig(out_dim=6, rank=2))
h, g = torch.rand(5, 6), torch.rand(5, 4)
out = lay(h, g, want=('score', 'message', 'matrix', 'factors'))
for k, v in out.items():
    print(' ', k, tuple(v.shape) if torch.is_tensor(v) else
          [tuple(t.shape) for t in v])
print('explain:', lay.explain())
print('penalties:', {k: float(v) for k, v in lay.penalties().items()})
lay.project_parameters()
print('U_min after project:', float(lay.pairing_ij.U.min()))
# 反向传播能通吗
loss = out['score'].sum() + out['m_ij'].sum() + sum(lay.penalties().values())
loss.backward()
print('no grad:', [n for n, p in lay.named_parameters() if p.grad is None])
print('grad ok:', lay.pairing_ij.U.grad is not None
      and lay.pairing_ji.V.grad is not None)
