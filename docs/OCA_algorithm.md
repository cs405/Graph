# OCA 算法细节（修订后：公式 · 实现 · 实测结果）

> **这份文档只放算法本体。** 每条公式对齐 [`modules/oca.py`](../modules/oca.py) 的真实计算图（不是设计稿里"应该这样"的写法），每个数字都标了取数命令，全部是实测产出、没有一个是估计值。
> 你原始设计的**目标、动机、主张、卖点、待验清单**在 [`docs/OCA.md`](OCA.md)，那份不再放实现细节，两份不重复维护：**数学与工程现状以本文为准，研究主张与目标以 OCA.md 为准。**
>
> 取数环境（换环境数字可能差最后一位）：Python 3.13.15 · torch 2.14.0+cu132 · torch-geometric 2.8.0.post1 · PyYAML 6.0.3 · 无 pytest（自带 `test/run_all.py`）。数值对拍一律 **CPU + float64** —— `index_add_` 在 CUDA 上不满足结合律，位级不一致会让 $10^{-16}$ 量级的断言随机失败。

------

## 0. 记号与三条硬约定

| 记号 | 含义 |
| :-- | :-- |
| $\mathcal V_i=\mathcal N(i)\cup\{i\}$ | 中心 $i$ 的竞争场，$\lvert\mathcal V_i\rvert=\deg_i+1$（代码里 `degp`） |
| $\hat{\mathcal E}=\mathcal E\cup\{(i,i)\}$ | 增广边集；**分数 $s$ 是 $\hat{\mathcal E}$ 上的量**，$s_{i,a}$ 逐 (中心, 槽位) |
| $c_a=\mathbb 1[a=i]$ | 槽位是否为中心（代码里 `c_a` = `1 - nb`，`nb = ~is_self`） |
| $H,\ d,\ d_h$ | 头数、层输出宽度、每头宽度 $d/D$；`out_dim % heads != 0` 在 `__post_init__` 直接 `assert` |
| $p$ | 本层输入维度 `in_dim`（由 `parse_model` 逐行推导，不写死） |

三条约定，后面所有推导都建立在它们上面：

1. **`edge_index` 用 [row = 中心 $i$，col = 槽位 $a$]**，与 PyG `MessagePassing` 的默认方向相反。无向数据集无差别，有向输入是否先 `to_undirected` 由 `OCAConfig.symmetrize`（默认 `True`）控制。
2. **$s$ 定义在增广边集上，不是节点上。** 同一节点 $a$ 在不同中心的竞争场里分数不同。这是 §2.9 因子化成立的前提，也是稠密 padding 写法（v1）掩盖掉的。
3. **$\beta\in(0,1)$、$\lambda_i\in(0,1)$、$\kappa\ge0$ 是实现的一部分，不是调参偏好。** `beta` 越界 `OCAConfig` 直接 `assert`；非负性由核的 $+1$ 偏移保证，Lemma 2 的两步不等式都吃它。

------

## 0.5 代码地图

| 路径 | 内容 |
| :-- | :-- |
| [`cfg/models/*.yaml`](../cfg/models) | **结构表**（拓扑 + `scales` + 算子默认超参）：`oca` / `oca1`（单尺度对照）/ `oca_deep`（深度缩放示范）/ `oca_gat`（同骨架 GAT 退化对照）/ `oca_nocomp`（只关竞争场的归因对照，§9.13）/ `gat` / `gcn` / `mlp` |
| [`cfg/train/default.yaml`](../cfg/train/default.yaml) | **训练超参**，四个块 `run` / `dataset` / `model` / `optimization`（§7） |
| [`config.py`](../config.py) | 嵌套 `dataclass` 配置树、YAML 加载与强校验、`config_used.yaml` 快照 |
| [`model_builder.py`](../model_builder.py) | `yaml_model_load` / `parse_model` / `GraphSequential` / `Repeat` / `build_model` / `model_info` / `MODULES` 注册表（§6） |
| [`model.py`](../model.py) | `class OCA`：`build` / `train` / `val` / `predict` / `info`，懒构建（`in_dim`/`nc` 只能来自数据，所以 `OCA(...)` 这一步不分配参数） |
| [`modules/`](../modules) | `oca.py`（`OCAConfig` / `OCALayer`，含 `gat_equivalent` 工厂与 `forward_dense` 参考实现）、`convs.py`（GAT / GCN / MLP 基线块）、`neck.py`（`Merge` / `Concat`）、`head.py`（`Classify`） |
| [`dataset/`](../dataset) | `GraphBundle` + 公共 splits + `synthetic.py`（BA / 异配合成图）+ `real.py`（`torch_geometric.datasets`）+ `hetero_probe.py`（**SOY/CORN 数据体检**，§9.5–§9.11 每个数字的取数脚本，不参与训练）+ `acad_probe.py`（把同一判据搬到标准异配基准上重测，§9.12，同样不参与训练）+ `acad_bench.py`（学术图外部基线表：mlp / floor+ / gcn / gat / oca 走同一条 `trainer.fit`，附逐配对单元存档，§9.13） |
| [`training/`](../training) | `trainer.py`（`fit` / `run_seeds` / `run_experiment` / `evaluate`）、`runs.py`（运行目录与落盘）、`checkpoints.py`（滚动 checkpoint）、`diagnostics.py`（$\lambda$-同质性相关性、门控落盘）、`sweep.py`（§5 消融表） |
| [`metrics/`](../metrics) | `scoring.py`（acc / P / R / F1 / macro-F1）；`homophily.py` 与算子**完全无关**地独立实现局部同质性，避免用它自己的输出反推自己 |
| [`utils/`](../utils) | `augment_edge_index`、`slots_by_center`、`scatter_add` 等图工具；`seed.py`（RNG 快照） |
| [`devices.py`](../devices.py) · [`log_setup.py`](../log_setup.py) | 设备解析/描述；标准 `logging` 的 console + file handler（本项目不装 loguru） |
| [`train.py`](../train.py) | CLI 入口：`--model` / `--scale` / `--set` / `--oca` / `--ablate` / `--dry-run`（§7） |
| [`test/`](../test) | `python -m test.run_all` 跑 10 个文件 59 个用例；按名字过滤 `python -m test.run_all oca parity`，单文件直跑 `python test/test_oca_energy.py` |

脚本里的对象接口（yolov8 风格）：

```python
from model import OCA

OCA('cfg/models/oca.yaml').train(data='cora', scale='n', epochs=200)
OCA('cfg/models/oca.yaml').info(in_dim=1433, nc=7, verbose=True)   # 不碰数据也能预演结构
```

------

## 1. 一层的七个阶段（与代码函数一一对应）

`OCALayer.forward(x, edge_index)`，形状 $x\in\mathbb R^{N\times p}$：

| # | 代码 | 产出 | 复杂度 |
| :-- | :-- | :-- | :-- |
| 1 | `utils.graph.augment_edge_index` | `ei`$=[2,\hat E]$、`is_self`$=[\hat E]$ | $O(\hat E)$ |
| 2 | `_gates` | $\lambda,\alpha\in\mathbb R^{N\times1}$ | $O(Np)$ |
| 3 | `self.W` → `_edge_terms` | $z\in\mathbb R^{N\times d}$；边上量 $b,\phi,c\in\mathbb R^{\hat E\times H}$ | $O(\hat E d)$ |
| 4 | `_kernel_ctx` | $\hat q$、$C_i$、$\kappa_{a,i}$、$\kappa_{aa}$、$D^{\text{raw}}$ | $O(\hat E d)$ |
| 5 | `_competition_step` × $T$ | $s\in\mathbb R^{\hat E\times H}$（含 §2.5 投影） | $O(T\hat E d)$ |
| 6 | `_temperature` | $\tau\in\mathbb R^{N\times1}$ | $O(NH+\hat E H)$ |
| 7 | `_readout` | $o\in\mathbb R^{N\times d}$ | $O(\hat E d)$ |

每次 `forward` 开头 `self.aux.clear()` —— 不这样会读到上一次（可能是另一套开关或后端）的残留。`aux` 暴露的量：`s`、`t`（边上中间量）、`z`、`ctx`、`tau`、`s_center`、`lambda`、`alpha`。逐节点 $\lambda$/$\alpha$ 落盘走 `run.dump_gates`。

第 1 步的顺序是必须的（`add_self_loops` 不为已有自环去重，直接加会重复计数）：

```python
def augment_edge_index(edge_index, num_nodes, symmetrize=True):
    ei, _ = remove_self_loops(edge_index)
    if symmetrize:
        ei = to_undirected(ei, num_nodes=num_nodes)
    ei = coalesce(ei, num_nodes=num_nodes)     # num_nodes 必须是整数
    ei, _ = add_self_loops(ei, num_nodes=num_nodes)
    return ei, ei[0] == ei[1]
```

`test/test_oca_parity.py::test_directed_dupe_input` 用带自环 + 重边 + 单向的输入验证这一步（实测 `E_hat=392`、`max_deg=13`）。

------

## 2. 公式（逐条对齐代码）

### 2.1 竞争场与投影

$$z_a=W h_a,\qquad a\in\mathcal V_i$$

中心和邻居走同一个 $W$，不搞中心专属变换。多头是 $z.\mathrm{view}(N,H,d_h)$。

### 2.2 打分与初值

纵向提案（`score_mode='dot'`，`b` 在中心槽位被 `nb` 屏蔽 —— 中心不给自己打分）：

$$b_{i,a}=\frac{\langle W_q h_i,\ W_k z_a\rangle}{\sqrt{d_h}}\,\mathbb 1[a\ne i]$$

`score_mode='gat'` 时换成 GAT 的加法式（这是 §3 七条件之一）：

$$b_{i,a}=\mathrm{LeakyReLU}_{0.2}\!\big(a_{\mathrm{src}}^{(h)}+a_{\mathrm{dst}}^{(h)}\big)\,\mathbb 1[a\ne i],\qquad [a_{\mathrm{src}}\,\|\,a_{\mathrm{dst}}]=\mathrm{Linear}(z_a)\in\mathbb R^{2H}$$

自身嗓门（中心槽位**同样有**，这是 v1 代码丢掉的一项）：$\phi(z_a)=\mathrm{MLP}(z_a)\in\mathbb R^{H}$，两层 `Linear(d,d)→ReLU→Linear(d,H)`。

**迭代初值 = 驱动项（设计稿口径，代码已对齐）：**

$$s_{i,a}(0)=c_{i,a}=\phi(z_a)+(1-\alpha_i)\,b_{i,a}$$

$T$ 轮 PGD 的每一步也都用同一个 $c$（`_competition_step` 里的 `t['c'] - lam_e * r`），所以初值与驱动项同式后，$T=0$ 与 $T\to\infty$ 之间不再有“哪个带 $\alpha$”的差别。Lemma 1 只要求驱动项为 $c$，对初值无约束 ⇒ 这一改动不动收敛点 $s^\star$，只改暂态。

代码口径落在一处：`s = t['c']`，但**稀疏与稠密两条路径各写一次** —— `forward` 的 $T>0$ 分支、$T=0$ 分支、`_forward_dense` 的 `s_out` 共三处，改一处必须三处一起改，否则 §8.1 的 sparse==dense 对拍会挂。这一度是代码与设计稿唯一剩下的字面差异（旧实现初值取 $\phi+b$），已按 §10 第 1 行的裁决改掉；由于 $\alpha_i\equiv0$ 时 $c=\phi+b$，严格 GAT 退化（§3）与所有关掉 `use_center_field` 的行**逐位不变**，被挪动的只有 $\alpha>0$ 的配置（§8.4 已重跑）。

### 2.3 抑制核、$\alpha$ 屏蔽、归一化

令 $q_a=q_\theta z_a$、$\hat q_a=q_a/\lVert q_a\rVert$：

$$\kappa_{ab}=\langle\hat q_a,\hat q_b\rangle+1\in[0,2]$$

$+1$ 偏移两个作用缺一不可：**(a)** 保非负 ⇒ 行和 $D_a$ 等于绝对值行和，GCN 式归一化的界才成立（Lemma 2）；**(b)** $\kappa$ 仍是 $[\hat q;1]$ 的 Gram 矩阵 ⇒ **线性可因子化**（§2.9）。代价是稀疏路径放弃 $\mathrm{ReLU}$：ReLU 分支需要 $O(\sum_i\deg_i^2)$，只在 `backend='dense'` 参考实现里保留，`backend='sparse', kernel='relu'` 直接抛 `NotImplementedError`。

$$M^{(i)}_{ab}=\alpha_i+(1-\alpha_i)(1-c_a)(1-c_b)=1-(1-\alpha_i)\big[c_a+c_b-c_ac_b\big]\in[\alpha_i,1]$$

邻居–邻居为 $1$，凡碰到中心的行列乘 $\alpha_i$；$\alpha_i=0$ ⇒ 中心的 $\kappa$ 行列整体消失（真 B 模式）。

$$A:=M^{(i)}\!\circ\kappa^{(i)}\ (\ge0),\qquad D_a=\sum_{b\in\mathcal V_i}A_{ab},\qquad D'_a=\max(D_a,1)$$

$$\tilde\kappa^{(i)}_{ab}=\frac{A_{ab}}{\sqrt{D'_a}\sqrt{D'_b}}\Big|_{a\ne b},\qquad \tilde\kappa^{(i)}_{aa}=0$$

`clamp(min=1)` 不是数值美化：$D_a$ 真的会趋于 0（孤立节点，或 $\alpha_i\to0$ 的中心行），而它只放大分母，按元素级只会让非负核变小 —— 这正是 Lemma 2 第一步要用的事实。

### 2.4 迭代 = 一步投影梯度下降

$$\boxed{\;s_{i,a}\leftarrow(1-\beta)s_{i,a}+\beta\Big(c_{i,a}-\lambda_i\!\sum_{b\in\mathcal V_i}\tilde\kappa^{(i)}_{ab}s_{i,b}\Big)\;}$$

| 项 | 含义 | 谁控制 |
| :-- | :-- | :-- |
| $(1-\alpha_i)b_{i,a}$ | 纵向提案，只作用邻居，$\alpha$ 越大越弱 | $\alpha_i$ |
| $\phi(z_a)$ | 自身嗓门，中心与邻居都有 | 无 |
| $\lambda_i\sum_b\tilde\kappa_{ab}s_b$ | 横向抑制，经归一化 | $\lambda_i$ |
| $M^{(i)}_{ab}$ | 中心进场程度（$\alpha$ 的第二条通路） | $\alpha_i$ |

`detach_iterations=True` 时前 $T-1$ 轮 `s.detach()`（最后一轮可微）。$T=0$（或 `use_competition=False`）时既不迭代也不中心化，直接 $s=\phi+(1-\alpha_i)b$：$\lambda$ 此时完全不进计算，但 $\alpha$ 仍通过初值进读出权重 —— 所以 `T0` 那一行是所有行里对初值口径最敏感的。

### 2.5 中心化 = 投影步

$$s_{i,a}\leftarrow s_{i,a}-\frac1{\lvert\mathcal V_i\rvert}\sum_{b\in\mathcal V_i}s_{i,b}$$

准确定位（四条，别多也别少）：

- 把 $s$ 投到 $\mathbf 1^\top s=0$ 子空间，等价于对 §4 的能量做**投影**梯度下降，不是无约束最小化；
- 对最终 readout 的 softmax 是 **no-op**（§2.8 只对 $\mathcal N(i)$ 归一，平移不变）⇒ 它的唯一作用是数值防漂移；
- 但它**确实影响后续迭代**（$\tilde\kappa$ 行和非零，漂移会被竞争项重新放大），所以每轮都要做；
- $\kappa$ 的 $+1$ 偏移在中心化后对分子无贡献，只通过 $D$ 起作用。

### 2.6 门控生成

$$\lambda_i=\sigma(w_\lambda^\top h_i),\qquad \alpha_i=\sigma(w_\alpha^\top h_i)$$

都由**原始输入** $h_i$（不是 $z_i$）生成。初始化：$\lambda$ 的 bias $=-2$（$\sigma(-2)\approx0.12$）、$\alpha$ 的 bias $=0$ ⇒ **默认不竞争，竞争要靠训练挣来**（v1 的"前 10 epoch 把门控钉在 0.5 再释放"已删：起点在中间会让"自动选模式"的故事静默失败，事后分不清是学出来的还是初始化剩的）。

注意 $\lambda,\alpha,\tau$ 三个门控同源 $h_i$，于是 $\lambda$ 随特征尺度漂（P1-3）；`center_stat='zscore'` 只解耦温度头，**不解耦 $\lambda/\alpha$**。

### 2.7 温度调制

$$\tau_i=\mathrm{softplus}\big(w_\tau^\top[h_i\,\|\,\text{stat}_i]\big)+\tau_{\min},\qquad \tau_{\min}=10^{-3}$$

代码里的 `stat`（默认 `center_stat='zscore'`，邻居集上求，不含中心槽位）：

$$\mu_i=\frac1{\deg_i}\sum_{j\in\mathcal N(i)}s_{i,j},\qquad
\text{stat}_i=\frac{s_{i,i}-\mu_i}{\sqrt{\deg_i\cdot\mathrm{var}_i}+\varepsilon},\qquad \mathrm{var}_i=\frac1{\deg_i}\sum_j s_{i,j}^2-\mu_i^2$$

> ⚠️ 分母是 $\sqrt{\deg_i\cdot\mathrm{var}_i}=\sqrt{\deg_i}\,\sigma_i$，**比标准 z-score 多一个 $\sqrt{\deg}$**（代码 `sd = var.clamp(min=1e-8).sqrt() * cnt.clamp(min=1.0).sqrt()`），所以“z-score”这个名字目前不准确。**§9.8 已拿真实度数分布量过这个因子**：它确实在按 $1/\sqrt{\deg}$ 缩统计量（attr 大 hub 上实测 113.5×），但对聚合锐度的影响是 $2\times10^{-4}$ 量级，且训练会让 $w_\tau$ 自己补偿（corn 6.9×）。结论：**数值不改，只改措辞**（下面的式子才是它真正的含义），去掉 `* cnt.sqrt()` 属行为变更、要重跑 §8.4 全部数字，收益不抵代价。
> 另一处：注释写「邻居数 0 ⇒ stat 0」，这只在 $T\ge1$ 时成立（中心化后孤立场的 $s_{i,i}=0$ 恰好为 0）；$T=0$ 时该分支会得到 $s_{i,i}/10^{-4}$ 量级的值，只是孤立节点本来没有邻居可聚合，不影响输出。

`center_stat='raw'` 直接把 $s_{i,i}$ 喂进去（消融 `raw_center_stat` 那一行）。

### 2.8 读出与输出

中心分数 $s_i(T)$ **不参与聚合**，只用于调温；仅对邻居做分组 softmax：

$$p_{i,j}=\frac{\exp(s_{i,j}(T)/\tau_i)}{\sum_{k\in\mathcal N(i)}\exp(s_{i,k}(T)/\tau_i)},\qquad
m_i=\sum_{j\in\mathcal N(i)}p_{i,j}z_j$$

输出融合（`out_mode='fusion'`）：

$$h_i'=\mathrm{LayerNorm}\big(P h_i+\mathrm{MLP}([W_s h_i\,\|\,m_i])\big)$$

残差必须走独立投影 $P$（`self.skip`）：v1 写 `norm(x + h)`，在 `in_dim != out_dim`（Cora 1433→64）时第一层就 shape crash。`out_mode='gat'` 则是 $h_i'=\mathrm{ELU}(m_i)$，无残差、无 LayerNorm、无 `out_mlp`，逐头拼接。

### 2.9 稀疏因子化（为什么能 $O(\hat E d)$）

目标是 $r_a=\sum_{b\ne a}\tilde\kappa_{ab}s_b$。记 $w_b:=s_b/\sqrt{D'_b}$，则一切含 $\kappa$ 的求和都过 $\hat q$：

$$\sum_{b}\kappa_{ab}w_b=\Big\langle\hat q_a,\ \underbrace{\sum_b w_b\hat q_b}_{U_i}\Big\rangle+\underbrace{\sum_b w_b}_{R_i},\qquad D^{\text{raw}}_a=\Big\langle\hat q_a,C_i\Big\rangle+\degp_i$$

$M$ 的屏蔽与去对角各一次修正：

$$D_a=D^{\text{raw}}_a-(1-\alpha_i)\big[c_aD^{\text{raw}}_a+(1-c_a)\kappa_{a,i}\big]$$

$$\sum_bM_{ab}\kappa_{ab}w_b=\text{num}^{\text{raw}}_a-(1-\alpha_i)\big[c_a\,\text{num}^{\text{raw}}_a+(1-c_a)\kappa_{a,i}w_i\big]$$

$$r_a=\Big(\text{num}_a-M_{aa}\kappa_{aa}w_a\Big)/\sqrt{D'_a},\qquad M_{aa}=1-(1-\alpha_i)c_a$$

两次 `index_add_`（`scatter_add`）实现，无 Python 循环、无 `max_deg`、无 padding 截断：

```python
D = ctx['Draw'] - one_ma * (c_a * ctx['Draw'] + (1.0 - c_a) * ctx['kap_ci'])
isd = D.clamp(min=1.0).rsqrt()
w = s * isd
U = scatter_add(w.unsqueeze(-1) * q[dst], src, N)      # Σ_b w_b qhat_b
Rd = scatter_add(w, src, N)                            # Σ_b w_b
num_raw = (q[dst] * U[src]).sum(-1) + Rd[src]           # Σ_b kappa_ab w_b
w_i = scatter_add(w * c_a, src, N)[src]                # 场中心的 w
num = num_raw - one_ma * (c_a * num_raw + (1.0 - c_a) * ctx['kap_ci'] * w_i)
r = (num - (1.0 - one_ma * c_a) * ctx['kap_aa'] * w) * isd      # (K̃ s)_a
s_new = (1.0 - beta) * s + beta * (t['c'] - t['lam_e'] * r)
mean = scatter_add(s_new, src, N) / degp.clamp(min=1.0)
return s_new - mean[src]                                # §2.5 投影
```

**复杂度** $O(T\,\hat E\,d)$，稠密参考实现 $O(\sum_i\deg_i^2H)$（只用于对拍与核消融，不是可训练路径）。高度数节点全部进场：实测 `max_deg=265` 无一丢弃（v1 在 $K=8$ 处静默截断，还系统性偏向小索引节点）。

### 2.10 单层参数量（可手算）

记输入 $p$、输出 $d$、头数 $H$，`out_mode='fusion'` 全开关打开时：

| 部件 | 参数量 |
| :-- | :-- |
| $W$ / $W_q$ / $W_s$ / $P$(skip) | $4(pd+d)$ |
| $W_k$ / $q_\theta$ | $2(d^2+d)$ |
| $\phi$（`Linear(d,d)`+`Linear(d,H)`） | $d^2+d+dH+H$ |
| $w_\lambda$ / $w_\alpha$ / $w_\tau$ | $(p+1)+(p+1)+(p+2)$ |
| `out_mlp`（`Linear(2d,d)`+`Linear(d,d)`） | $3d^2+2d$ |
| LayerNorm | $2d$ |

代入 $(p,d,H)=(1433,16,4)$ 得 **97,795**；$(16,16,4)$ 得 **2,856** —— 与 §6 逐层表逐位相符（`Merge` = 相加 + `Linear` + LN = $d^2+d+2d$ = 304；`Classify` = $64{\times}16{+}16+16{\times}7{+}7$ = 1,159）。关掉 $\phi$/温度/`out_mlp`（GAT 退化）后同尺寸层只剩个位数开销，这就是 §6 里 `oca_gat` 参数量骤降的来源。

------

## 3. 严格退化到 GAT：七个条件

$$\lambda=0\ \wedge\ \phi\equiv0\ \wedge\ \alpha=0\ \wedge\ \tau=1\ \wedge\ T=0\ \wedge\ \text{加法式打分}\ \wedge\ \text{GAT 式输出}$$

单靠 $\lambda_i=0$ **不够**：$\phi(z)$ 是 query 无关的节点偏置，会进 softmax 分数（破坏平移不变性）；点积打分与 GAT 的加法式不同族（GATv2 已论证点积式表达不了这种加性依赖）；`fusion` 输出多了残差/LN/MLP。

代码入口一次打开七条件，且**不分配**用不到的参数（`__init__` 里按开关建模块，所以没有死分支）：

```python
OCALayer.gat_equivalent(in_dim, out_dim=64, heads=4)
# = OCAConfig(T=0, score_mode='gat', use_proposal=True, use_phi=False,
#             use_competition=False, use_center_field=False,
#             use_temperature=False, out_mode='gat')
```

证据：`test/test_oca_degeneration.py::test_gat_degeneration_is_exact` 对独立实现的稠密 GAT，heads=1/4/8 下 `max|diff| = 2.22e-16`（含 8 头 concat 路径）；`test_gat_mode_has_no_dead_branches` 打印该用例尺寸下参数量 344，逐项对得上。结构表侧的等价物是 [`cfg/models/oca_gat.yaml`](../cfg/models/oca_gat.yaml)（七条件逐字段与工厂相等，`test_gat_yaml_matches_gat_equivalent` 钉住）。

------

## 4. 能量函数与两条引理

$\tilde\kappa^{(i)}$ 对称 ⇒ 下面 $E_i$ 是良定义的二次型：

$$E_i(s)=-\sum_{a\in\mathcal V_i}c_as_a+\frac{\lambda_i}2\,s^\top\tilde\kappa^{(i)}s+\frac12\lVert s\rVert^2$$

四项：$\to$ 各节点按自己嗓门激活；$\to$ 纵向提案拉高邻居分数（$\alpha$ 越大拉力越弱）；$\to$ 横向竞争惩罚"相似节点同时高激活"；$\to$ 正则防爆炸。

**Lemma 1（更新式 = 能量梯度）.** $\nabla_sE_i=-c+\lambda_i\tilde\kappa^{(i)}s+s$，于是 §2.4 恰为 $s\leftarrow s-\beta\nabla_sE_i$，§2.5 恰为向 $\mathbf 1^\top s=0$ 的投影。合起来：**在零均值子空间上做投影梯度下降**（措辞必须是这个，不能写"每轮把 $E$ 求到最小"）。

**Lemma 2（与度数无关的一致稳定性）.** $\big\lVert\tilde\kappa^{(i)}\big\rVert_2\le\big\lVert D^{-1/2}AD^{-1/2}\big\rVert_2=\rho(D^{-1}A)\le\lVert D^{-1}A\rVert_\infty=1$。三步依据：

- **(a)** $0\le\tilde\kappa\le D^{-1/2}AD^{-1/2}$ 元素级（抬 $D'$、去对角都只让非负元变小，**这一步吃 $\kappa\ge0$**）；对称非负矩阵的谱范数 = Perron 根，Perron 根对元素级序单调；
- **(b)** $D^{-1/2}AD^{-1/2}$ 与 $D^{-1}A$ 相似 ⇒ 同谱；
- **(c)** $D^{-1}A$ 行和恒为 $1$。$D_a=0$ 的行整行为零，不等式平凡成立，故可限制在 $D_a>0$ 的子图上用 (b)。

> ⚠️ 常见的**错误证明**是「归一化后行和 $\le1$ ⇒ 谱范数 $\le1$」。混合分母 $\sqrt{D_aD_b}$ 控制不住行和（要 $D_b\ge D_a$ 才行）。v1 和本文档的上一版写的都是这个错证明，已换掉。

推论：

1. $\nabla^2E_i=I+\lambda_i\tilde\kappa^{(i)}\succeq(1-\lambda_i)I\succ0$ ⇒ $E_i$ 强凸、唯一极小点 $s^\star$，由 KKT 系统 $\begin{bmatrix}I+\lambda\tilde\kappa&-\mathbf 1\\ \mathbf 1^\top&0\end{bmatrix}\begin{bmatrix}s\\\nu\end{bmatrix}=\begin{bmatrix}c\\0\end{bmatrix}$ 闭式给出；
2. 迭代矩阵谱半径 $\le\max\big(\lvert1-\beta+\beta\lambda_i\rvert,\lvert1-\beta-\beta\lambda_i\rvert\big)<1$，且 $\lVert s^{(T)}-s^\star\rVert\le\rho^T\lVert s^{(0)}-s^\star\rVert$；
3. 前提只有 $\beta\in(0,1),\lambda_i\in(0,1)$ —— **与 $\lvert\mathcal N(i)\rvert$ 无关**，这就是 hub 不再炸的原因（也是这份算法对农业异构图最相关的性质，见 §9）。

------

## 5. 超参全表（`OCAConfig`，默认值 = 现在生效的值）

| 字段 | 默认 | 语义 / 关掉是什么 |
| :-- | :-- | :-- |
| `out_dim` / `heads` | 64 / 1（结构表里 4） | $d$ / $H$；`out_dim % heads` 必须整除 |
| `T` | 2 | 迭代轮数；`T=0` 跳过竞争与中心化 |
| `beta` | 0.5 | PGD 步长，$\in(0,1)$ 由 `assert` 保证 |
| `kernel` | `shifted_cosine` | `relu` 仅 `backend='dense'`（稀疏路径抛 `NotImplementedError`） |
| `score_mode` | `dot` | `gat` = 加法式打分（§3 之一） |
| `use_proposal` | True | False ⇒ $b\equiv0$（w/o 纵向提案） |
| `use_phi` | True | False ⇒ $\phi\equiv0$ |
| `use_competition` | True | False ⇒ $\lambda:=0$（w/o 横向竞争） |
| `use_center_field` | True | False ⇒ $\alpha:=0$（**OCA-B**） |
| `use_temperature` | True | False ⇒ $\tau:=1$ |
| `center_stat` | `zscore` | `raw` = 喂原始 $s_i(T)$（消融 `raw_center_stat`）。**`zscore` 只是历史别名**：它是按 $1/\sqrt{\deg_i}$ 再缩放的中心-场差异，不是标准 z-score（§2.7；§9.8 已量过代价并裁决数值不改） |
| `out_mode` | `fusion` | `gat` = `ELU(m)`，无残差/LN/MLP |
| `gat_negative_slope` / `gat_activation` | 0.2 / `elu` | 只在 `score_mode='gat'` / `out_mode='gat'` 下生效 |
| `tau_min` | 1e-3 | 温度下界 |
| `lambda_bias_init` / `alpha_bias_init` | −2.0 / 0.0 | $\sigma(-2)\approx0.12$ ⇒ 默认不竞争；`+40` 强制 $\alpha\approx1$（**OCA-C**） |
| `backend` | `sparse` | `dense` = 逐节点参考实现（对拍专用） |
| `detach_iterations` | False | 前 $T-1$ 轮 stop-gradient（消融 `detach_iters`） |
| `symmetrize` | True | 有向输入是否先 `to_undirected` |

`cfg/train/*.yaml` 的 `model.oca` 与 CLI 的 `--oca k=v` 对这张表**逐键合并**（`TrainConfig.with_oca`），不是整块替换 —— 否则 `T4` 与 `full` 的差值就不只是 $T$ 的差值。

### 消融变体（`training/sweep.py::ABLATIONS`，17 个）

| 变体名 | 覆盖 | 验证什么 |
| :-- | :-- | :-- |
| `full` | — | OCA-full：$\lambda,\alpha$ 自适应，统一框架 |
| `no_competition` | `use_competition=False` | 无竞争时相对 GAT 的增益来自 $\phi$/打分而非竞争 |
| `no_proposal` | `use_proposal=False` | 纵向提案必要性 |
| `no_phi` | `use_phi=False` | $\phi$ 必要性 |
| `no_center_field` | `use_center_field=False`（$\alpha:=0$） | **OCA-B** |
| `center_in_field` | `use_center_field=True, alpha_bias_init=+40` | **OCA-C** |
| `no_temperature` | `use_temperature=False` | 中心调温必要性 |
| `raw_center_stat` | `center_stat='raw'` | §2.7 的替代方案 |
| `T0` `T1` `T2` `T4` `T8` | `T=...` | 收敛代价（Lemma 2 给了 $\rho^T$ 先验，别只报"T=2 最好"） |
| `detach_iters` | `detach_iterations=True` | stop-gradient 付了多少代价 |
| `gat_degenerate` | 七条件全套 | 同骨架同量级 GAT 退化，与外基线 `gat.yaml` 的差值才是诚实对照 |
| `single_scale` | `model.spec=cfg/models/oca1.yaml` | 多尺度 vs 单尺度（**同时动了深度**，见 §6 末） |
| `scale_s` `scale_l` | `model.scale=...` | 只改宽度，看增益是不是白涨参数来的 |

表头由 `format_table` 给出：`variant spec acc acc_std f1 f1_std params best_ep lambda_std r_lambda_homo`。`lambda_` 前缀是刻意保留的 —— 否则 `no_competition` 行会把 $\alpha$ 的相关性当成 $\lambda$ 报进表（它根本没有 $\lambda$ 键）。

**还没进表的行（需要额外开关或外部实现）：**

| 实验 | 现状 |
| :-- | :-- |
| OCA w/o 中心分数（温度输入去掉 $s_i(T)$） | `OCAConfig` **没有**这个开关，未实现 |
| 核消融 `backend='dense', kernel='relu'` | 能跑，但**这个 relu 分支也走同一套归一化，所以仍稳定**；v1 的病根是「无归一化」而不是「用了 ReLU」。稀疏路径不可用 ⇒ 只能小图，未进表 |
| v1 原式复现（无归一化 ReLU 核） | 已有反向对照用例（§8.2），不必进训练表 |
| GAT + 侧向抑制插件 | 外部实现，**未做**（这是"原算子 vs 插件"那条卖点的必要条件） |

------

## 6. 多尺度骨架（结构表驱动）

拓扑写在 [`cfg/models/*.yaml`](../cfg/models)，由 [`model_builder.parse_model`](../model_builder.py) 逐行推 $c_1/c_2$、复合缩放、构建。每行仍是 yolov8 四元组 `[from, repeats, module, args]`；`nc` 留空由数据注入；`scales: [depth_multiple, width_multiple, max_channels]`；文件名结尾 `-n`/`_s` 会被推成 scale（`gcn.yaml` 末尾那个 `n` 不会误判，有专门用例）。图侧四点差异：**(1)** 每层还要吃 `edge_index`（`GraphSequential` 按 `from` 路由，transductive 下 `edge_index` 恒定不进缓存）；**(2)** 没有空间下采样 ⇒「尺度」= 感受野跳数，融合只有 `Merge`（等宽相加→`Linear`→ReLU→LayerNorm，不等宽在构建期报错并提示改用 `Concat`）与 `Concat`（宽度由 builder 推导，args 省掉）；**(3)** `repeats` 用自定义 `Repeat`，一个节点只占**一个**层号，故 stage 内部中间层不可被 `from` 引用；**(4)** `Classify` 可多输入 list。

不变式：**任何被 `from` 引用的绝对行号必须在 `GraphSequential.save` 里**（引用未缓存的行不会立刻报错，只会拿到 `None`，然后在某个 `torch.stack` 里炸出一句跟结构表无关的话）。`oca.yaml` 的拓扑与宽度序列由 `test_oca_neck_topology_is_pinned` 钉死 —— 这条用例的存在理由是一次真实事故：bottom-up 两行曾写成 `[[4,6]]`/`[[5,7]]`，把 D1/D0 重复加了一遍，**前向 shape 全对、loss 照降**，能跑 ≠ 对。

**骨架与算子零耦合**：把 `OCAConv` 换成 `GATConv`/`GCNConv` 走同一条 `parse_model`，没有只在基线侧存在的胶水代码。所以「YOLO 式骨架」不能作为算法贡献卖，只能当工程配置；要保留就得靠 `single_scale` 证明增益来自融合而非参数量。

### 实测：行数与参数量

取数：`build_model(spec, in_dim=1433, nc=7, scale=...)` + `model_info`（`dropout=0.5`、`scale=n`）。复现：`python test/test_model_builder.py`（14 用例）。

| 结构表 | 行数 | OCA 层 | 参数量 | 说明 |
| :-- | :-- | :-- | :-- | :-- |
| `oca.yaml` | 11 | 4 | 109,346 | 多尺度融合（默认） |
| `oca1.yaml` | 3 | 2 | 101,042 | 单尺度对照 |
| `oca_deep.yaml` | 11 | 4 | 109,346 | $n$ 下 $\mathrm{round}(2\times0.33)=1$，深度尚未展开 |
| `oca_gat.yaml` | 11 | 4 | 27,287 | 同骨架 GAT 退化（七条件），骤降来自关掉 $\phi$/温度/`out_mlp` |
| `oca_nocomp.yaml` | 11 | 4 | 106,773 | 归因对照：与 `oca.yaml` 只差 `use_competition: false`（见下表注） |
| `gat.yaml` | 3 | 0 | 23,671 | 外基线 GAT |
| `gcn.yaml` | 3 | 0 | 23,607 | 外基线 GCN |
| `mlp.yaml` | 3 | 0 | 23,607 | 外基线 MLP（不用邻接，故与 GCN 同） |

`oca.yaml` 宽度阶梯（行数恒 11、OCA 层恒 4）：

| scale | n | s | m | l | x |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 每层宽度 | 16 | 32 | 48 | 64 | 80 |
| 参数量 | 109,346 | 237,906 | 390,018 | 565,682 | 764,898 |

深度缩放示范（`OCA 层数 / 参数量`）：`oca1` = 2/101,042 · 2/206,466 · 4/368,418 · 5/569,786 · 5/771,306；`oca_deep` = 4/109,346 · 4/237,906 · 4/390,018 · 8/734,162 · 8/1,026,690。`max_channels` 封顶实测 2048→384。

`oca.yaml` / `scale=n` 逐层表（`build_model(..., verbose=True)`）：

```text
       from  n    params  module    arguments
  0        -1  1     97795  OCAConv   [1433, 16]
  1        -1  1      2856  OCAConv   [16, 16]
  2        -1  1      2856  OCAConv   [16, 16]
  3        -1  1      2856  OCAConv   [16, 16]
  4    [2, 3]  1       304  Merge     [[16, 16], 16]
  5    [1, 4]  1       304  Merge     [[16, 16], 16]
  6    [0, 5]  1       304  Merge     [[16, 16], 16]
  7    [5, 6]  1       304  Merge     [[16, 16], 16]
  8    [4, 7]  1       304  Merge     [[16, 16], 16]
  9    [3, 8]  1       304  Merge     [[16, 16], 16]
 10  [6, 7, 8, 9]  1    1159  Classify  [[16, 16, 16, 16], 7]
```

两个读数：97k 参数全在第一层输入投影（$1433\times16$），`oca` 与 `oca1` 的差值 8,304 里 6 个 `Merge` 只占 **1,824**（其余是 2 个多出来的 OCA 层 $2\times2{,}856$ + 更宽 `Classify` 768）—— 多尺度融合本身几乎不带来容量差。而 `oca1` 在 `scale=n` 只有 **2** 个 OCA 层（`repeats=4` × depth 0.33 取整 = 1），`oca.yaml` 有 4 层 ⇒ **`single_scale` 同时动了融合与深度**，目前只能当"有没有 neck"的粗对照；严格隔离要补一份四行 backbone、无 `Merge` 的结构表（或把 `oca1.yaml` 里 $n$ 的 `depth_multiple` 改 1.0）。

`oca` − `oca_nocomp` 的 2,573 已逐张量核过（`build_model(in_dim=1433, nc=7)` 的 state_dict 差集）：每层 272（`q_proj` 权重 256 + bias 16）× 4 = 1,088，加 `w_lambda` 的首层 1,434（吃原始 1,433 维 + bias）与其余三层各 17（吃 16 维）。⇒ **这个开关撤掉的是整个场，不只是 λ**：`OCALayer.__init__` 里 `q_proj` 与 `w_lambda` 同进同出，`forward` 里那个 `cfg.T > 0 and cfg.use_competition` 又把 T 步迭代与中心化一并跳过，于是 s ≡ φ + (1-α)b —— 与 §2.4 那句「T=0（或 `use_competition=False`）时既不迭代也不中心化」是同一件事。因此 §9.13 里把它叫「只撤 λ」是错的措辞，正确说法是「只撤竞争场」。这条等价性是实测的（12 节点 toy 图、三网灌同一套权重、eval）：`oca` 且 `T=0` 与 `oca_nocomp` 前向 **max|Δ| = 0.000e+00**（逐位相同；state_dict 键数 140 vs 124，差的 16 个正是 4 层的 `q_proj`/`w_lambda` 各 w+b），而 `oca`（默认 $T{=}2$）与它们差 2.4e-02（相对 1.5e-02）⇒ 这个开关是「把整个场摘掉」而不是「把 λ 归零」，且 T=2 的场确实在改输出 —— 只是改的方向不划算（§9.13）。

------

## 7. 训练协议与产物（影响读数的部分）

配置树四个块（`run` / `dataset` / `model` / `optimization`），默认值与 `test_default_yaml_agrees_with_dataclass_defaults` 断言一致：

```yaml
run:   {run_name: oca, project: runs, device: auto, seed: 0, seeds: [0, 1, 2],
        vary_split: false, log_interval: 20, best_metric: val_acc,
        ckpt_interval: 0, max_ckpt: 5, save: true, dump_gates: true}
dataset: {name: synth, root: data, split_idx: 0, train_per_class: 20,
          homophily: 0.3, n_per_class: 60, fallback_to_synth: false}
model: {spec: cfg/models/oca.yaml, scale: n, dropout: 0.5, oca: {}}
optimization: {epochs: 400, patience: 100, optimizer: adam, lr: 0.01,
               weight_decay: 0.0005, momentum: 0.9, grad_clip: 0.0,
               class_weight: false}
```

- **优先级**：dataclass 默认 < `cfg/train/*.yaml` < `train(**kwargs)` / CLI `--set`。
- **选模型只看 val**：`best_metric` 写 `test_*` 直接 `ValueError`（`_metric_key`），`val_acc`/`val_loss` 都可用。
- **这组默认值是合成图上凑效的值**，真实数据必须按 val 重选 `lr`/`epochs`，不能拿它报真实数据结果。
- YAML 1.1 三个坑由 `_coerce` 兜住：`5e-4` 读成字符串、`off`/`yes` 读成 bool、`null` 与"没写这个键"必须可区分。
- 产物：`runs/<name>/train/{config_used.yaml, results.json, logs/train.log, weights/best.pt, checkpoints/}` + run_dir 根的 `gates.npz`（逐节点 $\lambda/\alpha$、局部同质性、标签与预测）。多种子时只有第一个种子占该目录，`best.pt` 只代表第一个种子。
- 已知缺口：`prepare_run_dir(resume=True)` 与 `capture_rng_state` 就绪，但 `fit` 固定传 `resume=False`、CLI 无 `--resume` ⇒ **checkpoint 只落盘、不续训**。

CLI：

```bash
python train.py                                   # 全用默认
python train.py --model cfg/models/gat.yaml       # 换结构表就是换模型
python train.py --dataset cora --seeds 0 1 2
python train.py --set optimization.lr=5e-3 model.scale=l
python train.py --oca T=4 beta=0.3 use_phi=false   # 覆盖 OCAConfig
python train.py --ablate full no_competition T0
python train.py --dry-run --dataset chameleon      # 只体检数据 + 打印逐层表
python train.py --epochs 20 --no-save              # 冒烟：不建 runs/
```

------

## 8. 实验结果（全部实测）

取数：`python -m test.run_all`（10 个文件 59 个用例，全绿；真实数据未预下载时其中一个打印 SKIP）。以下数字是**同一份代码在当前环境的一次实跑**，且已随 §10 第 1 行的初值改动（$s(0)=c$）**全部重跑过一遍**；表里未变的行不是因为没跑，而是跑出来确实没变。

> 设备口径：`test/` 的数值对拍一律 CPU + float64（位级一致），但 `--ablate` / 训练类用例走 `device=auto` ⇒ 本机是 CUDA。同一份代码在 CPU 与 GPU 上跑训练会因随机数流不同给出不同结果（实测 `gat_degenerate`：GPU 0.870 vs CPU 0.897），所以 §8.4 的 `--ablate` 表**必须连设备一起引用**。

### 8.1 数学正确性

| 主张 | 实测证据 |
| :-- | :-- |
| 稀疏因子化 == 稠密逐节点建矩阵 | 12 组配置（heads 1/2 × T 1/3 × $\alpha\to0$/中/1），最大绝对误差 **8.88e-16** |
| 更新式 == 投影梯度下降 | 全图 41 个场逐场验：$\max\lVert\nabla^{\rm num}E-P\nabla E\rVert=$ **1.99e-16**、$\max\lVert s^{(1)}-\mathrm{PGD}(s)\rVert=$ **1.11e-16** |
| 收敛到唯一极小点（KKT 闭式解） | $T=300$ 时 $\max_i\lVert s^{(T)}-s^\star\rVert_\infty<$ **1.11e-16**；速率界 $\lVert e^{2T}\rVert\le\rho^T\sqrt m\lVert e^T\rVert$ 成立（最差比 **0.407** $\le1$） |
| 无度数截断 | $\max\deg=265$ 全部进场（v1 在 $K=8$ 静默丢弃） |
| 自环/重边/有向输入健壮 | `E_hat=392`、`max_deg=13`，与手工去重结果一致 |
| 非法配置被拒 | `beta` 越界 / `heads` 不整除 / `sparse`+`relu` 三条路径各抛各的错 |

### 8.2 稳定性与退化（Lemma 2 的正反两面）

| 主张 | 实测证据 |
| :-- | :-- |
| hub + $\lambda\to1$ 不发散 | $\max\deg=670$、$\lambda_{\min}=1.000000$、$\alpha=1$、$T=40$ ⇒ $\max\lvert s\rvert=$ **0.619** |
| **反向对照**：归一化确为必需 | 换回 v1 的无归一化 ReLU 核，第 **10** 轮 $\max\lvert s\rvert=$ **1.134e+13** |
| 严格 GAT 退化 | heads=1/4/8，$\max\lvert\Delta\rvert=$ **2.22e-16**；gat 模式 344 参数无死分支 |
| $\alpha\to0$ 真解耦（B 模式） | 中心→邻居影响 $=\mathbf{0.00e{+}00}$（$\alpha_{\min}=8.2\text{e-}19$）；$\alpha=0.22$ 时 **4.11e-02**；$\alpha\to1$ 时 **7.80e-02** |
| 孤立节点 / 空边集不崩 | 两个用例分别 OK |

### 8.3 结构与缩放

`test/test_model_builder.py` 14 用例覆盖：8 张结构表全部 build/forward/路由 OK（oca=11 层，oca1=3，oca_deep=11，oca_gat=11，oca_nocomp=11，gat/gcn/mlp=3）；新增的 `test_nocomp_yaml_differs_only_by_lambda` 逐层断言两套 `OCAConfig` 的差集恰为 `{use_competition}`、且 `oca_nocomp` 的 state_dict 里没有任何 `w_lambda`（这个写法能过本身就是 §6 表注那条更正的由来：它同时也没了 `q_proj`）；`oca.yaml` 拓扑 `save={0..9}`、宽度序列 `[16]×10 + [4]`；`Concat` 省 args 推宽度（64+64→128→Classify(3)）；`Merge` 不等宽在构建期被拒；复合缩放 `[16,32,48,64,80]`；`max_channels` 封顶；逐层 OCA 超参覆盖生效（$T=[2,2,1,2]$）；结构表写错的六类情形报错点名到行。合成图侧：分层随机划分互斥且每类定量、同种子可复现、目标同质性 0.1/0.5/0.9 实测 0.113/0.480/0.883。

### 8.4 端到端（合成图，只能证管线）

| 项 | 实测 |
| :-- | :-- |
| 单种子训练 | test acc **0.9821**（chance 0.25）、f1 0.9821、best_ep **35/60**、params 14,356 |
| 可复现 | seed=5 两次 **0.982143 == 0.982143**（seed=6 为 0.964286，说明差异来自种子不是随机性泄漏） |
| 梯度真实流到门控 | 单层 `dL/dw_\lambda=` **1.462e-04**，$\lambda$ mean 0.132 / std **0.077**；整网 4 层 $\lambda/\alpha$ 均收非零梯度；诊断取数 4 层逐层 $\lambda=[0.121,0.124,0.129,0.142]$、std 0.029 |
| 消融开关确实改计算图（3 种子） | `full` **0.9732 ± 0.0126** · `no_competition` **0.9554 ± 0.0126** · `gat_degenerate` **0.3839 ± 0.1641**（末项改动前后逐位相同，正是 §2.2 那个改动不该碰退化路径的验证） |
| `--ablate` 一次表（`python train.py --ablate full T0 single_scale gat_degenerate`，synth、CUDA、seeds 0/1/2） | `full` **0.9967 ± 0.0058**（params 14,909、best_ep 10.3、$\lambda$ std 0.0126、$r_{\lambda,\text{homo}}=-0.022$）· `T0` 0.9967 ± 0.0058（best_ep 9.3、$\lambda$ std 0.0240、$r=-0.036$）· `single_scale`(oca1) 0.9933 ± 0.0115（params 6,605、best_ep 12.7）· `gat_degenerate` 0.870 ± 0.070（params 4,709、best_ep 96.3） |
| 产物与选模护栏 | `train`、`val_loss` 选模 best_ep=5；`config_used.yaml` roundtrip 一致 |
| 门控落盘 | `lambda(4,120)`、`h_local(120,)` 按节点对齐 |

`gat_degenerate` 掉到 0.87/0.38（两个数来自两套配置：`--ablate` 走 `cfg/train/default.yaml`，`test_training.py` 那行走用例自己的小配置）是**预期内**的：七条件关掉竞争后这个骨架没有可用的异配归纳偏置，它该去跟外基线 GAT 比，而不是跟 `full` 比。

> 这张表里有一条**旧数字被推翻**：改动前本节记的是 `full` 1.0 与 `gat_degenerate` 0.52，两个值在当前代码 + 当前配置下都复现不出来（CPU/GPU 各跑一次 gat_degenerate 得 0.870 / 0.897，没有一次接近 0.52）。当时那行没记命令行与设备，属于不可追溯读数，已删；往后 §8 的每个数都必须带命令 + 设备一起写。

### 8.5 目前没有证据的部分（别引用）

- 默认合成图（`n_per_class=60, homophily=0.3`）**基本饱和**（`full` acc=0.9967、三个种子 std 0.0058）⇒ §8.4 只能证"开关是活的、管线能跑"，不能当结论。$T=0$ 与 `single_scale` 也都在 0.99 上下 —— 这张图区分不了它们。
- **$\lambda$-vs-局部同质性散点图未画**，所以"同配/异配自适应"这条主张目前**没有直接证据**；合成图上 $\lambda$ 与 $h_i$ 相关性 $\approx0$（$|r|<0.04$），这张图只有在真实数据上才有意义。$\lambda$ 的 std 只有 0.077（`--ablate` 表里 0.013）也指向同一件事：竞争门控可能根本没在分化。
- **OCA 自己在真实图上的读数才刚开始**：`dataset/real.py` 接口接好了 11 个数据集，六张学术图已经可加载、且基线四行（mlp / floor+ / gcn / gat）已在同一套协议下量过（§9.13 表 3）—— 而 OCA 行目前**只有 chameleon 一个图**：单 run 探针 + 门控实测已入 §9.13，五折三种子的全表在跑。⇒ 现在能说出口的是「OCA 行在 chameleon 上真落后（选参折 val 0.5665 vs `gat` 0.7373）」，**不能说「六张都已评过」**。剩下的障碍是算力不是显存（§9.6 补测）。“本机下载 Cora 超时”仍成立（§10 第 10 行）。
- `OCA.md` §预期结果 那张表是 **v1 假设值**，且已核对低于 Chameleon/Squirrel/Actor 上 GGCN/FAGCN/ACM-GNN 的公开报告精度 —— 不得填进任何投稿材料。

------

## 9. 数据侧现状与接下来要做的事

### 9.1 已接入的（合成 + 学术真实图）

`GraphBundle` = 一张全图 + `x/edge_index/y` + 三个 mask + `num_classes`，`stats()` 打印体检表并 `assert_split_disjoint`。合成侧 BA / 异配可控图；真实侧 `torch_geometric.datasets`（Planetoid / WebKB / WikipediaNetwork）。**不做 mini-batch**（`dataset/base.py` 里写明：最大 7,600 节点放得下，硬上采样会让逐节点 $\lambda$ 分析失去意义）。**这条决定在 SOY/CORN 上已被 §9.6 实测推翻**（145k 节点 × 6 关系 × 4 层在 8 GiB 卡上装不下），但先不动 `base.py`：要不要上 Loader 取决于 §10 第 7 行的三选一。

### 9.2 新加的 SOY / CORN 保险异构图（本机实测结构）

取数（**已入库为常驻脚本**）：`python -m dataset.hetero_probe --dataset both --what structure`；下表最后一行是元文件里的静态字段，直接读 [`data/`](../data)。

注意：这两个 `.pt` 是自定义类封装的，**必须**传 `weights_only=False`，PyTorch 2.6+ 的默认安全加载会直接拒。

| 项 | corn | soy |
| :-- | :-- | :-- |
| 容器 | `HeteroData`，7 种节点 / 13 种关系 | 同 |
| `record` | 145,544 × 12 维 | 133,422 × 12 维 |
| record 上的字段 | `x, y_cls, loss_level, y_reg, per_indemnity, train_mask, test_mask` | 同 |
| 其余节点 | `state`7 · `year`15 · `stage`7 · `month`12 · `plan`7 · `coverage`2，**全部无特征** | 同 |
| 边 | record↔6 类属性双向各 145,544 + `state-neighbor` 22 有向 | 各 133,422 + 同 22 |
| record 的出度 | **每条属性关系上恒为 1**（`outdeg mean=1.00 max=1`）⇒ 一个 record 总共只有 6 个邻居，全是属性节点；**record 之间没有任何直接边** | 同 |
| attr 侧竞争场规模（max） | `coverage` 92,643 · `plan` 87,358 · `stage` 70,943 · `month` 32,652 · `state` 29,478 · `year` 13,221 | 89,765 · 79,187 · 69,042 · 33,710 · 27,125 · 13,278 |
| $\sum_r\lvert\mathcal E_r\rvert$ | **1,746,550** | 1,601,086 |
| split | train 120,497 / test 25,047，按年切（2020–22 测试），**无 `val_mask`** | 110,077 / 23,345 |
| `y_cls` | 正类 36,386 = **25.0%** | 33,356 = 25.0% |
| `loss_level` | 三分类 **完全均衡**（48,514/48,515/48,515） | 均衡（44,474×3） |
| `y_reg` | log 赔付额 0.0098–8.5364，mean 4.883 | 0.0092–7.8084，mean 4.501 |
| 特征质量 | 无 NaN/Inf、std≈1.01，但极值到 **37σ** | 同，极值 **72σ** |
| 年份映射 | `year` id 0–14 = 2008–2022（`graph_meta.json` 的 `node_index_maps`）；train = id 0–11（2008–2019），test = id 12–14 | 同 |
| 元文件 | `graph_meta.json` · `county_year.csv`（36 列原始表）· `scaler.pkl` · `processing_log.txt` | 同名但**带 `soy_` 前缀**（`soy_graph_meta.json` 等） |

特征名（`graph_meta.json.record_feats`，顺序即 $x$ 的列序）：`prcp_mean, prcp_max, prcp_cv, tmax_mean, dtr, tmin_min, tmax_max, srad_mean, gdd_sum, heat_days, frost_days, quantity`。第 11 列 `quantity` 就是那个 37σ/72σ 的重尾列（corn max 37.20 / soy max 72.27，其余列 max ≤ 9.1）。

两个标签之间是**嵌套关系**，不是两个独立任务：`y_cls=1` 的 36,386 条（soy 33,356）**全部**落在 `loss_level=2` 里，而 `loss_level=2` 另有 12,129（soy 11,118）条 `y_cls=0`；`corr(y_reg, loss_level)` = 0.870（soy 0.897）。$\mathrm{mean}(y_{reg})$ 按 level 为 3.576 / 5.081 / 5.993。⇒ T1 的 AUC 0.805 与 T2 的 0.583 **不能当成互不相干的两个证据**，而且从 `loss_level` 三分类推出二分类阈值 ≠ 现成的 `y_cls`。

源表 `county_year.csv` 的几条硬事实（影响怎么读这些数字）：`stage` 列注释为 **`MAX(12个物候期ID)`**（由逐日气象派生的季末累计量，不是出险后才可知的字段），且 CSV 里**本来就带 `stage_H … stage_OTHER` 七个 one-hot 列** —— 即 `stage` 在数据生产方那里就是**特征**，不是结构。另外 `cpi_indemnity` 被 README 明令禁作特征，`per_indemnity/loss_ratio/high_loss/loss_level` 全是标签侧。清洗日志：145,550 个源文件中 6 个因 `stage` 为空被丢弃。

**关于地理粒度（上一版此处我说错了，已核表订正）**：文件名里的 `county` 是**误称**。`county_year.csv` 的 36 列里**没有任何 county 列**；`id` 逐行唯一（corn 145,544 行 / 145,544 个 id，soy 133,422 / 133,422）⇒ 它是理赔记录号而**不是**可复用的地理单元；`colc` 只有 $\{11,31\}$ 两个值，就是图里被叫作 `coverage` 的那一列（与 `state` 组合出 14 = 7×2 个格子，也不可能是县级码）。地理可得的**最细粒度就是 `state`（7 值）**，且已在图里。⇒ **§10 第 7 行里“县级邻接”这条造边路径在数据上不成立，已划掉。**

同一个核表里唯一的好消息：**CSV 行序 == `.pt` 的 record 节点序**，逐位比对 5 列属性（`state/year/month/plan/stage`）与 2 个标签（`y_cls/loss_level`）均为 145,544/145,544（soy 133,422/133,422）完全一致 ⇒ 任何新造的边都能按行号直接挂回 record，而且表里还有 20 多列（`n_days`、`prcp_std`、`tmin_mean`、`prcp_cum`、`per_indemnity` …）根本没进 $x$。

已有基线（**来自 `data/*/train_metrics.json`，我没有复跑**）：T1 `high_loss` 二分类 corn AUC 0.805 / PR-AUC 0.555 / acc 0.615，soy AUC 0.894 / PR-AUC 0.756 / acc 0.797。这两份 `model_best.pt` 是他们在 `yolo8` 环境（torch 2.7 / PyG 2.6）+ 自定义脚本下训的，能否在本环境加载未验。

### 9.3 算法侧要接住它，缺这 7 件（按依赖顺序）

1. **`GraphBundle` 只装单类型图**。要么扩成多 `edge_index` + 节点类型 embedding，要么加一个 `HeteroBundle`。这是所有后续工作的前置。
2. **6 类属性节点无特征** ⇒ 必须给可学习 embedding（新超参 `emb_dim`），否则 $\kappa$、$b$、$\phi$ 全无定义。
3. **`metrics/` 没有 ROC-AUC / PR-AUC**（本机也没有 sklearn）⇒ 要自实现（AUC 用带 tie 校正的 Mann-Whitney U，PR-AUC 用阶梯积分），否则 corn 0.805 这个 must-beat 基线根本读不到。
4. **没有 `val_mask`** ⇒ 现有 `_masks` 会从 train 里**随机**抠 val。时间协议下更该显式给（例如 2019 作 val），不然报数时"随机 val + 按年 test"这个混合协议必须写明。
5. **关系怎么进竞争场**（真正的算法选择，两条都做）。**注意：§9.5–§9.7 的实测已动摇了本条的前提，先看那三节再开工。**【已更新：route B 按 §9.9 的 go/no-go 实测 **不过线**（三种造边的邻域均值全部不能越过地板，而组级标签率这种最强形式反而最差），route A 开工与否等你裁决】
   - **A 主方法 · 关系条件化**：每条 relation 保留自己那条 `edge_index`（**不做两跳展开**），$\kappa$/打分用 relation-specific 投影 ⇒ 每关系一组 $\lambda_r$，学"哪类关系该竞争"；record 与 attr 双侧都跑场。边数 $\sum_r\lvert\mathcal E_r\rvert=12\times145{,}544+22=1.75\times10^6$（双向合并后 $8.7\times10^5$），线性可控。
   - **B 对照 · record 同构图**：`state-neighbor`(22) + 自算 kNN（`graph_meta.json` 里预留 `knn: {enabled: false, k: 10}`）造纯 record 图，现有管线全复用，作为消融行。
   - **不能走的路**：把 record↔attr 折叠成 record–record 两跳。145,544 条 record→`year` 边只落到 15 个 year 节点 ⇒ 同年 record 全连接，边数量级 $10^8$。
6. **hub 压力正好顶到 Lemma 2**：`year` 节点平均度 $\approx9{,}700$、`state` $\approx20{,}800$，而 record 的入向槽位只有 6 个 attr 邻居（**record 之间没有直接边**）。所以这张图里"横向竞争"首先发生在 **attr 侧**：同一 `year`/`state` 下的上万个 record 会拿到高度相似的 attr 消息 ⇒ 既是 OCA 的用武之地（$\lambda_r$ 压制与自己过像的槽位），也是最大的风险（record 的相似性只能靠共享 attr 节点间接表达）。这件事**必须先用一个诊断实验确认**，别急着调网络。
7. **"不做 mini-batch"这条决定被顶穿了**：145k 节点 × 13 关系 × 4 层 × $T{=}2$，反向要存 $\hat E\times d\times H$ 级别的中间量。需要先做一次显存/单 epoch 时间实测，再决定是加 `NeighborLoader`（会破坏逐节点 `gates.npz` 对齐）还是按 `state`/`year` 分块训练。

### 9.4 首个任务（已定：T2 三分类）

`loss_level` 三分类完全均衡、CE 与 `acc`/`macro_f1` 现成 ⇒ 改动只落在 1、2、4 三件，能最快把 hetero 管线打通并拿到第一份可比较数字。之后再上 T1（要 3 的 AUC + 不平衡处理）与 T5（要回归头与 RMSE/MAE）。

### 9.5 先做了诊断：这张图的结构本身几乎不带标签信息（重要）

取数：`python -m dataset.hetero_probe --dataset both --what structure`（[`dataset/hetero_probe.py`](../dataset/hetero_probe.py)，脚本头写了它要回答的三个问题）。协议统一为 train = 2008–2018、**val = 2019（从 train 里抠出来）**、test = 2020–2022。

**（a）record 侧的竞争场是退化的。** 六条属性关系上 record 的出度恒为 1 ⇒ 单关系上中心的竞争场只有 $\{i, a\}$ 两个槽位，对**唯一**一个邻居做分组 softmax 恒得 $p_{i,a}=1$ —— 没有任何“竞争”可学。真正有多槽位的只有两处：attr 节点作为中心（§9.2 的 92k/87k/71k 大场），以及把六条关系合并后 record 的 6 个 attr 邻居 + 自环 = 7 槽位。所以路线 A 必须写成**二部两层**（record→attr 带 attr 侧竞争，attr→record 让 record 在自己的 6 个属性消息上竞争），而不能“逐关系各自跑算子”了事 —— 后者逐关系看都是单邻居，等于没跑。

**（b）共同 attr 的 record 在标签上接近随机。** 2-hop 同组内成对标签一致率（随机基线：`y_cls` 0.6250 = $p^2+(1-p)^2$，`loss_level` 0.3333）：

| attr | $\eta^2$（组间方差占特征总方差，12 列均值） | `y_cls` 组内一致率 | `loss_level` 组内一致率 |
| :-- | :-- | :-- | :-- |
| `state` | 0.082 | 0.6249 | 0.3377 |
| `year` | 0.098 | 0.6282 | 0.3512 |
| **`stage`** | 0.049 | **0.7175** | **0.3968** |
| `month` | **0.477** | 0.6127 | 0.3445 |
| `plan` | 0.021 | 0.6320 | 0.3343 |
| `coverage` | 0.095 | 0.6210 | 0.3377 |

除 `stage` 外，六条关系**全部贴到随机水平**（soy 同构：只有 `stage` 高 +0.16，其余 ≤ ±0.007）。⇒ §9.3-6 预感的“横向竞争首先发生在 attr 侧”在数字上表现为：**attr 侧的共享只能间接表达 record 相似性，而这张图里 record 的标签相似性本来就没有多少**。

**（c）标签信息几乎全在 `stage` 一个属性里。** corn train（剔去 2019）各组 `high_loss` 率：

| attr | 组间 `high_loss` 率的极差 | 极端组 |
| :-- | :-- | :-- |
| **`stage`** | **70.5 pp** | `R` 组 13,393 条 **0.0%**；`PT` 5,455 条 70.5%；`UH` 23,294 条 43.1% |
| `year` | 34.4 pp | 2012 年 46.9%，2017 年 12.5% |
| `plan` | 16.3 pp | plan42 34.1% vs plan90 17.8% |
| `month` | 15.7 pp | 6 月 31.8% vs 12 月 16.1%（但 1–3 月只有 569/142/562 条） |
| `state` | 12.7 pp | 20 号州 16.9% vs 29 号州 29.6% |
| `coverage` | 1.7 pp | 基本无用 |

另注意小样本组：`stage=OTHER` 只 19 条、`stage=PF` 425 条（soy 23 / 399），`plan` 有一组 167 条 —— 这些组上的百分比没有意义，建模时要么合并要么按频率正则。

**（d）`stage` 不是来路不正的字段，也不是特征的另一面。** 拿 12 维气象去预测 `stage`（MLP 128×3，val 早停，80 epochs，3 个种子）：corn 六次测量（含旧临时脚本在 60 epochs 下的一次）test acc 落在 **0.509–0.528**、macro-F1 **0.14–0.21**，而多数类只有 0.497；soy 稳得多：0.588–0.590 / 0.223–0.226（多数类 0.525）。七类里能预测好的只有大类，`OTHER`/`PF` 几乎全猜错 ⇒ **`stage` 带着这 12 列（均值/极值/求和）之外的信息**（它是逐日气象的 MAX 物候期，信息在逐日序列里），所以不构成“同一个量被写了两遍”的假增益。但它本身是**源表里已 one-hot 好的一列类别特征**，不是图结构。（init 修好后的重跑：corn 0.5179±0.0042 / 0.1792±0.0342，soy 0.5885±0.0017 / 0.2247±0.0036 —— 这次三个种子是真的三份 init。）

> corn 这一行方差很大：三个种子彼此可差到 ±0.007，而**重新跑一个进程**（重启解释器、同种子 0）同一配置能给出 acc 0.509 / 0.514 / 0.528、macro-F1 0.14 / 0.16 / 0.21。当时我把这归到 GPU 归约的非确定性（`index_add_`/cuBLAS 原子加），只对了一半 —— 主因是权重 init 根本没被种子覆盖（见下面第二条注的更正：本行用的 `_fit_eval_many` 正是重灾区，三个“种子”其实共用一份 init）。本行的结论（“复原得很差”）在这个方差量级下都成立，但任何小数点后两位的比较都不能拿它当证据。
>
> 同一条非确定性也适用于 §9.7 的 T1 行（当时那句「soy 与 §9.7 所有行可逐位复现」写高了，已改）：用 **argmax** 的行（T2 的 acc / macro-F1）重启解释器后仍逐位一致，而用**连续分数**的行（T1 的 AUC / PR-AUC）重启一次能漂 0.006 —— corn 的 $x$+onehot 在七次测量（§9.7 一次、§9.9 两版共五次、中途旧代码一次）里是 0.8147 / 0.8162 / 0.8185 / 0.8192 / 0.8195 / 0.8207 / 0.8221（**极差 0.0074**），而同一行的 T2 七次全给 0.5829 / 0.5871（soy 那两行的六次测量也逐位不变）。
>
> **那 0.0074 的主因已查明、已修，而且不是 GPU。** 旧代码把建网写在 `_fit_eval(...)` 的实参位置上：`_fit_eval(xx…, _mlp(62, 128, 3, 1), …)` —— 实参在外层就求值完了，函数里的 `torch.manual_seed(seed)` 管不到权重。实测：两个进程里 `torch.initial_seed()` 分别是 125052124035100 / 125053567709800，同一个不加种子的 `nn.Linear(62,128)` 两次给出发散的权重；而 `manual_seed(0)` 之后再建网则逐位相同。推论有两层，第二层才是要命的：**一个进程里只有第一个建的网络吃到随机 init，之后每个网络都从 seed 之后那份固定状态抽样。** `probe_edges` 里第一个建的网络恰好就是地板行 ⇒ 旧版“必须同进程内比”只做到了候选行彼此同 init，而它们要减去的地板是全场唯一一份随机 init —— 配对比较恰好在最该配对的那一侧断了。（`_fit_eval_many` 同一个洞：三种子共用一份 init ⇒ 本节 (d) 与 §9.7 stage 复原行报的 ±std 系统性低估。）
>
> 修法：`_fit_eval` 收构造函数、建网移到 `manual_seed` 之后，于是 init 只由（种子, 形状）决定，跨行跨进程都可比；`probe_edges` 加了 `--seeds`，多于一元时逐行报**与地板按种子配对**的差与 ±std。后果：**任何小于配对 ±std 的“增益”不能当证据**，§9.9 已按这个口径重跑。

### 9.6 全图全批在现有实现下装不下（实测）

设备：**RTX 5060 Laptop，8.0 GiB**（CUDA 可用；此前 §8 所有断言均在 CPU）。取数：`python -m dataset.hetero_probe --dataset corn --what cost`。把 7 种节点扫平成 $N=145{,}594$、六条属性关系 + `state-neighbor` 合并增广后 $\hat E=1{,}892{,}144$，单层 `OCALayer(d=16, H=4)` 的 fwd+bwd（Adam，5 次取均，峰值显存扣除基线；下表为一次运行的实测，跨次差几个百分点）：

| 配置 | 时间/次 | 峰值显存 |
| :-- | :-- | :-- |
| 单条 `record\|in_year\|year`（$\hat E$=436,682，max_deg 13,222） | 134 ms | +858 MiB |
| 单条 `record\|has_coverage\|coverage`（同 $\hat E$，**max_deg 92,644**） | **537 ms** | +857 MiB |
| 六关系合并 $T=0$ | 196 ms | +971 MiB |
| 六关系合并 $T=1$ | 618 ms | +2,673 MiB |
| 六关系合并 $T=2$（现行默认） | **801 ms** | **+3,542 MiB** |
| 六关系合并 $T=4$ | 1,195 ms | +5,279 MiB |

三个读数：

> 复核记录：`flatten_graph` 重构之后重跑了一次 `--what cost`，$N$、$\hat E$、max_deg 逐位相同，时间/显存的差别在 ±5% 内（$T{=}2$ 全量：806.8 ms / +3,513 MiB），所以下面三行读数不变。

1. **层数是墙**：单层 $T=2$ 就吃掉 3.5 GiB，`oca.yaml` 的 4 层 backbone 反向需存每层激活 ⇒ ~14 GiB > 8 GiB。**“不做 mini-batch”这条旧决定到此作废**，§9.3-7 的担心成立。可选出路（按代价升序）：`detach_iterations=True`（已有开关；代价在下面的补测第 2 条里量了，**合成图上报不出代价**）、逐关系分块前向（手接管子释放）、按 `state` 分块训练（破坏全局 $\lambda$ 统计）、`NeighborLoader`（破坏 `gates.npz` 的逐节点对齐）。
2. **耗时吃的是最大场，不是边数**：两条关系 $\hat E$ 完全相同，`coverage`（max_deg 92,644）比 `year`（13,222）慢 **4.0×**。这是 `index_add_` 在超大中心上的归约争用，**不是**复杂度问题（仍是 $O(\hat E d)$），但它意味着 attr 侧竞争的实际常数由最大 hub 决定。 Lemma 2 保证的“稳”仍然成立，不保证“快”。
3. $T$ 的显存代价不是一口价：$T=0\to1$ 一次性 $+1.7$ GiB（计算图里开始存 $s$ 与逐轮中间量），此后每轮 $\approx+0.85$ GiB。$T=4$/$T=8$ 消融行在真实图上必须先证明 $T\le2$ 跑得通再说。

**补测（§9.13 之后）：学术图这一侧，墙只有一张 —— squirrel**。同一口径（全图全批、一次 fwd+bwd+Adam step、`max_memory_allocated` 扣基线），但模型是 `oca.yaml` 的**完整 4 层 backbone + PANet**（不是上表的单层代理），所以本表可直接当预算用。取数：一次性脚本（4 step 丢首轮），**当时并发着 §9.13 表 3 的后台任务 ⇒ s/step 是上界**（上界有多宽已量到：§9.13 的 OCA 探针在无争用下重测 chameleon $l$，0.33 → **0.18 s/step**，即本表那列普遍虚高约 1.8×）；峰值是按进程独立计的，但 7 GiB 那两行本身就在溢出边界上，只当量级读。

| 配置（fold0） | params | s/step | 峰值显存 |
| :-- | :-- | :-- | :-- |
| squirrel $l$ $T{=}2$（现行默认） | 735,456 | **7.73** | **+8,983 MiB** $\dagger$ |
| squirrel $l$ $T{=}1$ | 735,456 | 1.49 | +7,058 MiB |
| squirrel $l$ $T{=}0$ | 735,456 | 0.20 | +2,286 MiB |
| squirrel $l$ $T{=}2$ + `detach_iterations` | 735,456 | **1.20** | +7,084 MiB |
| squirrel $n$ $T{=}2$ | 153,264 | 0.41 | +3,175 MiB |
| chameleon $l$ $T{=}2$（$E$=62,742） | 796,580 | 0.33 | +1,547 MiB |
| actor $l$ $T{=}2$（$E$=53,318） | 435,793 | 0.33 | +1,516 MiB |
| wisconsin $l$ $T{=}2$（$E$=900） | 635,482 | 0.23 | +101 MiB |
| 对照：同图 gcn $l$ / gat $l$ / mlp $l$ | 142,405 / 142,661 / 142,405 | 0.02 / 0.05 / 0.01 | +283 / +666 / +76 MiB |

$\dagger$ 卡容量只有 8,151 MiB ⇒ 这一档根本没装进显存，Windows 把它溢到共享内存，代价就是 7.73 vs 1.49 s/step 那 5.2×。

> 参数量这一列为什么跳来跳去：OCA 的第一层吃 $x$（$F$ 从 2,325 到 932）而基线只有 2 层，所以 `scale l` 在六张图上不是同一个数（796,580 / 735,456 / 435,793 / 635,482）；`mlp`/`gcn` 同 params 是因为它们第一层形状相同。

三条读数：

1. **只有 squirrel 装不下**。它的无向边数 $E=396{,}706$ 是其余五张的 6–440 倍（chameleon 62,742、actor 53,318、WebKB 三张 $<1{,}000$），而节点数只有 5,201 —— 成本全在边上。⇒ **OCA 行在 5/6 张图上无需任何折衷**，只有 squirrel 需要改 $T$ 或开 `detach_iterations`。
2. **`detach_iterations=True` 拆掉了这堵墙，而它在现有能测的地方报不出代价**：8,983→7,084 MiB、7.73→1.20 s/step（同一配置、只改这一个开关）。它的价格：`python train.py --ablate full detach_iters`（synth、CUDA、seeds 0/1/2）两行**逐位同分** —— 都给 0.9967±0.0058 / best_ep 10.3 / 14,909 参数，只有 $\lambda$ std 0.0126→0.0127、$r_{\lambda,\text{homo}}$ −0.022→−0.025。⇒ 但 §8.5 已说明那张图饱和，所以**不能外推成「学术图上也不贵」**；要断言就得在 OCA 行上单独加一档 `oca(detach)` vs `oca`（只有 squirrel 必须这么做）。
3. **成本对账（得写进 §9.13 的表注）**：同一张 squirrel 上一步的成本是 gcn $l$ 的 **60×（detach）到 390×（默认）**（1.20 / 7.73 vs 0.02 s/step），参数量 5.2×（735k vs 142k），而且它是 4 层 $\times(T{+}1)$ 跳的感受野 vs 基线的 2 跳。⇒ **「OCA 与文献宽度的 GCN 同规模」这句话在 scale $l$ 下不成立**，而 §9.13 协议修正的第 2 条只对齐了 hidden 宽度，对不齐参数量与感受野 —— 表里必须并排 params 与 s/run，**OCA 跟基线打平不算增益**。

### 9.7 地板值：现有 GNN 基线可以被“一条边都不用”的模型复现

同一个协议（train 2008–18 / val 2019 / test 20–22，MLP 128×3、Adam lr 3e-3、batch 8192、2019-val 早停、单种子）。取数：`python -m dataset.hetero_probe --dataset both --what floors`。“onehot” = 6 类属性各取 one-hot 拼接（50 维）；“LOO 组均值” = 每条关系上用 **train 记录**算的 leave-one-out 特征均值（72 维）—— 后者就是消息传递在做的事（邻居信息的池化），只是不可学。

**T1 `high_loss`（AUC / PR-AUC）**：

| 输入 | corn | soy |
| :-- | :-- | :-- |
| 生产方的 GNN（`train_metrics.json`，未复跑） | 0.805 / 0.555 | 0.894 / 0.756 |
| **仅 `stage` 一列（7 维，零边、零气象）** | **0.762 / 0.422** | **0.858 / 0.667** |
| $x$（12） | 0.574 / 0.269 | 0.686 / 0.391 |
| $x$，256×4 容量控制 | 0.590 / 0.274 | 0.660 / 0.381 |
| **仅 onehot（50）** | **0.803 / 0.525** | **0.882 / 0.729** |
| $x$ + onehot（62） | 0.819 / 0.597 | 0.892 / 0.737 |
| $x$ + LOO 组均值（84） | 0.812 / 0.585 | 0.888 / 0.729 |
| $x$ + onehot + LOO（134） | 0.819 / 0.601 | 0.892 / 0.735 |

**T2 `loss_level` 三分类（acc / macro-F1）**：

| 输入 | corn | soy |
| :-- | :-- | :-- |
| 仅 `stage` 一列（7 维，零边） | 0.555 / 0.547 | 0.686 / 0.691 |
| $x$（12） | 0.400 / 0.385 | 0.478 / 0.469 |
| 仅 onehot（50） | 0.571 / 0.569 | 0.687 / 0.692 |
| **$x$ + onehot（62）** | **0.583 / 0.587** | **0.688 / 0.693** |
| $x$ + onehot + LOO（134） | 0.581 / 0.585 | 0.688 / 0.694 |
| $x$，clamp $\pm4\sigma$（12） | 0.414 / 0.410 | 0.478 / 0.472 |
| $x$ + onehot，clamp $\pm4\sigma$（62） | 0.582 / 0.586 | 0.689 / 0.695 |
| chance | 0.333 | 0.333 |

四行结论，都是难听的但必须先知道：

1. **无图模型复现了生产方的 GNN 基线**：corn 0.803 vs 0.805，soy 0.882 vs 0.894 —— 只差 0.002–0.012 AUC。README 里“借助 stage/state/year 等关系结构把区分度提至 0.80~0.89”这句话，可复现的部分完全由**属性当特征**解释，不需要邻接矩阵。更极端的一行：**只拿 `stage` 这一列（7 个 one-hot，不碰另外 5 类属性、不碰 12 列气象、不碰任何边）就到 corn 0.762 / soy 0.858**。按“超过 0.5 的部分”算，它吃掉了生产方基线的 85%（corn）/ 91%（soy）—— 一个列向量。
2. **池化（消息传递的实质）的增益 $\le$ 0.001 AUC**：$x$+onehot 0.8192 → 加 LOO 组均值 0.8191（corn）、0.8917 → 0.8916（soy）。T2 上加池化反而略降（0.5871 → 0.5845）。这就是路线 A 的**信息上限估计**。
3. **因此 T2 的地板不是 0.333 而是 0.583/0.587（corn）、0.688/0.693（soy）**，hetero OCA 必须拿这组数当对照而不是当 chance；且这些是我单种子未调参的快跑值，只能低估不能高估（GBM/调参 MLP 会更高）。
4. **单靠 12 列气象做不到 0.8**（AUC 0.574→0.590 随容量停止增长），所以真实增益来自 `stage`；而 `stage` 已在源表里 one-hot。→ 拿这张图当“图学习有用”的证据，**当前形态下不成立**；要让它成立必须先把图造出信息（§9.3-5 里的 route B）。【route B 已按你的裁决先造边再复测：**不过线**（§9.9 第三版，按种子配对）。但当时那句“连组级标签率（目标编码）喂进去都是负的”要收窄：corn −0.0163±0.0080 是真的，而它的原因不是先验不迁移 —— 同一张裸查表在 test 上有 0.7991，喂进 MLP 得 0.7996（零边际贡献），见 §9.10】

以上实测直接动了 §9.3-5 路线选择的性价比判断，已记入 §10。

### 9.8 hub 诊断：$\sqrt{\deg}$ 确在改尺度，但拦路的是它前面那个 0

取数：`python -m dataset.hetero_probe --dataset both --what hub --steps 200`。代理是**单层** $d{=}16,H{=}4,T{=}2$，六关系合并扫平（corn $N=145{,}594$、$\hat E=1{,}892{,}144$；soy $N=133{,}472$、$\hat E=1{,}734{,}558$），建层前 `torch.manual_seed(0)`。$A$ = 当前数据形态（attr 无特征），$B$ = 给 attr 随机 embedding，用来代理 §9.3-2 修好后的形状。

**(a) 那个因子确实在按 $1/\sqrt{\deg}$ 吞信号，但只吞尺度。** 两版统计量（严格 $z$ / 现行代码）的逐桶比值等于 $\sqrt{\deg}$：record 侧 $\deg=6$ 比值 **2.4**，attr 大 hub（均度 19,352）比值 **113.5**（该桶 $\sqrt{\deg}$ 均值 127.7，差异来自桶内度数分布）。soy 同形（101.6 vs 121.9）。推断成立，没有意外。

**(b) 锐度不跟它走。** 聚合锐度用参与率 $PR_i=1/\sum_j p_{i,j}^{2}$（= 有效平均了几个邻居）除以 $\deg_i$，$PR/\deg\to1$ 就是均值池化。init 时 $B$ 形态 record 侧 $0.9876$ vs 去掉 $\sqrt{\deg}$ 后 $0.9878$ —— 差 $2\times10^{-4}$。原因：$\tau$ 的先验激活里 stat 那列的贡献 std 只有 **0.0228**，而 12 列原始气象是 **0.5299**（同一个 $w_\tau$、同一尺度）—— stat 无论怎么缩放，还轮不到它说话。

**(c) 学习会自己补回这个因子。** 拿真实 `loss_level` 训一个线性头 200 步（corn val loss 0.803）：$w_\tau$ 的 stat 列 $+0.0731\to+0.5016$（**6.9×**），$PR/\deg$ 从 0.988 掉到 **0.43**（record）/ 0.19–0.44（attr）。锐度是学出来的，$\sqrt{\deg}$ 没挡住它。soy 同向但不一致（stat 列 $\to+0.0404$，0.6×，而 $PR/\deg$ 同样掉到 0.43）⇒ stat 不是唯一通路。

**(d) 真正的拦路者：$A$ 形态下 record 的竞争场是零展开的。** 两遍算法算的场内绝对偏差 $\mathrm{mad}$ 在 corn/soy 都是 $p_1/p_{50}/p_{99}=0\,/\,0\,/\,3.7\times10^{-9}$，且**全部 145,544（soy 133,422）个 record 中心**的场内方差贴住 `var.clamp(min=1e-8)` 下界。⇒ 这不是 float32 单遍抵消误差（两遍算法也是 0），是**真零方差**：attr 节点的 $x$ 整行是 0，六个邻居的分数逐位相同。后果链：$\text{stat}$ 由 $10^{-4}$ 硬除出来 $\to\lvert\text{stat}\rvert\approx211$（soy 203）；$\tau$ 跨 5 个数量级（$p_1=0.001$ 贴着下限、$p_{99}=40$、max 80；soy max 176）；而 $PR/\deg$ 恒为 $1.0000$ —— 场里没有任何可分辨的差别，**温度在当前 record 侧根本不起作用**，与 $\sqrt{\deg}$ 无关。

> 顺带纠正我自己先前的两个误判。① 上一轮我说「$A$ 形态下读出 NaN」—— 清点非有限值后 out/s/tau/stat 的计数**全为 0**，模型侧从来不出 NaN（`tau_min` 下限有效）；当时的 nan 出在探针自己漏写 $+\tau_{\min}$（$\tau$ 在 fp32 下溢到 0、探针里的 $s/\tau$ 先得 inf），已在 `dataset/hetero_probe.py::_field_quantities` 与 `modules/oca.py:392` 对齐。② 第一版 hub 读数不可复现（同一配置重跑，$w_\tau$ 的 stat 列符号会从 $-0.11$ 变成 $+0.21$），因为建层前没固定种子；现在是固定后的数。

**裁决建议（§10 第 2 行）**：**不改数值，只改名字**。理由：(a) 它的作用可预测且不越界；(b) 对锐度实测无影响；(c) 去掉它要重跑整张 §8.4 的端到端与消融表，而换来的只是“z-score 这个名字变准确”。低成本的正确做法已写在 §2.7：把它叫「按 $1/\sqrt{\deg_i}$ 再缩放的中心-场差异统计量」，`center_stat='zscore'` 这个键名当历史别名留着。该排到前面的是 §9.3-2（attr embedding）：$A$ 形态不修，温度与横向竞争在 record 侧都是空转。

### 9.9 route B 造边的 go/no-go（第三版：配对之后，16 行候选里几乎没正号）

> **第一版的六行读数作废，原因是我自己的造边口径坏了。** 第一版把格元邻域定义成「同格元且年份差 $\le1$ 的 train 记录」，可表源里 train 行的年份最大就是 2018，而 test 行在 2020–22 ⇒ 带宽**必然越界**。实测（corn，按 split 拆零邻居数）：`state×stage`、`state×month×stage`、对侧格元三行的 test 零邻居是 **25,047 / 25,047**，一行不剩；以 `state×month×stage` 为例，那 12 列邻域特征在 test 上 $\max|x| = 0.000$，在 train 上是 $9.083$。所以那六行（含依赖它们的标签通路两行）测到的是「一列特征整列塌成常数、而模型已把权重押上去」，不是「邻域有没有标签信息」。修法两条：格元行改成**跨全窗口池化**（对该轴求和，train 行仍 LOO ⇒ train/val/test 同口径，且源侧只有 train 行，无泄漏）；kNN 的候选集限在 train 行内（第一版没限，test 行的十近邻平均只剩 **3.07** 个 train 邻居、train 行有 8.94 个，同样是构造自造的失配）。唯一没坏的是 `state×plan×coverage, month±1` 那行（轴是 month，test 上只有 1 行越界）。

> **第二版错在地板行没配对，本版已修，结论因此又变一次。** 上一版是单种子，且把建网写在 `_fit_eval` 的实参位置 ⇒ 地板行作为该进程里第一个建的网络，init 抽自进程随机的默认 RNG，那一次抽到 **0.8221**；三个种子配平以后地板是 **0.8159**。基准被一次偏高的抽样抬走了 0.006，于是所有候选行的 $\Delta$ 一起下移，而全表唯一的正号恰好落在那行上。本版跑 3 个种子、逐行与地板**按种子配对**相减并给 ±std ⇒ “交叉项是唯一干净的正信号”撤销，宽度税从 0.008 缩到 0.004。

取数：`python -m dataset.hetero_probe --dataset both --what edges --epochs 80 --seeds 0,1,2`。评价口径与 §9.7 同（train 2008–18 / val 2019 / test 20–22，MLP 128×3、val 早停），差别是本节跑 3 个种子且每一行都减去**同进程、同一种子**的地板行。corn 地板（62 维）：T1 AUC 0.8159±0.0023、PR 0.5859±0.0076、T2 acc 0.5776±0.0040、F1 0.5788±0.0063；soy 地板：0.8911±0.0005、0.7391±0.0017、0.6874±0.0002、0.6929±0.0003。注意 §9.7 表里那个 0.8192 是**种子 0 的单跑**（刚重跑了 `--what floors`，那一行仍给 0.8192），与这里 0.8159 的三种子均值差 0.003 —— 两张表的 $\Delta$ 各按自己的地板算，不可跨表混减。

三个候选邻域（「县级邻接」已由 §9.2 的事实划掉）：**特征空间 kNN**（标准化 + clamp $\pm4\sigma$ 的 12 维 $x$ 上取 $k=10$，全局 / 限同 `state`，候选 = train 行、不含自身）；**属性组合格元**（$(state,stage)$ 与 $(state,month,stage)$，邻域 = 同格元的**全部** train 记录，LOO 扣自身）；**对侧作物同格元**（corn 的邻域 = soy 落在同一 $(state,month,stage)$ 的 train 记录）。另外两段是尺子：**标签通路**（把格元的 train 标签率 $\hat P(y_{cls}{=}1)$ 与 $\hat E[loss\_level]$ 当特征 = 目标编码，组级标签先验的最强形式）与**无信息对照**（同样加 12 列：iid 噪声，或把上面那些邻域均值按行打乱 —— 保尺度、保列间相关，只破坏与样本的对应）。

**corn**（括号外是 3 个种子的 mean±std；$\Delta$ 列 = 与地板按种子配对的差）

| 候选邻域 | 邻居/中心 | 零邻居 | $y_{cls}$ excess | $loss\_level$ excess | T1 AUC | T1 $\Delta$配对 | T2 acc | T2 $\Delta$配对 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 参照：`stage` 单属性 | 34,171 | 0 | +0.0832 | +0.0618 | — | — | — | — |
| 参照：另 5 个单属性 | 7.7k–58k | 37,923（`year`） | −0.0179 … +0.0031 | +0.0013 … +0.0211 | — | — | — | — |
| kNN 全局 | 10.0 | 0 | +0.0319 | +0.0535 | 0.8123±0.0046 | −0.0036±0.0037 | 0.5762±0.0029 | −0.0014±0.0040 |
| kNN 限同 `state` | 10.0 | 0 | +0.0359 | +0.0613 | 0.8140±0.0049 | −0.0019±0.0039 | 0.5778±0.0023 | +0.0002±0.0040 |
| $state\times stage$，全窗口 | 5,489.4 | 11 | +0.0913 | +0.0744 | 0.8137±0.0069 | −0.0022±0.0048 | 0.5798±0.0028 | +0.0022±0.0027 |
| $state\times month\times stage$，全窗口 | 880.3 | 524 | +0.1049 | +0.1275 | 0.8090±0.0062 | −0.0069±0.0041 | 0.5760±0.0010 | −0.0016±0.0031 |
| 对侧（soy）同格元 | 798.5 | 629 | **+0.1419** | **+0.1598** | 0.8100±0.0041 | −0.0059±0.0025 | 0.5765±0.0020 | −0.0011±0.0028 |
| $state\times plan\times coverage$，$month\pm1$ | 1,492.0 | 15 | −0.0059 | +0.0242 | 0.8141±0.0034 | −0.0019±0.0014 | 0.5782±0.0008 | +0.0007±0.0037 |
| 标签通路：self 格元的标签率 | 880.3 | — | — | — | 0.7996±0.0058 | **−0.0163±0.0080** | 0.5705±0.0020 | −0.0071±0.0022 |
| 标签通路：对侧格元的标签率 | 798.5 | — | — | — | 0.8065±0.0052 | −0.0095±0.0029 | 0.5705±0.0029 | −0.0071±0.0019 |
| 无信息对照：iid 噪声 12 列 | — | — | — | — | 0.8101±0.0037 | −0.0059±0.0032 | 0.5743±0.0028 | −0.0032±0.0021 |
| 无信息对照：4 个行置换 | — | — | — | — | 0.8098–0.8143 | −0.0017 … −0.0061 | 0.5761–0.5792 | −0.0015 … +0.0016 |

**soy**（地板 0.8911±0.0005 / 0.6874±0.0002；soy 的种子间抖动只有 corn 的四分之一，所以它的 $\Delta$ 更小也更可信）

| 候选邻域 | 邻居/中心 | 零邻居 | $y_{cls}$ excess | $loss\_level$ excess | T1 AUC | T1 $\Delta$配对 | T2 acc | T2 $\Delta$配对 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 参照：`stage` 单属性 | 33,151 | 0 | +0.1566 | +0.1229 | — | — | — | — |
| 参照：另 5 个单属性 | 7.0k–56k | 34,751（`year`） | −0.0068 … +0.0087 | +0.0024 … +0.0180 | — | — | — | — |
| kNN 全局 | 10.0 | 0 | +0.0332 | +0.0613 | 0.8895±0.0017 | −0.0016±0.0022 | 0.6866±0.0017 | −0.0008±0.0015 |
| kNN 限同 `state` | 10.0 | 0 | +0.0385 | +0.0716 | 0.8903±0.0012 | −0.0008±0.0016 | 0.6826±0.0064 | −0.0047±0.0064 |
| $state\times stage$，全窗口 | 5,344.6 | 8 | +0.1621 | +0.1372 | **0.8917±0.0009** | **+0.0006±0.0007** | 0.6861±0.0019 | −0.0012±0.0021 |
| $state\times month\times stage$，全窗口 | 924.7 | 416 | **+0.1856** | **+0.2136** | 0.8911±0.0014 | −0.0000±0.0015 | 0.6879±0.0012 | +0.0005±0.0011 |
| 对侧（corn）同格元 | 873.6 | 464 | +0.1344 | +0.1569 | 0.8902±0.0015 | −0.0009±0.0012 | 0.6827±0.0079 | −0.0047±0.0079 |
| $state\times plan\times coverage$，$month\pm1$ | 1,403.1 | 15 | +0.0008 | +0.0369 | 0.8909±0.0008 | −0.0002±0.0012 | 0.6851±0.0005 | −0.0023±0.0003 |
| 标签通路：对侧格元的标签率 | 873.6 | — | — | — | 0.8902±0.0006 | −0.0009±0.0011 | 0.6856±0.0012 | −0.0017±0.0011 |
| 标签通路：self 格元的标签率 | 924.7 | — | — | — | 0.8878±0.0013 | −0.0033±0.0018 | 0.6873±0.0017 | −0.0001±0.0016 |
| 无信息对照：iid 噪声 12 列 | — | — | — | — | 0.8872±0.0011 | −0.0038±0.0011 | 0.6806±0.0050 | −0.0068±0.0051 |
| 无信息对照：4 个行置换 | — | — | — | — | 0.8892–0.8904 | −0.0007 … −0.0019 | 0.6846–0.6859 | −0.0014 … −0.0027 |

五条读数（配对口径）：

1. **判据（2）这次是真的不过线**。16 个候选行（每作物 6 行邻域均值 + 2 行标签率）里 corn **0 行**为正，soy 只有 $state\times stage$ 一行 **+0.0006±0.0007**（配对差的标准差比它自己还大）。上一版那个“corn +0.0009 的唯一正号”是基准行的抽样偏高，不是信号。
2. **宽度税缩到 corn $\approx0.004$、soy $\approx0.002$**（上一版报的 0.008 里多出的部分就是地板那个 0.006 的镜像）：无信息带 corn −0.0017 … −0.0061（均值 −0.0037）、soy −0.0007 … −0.0038（均值 −0.0020）。**所有候选行都落在这条带子里或带子下方** ⇒ 修正后既不能说格元有害，也不能说它有功。
3. **上一版的读数 3（“全表唯一干净的正信号是属性交叉项”）撤销**：corn 的 $state\times stage$ 现在是 −0.0022±0.0048，与无信息带不可区分；soy 的 +0.0006±0.0007 只有 0.9 个 std。⇒ “route A 该去学可学的属性交叉项”这条建议在本节**没有证据了**；剩下的只有 §9.7 那个间接事实（属性当特征就够 0.803 / 0.882，池化不再涨）。
4. **标签通路（目标编码）仍是全表最差，但从“有害”改口为“零边际贡献”**：corn 的 self 行 −0.0163±0.0080（仍显著低于带内均值 −0.0037），可它的绝对值 0.7996±0.0058 与那张裸查表在 test 上的 pooled AUC **0.7991**（§9.10）**打平**。上一版说“模型输给了自己的一个输入”（0.7925 $<$ 0.7991）用的也是未配对的单跑，现在只能说：**一张单练值 0.7991 的表交给 MLP，它做不到比这张表更好，还要在它上面倒贴 0.016**。soy 的 self 行 −0.0033±0.0018 已在带内（上一版那句“soy −0.0053 有害”同样是基准抽样假象）。信息侧的结论不变：落差在拟合通路（计数 30 的小格元与计数 3 万的大格元同权、没按可信度收缩、尺度与 one-hot 不一致）。
5. **本节的分辨率，下次比数前先拿它卡**：行内种子间 std corn 达 ±0.007（$state\times stage$ 行 0.0069）、soy 只 ±0.0015；配对差 std corn 0.0014–0.0080、soy 0.0007–0.0022 ⇒ 这张图上能被断言的最小非零效应，corn 约 0.01、soy 约 0.003。未覆盖的两处照旧：（i）本节测的是“不可学的邻域池化”，它是消息传递的替身而非上界；（ii）图里唯一真正“关系型”的 `state-neighbor`（22 条州邻边）未单独测，而 `state` 单属性 excess −0.0056 已贴随机。

**结论：route B 仍判 no-go，而且现在是“配对着判”的**（按你定的 gate：“造完复测，过线才开工 route A”）。三种造边加上目标编码，在按种子配对的设计里没有一行给出可信增益。手里剩下的硬事实只有一条：组级标签率先验在 test 上值 0.7991，而 MLP 拿它做不到比它更好。这仍是**拟合侧**的问题（计数没做收缩、可信度没进门、尺度没对齐），不是“信息不迁移”（§9.10），也不是“信息有害”（本版）。§10 第 7 行已据此改写，并把上一版“route A 该学交叉项”的建议一并降级。

> 两版写错、现已推翻的读数，留着当错账。**v1（`year\pm1` 越界 + kNN 候选未限 train）**：甲“格元四行比噪声带还差 4–10 倍，是确定的有害”、乙“组级先验在样本内真实存在但不随年份迁移”、丙“kNN 两行零邻居 16,512”。**v2（地板行 init 未配对）**：丁“corn $state\times stage$ +0.0009 是全表唯一干净的正信号，所以 route A 该学属性交叉项”、戊“宽度税 corn $\approx0.008$”、己“soy 标签率 −0.0053 有害”、庚“模型输给裸查表（0.7925 $<$ 0.7991）”。v2 四条同源于一个未被种子覆盖的 init（§9.5 的注已更正）—— 同一个错误把三个方向的读数各自推歪了一次，只因为它是被减的那个基准。

### 9.10 组级先验的年份漂移实测（“不迁移”那句是错的）

取数：`python -m dataset.hetero_probe --dataset both --what drift`。表源 = 2008–18 的 **train** 行（corn 107,621 / soy 98,671），目标年用该年**全部**行；分数 = 格元在表源里的 $y_{cls}$ 率，unseen 退回表源全局率。三段读法：(a) lag 曲线（单一年份表 $\to$ 相隔 $L$ 年的那一年，forward 用过推未 / backward 未推过）；(b) 窗口表（目标年若在窗口内则整年从表里剔除）$\to$ 逐年，配格元级 Spearman 与加权 $|\Delta p|$；(c) pooled 三口径：逐年均值｜跨年混排｜oracle 年度重标定（把每年分数均值掰回该年真实正类率，用了标签 $\Rightarrow$ 上界）。反序对照（行 $\to$ 格元映射整体打乱）10 组全部落在 0.497–0.501。

> 一个口径修正：目标侧的率必须用该年**全部**行算。源侧只能统计 train 行（不然就是自拟合），而 test 年一行 train 都没有 —— 第一版拿 train 做分母，2020–22 的 Spearman 全变 nan，恰好丢掉本节要问的那三年。

**corn**（窗口全局正类率 0.2591）

| 粒度 | 窗口内格元数 / 计数 $\ge$30 | 窗口内均值 $\to$ 2020–22 | 漂移 | test 年 Spearman | test 年加权 $|\Delta p|$ | unseen 峰值 | pooled：逐年｜混排｜oracle |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| `stage` | 7 / 6 | 0.7612 $\to$ 0.7688 | +0.0076 | 0.90–1.00 | 0.046–0.108 | 0.0% | 0.7688｜0.7618｜0.7996 |
| `state` | 7 / 7 | 0.5393 $\to$ 0.5444 | +0.0050 | 0.04–0.75 | 0.084–0.100 | 0.0% | 0.5444｜0.5279｜0.6141 |
| $state\times stage$ | 48 / 40 | 0.8050 $\to$ 0.8146 | +0.0096 | 0.95–0.96 | 0.066–0.112 | 0.1% | 0.8146｜0.8068｜0.8260 |
| **$state\times month\times stage$** | 425 / 224 | 0.8043 $\to$ 0.8040 | **−0.0003** | **0.87–0.88** | 0.068–0.111 | 2.3%（2020） | 0.8040｜**0.7991**｜**0.8155** |
| $state\times plan\times coverage$ | 87 / 84 | 0.5339 $\to$ 0.5569 | +0.0230 | 0.006–0.37 | 0.082–0.119 | 0.0% | 0.5569｜0.5614｜0.6186 |

**soy**（窗口全局正类率 0.2447）

| 粒度 | 窗口内格元数 / 计数 $\ge$30 | 窗口内均值 $\to$ 2020–22 | 漂移 | test 年 Spearman | test 年加权 $|\Delta p|$ | unseen 峰值 | pooled：逐年｜混排｜oracle |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| `stage` | 7 / 6 | 0.8445 $\to$ 0.8618 | +0.0173 | 1.00 | 0.020–0.053 | 0.0% | 0.8618｜0.8576｜0.8793 |
| `state` | 7 / 7 | 0.5635 $\to$ 0.5852 | +0.0217 | 0.32–0.82 | 0.034–0.083 | 0.0% | 0.5852｜0.5759｜0.5960 |
| $state\times stage$ | 48 / 40 | 0.8630 $\to$ 0.8855 | +0.0225 | 0.90–0.95 | 0.044–0.074 | 0.1% | 0.8855｜0.8804｜0.8901 |
| $state\times month\times stage$ | 426 / 216 | 0.8636 $\to$ 0.8809 | +0.0173 | 0.91–0.92 | 0.040–0.069 | 2.0%（2020） | 0.8809｜0.8766｜0.8835 |
| $state\times plan\times coverage$ | 87 / 85 | 0.6059 $\to$ 0.6420 | +0.0361 | 0.47–0.66 | 0.062–0.093 | 0.0% | 0.6420｜0.6389｜0.6582 |

四条读数：

1. **排序迁移得很好，水平也几乎不漂**。最细粒度（$state\times month\times stage$，224 个合格格元）的窗口表搬到三年后的 test 上，corn 的逐年 AUC 只动 **−0.0003**，格元级 Spearman 在 test 年是 **0.87–0.88**（$state\times stage$ 更高，0.95–0.96）；soy 漂 +0.017、Spearman 0.91–0.92。而 $|\Delta p|$（率的绝对偏移）只有 0.07–0.11。
2. **没有可检出的时间趋势**。lag 曲线 forward 与 backward 基本对称（corn $state\times month\times stage$：forward 0.7781/0.7689/0.7515/0.7622 at $L=1/2/3/10$，backward 0.7670/0.7549/0.7417/0.7449），差别在年际噪声量级（2011、2012 两个灾年把窗口内均值拉低，而 2012 一年的加权 $|\Delta p|$ 就 0.279）。“不随年份迁移”需要的是一个方向性衰减，这里没有。
3. **水平漂移存在，但很便宜**。跨年混排比逐年均值低 0.005（corn 0.7991 vs 0.8040）到 0.004（soy 0.8766 vs 0.8809）；用标签做 oracle 年度重标定也只拿回到 0.8155 / 0.8835。⇒ “每年基率不同”一共值 $\approx0.01$ AUC，与 §9.9 标签通路那一行的落差（−0.0163±0.0080）同量级 —— 也就是说目标编码那行的损失全在拟合侧。
4. **粗粒度那两行不是漂移，本来就没信息**：`state` 单属性 pooled 0.5279–0.5451（贴随机），$state\times plan\times coverage$ 0.5614/0.6389；它们的 Spearman 也乱（0.006–0.75）——7 个或 87 个格元，两侧计数都不够稳。

**⇒ ② 的卖点要换。** “组级先验不随年份迁移”不成立，而 §9.9 的标签通路行在全表最差是真的（配对后仍有 −0.0163±0.0080）。两件事拼起来的硬事实是：**一张裸查表在 test 上有 0.7991 的 pooled AUC，把这列喂进 74 维 MLP 后得到 0.7996±0.0058 —— 零边际贡献，而且相对地板倒贴 0.016。**（上一版写的“模型输给了它自己的一个输入（0.7925 $<$ 0.7991）”拿未配对的单跑当了结论，已撤销；两头的数字现在打平，“输给”不成立。）要修的机制在拟合侧：格元先验要按计数收缩（经验贝叶斯 / $k$-smoothing）、可信度要作为第二列进模型、尺度要与 one-hot 对齐。这也回答了 route A 该长成什么样：与其让 attr embedding 去“抑制一份不迁移的先验”，不如让它显式携带**格元的样本量**。（上一版追加的“并学交叉项”已随 §9.9 读数 3 一起撤销。）**→ 本段那句“要修的机制在拟合侧”已由 §9.11 实测限定：拟合侧能修好危害、修不出增益，真正的天花板是“裸分 vs 地板”。**

### 9.11 目标编码“怎么喂”的对照：能把倒贴抹平，但拿不到正号

取数：`python -m dataset.hetero_probe --dataset both --what te --epochs 80 --seeds 0,1,2`。动机是 §9.10 末句那个断言——“落差在拟合侧”。本节不改模型、不加参数，只改喂进去的那几列，把它当假设来证伪：

- **收缩**（经验贝叶斯 / $k$-smoothing）：$r=(a+\tau p)/(n+\tau)$，$a$ 是格元内命中数、$n$ 是格元的 train 记录数、$p$ 是全局 train 率。它只做一件事：$n$ 小的时候不把 $a/n$ 当证据。
- **计数列**：额外给一列 $\log(1+n)$，让模型能自己做可信度门控（原式把率与计数压进了同一列）。
- **粒度**：$state\times stage$（每格 train 数中位 1,278 / 1,112）与 $state\times month\times stage$（中位 36 / 32，即 §9.9 那两行的格元）。

每行仍是 $x+onehot$ 打底 $+\,12$ 列（不足补 iid 噪声），与 §9.9 同宽度，所以宽度税与无信息带照旧适用。

**裸查表**（不进模型，分数 = 收缩后的 $P(y_{cls}=1)$，test 行；这一列与模型无关，所以是确定值）：

| 作物 · 粒度 | $\tau$=0 | 1 | 2 | 3 | 5 | 10 | 20 | 30 | 100 | 300 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| corn $s\times st$ | **0.8068** | 0.8069 | 0.8068 | 0.8067 | 0.8065 | 0.8058 | 0.8051 | 0.8042 | 0.7995 | 0.7937 |
| corn $s\times m\times st$ | 0.7991 | 0.8003 | 0.8011 | 0.8013 | **0.8015** | 0.8008 | 0.7993 | 0.7985 | 0.7930 | 0.7871 |
| soy $s\times st$ | 0.8804 | 0.8805 | 0.8805 | 0.8805 | 0.8806 | **0.8808** | 0.8800 | 0.8796 | 0.8772 | 0.8720 |
| soy $s\times m\times st$ | 0.8766 | 0.8786 | 0.8793 | **0.8794** | 0.8790 | 0.8778 | 0.8767 | 0.8759 | 0.8716 | 0.8673 |

**喂进 MLP**（T1 AUC，括号内是与地板按种子配对的 $\Delta\pm$std；地板 corn 0.8159±0.0023 / soy 0.8911±0.0005）：

| 喂法（12 列） | corn $s\times st$ | corn $s\times m\times st$ | soy $s\times st$ | soy $s\times m\times st$ |
| :-- | :-- | :-- | :-- | :-- |
| §9.9 原样（未收缩，兜底反向） | 0.7982 (−0.0178±0.0055) | 0.8000 (−0.0160±0.0093) | 0.8899 (−0.0012±0.0011) | 0.8859 (−0.0052±0.0013) |
| 率 $\tau=0$（兜底方向自洽） | 0.8053 (−0.0106±0.0113) | 0.7969 (−0.0190±0.0072) | 0.8894 (−0.0016±0.0010) | 0.8863 (−0.0048±0.0005) |
| 率 $\tau=0$ + $\log n$ | 0.8141 (**−0.0018±0.0031**) | 0.8102 (−0.0057±0.0022) | 0.8895 (−0.0016±0.0008) | 0.8879 (−0.0032±0.0013) |
| 仅 $\log n$ | 0.8106 (−0.0054±0.0043) | 0.8088 (−0.0071±0.0034) | 0.8870 (−0.0040±0.0003) | 0.8890 (−0.0021±0.0003) |
| 率 $\tau=3$ | 0.8052 (−0.0107±0.0120) | 0.8073 (−0.0086±0.0012) | 0.8902 (−0.0009±0.0019) | 0.8877 (−0.0034±0.0019) |
| 率 $\tau=3$ + $\log n$ | 0.8148 (**−0.0012±0.0020**) | 0.8116 (−0.0043±0.0019) | 0.8882 (−0.0029±0.0015) | 0.8893 (−0.0018±0.0011) |
| 率 $\tau=10$ | 0.7999 (−0.0160±0.0114) | 0.8093 (−0.0066±0.0008) | 0.8885 (−0.0026±0.0005) | 0.8878 (−0.0033±0.0014) |
| 率 $\tau=10$ + $\log n$ | 0.8114 (−0.0045±0.0036) | 0.8117 (−0.0042±0.0013) | 0.8884 (−0.0027±0.0015) | 0.8904 (**−0.0007±0.0011**) |
| 率 $\tau=100$ | 0.8069 (−0.0090±0.0030) | 0.8080 (−0.0079±0.0016) | 0.8888 (−0.0023±0.0015) | 0.8896 (−0.0015±0.0010) |
| 率 $\tau=100$ + $\log n$ | 0.8092 (−0.0067±0.0033) | 0.8102 (−0.0057±0.0027) | 0.8892 (−0.0019±0.0008) | 0.8903 (−0.0008±0.0007) |
| — iid 噪声 12 列（无信息带） | 0.8064 (−0.0095±0.0017) | 同左 | 0.8878 (−0.0033±0.0011) | 同左 |
| — 率 $\tau=0$ 的行置换（无信息带） | 0.8090 (−0.0069±0.0017) | 0.8087 (−0.0072±0.0023) | 0.8869 (−0.0041±0.0010) | 0.8878 (−0.0033±0.0015) |

T2 那一侧不单独列表：40 个读数里最大的是 +0.0005（soy $s\times st$，$\tau=3$），没有任何一行离开噪声，与 T1 同结论。

六条读数：

1. **两处独立复现校核通过**。地板在本函数里重跑得 corn 0.8159±0.0023 / soy 0.8911±0.0005，与 §9.9 逐位相同；“§9.9 原样”那行的 corn 细格元是 −0.0160±0.0093，§9.9 记的是 −0.0163±0.0080——不同进程、不同噪声列、不同函数，差 0.0003。这是 init 修进 seed 之后第一次跨进程复现，说明 §9.9 的配对口径站得住。
2. **收缩只在细格元上有增益，而且很小**：corn $s\times m\times st$ 从 0.7991 抬到 $\tau=5$ 的 0.8015（**+0.0024**），soy 从 0.8766 抬到 $\tau=3$ 的 0.8794（**+0.0028**）。粗格元上 $\tau=0$ 就是最优，收缩只会伤（corn $s\times st$ 到 $\tau=300$ 掉 0.013）。⇒ 小计数确有毒，但**粒度比收缩重要**：粗格元不收缩的 0.8068 仍高于细格元收缩后的最优 0.8015，差 0.005 —— 是“该合并哪些格元”的问题，不是“该把率往先验收多少”的问题。
3. **写本节时怀疑过的一个 bug，实测不值钱**。`_label_rate` 的第 0 列是 $P(y_{cls}=0)$，而它的零邻居兜底填的是 $p=P(y_{cls}=1)$——方向错了一格（corn 细格元 524 行、粗格元 11 行）。我把它单独做成一行来量：四个“作物 $\times$ 粒度”上，方向自洽版相对原样是 +0.0072 / −0.0030 / −0.0004 / +0.0004，两好两坏且都在 $\pm$std 内。⇒ **它不是那 0.016 的解释**，也因此没有去改 `_label_rate`（改了会使 §9.9 已入档的读数失去可比性）。
4. **强形式被否：拟合侧能修好危害，修不出增益**。corn 细格元从 −0.0190（裸喂未收缩的率）抬到 −0.0042（$\tau=10+\log n$）、corn 粗格元抬到 **−0.0012**、soy 细格元抬到 **−0.0007** —— 最好的几行已经与 0 不可区分（0.6$\sigma$），但 **40 个候选读数（10 种喂法 $\times$ 2 作物 $\times$ 2 粒度）里没有一行显著为正**。弱形式成立的那一半是：−0.016 里大约有 0.012 是“没收缩”造成的自家伤害，收缩之后剩下的 −0.007 已落进**本节自己的**无信息带（iid −0.0095、行置换 −0.0069 / −0.0072），即与“加了 12 列纯噪声”不可区分。
5. **计数列确实赎回来一部分，但只够到打平**。每一档粒度上“率 $+\log n$”都比“率”单用好（corn 粗 −0.0106→−0.0012、corn 细 −0.0190→−0.0042、soy 细 −0.0048→−0.0007），而“仅 $\log n$”单用是 −0.0021…−0.0071（不比 iid 噪声好）——两者只有搭配才起作用，这正是“可信度门控”的定义，也是对假设 (ii) 唯一支持的证据。但**越过无信息带 ≠ 越过地板**。
6. **统一的解释（本节真正留下的东西）**：一列组级统计量能不能涨，先看它的裸分越不越过地板。上面那张 $\tau$ 表里 40 个裸分读数，峰值 0.8068 / 0.8015 / 0.8808 / 0.8794，**没有一个越过各自的地板 0.8159 / 0.8911**；而 40 个 MLP 候选读数**没有一行显著为正**。两件事严格一致并非巧合：目标编码就是从 one-hot 里那几个属性算出来的，它是地板已经掌握的信息的一个函数，天花板天然在下面。

**⇒ §9.10 那句“落差在拟合侧”要加一半限定，而且它反过来说明了另一件事。** 拟合侧的修正确实存在（收缩 + 计数列，把 −0.019 抬到 −0.001），但修到头也只是打平。对 route A 的后果是具体的、可执行的：attr embedding 想学的就是这种组级查表，而**开工前先量“要学的那个统计量的裸分 vs 地板”**——越不过地板就不要为它改模型。这条判据顺手解释了 §9.9 全表：三种造边 0 行增益不是运气，是它们的可表信息都在地板之下。

> 本节没有推翻任何已入档数字，但把 §9.10 末段与 §10 第 7 行那句“问题在拟合侧”降级为“拟合侧的症状，病因是天花板”。
>
> **→ §9.12 拿 6 张标准异配图重测了本节的判据：判据通过（6/6 图裸分不越地板、6/6 图没有专属增益），但它的表述要改：「裸分 vs 地板」必须配一条**同尺度**的同宽度对照才可用，而且真正的分野是「聚合邻居的 $y$ 还是聚合邻居的 $x$」。**

### 9.12 判据的迁移检验：拿到标准异配基准上重测

取数（权威档是第一条，带 `--feat-z`）：

```bash
python -m dataset.acad_probe --datasets chameleon,squirrel,actor,wisconsin,texas,cornell \
       --folds 0,1,2,3,4 --seeds 0,1,2 --epochs 300 --feat-z     # 公平档（下表）
python -m dataset.acad_probe --datasets chameleon --folds 0,1,2,3,4 --seeds 0,1,2 --epochs 300                  # quiet 档
python -m dataset.acad_probe --datasets chameleon --folds 0,1,2,3,4 --seeds 0,1,2 --epochs 300 --no-scale       # loud 档
```

载体是新文件 `dataset/acad_probe.py`。它复用 §9.5–§9.11 的同一个 `_fit_eval`（128×3 MLP、Adam lr 3e-3、wd 1e-5、batch 8192、按 val 早停），只换三样东西：把「属性格元」换成**图上的 1 跳邻域**，把「组级标签率」写成同一个收缩式 $r=(a+\tau p)/(n+\tau)$（$\tau\in\{0,1,5,20\}$，正反两个方向：异配图上「邻居里最不可能的类」也是一个规则），把对照从两条加到**三条同宽度**：行置换 / iid 噪声 / **$x$ 的固定随机投影**（后者是「同样加宽、但信息量是 $x$ 的子集」，专治 §9.11 没有的那个病）。配对单位是（官方折, 种子）$=15$ 元，全部用 geom-gcn 自带 10 折，不用随机划分。

**为什么值得换图重测**：§9.11 那条判据是从一张「地板异常高（属性 one-hot 单用就有 0.805）+ 图异常弱（邻居特征池化增益 $\le0.001$）」的图上归纳来的。两个条件同时成立时，“裸分越不过地板”可能是规律，也可能只是环境。本节把它放到反面条件下：地板是词袋上的 MLP（0.30–0.48），而 1 跳邻居特征均值单加进去就值 $+0.10\ldots+0.14$。

**先把射程说清（环境事实）**：本机 `raw.githubusercontent.com:443` 通、`github.com:443` 不通 ⇒ geom-gcn 那批能下，**Planetoid 三张（Cora / CiteSeer / PubMed）下不下来**；`data/Cora/raw` 也补不出（只有 5 个文件，缺 `ind.cora.graph` 与 `test.index`，`ind.cora.ty` 是 0 字节）。⇒ 本节只覆盖「异配」这一侧，**同质那一侧目前没有真实图可测**（`--datasets 'synth:h=0.85:sep=1.5'` 能填形状，但合成图的准确率按本仓库的既定口径不能当论文卖点）。

**表 1 · 公平档**（$\pm$ 是 15 个配对单元的总体 std，不是标准误；$t=$ 均值 $/(\mathrm{std}/\sqrt{15})$，只当分辨率看）：

| 图 | $h_{edge}$ | 地板 acc / F1 | 地板+（$x+$邻居特征均值）$\Delta$acc | 裸分峰值 acc / F1 | 最好 fed $\Delta$acc | 行置换 | iid | 随机投影$(x)$ |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| chameleon | 0.2299 | 0.4842 / 0.4676 | **+0.1383±0.0314** | 0.3004 / 0.2911 | +0.0113±0.0221 ($\tau$=5) | +0.0007 | −0.0020 | +0.0048 |
| squirrel | 0.2221 | 0.3018 / 0.2948 | **+0.1003±0.0079** | 0.2402 / 0.2320 | +0.0050±0.0080 ($\tau$=5) | +0.0026 | +0.0025 | +0.0026 |
| actor | 0.2167 | 0.3193 / 0.2123 | −0.0066±0.0117 | 0.2618 / 0.2001 | +0.0043±0.0113 ($\tau$=20) | +0.0019 | +0.0036 | +0.0028 |
| wisconsin | 0.1778 | 0.7516 / 0.5086 | −0.0157±0.0406 | 0.6275 / 0.2375 | +0.0105±0.0410 ($\tau$=5) | +0.0078 | +0.0013 | +0.0000 |
| texas | 0.0609 | 0.7261 / 0.4715 | +0.0144±0.0764 | 0.6486 / 0.1586 | −0.0054±0.0485 ($\tau$=0) | −0.0072 | −0.0000 | −0.0036 |
| cornell | 0.1227 | 0.7099 / 0.5025 | −0.0523±0.0612 | 0.5405 / 0.1854 | −0.0054±0.0322 ($\tau$=1) | −0.0144 | −0.0144 | −0.0180 |

裸分峰值取的是 8 个「$\tau$ × 方向」组合里最好的那个 acc（以及它对应的 F1）；地板+ 与三条对照都是与地板按 15 元配对的差。

**表 2 · 尺度档（chameleon，同一个统计量、同一个模型，只改两块输入的比例）**：

| 档 | rate 列 : $x$ 列的 std | 地板 acc | 最好 fed $\Delta$acc | 三条对照 $\Delta$acc 的最大值 |
| :-- | :-- | :-- | :-- | :-- |
| loud（rate 单位方差，$x$ 按行归一） | 1 : 0.0064 ⇒ **156×** | 0.4004 | **+0.0817±0.0247** | +0.0171 |
| quiet（rate 缩到 $x$ 的列 std） | 1 : 1 | 0.4004 | +0.0018±0.0268 | +0.0009 |
| fair（两块都单位方差） | 1 : 1 | **0.4842** | +0.0113±0.0221 | +0.0048 |

六条读数：

1. **判据通过**。12 个「图 $\times$ 指标」的裸分峰值**没有一个**越过各自地板（缺口从 actor 的 F1 $-0.012$ 到 cornell 的 F1 $-0.317$）；24 个 fed 读数（$\tau$ 4 档 $\times$ 6 张图）最大只有 $+0.0113$。
2. **但“没越过”不等于“不涨”，要靠同宽度对照才能分开**。每一张图上都有一条同宽度的正偏移可以拿：squirrel 三条对照 $+0.0025/+0.0026/+0.0026$，chameleon 的随机投影 $+0.0048$，actor 的 iid 噪声 $+0.0036$（比它自己那行 fed 的 $+0.0043$ 只差 $0.0007$）。⇒ **标签率的专属增量 $\le+0.007$，全部在噪声里**。没有第三条对照时，chameleon 那行 $+0.0113$（$t\approx2.0$）会被当成增益写进表。
3. **同一个 1 跳聚合，换成邻居的特征就值钱**：chameleon $+0.1383\pm0.0314$（$t\approx17$）、squirrel $+0.1003\pm0.0079$（$t\approx49$），而同两张图上标签率值 $\le+0.011$。（actor $-0.0066$、cornell $-0.0523$、wisconsin $-0.0157$ ⇒ 特征通路也不是处处有用，这与文献里 Actor 上 GNN 不敌 MLP 一致。）所以真正的分野不是「组级统计量 vs 其余」，而是 **聚合 $y$ 还是聚合 $x$**。
4. **尺度这一档能自己造出一个假正**（表 2）：loud 档下 fed 得 $+0.0817\pm0.0247$（$t\approx12.8$）、三条对照都 $\le+0.017$，看起来是一个很强的真涨；把 $x$ 也做成单位方差，同一行变成 $+0.0113\pm0.0221$。而且**地板自己对预处理更敏感**：$0.4004\to0.4842$（$+0.084$），与那个假涨同量级 —— 不做公平档，这两个 0.08 会互相冒充。边界：SOY/CORN 不在这个 confound 的射程里，实测那几块本来就同量级（$x$ 列 std 1.013（已 z-score）、one-hot 0.274、率 0.204 / 0.466、$\log n$ 1.04），而且最响的是 $x$ 自己 ⇒ **§9.11 的 40 个读数不需要重跑**。
5. **这条通路的病不是“train-only 屏蔽卡住了它”**。把邻居标签换成全集（含 test 自身标签，纯粹为了量上限）后，裸分只从 0.2263 到 0.2354（squirrel），其余五张图反而**下降**（chameleon 0.2623→0.2496、texas 0.3622→0.0324）。⇒ 它是信息量本身不够，不是被正确做法锁住。
6. **为什么不够，一行数字就能着地**：chameleon 的 test 行里 train 邻居数中位 **5**（均值 12.6，29 行为 0）、squirrel 中位 8、actor 中位 2、WebKB 三张中位 1——**在 5 类问题上用 1–5 个样本的直方图定后验**。而同一批邻居的 2325 维特征取均值是降噪。**同一批邻居，取 $y$ 得到噪声，取 $x$ 得到信号** —— 这是本节全部数字里最该留下的一句。

**⇒ 结论：判据保留，表述换掉。** §9.11 那句“先看裸分越不越过地板”作为开工前的否决器仍然成立（它在两类数据形态上都没误判），但它把两种不同的失效混成了一个读数：“地板已含此信息”与“此信息本身太稀”。完整流程是三步：（i）构造上判它是不是地板输入的函数；（ii）量裸分 vs 地板；（iii）用**同尺度、同宽度**的对照扣掉“加列”自己的偏移。§9.11 有（i）（ii），本节证明（iii）不可省。

**对 route A 的后果（否决不撤销，但理由换成一条更有用的）**：attr embedding 要拿的增益在邻居特征那一侧，靶子现在量出来了 —— **同一预处理下的基线是 chameleon 0.6225 / squirrel 0.4021 acc（= $x$ 加不可学的邻居特征均值）**，不是 MLP 的 0.4842 / 0.3018。而 SOY/CORN 上这条线等于地板（§9.7 池化增益 $\le0.001$）⇒ **那张图当不了 route A 的载体，③ 不是锦上添花而是前提**。下一步因此是确定的：把 OCA 管线跑在这 2–3 张图上，与地板 / 地板+ 对表。

> 本节否的是“**把真标签当特征喂进去**”这一条通路。标签传播 / C&S 那一族（用全图节点的预测做两阶段后处理）不在本表射程里 —— 本节不为它背书，也不否它。

### 9.13 外部基线表：对手从「地板+」换成可学聚合

**为什么这张表必须存在**（不是补数据，是换问题）。§9.12 把 route A 的靶子定成「地板+」= $x$ 加**不可学的**邻居特征均值，但那一行数字来自诊断脚本自己那套拟合配方（`hetero_probe._fit_eval`：128×3 的 MLP、Adam lr 3e-3、wd 1e-5、无 dropout）。仓库的正式训练路径（`train.py` → :func:`training.trainer.fit`：lr 1e-2、wd 5e-4、dropout 0.5）是另一套，而**同一个 chameleon、同一份 `row_normalize` 特征，MLP 地板在这两套下分别是 0.4842 与 0.3858**（后者是 scale n；宽度也换到文献档则 0.4401）。两套之间差的不仅是 lr/wd/dropout，还有 hidden 128×3 vs 16，所以不能把 0.098 全记在协议头上 —— 但 0.04–0.10 这个量级已经与本节要量的增益同阶。⇒「GCN 打不打得过一个加了邻居均值的 MLP」这种问题跨表问就没有答案，必须把几行塞进同一套协议、同一批折、同一批种子重测。

**载体**：`dataset/acad_bench.py`。四行走同一条 :func:`training.trainer.run_once`（= §8 主结果那条路），差别只在结构 YAML：`mlp` 完全不读 `edge_index`；`floor+` 把 $[\,x \mid A_{\mathrm{norm}}x\,]$ 拼进输入喂同一个 MLP（聚合式直接复用 `acad_probe._spmm`，不另写一遍，口径由 `test/test_dataset.py::test_acad_bench_pooling_is_neighbor_mean_no_selfloop` 用 4 节点手推图钉住：度不等的中心、孤立点给 0 向量而非 NaN、前 $F$ 列原样保留）；`gcn` / `gat` 是文献基线。配对单位是（官方折, 种子），$\Delta$ 是配对差，选点只看 val（:func:`_metric_key` 拒 `test_*`）。`gat` 那行 `add_self_loops=False` 写死（`modules/convs.py::GATBlock`，为了与 OCA 的中心槽位同口径），且它要求的「数据已是无向图」已被 `real.py:113-114` 的 `remove_self_loops` + `to_undirected` + `coalesce` 满足，并由 `test_real_dataset_optional` 断言钉住。

**协议被实测改了三次**，每次都是一个会让结论翻掉的坑：

1. **轮数 400 → 3000（patience 100 → 300）**。400 轮时 `gcn` 与 `floor+` 的 `best_epoch` 顶在上限（= 欠训练的稻草人）；1500 轮时 chameleon 上 gcn 1308、gat 1475 仍近上限；3000 轮得 619 / 564 / 1678 / 1448，四行无一顶限。表 1 全部是 3000 轮档。
2. **宽度 scale n → scale l**。`n` 的 `width_multiple` 0.25 把 hidden 64 压成 16（chameleon 上 37,845 参数），而文献宽度是 157,509；同 scale 下 **OCA 自己就有 169,076 参数（scale l 是 796,580）** ⇒ 不对齐宽度，「参数量差」会被当成「能力差」。表 1 留在 `n` 档（当时未知此坑），表 2 起全在 `l` 档。
3. **wd 从共享旋钮改成每格按 val 选**。见下。

**表 1 · 共享协议下的全表**（scale n；lr 1e-2、wd 5e-4、dropout 0.5、epochs 3000 / patience 300、`val_acc` 选点；官方折 0-4 × 种子 0-2 = 15 个配对单元；$\pm$ 是配对间**总体 std 不是标准误**；每格是 acc / macro-F1）：

| 图 | $h_{edge}$ | mlp | floor+（不可学聚合） | gcn | gat |
| :-- | :-- | :-- | :-- | :-- | :-- |
| chameleon | 0.2299 | 0.3858±0.0189 / 0.3667 | 0.5279±0.0365 / 0.5163 | 0.5822±0.0284 / 0.5738 | **0.6171±0.0484 / 0.6131** |
| squirrel | 0.2221 | 0.2598±0.0507 / 0.2148 | 0.3191±0.0508 / 0.2977 | **0.4190±0.0091 / 0.3892** | 0.1959±0.0089 / 0.0687 $\dagger$ |
| actor | 0.2167 | **0.3551±0.0085 / 0.2885** | 0.3469±0.0116 / 0.2512 | 0.2820±0.0187 / 0.1786 | 0.2755±0.0189 / 0.1623 |
| wisconsin | 0.1778 | **0.8013±0.0545 / 0.4864** | 0.7660±0.0575 / 0.4552 | 0.5373±0.0973 / 0.2327 | 0.6588±0.0598 / 0.3672 |
| texas | 0.0609 | 0.6973±0.0861 / 0.4189 | **0.7027±0.0592 / 0.4339** | 0.6180±0.0530 / 0.2915 | 0.6937±0.0652 / 0.4222 |
| cornell | 0.1227 | **0.6883±0.0492 / 0.4199** | 0.6198±0.0517 / 0.3396 | 0.4775±0.0598 / 0.2101 | 0.5459±0.0569 / 0.2811 |

参数量（列序同表）：chameleon 37,845 / 75,045 / 37,845 / 37,909；squirrel 34,069 / 67,493 / 34,069 / 34,133；actor 15,557 / 30,469 / 15,557 / 15,621；WebKB 三张 27,893 / 55,141 / 27,893 / 27,957。$\dagger$ = 该行不可训练，见下「GAT 之死」。

表 1 三条读数：

1. **只有 chameleon / squirrel 奖励聚合**。actor 与 WebKB 三张上「完全不读图」的 MLP 最强：actor 0.3551 vs gcn 0.2820 / gat 0.2755；wisconsin 0.8013 vs 0.5373；cornell 0.6883 vs 0.4775。这与文献一致（Actor、WebKB 是「MLP 打败 GNN」的招牌例子），也直接缩小了 route A 的射程：**能承载「结构增益」的只有 2 张图**。
2. **可学聚合确实压过不可学的**。以 `floor+` 为基准（不是以 MLP）：chameleon gcn $+0.054$、gat $+0.089$；squirrel gcn $+0.100$。⇒ §9.12 定的靶子「地板+ 0.6225 / 0.4021」只是**下限**，OCA 要赢的是可学的聚合。
3. **F1 与 acc 会分家，别只看 acc**。WebKB 三张只有 183-251 个节点、test 37-51 个，一行 acc 的粒度是 $1/50=0.02$；wisconsin 上 mlp→gcn 是 acc $-0.264$、F1 $-0.254$（掉进「只预测多数类」），而 texas 上 mlp 与 gat 的 acc 差 $0.004$、F1 差 $0.003$ —— 小图上的 $\pm0.05$ 级别差没有含义。折间散度也比大图大得多：cornell 逐折 mlp 0.7027/0.6306/0.7207/0.7387/0.6486，同一折内四行名次会整体翻转。

**表 2 · wd 敏感性**（scale l、官方折 0、种子 0-2；括号内是同一 run 的 **val** acc；$\Delta$ = wd 0 减 wd 5e-4）：

| 图 | 行 | wd 5e-4 | wd 0 | $\Delta$acc |
| :-- | :-- | :-- | :-- | :-- |
| chameleon | mlp | 0.4401 (0.4723) | 0.4635 (0.4966) | +0.023 |
| chameleon | floor+ | 0.5885 (0.6118) | 0.6031 (0.6411) | +0.015 |
| chameleon | gcn | 0.6513 (0.6831) | 0.6645 (0.6955) | +0.013 |
| chameleon | gat | 0.6754 (0.7380) | 0.6988 (0.7398) | +0.023 |
| squirrel | mlp | 0.3189 (0.3125) | 0.3157 (0.3151) | −0.003 |
| squirrel | floor+ | 0.3983 (0.4101) | 0.4371 (0.4547) | +0.039 |
| squirrel | gcn | 0.5130 (0.5200) | 0.5786 (0.5885) | +0.066 $\dagger$ |
| squirrel | **gat** | **0.1963 (0.2001)** | **0.6100 (0.6120)** | **$+0.414$** $\dagger$ |
| actor | mlp | 0.3638 (0.3794) | 0.3636 (0.3824) | −0.000 |
| actor | floor+ | 0.3414 (0.3732) | 0.3487 (0.3751) | +0.007 |
| actor | gcn | 0.2974 (0.3187) | 0.2917 (0.3166) | −0.006 |
| actor | gat | 0.2901 (0.3073) | 0.2943 (0.2991) | +0.004 |
| wisconsin | mlp | 0.7974 (0.8708) | 0.7647 (0.8500) | −0.033 |
| wisconsin | floor+ | 0.7712 (0.8708) | 0.6863 (0.8250) | −0.085 |
| wisconsin | gcn | 0.5686 (0.6417) | 0.5621 (0.6167) | −0.007 |
| wisconsin | gat | 0.6340 (0.7667) | 0.5948 (0.7333) | −0.039 |

$\dagger$ = 该行 `best_epoch` 顶在协议上限（squirrel/gat 是 2、squirrel/gcn 5e-4 档是 2763/3000），数字本身不可用。**wd 不是小旋钮**：同一格最大差 0.414，方向还随图翻转（chameleon/squirrel 偏向 wd 0，wisconsin 四行全偏向 wd 5e-4）。

**squirrel 上 GAT 之死：是协议杀的，不是模型不行**（下面每一数字出自一个一次性 trace 脚本，未入库；数字全录于此）：

* 现象：loss 从 1.6118 落到 $\ln 5=1.6094$ 附近并钉死，train acc 恒 0.2059（= 某一类在 train 里的占比）；每折的 test acc 就等于那一个常数类在该折 test 集里的占比（squirrel fold0 = 0.1931 = 类 3 的占比）⇒ 模型退化成常数预测。表 1 那行 `best_epoch`=6 的意思是：第 6 轮之后它再也没胜过自己的起点。
* **唯一的开关是 wd**。lr $\in\{0.01,0.005,0.003,0.001\}$ 加 grad-clip $\in\{0,1\}$ 给出逐位相同的 0.1931 / F1 0.0647；开自环（`add_self_loops=True`）照样塌；去掉 conv 后的 ELU 照样塌；把 wd 置 0，同图同折同种子从 0.1963 变 0.6100（表 2）。
* 机制（逐张量量到了）：init 时 GAT 的 4 个打分张量 `att_src/att_dst` 满足 $\lvert g\rvert/(\mathrm{wd}\lVert w\rVert)=5\text{e-}4\sim1\text{e-}3$（**基本接不到梯度**），两个 `conv.lin.weight` 是 0.12 / 0.03，而末端 head 是 23–470。Adam 的步长是 $\mathrm{lr}\cdot\hat g/\sqrt{\hat v}$，它归一化的是**含 L2 项的合成梯度**，所以在比值 $<1$ 的张量上「每轮乘性缩权重」与真实梯度多小无关。实测 40 轮内 $\lVert w\rVert$ 15.7 → 2.55、$\mathrm{std}(\mathrm{logits})$ 0.080 → 0.028；wd 置 0 后 $\lVert w\rVert$ 反向涨到 52.0、loss 降到 1.3470、train acc 0.4716。
* 为什么 `gcn` 在同一档 wd 下没死，**这一条我没解释干净**。已测并排除三种说法：（i）注意力 softmax 饱和 —— 没有，每边 $-\alpha\log\alpha$ 均值 0.040（纯均匀时 0.058，$\bar d=76$），且 40 轮里 $\overline{\max_j\alpha_{ij}}$ 恒为 $1/\bar d=0.013$；（ii）GAT 信号天生小 —— 也不是，层 1 输出绝对均值 GAT 1.02e-02 **大于** GCN 4.76e-03（GCN 的 ReLU 还杀掉 53% 激活），init 总梯度范数 3.8e-2 vs 4.9e-2 同量级，GCN 自己的 `conv.lin.weight` 比值 0.09/0.03 与 GAT 几乎相同 —— 两边都是 wd 主导；（iii）容量 —— 两行参数量只差 256。留下的事实只有：GCN 在 ep20-40 自己爬出来（$\lVert g\rVert$ 0.031 → 0.148、loss 1.604 → 1.519、$\lVert w\rVert$ 回到 10.0），GAT 同区间 $\lVert g\rVert$ 反而缩（ep10 的 1.12e-2 → ep40 的 9.8e-3）、loss 一动不动。差别大概率在打分通路：它一直接不到梯度，于是 $W$ 与分数同时缩，聚合退成固定均匀平均后能学的只剩 `lin` 与 head，而它们仍在 wd 的收缩侧。
* ⇒ 处理：不再追这条线，改协议（下段）。读表人需要知道的只有两点：共享 wd 下 squirrel/gat 那一格不可用；按 val 选 wd 之后它是本表最强的一行。

**⇒ 正式协议据此定**：共享 `dropout 0.5`、`epochs 3000`、`patience 300`、`val_acc` 选点、**scale l**，而 $(lr, wd)$ **每个（图, 行）单独按 val 选**，网格 $lr\in\{0.01,0.005\}\times wd\in\{5\text{e-}4,0\}$ 共 4 档，选参只用官方 10 折里**不报出来的 5、6 折**（`--tune --tune-folds 5,6 --tune-seeds 0`），所以折 0-4 的 test 从头到尾没被读过一次。按 val 选 wd 靠不靠谱，表 2 内部就能自查：16 格里 val 与 test **同向 12 格**，4 格反向但 $|\Delta_{test}|\le0.006$；而三格大分歧（squirrel 的 gat $+0.414$、gcn $+0.066$、floor+ $+0.039$）**val 全部指对了方向**。

**表 3 · 正式协议下的基线全表**（`python -m dataset.acad_bench --tune --tune-folds 5,6 --tune-seeds 0 --datasets chameleon,actor,wisconsin,texas,cornell,squirrel --models mlp,floor+,gcn,gat --folds 0,1,2,3,4 --seeds 0,1,2`；每格 acc / macro-F1，$\pm$ 是配对（官方折, 种子）间**总体 std 不是标准误**，15 个单元；**粗体** = 该图最强行）：

| 图 | $h_{edge}$ | mlp | floor+（不可学聚合） | gcn | gat |
| :-- | :-- | :-- | :-- | :-- | :-- |
| chameleon | 0.2299 | 0.4756±0.0231 / 0.4647 | 0.6098±0.0256 / 0.6078 | 0.6721±0.0168 / 0.6721 | **0.7218±0.0210 / 0.7219** |
| squirrel | 0.2221 | 0.3188±0.0138 / 0.3183±0.0138 | 0.4264±0.0094 / 0.4243±0.0096 | 0.5869±0.0117 / 0.5848±0.0118 | **0.6191±0.0135 / 0.6157±0.0132** |
| actor | 0.2167 | **0.3631±0.0058 / 0.3233** | 0.3500±0.0053 / 0.2963 | 0.3004±0.0086 / 0.2018 | 0.2904±0.0114 / 0.2131 |
| wisconsin | 0.1778 | **0.8392±0.0522 / 0.6446** | 0.7974±0.0333 / 0.5956 | 0.5137±0.0734 / 0.2569 | 0.6850±0.0753 / 0.4351 |
| texas | 0.0609 | 0.7784±0.0454 / 0.6091 | **0.7838±0.0632 / 0.6612** | 0.5964±0.0571 / 0.3179 | 0.6793±0.0654 / 0.4459 |
| cornell | 0.1227 | 0.7279±0.0435 / 0.5350 | **0.7387±0.0652 / 0.5655** | 0.4865±0.0616 / 0.2673 | 0.5514±0.0574 / 0.3438 |

每格选中的是哪个 $(lr, wd)$（括号内是该档在 fold 5/6 上的 val，即选参依据）：

| 图 | mlp | floor+ | gcn | gat |
| :-- | :-- | :-- | :-- | :-- |
| chameleon | 0.01/0 (0.4849) | 0.01/0 (0.6351) | 0.01/0 (0.6927) | 0.01/0 (0.7428) |
| squirrel | 0.01/5e-4 (0.3311) | 0.01/0 (0.4435) | 0.01/0 (0.5862) | 0.005/0 (0.6121) |
| actor | 0.01/5e-4 (0.3925) | 0.01/5e-4 (0.3694) | 0.005/0 (0.3164) | 0.005/5e-4 (0.2989) |
| wisconsin | 0.01/5e-4 (0.8687) | 0.01/5e-4 (0.8500) | 0.01/5e-4 (0.6188) | 0.01/5e-4 (0.7563) |
| texas | 0.01/5e-4 (0.9068) | 0.01/5e-4 (0.8559) | 0.01/5e-4 (0.6610) | 0.01/5e-4 (0.7712) |
| cornell | 0.01/0 (0.7881) | 0.01/5e-4 (0.7288) | 0.01/5e-4 (0.5678) | 0.01/5e-4 (0.5932) |

**表 3b · 同一批格的训练长度与成本**（每格 `best_epoch` / s·run⁻¹，15 单元均值；一次 run = 一个（官方折, 种子），早停生效）：

| 图 | mlp | floor+ | gcn | gat |
| :-- | :-- | :-- | :-- | :-- |
| chameleon | 74.5 / 2.5 | 329.1 / 4.5 | 401.2 / 7.9 | 561.5 / 11.1 |
| squirrel | 291.7 / 3.5 | 223.7 / 3.6 | **1492.9 / 45.4** | **1384.1 / 91.5** |
| actor | 50.8 / 2.3 | 78.1 / 2.8 | 57.8 / 4.2 | 100.2 / 5.1 |
| wisconsin | 376.5 / 3.3 | 396.8 / 3.4 | 161.7 / 3.5 | 184.4 / 3.9 |
| texas | 437.1 / 3.8 | 364.8 / 3.6 | 179.9 / 4.0 | 182.9 / 4.2 |
| cornell | 407.1 / 3.6 | 458.0 / 4.0 | 146.9 / 3.8 | 163.3 / 4.2 |

参数量（列序同表，四行一组）：chameleon 157,509 / 306,309 / 157,509 / 157,765；squirrel 142,405 / 276,101 / 142,405 / 142,661；actor 68,357 / 128,005 / 68,357 / 68,613；WebKB 三张 117,701 / 226,693 / 117,701 / 117,957。Δ 的基准行取 `mlp`，但本次没开 `--archive`（该选项是在这一轮跑完之后才加的）⇒ **逐配对单元的存档为空**：表 3 只有日志里的均值与逐折拆分可用，将来补一行必须把基线行一起跑（见下）。

表 3 五条读数：

1. **两张异配维基图的最强行都是 `gat`，而它正是表 1 里那个「不可训练」的行**：chameleon 0.7218、squirrel 0.6191（表 1 给的是 0.6171 与 0.1959$\dagger$）。⇒ §9.12 定的靶子「地板+ 0.6225 / 0.4021」再抬一次：**route A 的正面对手是 chameleon 0.7218 / squirrel 0.6191**，`floor+` 那两行（0.6098 / 0.4264）现在只是下界。
2. **宽度这一档值 0.06–0.16**：表 1（scale n）→ 表 3（scale $l$）同图同行 chameleon gat 0.6171→0.7218、squirrel gcn 0.4190→0.5869、squirrel floor+ 0.3191→0.4264。⇒ 任何拿 scale n 基线当「够得着」的论据作废。
3. **另四张图仍是 MLP 最强或与之打平**（actor 0.3631、wisconsin 0.8392；texas 0.7784 vs floor+ 0.7838、cornell 0.7279 vs 0.7387）。后两图那次「floor+ 反超」的差是 test 37 个节点里的 0.2–0.4 个，而配对 std 有 0.045–0.065 ⇒ **不可断言**；而表 1 上 cornell 是 mlp 最强，换协议就翻过来 —— 这正是小图没有分辨率的演示。
4. **成本极不对称**：squirrel 的 gcn/gat 是 45.4 / 91.5 s/run、best_ep 1492.9 / 1384.1，其余五张同一行只有 2–11 s/run、best_ep $\le$ 562。反算单步：squirrel/gcn $45.4/1493\approx0.030$ s/step，与 §9.6 补测独立测到的 0.02 对得上 ⇒ 两处成本口径互校。
5. **「没被 3000 轮截断」这句还没有全证**：表 3b 里最长的是 squirrel/gcn 的 1492.9，若有单元跑满了 3000 会把均值拉得更高，所以大概率没截，但本轮日志只记了 `best_ep`。已给 `report_one` 补一列 `stop_ep`（= `RunResult.epochs_ran`，早停与顶限两种情形从此可分），从 OCA 行起每一格直接给停止轮。

**⇒ 对 route A 的后果（判定线在看 OCA 行之前写下，免得事后挑口径）**：

* **对手**：同格按 val 选过参的 `gat`/`gcn` 行，不是 `floor+`、更不是 `mlp`；射程只有 chameleon 与 squirrel（读数 1、3），另四张上 OCA 只能当「不掉分」的一致性检查。
* **断言形式**：配对差 $\Delta$acc（vs `gat`），同协议同折同种子。分辨率从已有数据估：squirrel 四行配对 std 0.0094–0.0138、chameleon 0.0168–0.0256 ⇒ **15 个单元上能被断言的最小正效应约 $+0.02$**（$\approx2\sigma_{\text{配对}}$）。
* **归因必须先过这一关**：`oca` 与 `gat` 的差别在「竞争/抑制」而不在「注意力容量」，而这两张图上线性注意力的 `gat` 已经把 `gcn` 甩开 0.03–0.04 —— 说明**可学的边权分配本身就值钱**。拆法仓库里已有：`cfg/models/oca_gat.yaml` 与 `oca` 同一骨架（4 尺度 + PANet）但算子按 §3 七条件退成 GAT，当时写的是 `oca` − `oca_gat` = 竞争算子的净贡献。**这句已在 §9.13 更正**：七条件连残差与 LayerNorm 一起关，所以两行差里混着「带融合读出的输出路径」与「竞争场」两件事。单开关对照另建 `cfg/models/oca_nocomp.yaml`（与 `oca.yaml` 只差 `use_competition: false`，撤掉整个竞争场），分解式改成 `oca` − `oca_nocomp` = 竞争场净贡献、`oca_nocomp` − `gat` = 其余全部。三行均已加进 `acad_bench.MODELS`（cornell 冒烟：`oca` 635,482 参数 / `oca_gat` 166,117 —— 后者与基线同量级）。
* **成本账要并排报**：OCA scale $l$ 的参数量是基线的 5.2×、单步 40–60×、感受野 $4\times(T{+}1)$ 跳 vs 基线 2 跳（§9.6 补测）。⇒ **打平不是结果，落在 `gat` 之下也不是结果**，只有跨过上面那条 $+0.02$ 线并把归因拆干净才算。

**OCA 行的开工顺序（成本决的）**：先 chameleon（$\le$ 1.6 GiB、0.33 s/step，全表含选参约 2–4 h），后 squirrel（必须 `--oca detach_iterations=true`，1.20 s/step、best_ep 按基线推在 1400–1800 轮 ⇒ 一次 run 半小时以上，单行 15 单元按 5–10 h 计）。两行都要带自己的 $(lr,wd)$ 选参档（同表重跑 `--tune`）：**不能沿用共享 wd 5e-4** —— 那正是表 2 杀掉 `gat` 的那一档，而 OCA 的 $\lambda$ 门控 bias 初值是 $-2$（§2.6），属于「梯度本来就小」的那类张量。

**OCA 行的第一档实测**（chameleon、fold0 seed0 单 run，只跑 `oca` 与 `gcn` 两行，`--archive=` 不存盘）。目的：在把几小时预算花出去之前，先量三件当时不知道的事 —— 单 run 成本、$(lr,wd)$ 敏感性、早停形态。取数：`--datasets chameleon --models oca,gcn --folds 0 --seeds 0`，两档 wd 各一次。

| wd | 行 | test acc | f1 | val | best_ep | stop_ep | s·run⁻¹ |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 5e-4 | `oca` | 0.2456 | 0.1840 | 0.2798 | 32 | 332 | 61.7 |
| 5e-4 | `gcn` | 0.6645 | 0.6671 | 0.6818 | 1242 | 1542 | 15.8 |
| 0 | `oca` | **0.5417** | 0.5412 | 0.5857 | 189 | 489 | 88.4 |
| 0 | `gcn` | 0.6754 | 0.6755 | 0.6927 | 508 | 808 | 7.5 |

四条读数：

1. **§10 第 11 行那个坑对 OCA 成立，而且更狠**：共享 wd 5e-4 把 `oca` 打成近似退化预测 —— test 0.2456 只比该折 test 的多数类率 $102/456 = 0.2237$ 高 0.022，macro-F1 0.1840，且 $best\_ep = 32$ 之后再没胜过起点。⇒ **OCA 行必须自带选参档**（已按协议交给 `--tune`，不能跟着基线那一档走）。
2. **wd 0 是可用档，但差距仍然很大**：同折同种子 `oca` 0.5417 vs `gcn` 0.6754（$-0.134$）、vs 表 3 的 `gat` 0.7218（$-0.180$）。val 侧同向（0.5857 vs 0.6927），所以这不是选点问题，是真差距。单 run 不足定案（表 3 折间 std 0.017–0.026），已把 chameleon 全表连同 `oca_gat` 挂后台。**当时写下的判据（二选一）已经被选参档回答，读的是后者**：`oca_gat` 在 lr 0.005 / wd 0 上给选参折 val **0.7010**，而同批折上 `oca` 自己最优只有 **0.5665**（lr 0.005 / wd 0）、`gat` **0.7373**（lr 0.01 / wd 0）。⇒ 骨架本身训得动，被拖累的是带竞争场的那一支。**但这个 0.135 不能全记到竞争项头上** —— `oca_gat` 与 `oca` 之间还隔着 out_mode（无残差/无 LayerNorm/无 `out_mlp`）与 φ/α/τ，要干净量得等 `oca_nocomp`。另一条自我更正：`oca_gat` 在 lr 0.01 只有 val 0.2627，我一度据此判「4 层无残差的栈自己训不动」，**实为深行对 lr 敏感**，那句话已从 `oca_nocomp.yaml` 注释里改掉。
3. **§9.6 补测那列 s/step 至少虚高 1.8×**：本轮卡上无争用，`oca` 0.18 s/epoch（88.4 s ÷ 489 轮，与 wd 5e-4 档的 61.7 ÷ 332 = 0.186 互校）、`gcn` 0.0093 s/epoch ⇒ 单 epoch 成本比 19×；而 §9.6 补测同一格给的是 0.33 / 0.02 = 16.5×（比值对得上，绝对值差 1.8×，那次正与表 3 抢 GPU）。⇒ 据此外推的「squirrel 一行 5-10 h」只能当上限读。
4. **`stop_ep` 这一列当场证明有用**：四格全是 stop_ep = best_ep + 300（patience），即每一格都是早停而非顶限 ⇒ 表 3 读数 5 那句「大概率没被 3000 轮截断」在能验的格子里全部为真。

**同一张图上的门控实测**（chameleon fold0 seed0、CPU、120 轮、lr 0.01 wd 0、`oca.yaml` 默认档 $T{=}2$；该跑给 val 0.5213 / test 0.5066 @ best_ep 109，与上面探针同一配置、只是轮数截在 120）。读数口径：`training.diagnostics.collect_gates`，$L{=}4$ 层 $\times$ $N=2{,}277$ 节点 $=9{,}108$ 个按层均值后的逐节点值。

| 量 | init：mean±std / p05 / p50 / p95 | 训 120 轮后：mean±std / p05 / p50 / p95 |
| :-- | :-- | :-- |
| $\lambda$ | 0.1533±0.0258 / 0.1189 / 0.1515 / 0.1940 | 0.0979±0.1515 / 0.0033 / 0.0726 / 0.2395 |
| $\alpha$ | 0.5031±0.0770 / 0.4134 / 0.4956 / 0.6314 | 0.3200±0.2900 / 0.0027 / 0.3263 / 0.9202 |
| $\tau$ | 2.985±14.35 / 0.6973 / 1.3180 / 2.2382 | 1284±11,045 / 0.0013 / 1.3138 / 10.7386 |
| 场 $\overline{\lvert s\rvert}$（不含中心槽） | 0.0123±0.0137 / 0.0006 / 0.0088 / 0.0385 | 39.9±46.6 / 0.2276 / 20.49 / 140.0 |

门控输入自己的尺度：row-normalize 后的 2,325 词袋，每行平均 12.8 个非零元（p05 0 / p95 43）、非零元 $\lvert v\rvert$ 均值 0.0701、行范数均值 0.385 —— 比 hetero 那 12 列气象（std $\approx0.53$）小一个量级，但不为零。

三条读数：

1. **门控确实在个性化，不是贴在先验上**：$\lambda$ 的跨节点 std 0.026 → 0.152（$6\times$）、$\alpha$ 0.077 → 0.290（$3.8\times$），分位数张到 $[0.003,0.24]$ 与 $[0.003,0.92]$。⇒ §10 第 6 行那句「自适应故事可能为空」在 chameleon 上被否掉一半：门控在用，只是方向不对。
2. **方向是往下**：$\lambda$ 均值 0.153 → 0.098、$\alpha$ 0.503 → 0.320 —— 优化器在**削弱**横向抑制与中心场。§2.6 的 $\lambda_{\text{init}}=-2$ 本来就把起点压在 0.119 附近（表里 init 的 p05 0.1189 就是它），而梯度又朝下 ⇒ 「竞争要靠训练挣来」在这张图上读反了：**竞争是靠训练省掉的**。
3. **$\tau$ 在 hub 上仍炸**：中位数几乎没动（1.318 → 1.314），但均值 1,284、std 11,045、p95 10.7 —— §9.8(d) 那个「跨 5 个数量级」的形态在学术图上复现，且训练后更严重。（场的绝对值 0.0123 → 39.9 不能归给竞争：这一档 wd 0，激活整体在长；拿 $\lambda\times$ 场 $\approx0.098\times39.9\approx3.9$ 看，竞争通道在数值上不是 0，上边那句「被削弱」是从均值方向说的。）



**已知漏点核查：重复节点**。geom-gcn 那两张维基图以「同一篇网页被切成若干特征逐位相同的节点」出名（Platonov et al. 2023 因此发布了去重版），这类孪生若跨 train/test 就是白送的答案。本加载器（`WikipediaNetwork(geom_gcn_preprocess=True)` + 官方 10 折）实测三折均值，「全零特征」单列剔除空词袋造成的假重复：

| 图 | $N$ | 唯一特征行 | 重复行占比 | 组内标签全同的重复 | 孪生跨 train/test 的 test 节点 | 其中标签也相同 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| chameleon | 2,277 | 1,950 | 6.24% | 3.25% | 0.83% | **0.42%** |
| squirrel | 5,201 | 4,980 | 1.92% | 0.67% | 0.22% | **0.04%** |
| actor | 7,600 | 6,014 | 26.00% | 1.61% | 4.40% | **0.20%** |
| wisconsin | 251 | 248 | 1.99% | 1.99% | 0.13% | 0.13% |
| texas / cornell | 183 | 183 | 0.00% | 0.00% | 0.00% | 0.00% |

⇒ 真正有害的那一格（test 节点在 train 里有特征逐位相同**且标签相同**的孪生）六张图全部 $\le0.42\%$；actor 那 4.40% 的跨集孪生里只有 0.20 个百分点标签相同（其余 95% 标签不同 —— 它的特征是 932 维稀疏「单词」指示位，撞行容易、撞标签不容易）。所以本表绝对值偏高**不能**用重复节点解释，也不能拿它当去掉文献质疑的挡箭牌 —— 见下条。

**绝对值不可与文献表并列**。本表协议 = 3000 轮 / patience 300 + `row_normalize` + 全图无向化去自环 + GAT 不开自环，与常见复现档（1000 轮 / patience 100 / 带自环 / 或去重版数据）不同；表 2 里 squirrel 的 GAT 0.6100 就明显高于多数复现表给的那批 0.45-0.50。**本节只支持行间断言（同协议、同折、同种子、同选点规则），不支持与任何外部表格竖比。** 只有一个反向检查：表 1 的 `mlp` 行落在文献 MLP 行的常见区间内（凭记忆给的宽区间：chameleon 0.36-0.46、actor 0.34-0.37、Wisconsin 0.74-0.85 —— **正式引用前必须回原文核对**），说明换图后的装载与划分没跑偏。

------

## 10. 核对代码与数据时确认的“不干净”清单（供你逐条裁决）

| # | 事实 | 影响 | 我的建议 |
| :-- | :-- | :-- | :-- |
| 1 | ~~迭代初值~~ **已按你裁决改掉**：$s(0)$ 现为 $c=\phi+(1-\alpha_i)b$，三条路径（sparse $T>0$ / sparse $T=0$ / `s_out`）一起改，代码口径 == 设计稿口径 | §8 已全量重跑：能量对拍 1.94e-16 → 1.99e-16、收敛 err $<2.22\text{e-}16$ → $<1.11\text{e-}16$、$dL/dw_\lambda$ 1.784e-04 → 1.462e-04、消融 `full` 0.9821 → 0.9732、`no_competition` 1.0000 → 0.9554；GAT 退化与 $\alpha\equiv0$ 各行逐位不变（§2.2、§8.4） | 无需再动。唯一遗留：§8.4 的 `--ablate` 表里两条旧读数（`full` 1.0 / `gat_degenerate` 0.52）当时未记命令行与设备，现已不可复现，已用重跑值替换并在文档里注明 |
| 2 | 温度统计量分母多乘 $\sqrt{\deg_i}$（§2.7） | 名字不准确（它不是 z-score），但**实测对行为几乎无影响**（§9.8） | **已诊断完：建议不改数值、只改措辞**。锐度差不超过 $2\times10^{-4}$，而训 200 步后 $w_\tau$ 的 stat 列自己涨到 6.9× —— 这个因子可以被学习补偿；去掉它要重跑整张 §8.4。**已定（你已裁决）：不改数值、只改措辞** —— §2.7 的式子、§5 表的 `center_stat` 行、`modules/oca.py:75` 与 `_temperature` 的注释已按这个口径同步 |
| 3 | `dense` 的 `relu` 核分支把每侧 $q$ 除 $\sqrt{d_h}$ ⇒ $\mathrm{ReLU}(\langle q_a,q_b\rangle)/d_h$，v1 手写的 $\sqrt d$ | 只是与 $d$ 相关的常数差，不改变"无界 ⇒ 发散"的结论 | 在消融表里把它叫「无归一化 ReLU 核」，不要写成 v1 原式 |
| 4 | `--resume` 缺失（§7） | 长训练断了只能从头跑；hetero 数据上成本会明显放大 | 接权重 + epoch + 三套 RNG，并补「存→读→接着跑」用例 |
| 5 | `single_scale` 同时动了融合与深度（§6 末） | 那行读数不能支撑"多尺度有增益" | 补一份四行 backbone、无 `Merge` 的结构表 |
| 6 | 门控 $\lambda,\alpha$ 吃原始 $h_i$，尺度漂（P1-3） | 自适应故事可能为空 | 先画 §8.5 那张散点图，再决定是否给三个门控各配独立投影 |
| 7 | **SOY/CORN 这张图的结构信息已被三轮实测封住**：（i）无图模型复现生产方 GNN 基线、池化增益 $\le$ 0.001 AUC（§9.7）；（ii）route B 的三种造边（kNN / 属性组合格元 / 对侧作物同格元）在按种子配对的 16 行候选里 corn **0 行**为正、soy 最大 **+0.0006±0.0007**（§9.9 第三版；v2 那句“最大 +0.0013”是地板行 init 未配对造成的）；（iii）最强形式（格元的组级标签率 = 目标编码）仍是最差行：corn −0.0163±0.0080 / soy −0.0033±0.0018（v1 的 −0.052/−0.040 是 `year±1` 越界使 12 列在 100% 的 test 行上塌成 0；v2 的 −0.0296/−0.0053 多扣了一份偏高的地板 init —— 两处都已修） | 拿它当“图学习有用”的证据会被一个 MLP 驳掉；而 route A 的 attr embedding 能学的正是同一种组级先验，所以先验上它更可能重演（iii）而不是涨点 —— （iii）的机制曾定在拟合侧（§9.10），但 §9.11 已实测把它改判为天花板（拟合侧修得好危害、修不出增益） | route B 已按你裁决做完并划掉，剩下三条（**已裁决：② + ③ 同时做**）：② 改卖“分布漂移下的稳健性”—— 现在的实测恰好把卖点递上来了：拿 §9.9 的标签率行当反例，量一个“组级统计量随年份漂移”的指标（逐格元 train/test 标签率的 Spearman + train/test gap），再问 OCA 的竞争/门控能不能抑制它。不需要新数据，且是这张图独有的；③ 算法主张回到 Chameleon/Squirrel/Actor/Cora（§8 的全部正确性证据已在那些图上），SOY/CORN 降为应用节；④ 重找真有县级/地块地理单元的数据（唯一能让 route B 过线的途径，但要重走数据获取）。**建议 ② + ③ 同时做**：它们不互斥，且 ② 的唯一风险（“漂移下 GNN 反而更差”）本身就是一个可写的负面结果。已按此开工：② 的前置量（组级统计量的年份漂移）不需要 route A 就能测，先测它（已完，§9.10）。**② 的实测把“漂移”这个卖点否掉了**：窗口表在 test 年漂 −0.0003 ~ +0.023、格元级 Spearman 0.87–0.96、反序对照贴 0.5 —— 组级先验是这张图上**最可迁移**的信息。它换成的事实同样独有、但没那么戏剧：**一张裸查表在 test 上有 0.7991，把这列喂进 MLP 后得 0.7996±0.0058 —— 零边际贡献，还要在它上面倒贴 0.0163±0.0080**（§9.9 读数 4、§9.10 末条；v2 那句“模型输给了自己的一个输入 0.7925 $<$ 0.7991”是未配对读数，已撤）。因此 route A 的形态据此定：attr embedding 显式携带**格元样本量**（计数收缩 / 可信度门控），而不是去“抑制一份不迁移的先验”；v2 追加的“并学属性交叉项”已随 §9.9 读数 3 一起撤，目前没有证据。**而 §9.11 把这句再推了一步，结论掉头 180°**：按计数收缩 + 单给一列 $\log n$ 确实能把 −0.019 的倒贴抬到 −0.001 … −0.004（已越过本节的无信息带），但 40 个候选读数**没有一行显著为正**；而 40 个裸分读数也没一个越过地板（corn 0.8068 $<$ 0.8159、soy 0.8808 $<$ 0.8911）。⇒ 落点从“拟合侧”改判为“**天花板在拟合之前**”：组级统计量是 $x+onehot$ 的函数，学它超不过地板。开工前先量“要学的那个统计量的裸分 vs 地板”，越不过就不要为它改模型。**§9.12 已把这条判据拿到 6 张标准异配图重测：判据通过（12 个「图 $\times$ 指标」的裸分峰值无一越地板；24 个 fed 读数最大 $+0.0113$，且被同尺度同宽度对照吃掉），但表述必须补第三步，且输入尺度本身可以单独决定符号**。route A 的靶子据此从 MLP 地板改成「地板+」：$x$ 加不可学的邻居特征均值才是基线（chameleon 0.6225 / squirrel 0.4021 acc），而不是 0.4842 / 0.3018。**但这个靶子已被 §9.13 再推两次**：表 1 里可学的聚合压过不可学的；表 3（正式协议：scale $l$ + 每格按 val 选 $(lr,wd)$）把对手又抬高一档 —— **chameleon 0.7218 / squirrel 0.6191，两图最强行都是 `gat`**，而 `gat` 在表 1 的共享 wd 下本来是一行不可训练的 0.1959（§9.13 表 3 读数 1）。⇒ 地板+ 只是下限，route A 的正面对手是选过参的 `gat`/`gcn`，而且只能在 chameleon / squirrel 上比（另四张图上 MLP 最强，见 §9.13 读数 1 与 3）；能被 15 个单元断言的最小正效应约 $+0.02$，且必须带归因对照行才能把「竞争项有用」与「又一个可学的边权」分开 —— 先写的是 `oca_gat`（同骨架、算子退成 GAT），**已证不干净**（七条件连残差与 LayerNorm 一起关），干净的那行是 `oca_nocomp`（与 `oca` 只差 `use_competition`，即撤掉整个竞争场） |
| 8 | **全图全批装不下**：hetero 图上单层 $T=2$ 已 +3.5 GiB，4 层需 ~14 GiB 而卡只有 8 GiB（§9.6）；学术图这一侧已量清（§9.6 补测）：六张里**只有 squirrel 装不下**（$l$、$T{=}2$ 峰值 +8,983 MiB > 卡的 8,151 MiB，溢到共享内存后 7.73 s/step），而 `detach_iterations=True` 把它压回 +7,084 MiB / 1.20 s/step，其余五张 $\le$ 1.6 GiB | “不做 mini-batch”这条旧决定在 hetero 图上失效；学术图上则**显存已不是障碍，剩下的障碍是算力**：squirrel 的 OCA 行按基线的 best_ep（1384-1493）外推是 5-10 h/行，而 `gcn` 只要 45 s/run | 先试 `detach_iterations=True`（零新代码）+ 逐关系分块；真不行再上 Loader，并同步记一笔“`gates.npz` 对齐在 Loader 下怎么保证”。**需你裁决的是范围不是显存**：OCA 行先只在 chameleon 上跑全表（含 `oca_gat` / `oca_nocomp` 两条归因对照，约 2-4 h），还是六张全跑（按外推 15-25 h），还是 squirrel 降 $T$ / 降折数（降了就与表 3 不同协议，只能另起一行） |
| 9 | $x$ 第 12 列 `quantity` 重尾（corn max $\lvert z\rvert$ 37.2、soy 67.0）且门控吃原始 $h_i$（第 6 行） | $\lambda_i/\alpha_i/\tau_i$ 会被单列主导，变成近似阈值函数 | 入库前先 winsorize / rank-gauss 这列，或给三个门控各自先 LayerNorm。实测（`--what floors` 最后两行）：全体 clamp 到 $\pm4\sigma$ 使 corn T2 的 $x$ 从 0.4004/0.3851 升到 **0.4139/0.4099**（soy 不动），但 $x$+onehot 几乎不动（0.5829/0.5871 → 0.5816/0.5857）⇒ **重尾只伤弱基线**。副作用：在入库处 clamp 同时也把门控的输入从 37σ/67σ 压到 4σ，风险随之下降；但门控到底有没有在用这一列，仍要靠 §8.5 那张相关性散点看，不能靠预处理断言 |
| 10 | **③ 的学术图管线从来没跑通过，而错误信息把它报成了网络问题**：`dataset/real.py` 旧表把 chameleon / squirrel / actor 三条主名指向 `WebKB`，但 PyG 2.6.x 的 `WebKB` 只收 cornell / texas / wisconsin（`webkb.py:74` 的 assert），而 `Actor.__init__` 根本没有 `name` 形参（`actor.py:50-56`）⇒ 两者都在**发请求之前**就抛，又被 `except Exception` 兜成「加载失败（无网时请用 `--dataset synth`）」 | 之前任何「学术图跑不动」的记录都是假故障，真实原因被措辞遮住；③ 至今没有一个来自真实学术图的读数（§9.12 是第一次） | **已修**：映射改 `WikipediaNetwork` / `Actor`，`name` 按形参条件传（`REAL_DATASETS` 元素改 `Optional[str]`），删掉本版本不存在的 `crocodile`；六张图全部可加载并已入 §9.12 表 1。另记两条环境事实：本机 `raw.githubusercontent.com:443` 通而 `github.com:443` 不通 ⇒ Planetoid 三张（Cora / CiteSeer / PubMed）下不下来，`data/Cora/raw` 也只有 5 个文件（缺 `ind.cora.graph` 与 `test.index`，`Citeseer`/`Pubmed` 目录为空）⇒ **同质那一侧目前没有真实图可测**。两条出路已按你裁决定为后者：**论文数据集清单改成 geom-gcn 六张**（chameleon / squirrel / actor / texas / cornell / wisconsin），同质对照用 `synth:h=…` 只填判据曲线形状、按既定口径不当论文卖点；Cora 那一侧不再阻塞主线（若以后拿到代理再补，补上就是加一行而非改结论）。基线表见 §9.13 |
| 11 | **共享的 weight decay 会静默杀掉一行基线**：`cfg/train/default.yaml` 的 wd 5e-4 使 squirrel 上的 GAT 退化成常数预测（loss 钉在 $\ln 5$、test 0.1963、`best_epoch` 6），而同一档 wd 在 wisconsin 上是**四行全更好**的一档（表 2） | 若把「一行崩了」当成该模型在该图上不行，就会把协议缺陷写成 GAT 的能力上限；反过来单独为 GAT 改 wd 又会被读成调参偏袒 | **已改掉**：$(lr, wd)$ 每格按 **val** 选，选参只用官方 10 折里不报出来的 5/6 折（§9.13）。附带两条硬要求：表里每行必须带 `best_epoch`（顶限 = 欠训练信号，本轮就是靠它发现 400/1500 轮档不可用的），且协议改了之后旧档（§9.13 表 1，scale n + 共享 wd）只当内部对照、不再被引用为「基线水平」 |

------

## 11. v1 → v2 的六处实质修改（自设计文档移入）

下表里的 §2.3 / §三 / §四 / §五 / §六 指 [`docs/OCA.md`](OCA.md) 的节号（设计稿编号），本文内的对应用 §0–§10 的号。

| # | v1 的问题 | v2 的改法 |
| :-- | :-- | :-- |
| 1 | 抑制核 $\kappa=\mathrm{ReLU}(\langle q_a,q_b\rangle)/\sqrt d$ 无界 ⇒ 迭代矩阵谱半径可 $>1$，hub 节点若干轮内发散 | 改为 shifted-cosine 核 + GCN 式对称归一化，得 $\lVert\tilde\kappa\rVert_2\le 1$（§2.3、§4 Lemma 2） |
| 2 | $K=8$ 稠密 padding + `for i in range(N)` Python 循环：静默丢弃高度数节点的邻居、系统性偏向小索引节点、不可扩展 | 分数张量定义在增广边集 $\hat{\mathcal E}$ 上，竞争项精确因子化为两次 `index_add_`，$O(\lvert\hat{\mathcal E}\rvert d)$，无截断（§2.9） |
| 3 | $\alpha_i$ 只缩放纵向提案，中心恒在竞争场内 ⇒ 设计稿 §三 表格声称的「$\alpha\to0$ 得到 B 模式」并未实现 | $\alpha_i$ 同时衰减中心的 $\kappa$ 行列（$M_{ab}$）并反向缩放提案（§2.3） |
| 4 | 「$\lambda_i=0$ 退化为 GAT」不成立（$\phi(z)$ 是 query 无关偏置；点积打分也非 GAT 的加法式） | 给出严格退化的**七个条件**并逐元素验证（§3） |
| 5 | 代码 bug：`_get_phi` 丢掉中心的 $\phi(z_i)$；残差 `norm(x+h)` 在 `in_dim!=out_dim` 时 shape crash；`gather_neighbors_padded` 依赖 `edge_index` 已按 row 排序；`-1e9` 参与 `bmm` | 均已修（稀疏实现无 padding，也就无需 `-inf`） |
| 6 | 骨架写死在 `model/oca_net.py`：换宽度要改代码，`scale` 无从表达，而且**拓扑写错了也能跑**（旧版 neck 就是这么带过来的） | 拓扑下沉 `cfg/models/*.yaml`，由 `model_builder.parse_model` 逐行推宽度并复合缩放（§6）；`from` 下标本身由用例钉成断言 |

> v1 的原文已无法逐字恢复：仓库里最早的提交 `848d0c1` 已经包含改后的版本，会话记录不存工具输出。上表的“v1 的问题”列是当时逐条引过的原文表述，是目前能给出的最完整记录。
