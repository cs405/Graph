# OCA 统一框架：B + C + 同配自适应

> **这一份是设计文档** —— 研究目标、动机、数学主张、要做的实验与数据。
> 算法的**实现口径与实测结果**在 [`OCA_algorithm.md`](OCA_algorithm.md)：那里每条公式对齐 [`modules/oca.py`](../modules/oca.py) 的真实计算图，每个数字带取数命令，另有 `OCAConfig` 全表、结构表参数量、消融变体全集、以及新加的 SOY/CORN 数据体检。
> 分工：公式细节与工程现状以 `OCA_algorithm.md` 为准，研究主张与目标以本文为准。
>
> v1 → v2 有 6 处实质修改（都是为了可证明稳定与可扩展），已在下面 §二 就地订正，逐条对照与证据在 [`OCA_algorithm.md`](OCA_algorithm.md) §11。未处理的遗留见文末 §十一（基线选择、novelty 撞车、门控同源梯度纠缠、异构图接入）。

## 一、整体设计哲学

一个算子，三种行为模式，由两个可学习门控自动切换：

| 门控 | 作用 | 控制什么 |
| :-- | :-- | :-- |
| $\lambda_i$ | 竞争强度 | 该不该横向竞争（同配/异配自适应） |
| $\alpha_i$ | 中心参与度 | 中心该不该进竞争场（B→C 连续过渡） |

- $\lambda_i\to 0$：无横向竞争（严格回到 GAT 还需 §三 的另外几个条件）
- $\lambda_i\to 1$：启用横向竞争，适合异配图
- $\alpha_i\to 0$：B 模式（中心只提案，不进竞争场）
- $\alpha_i\to 1$：C 模式（中心进竞争场）

------

## 二、统一公式

记号：竞争场 $\mathcal V_i=\mathcal N(i)\cup\{i\}$，$c_a:=\mathbb 1[a=i]$ 标记槽位是否为中心。
分数定义在**增广边集** $\hat{\mathcal E}=\mathcal E\cup\{(i,i)\}$ 上，即 $s$ 是逐 (中心 $i$, 槽位 $a$) 的量 $s_{i,a}$——同一节点 $a$ 在不同中心 $i$ 的竞争场里分数不同。这一点是 §2.3 因子化成立的前提，也是 v1 稠密 padding 写法掩盖掉的。

### 2.1 竞争场

$$\mathcal V_i=\mathcal N(i)\cup\{i\},\qquad z_a=W h_a,\ a\in\mathcal V_i$$

中心和邻居一起进入竞争场，所有节点用同一变换。

### 2.2 初始竞争分数

$$s_{i,a}(0)=\phi(z_a)+(1-\alpha_i)\,b_a\,\mathbb 1[a\ne i],\qquad b_a=\frac{\langle W_q h_i,\ W_k z_a\rangle}{\sqrt d}$$

中心没有纵向提案项（它不需要给自己打分），故 $a=i$ 时 $s_{i,i}(0)=\phi(z_i)$。

### 2.3 统一竞争迭代（归一化抑制核）

**核。** 令 $q_a=q_\theta z_a$、$\hat q_a=q_a/\lVert q_a\rVert$，

$$\kappa_{ab}=\langle\hat q_a,\hat q_b\rangle+1\in[0,2]$$

$+1$ 偏移有两个作用，缺一不可：

1. **保非负**，于是行和 $D_a$ 等于绝对值行和，才能做 GCN 式归一化并把谱范数压到 $\le1$（§四 Lemma 2）；
2. **保持线性可因子化**——$\kappa$ 仍是 $[\hat q;1]$ 的 Gram 矩阵，故 $\sum_b\kappa_{ab}w_b=\langle\hat q_a,U_i\rangle+R\,d_i$ 可写成两次 scatter-add（§五）。这正是放弃 $\mathrm{ReLU}(\cdot)$ 的代价：ReLU 分支需要 $O(\sum_i\lvert\mathcal N(i)\rvert^2)$，只在 `backend='dense'` 的参考实现里保留，供小图消融。

**$\alpha$ 屏蔽中心行列。**

$$M^{(i)}_{ab}=\alpha_i+(1-\alpha_i)(1-c_a)(1-c_b)\in[\alpha_i,1]$$

即邻居–邻居之间权重为 $1$，凡是碰到中心的行列权重为 $\alpha_i$。$\alpha_i=0$ 时中心的 $\kappa$ 行列整体消失 ⇒ 真 B 模式。

**归一化。**

$$D^{(i)}_a=\!\!\sum_{b\in\mathcal V_i}\!\!M^{(i)}_{ab}\kappa_{ab},\qquad
\tilde\kappa^{(i)}_{ab}=\frac{M^{(i)}_{ab}\kappa_{ab}}{\sqrt{\max(D^{(i)}_a,1)}\sqrt{\max(D^{(i)}_b,1)}}\Big|_{a\ne b},\quad \tilde\kappa^{(i)}_{aa}=0$$

$\max(\cdot,1)$ 的截断只放大分母，因而按元素级只会把归一化核变小（$\kappa\ge0$ 是用得上的前提）；它同时保证孤立节点（$\mathcal V_i=\{i\}$，$D=0$）与 $\alpha_i\to0$ 的中心行不出现 $0/0$。

**迭代。** 驱动项 $c_a:=\phi(z_a)+(1-\alpha_i)b_a\mathbb 1[a\ne i]$，

$$\boxed{\;s_{i,a}\leftarrow(1-\beta)s_{i,a}+\beta\Big(c_a-\lambda_i\sum_{b\in\mathcal V_i}\tilde\kappa^{(i)}_{ab}s_{i,b}\Big)\;}$$

逐项解释：

| 项 | 含义 | 门控 |
| :-- | :-- | :-- |
| $(1-\alpha_i)\mathbb 1[a\ne i]b_a$ | 纵向提案，只作用于邻居，$\alpha_i$ 越大越弱 | $\alpha_i$ |
| $\phi(z_a)$ | 自身嗓门，中心和邻居都有 | 无 |
| $\lambda_i\sum_b\tilde\kappa_{ab}s_b$ | 横向抑制，经归一化，$\lambda_i$ 越大竞争越强 | $\lambda_i$ |
| $M^{(i)}_{ab}$ | 中心进场程度（v2 新增，$\alpha_i$ 的第二条通路） | $\alpha_i$ |

**$\beta\in(0,1)$、$\lambda_i\in(0,1)$ 是硬约束**，不是调参偏好：`OCAConfig.__post_init__` 对越界的 `beta` 直接 `assert`。

### 2.4 门控生成

$$\lambda_i=\sigma(w_\lambda^\top h_i),\qquad \alpha_i=\sigma(w_\alpha^\top h_i)$$

两者都由中心节点 $h_i$ 生成。初始化上 $\lambda$ 的 bias 取负（`lambda_bias_init=-2`，$\sigma(-2)\approx0.12$），即**默认不竞争，竞争要靠训练挣来**；理由见 §十一 的 P1-3。

### 2.5 中心化 = 投影步

每轮结束后

$$s_{i,a}\leftarrow s_{i,a}-\frac1{\lvert\mathcal V_i\rvert}\sum_{b\in\mathcal V_i}s_{i,b}$$

准确定位（v1 说得含糊）：

- 它是把 $s$ 投影到 $\mathbf 1^\top s=0$ 子空间，等价于对 §四 的能量做**投影**梯度下降，而非无约束最小化；
- 对**最终 readout 的 softmax 是 no-op**（§2.7 只对 $\mathcal N(i)$ 归一化，平移不变），所以它的唯一作用不是"让分数更好玩"，而是**数值防漂移**；
- 但它**确实影响后续迭代**（$\tilde\kappa$ 行和非零，漂移会被竞争项重新放大），所以每轮都要做；
- $\kappa$ 的 $+1$ 偏移中心化后对分子无贡献，只通过 $D$ 起作用。

### 2.6 温度调制

$$\tau_i=\mathrm{softplus}\big(w_\tau^\top[h_i\,\|\,s_i(T)]\big)+\tau_{\min}$$

中心自己的竞争结果 $s_i(T)$ 参与调温：中心被抑制（$s_i(T)$ 低）说明它与邻居重复，温度升高、聚合更分散。喂给温度头的中心统计量默认取 **z-score**（`center_stat='zscore'`），否则 $\lambda$ 门控会跟着特征尺度漂（§十一 P1-3）。

### 2.7 softmax 读出（仅邻居）

$$p_j=\frac{\exp(s_j(T)/\tau_i)}{\sum_{k\in\mathcal N(i)}\exp(s_k(T)/\tau_i)},\qquad j\in\mathcal N(i)$$

中心分数 $s_i(T)$ **不参与聚合**，只用于调温。

### 2.8 加权聚合

$$m_i=\sum_{j\in\mathcal N(i)}p_j z_j$$

### 2.9 输出融合

$$h_i'=\mathrm{LayerNorm}\big(P h_i+\mathrm{MLP}([W_s h_i\,\|\,m_i])\big)$$

残差必须走独立投影 $P$（`self.skip`）。v1 写 `self.norm(x + h)`，在 `in_dim != out_dim`（如 Cora 1433→64）时第一层就 shape crash。

------

## 三、三种模式的连续过渡

| 条件 | $\lambda_i$ | $\alpha_i$ | 行为 | 等价于 |
| :-- | :-- | :-- | :-- | :-- |
| 同配图 | 0 | 任意 | 无横向竞争 | **无竞争的 OCA**（≠ GAT，见下） |
| 异配图 + B | 1 | 0 | 纵向提案 + 邻居间横向竞争 | OCA-B |
| 异配图 + C | 1 | 1 | 统一竞争场（中心参与抑制） | OCA-C |
| 中间状态 | 0.5 | 0.5 | 部分竞争 + 部分提案 | 平滑过渡 |

**"退化为 GAT"需要七个条件同时成立**，$\lambda_i=0$ 只是其中一个：

$$\lambda=0\ \wedge\ \phi\equiv0\ \wedge\ \alpha=0\ \wedge\ \tau=1\ \wedge\ T=0\ \wedge\ \text{加法式打分}\ \wedge\ \text{GAT 式输出}$$

- $\phi\equiv0$：v1 保留 $\phi$，它是 query 无关的节点偏置，会加进 softmax 分数，不满足 softmax 平移不变性 ⇒ 不等于 GAT；
- **加法式打分**：GAT 用 $\mathrm{LeakyReLU}(a_{\mathrm{src}}^\top z_i+a_{\mathrm{dst}}^\top z_j)$，与 §2.2 的 $q^\top k$ 点积不同族（GATv2 已论证点积式无法表达这种加性依赖）；
- GAT 式输出：$h_i'=\mathrm{ELU}(m_i)$，无残差、无 LayerNorm、无 `out_mlp`。

这七个条件在代码里是 `OCAConfig` 的一组开关组合，工厂函数 `OCALayer.gat_equivalent(in_dim, out_dim, heads)` 一次打开，且**不分配**用不到的参数（与 GAT 参数量可比）。`test/test_oca_degeneration.py::test_gat_degeneration_is_exact` 用独立实现的稠密 GAT 做参照，heads=1/4/8 下逐元素误差 $<10^{-12}$。

------

## 四、能量函数（统一解释）

$\tilde\kappa^{(i)}$ 对称，故下面这个 $E_i$ 是良定义的二次型（v1 用无界 ReLU 核时，能量与更新式也自洽，但凸性无保证）：

$$E_i(s)=-\sum_{a\in\mathcal V_i}c_a s_a+\frac{\lambda_i}2\,s^\top\tilde\kappa^{(i)}s+\frac12\lVert s\rVert^2,\qquad c_a=\phi(z_a)+(1-\alpha_i)b_a\mathbb 1[a\ne i]$$

四项含义：

1. $-\sum_a\phi(z_a)s_a$：所有节点想按自己嗓门激活；
2. $-(1-\alpha_i)\sum_j b_j s_j$：纵向提案拉高邻居分数，$\alpha_i$ 越大拉力越弱；
3. $+\frac{\lambda_i}2 s^\top\tilde\kappa^{(i)}s$：横向竞争惩罚相似节点同时高激活；
4. $+\frac12\lVert s\rVert^2$：正则，防爆炸。

**Lemma 1（更新式 = 能量梯度）.** $\nabla_s E_i=-c+\lambda_i\tilde\kappa^{(i)}s+s$，于是 §2.3 的迭代恰为 $s\leftarrow s-\beta\nabla_sE_i$，§2.5 恰为向 $\mathbf 1^\top s=0$ 的投影。即：**在零均值子空间上对 $E_i$ 做投影梯度下降**（不是"每轮都把 $E$ 求到最小"）。

**Lemma 2（与度数无关的一致稳定性）.** 记 $A:=M^{(i)}\!\circ\!\kappa^{(i)}\ (\ge0)$、$D_a=\sum_b A_{ab}$、$D'_a=\max(D_a,1)$，则 $\tilde\kappa^{(i)}=\big(D'^{-1/2}AD'^{-1/2}\big)\big|_{\text{去对角}}$，且

$$\big\lVert\tilde\kappa^{(i)}\big\rVert_2\le\big\lVert D^{-1/2}AD^{-1/2}\big\rVert_2=\rho(D^{-1}A)\le\lVert D^{-1}A\rVert_\infty=1 .$$

三步依据：**(a)** $0\le\tilde\kappa\le D^{-1/2}AD^{-1/2}$ 元素级（抬 $D'$、去对角都只让非负元变小），而对称非负矩阵的谱范数等于其 Perron 根，Perron 根对元素级序单调；**(b)** $D^{-1/2}AD^{-1/2}$ 与 $D^{-1}A$ 相似 ⇒ 同谱；**(c)** $D^{-1}A$ 行和恒为 $1$。$D_a=0$ 的行在 $\tilde\kappa$ 中整行为零，不等式平凡成立，故可限制在 $D_a>0$ 的子图上用 (b)。

> ⚠️ 常见的**错误证明**是「归一化后行和 $\le1$ ⇒ 谱范数 $\le1$」。混合分母 $\sqrt{D_aD_b}$ 控制不住行和（需要 $D_b\ge D_a$ 才行），所以这一步不成立。$v1$ 与本文件上一版都写的就是这个错证明，已换掉。

推论：

1. $\nabla^2E_i=I+\lambda_i\tilde\kappa^{(i)}\succeq(1-\lambda_i)I\succ0$ ⇒ $E_i$ 强凸、存在唯一极小点 $s^\star$（由 KKT 系统 $\begin{bmatrix}I+\lambda\tilde\kappa&-\mathbf 1\\ \mathbf 1^\top&0\end{bmatrix}\begin{bmatrix}s\\ \nu\end{bmatrix}=\begin{bmatrix}c\\0\end{bmatrix}$ 闭式给出）；
2. 迭代矩阵 $(1-\beta)I-\beta\lambda_i\tilde\kappa^{(i)}$ 的谱半径 $\le\max\big(\lvert1-\beta+\beta\lambda_i\rvert,\ \lvert1-\beta-\beta\lambda_i\rvert\big)<1$。因 $s^{+}=P\big[(1-\beta)I-\beta\lambda_i\tilde\kappa^{(i)}\big]s+\beta Pc$ 且 $\lVert P\rVert_2=1$、方括号内对称，得 $\lVert s^{(T)}-s^\star\rVert\le\rho^T\lVert s^{(0)}-s^\star\rVert$；
3. 前提只有 $\beta\in(0,1),\lambda_i\in(0,1)$ —— **与 $\lvert\mathcal N(i)\rvert$ 无关**，这正是 hub 节点不再炸的原因。

实测（`python -m test.run_all oca`，float64/CPU，合成 BA 图）：

| 主张 | 证据 |
| :-- | :-- |
| 稀疏因子化 == 稠密逐节点建矩阵 | 12 组配置（heads 1/2 × T 1/3 × α→0/中/1）最大绝对误差 $8.88\times10^{-16}$ |
| 更新式 == 投影梯度下降 | 全图 41 个场逐场验：$\max\lVert\nabla^{\rm num}E-P\nabla E\rVert=1.94\times10^{-16}$、$\max\lVert s^{(1)}-\mathrm{PGD}(s)\rVert=1.11\times10^{-16}$ |
| 收敛到唯一极小点 | $T=300$ 时 $\max_i\lVert s^{(T)}-s^\star\rVert_\infty<2.22\times10^{-16}$；速率界 $\lVert e^{2T}\rVert\le\rho^T\sqrt m\lVert e^T\rVert$ 成立 |
| hub + $\lambda\to1$ 不发散 | $\max\deg=670$、$\lambda=1.000000$、$\alpha=1$、$T=40$ ⇒ $\max\lvert s\rvert=0.619$ |
| 归一化确为必需（反向对照） | 换回 v1 的 $\mathrm{ReLU}(\langle q,q\rangle)/\sqrt d$ 无归一化核，第 10 轮 $\max\lvert s\rvert=1.134\times10^{13}$ |
| 无度数截断 | $\max\deg=265$ 全部进竞争场（v1 在 $K=8$ 处静默丢弃） |
| $\alpha\to0$ 真解耦 | 中心→邻居影响 $=0.00{\times}10^{0}$；$\alpha=0.62$ 时 $4.11\times10^{-2}$ |
| 严格 GAT 退化 | heads=1/4/8，$\max\lvert\Delta\rvert<10^{-12}$；且 gat 模式无死分支（344 参数） |

------

## 五、两条属于算法定义的约定（代码地图已迁至算法文档）

实现按层拆包，目录布局对齐 yolov8：**`model` 是一个文件而不是文件夹** —— 根 [`model.py`](../model.py) 里是 `class OCA`（yolov8 `class YOLO` 的图版），它只做「结构表 + 配置 → 构建/训练/评估/查询」这一件事，展开逻辑在 `model_builder.py`，算子本体在 `modules/oca.py`。**逐文件职责表、`OCA(...).train()` 接口、CLI 与运行产物都在 [`OCA_algorithm.md`](OCA_algorithm.md) §0.5、§7**，本节只留两条写反就整个作废的约定：

1. **`edge_index` 用 [row = 中心 $i$，col = 槽位 $a$]**，与 v1 里 `row, col = edge_index; Z_pad[i] = z[col[...]]` 一致，**与 PyG `MessagePassing` 默认相反**。这不是工程口味：$s$ 是增广边集 $\hat{\mathcal E}$ 上的逐 (中心, 槽位) 量（§二），方向定反了竞争项整个作废。无向数据集无差别，directed 输入由 `OCAConfig.symmetrize` 控制是否先 `to_undirected`。
2. **竞争场 $\mathcal V_i=\mathcal N(i)\cup\{i\}$ 靠「先去自环 → `coalesce` 去重 → 再加自环」构造**（`add_self_loops` 不为已有自环去重，直接加会重复计数）。$\lvert\mathcal V_i\rvert=\deg_i+1$ 与“无度数截断”两条主张吃它。

验证在 **[`test/`](../test)**：`python -m test.run_all` 一次跑完 10 个文件 57 个用例（真实数据未预下载时，其中一个会在用例内部打印 SKIP），无需 pytest；按名字过滤 `python -m test.run_all oca parity`，单文件直跑 `python test/test_oca_energy.py`。数值对拍一律 CPU + float64（`index_add_` 在 CUDA 上不满足结合律，位级不一致会让 $10^{-16}$ 量级的断言随机失败）。

增广边集的具体构造、$\sum_b\tilde\kappa_{ab}s_b$ 的两次 `index_add_` 因子化（$O(\lvert\hat{\mathcal E}\rvert d)$、无 Python 循环、无 `max_deg`、无 padding 截断）、`OCAConfig` 全字段与消融开关的对应关系、稠密参考实现 `OCALayer.forward_dense`（逐节点显式建 $\tilde\kappa^{(i)}$ 矩阵，$O(\sum_i\deg_i^2)$，只用于对拍与 ReLU 核消融 —— 它是因子化的地面真值，不是可训练路径），全部按代码写在 [`OCA_algorithm.md`](OCA_algorithm.md) §1、§2.9、§5。

------

## 六、YOLO 式骨架：拓扑在结构表里

v1 把「4 层 backbone + PANet 式 neck + head」写死在 `model/oca_net.py`，代价有两处，而且都是静默的：换宽度要改代码（`scale` 根本没有地方表达）；**拓扑写错了照样能跑** —— bottom-up 两行的输入下标曾写成 `[[4,6]]`/`[[5,7]]`，把 D1/D0 重复加了一遍，前向 shape 全对、loss 照降，自称「逐行核对过」也没人发现。能跑 ≠ 对，所以 v2 把拓扑搬到 [`cfg/models/oca.yaml`](../cfg/models/oca.yaml)，由 [`model_builder.parse_model`](../model_builder.py) 逐行推宽度、复合缩放，并把下标本身钉成断言（`test/test_model_builder.py::test_oca_neck_topology_is_pinned`）。

> 顺带订正 v1 §六 的另一条：当时怀疑 `GraphNeck.forward` 里 `self.td[i]` 配对有 off-by-one，重读确认没有（`td[-1]` 就是上一层的对角项）—— 那个结论仍然成立；出事的是把这套拓扑**翻译成结构表下标**的那一步。

结构表每行仍是 yolov8 的四元组 `[from, repeats, module, args]`：

```yaml
backbone:                              # 4 个 OCA 层，逐层输出就是一个「尺度」
  - [-1, 1, OCAConv, [64]]             # 0 尺度 f0（1 跳）
  - [-1, 1, OCAConv, [64]]             # 1 尺度 f1（2 跳）
  - [-1, 1, OCAConv, [64]]             # 2 尺度 f2（3 跳）
  - [-1, 1, OCAConv, [64]]             # 3 尺度 f3（4 跳）
head:                                  # PANet 式双向融合
  - [[2, 3], 1, Merge, [64]]           # 4 = f2 + f3
  # ...
  - [[6, 7, 8, 9], 1, Classify, [nc]]  # 10 多尺度读出
```

`nc` 在结构表里留空，由数据集在构建时注入（同 yolov8 用 `data.yaml` 覆盖）；`scales` 是 `[depth_multiple, width_multiple, max_channels]`；文件名结尾的 `-n` / `_s` 会被 `yaml_model_load` 推成 scale，但 `gcn.yaml` 末尾那个 `n` 不会被误判（有专门用例钉住）。

与图像侧只有四点差别，全部源于「图上没有空间维度」：

1. **每层都要拿 `edge_index`**：前向不是「张量进张量出」，而由 `GraphSequential` 把 `(x, edge_index)` 与按 `from` 缓存的历史层输出一起路由；transductive 全图下 `edge_index` 恒定，不进缓存。
2. **没有下采样**：图像 neck 靠 stride/pool 造金字塔，图侧的「尺度」= **感受野跳数**（第 $k$ 层是 $k{+}1$ 跳）。融合只两种：`Merge`（等宽相加，`modules/neck.py`；不等宽在**构建期**就报错并提示改用 `Concat`）与 `Concat`（沿特征维拼接，输出宽度 = 各输入之和，由 builder 推导，args 直接省掉）。
3. **`repeats` 用自定义 `Repeat` 而不是 `nn.Sequential`**：后者只会把单个输入往下传，而这里每层还要吃 `edge_index`。一个 `Repeat` 节点仍只占**一个**层号（缓存协议要求一对一），所以 stage 内部的中间层不能被 `from` 引用 —— 这是 `oca_deep.yaml` 的代价，也是为什么 `oca.yaml` 每行 `repeats` 都是 1。
4. **`Classify` 可以多输入**（`modules/head.py`）：收 list（多尺度读出）或单 tensor，内部拼接 → 隐藏层 → 分类头，`dropout` 由 `cfg.model.dropout` 注入。

拓扑读法（backbone 的 0..3 = f0..f3）：top-down `4=f2+f3`、`5=f1+4`、`6=f0+5`；bottom-up `7=D1+D0=[5,6]`、`8=D2+U1=[4,7]`、`9=f3+8`；读出只看 4 个**融合后**的结果 `[6,7,8,9]`（backbone 那 4 行已被融合吸收，不再单独看）。`parse_model` 顺带校验一条不变式：**任何被 `from` 引用的绝对行号都必须在 `save` 集合里** —— 引用未缓存的行不会报错，只会拿到 `None`，然后在某个 `torch.stack` 里炸出一句跟结构表无关的话。写错结构表的六类典型情形（未注册模块 / 缺 `nc` / 多输入却 `repeats>1` / 前向引用 / 下标越界 / 列数不足）都有用例检查报错文案是否点名到行。

三处**有意的偏离**（不是漏实现，是不想在没做消融前多堆组件）：`Merge` 只在融合完做 **1 次** LayerNorm（yolov8 的 CSP 每个分支都归一化）；`Classify` 只有一层隐藏；`oca.yaml` 各 `repeats=1` 故 `n/s/m/l/x` **只差宽度**，深度缩放由 `oca1`/`oca_deep` 示范。

**骨架与算子零耦合**：neck 只是多尺度特征融合，把 `OCAConv` 换成 `GATConv`/`GCNConv` 同样能跑（基线结构表走的就是同一条 `parse_model` 路径，没有任何只在基线侧存在的胶水代码）。因此「YOLO 式骨架」**不能作为 OCA 的贡献卖**，只能作为工程配置；若要保留，必须在 §八 单列 `single_scale` 一行并证明增益来自融合而非参数量。

### 实测：行数、缩放与参数量

全在 [`OCA_algorithm.md`](OCA_algorithm.md) §6（七张结构表的行数/OCA 层数/参数量、`scale` 阶梯、深度缩放、逐层表，以及单层参数量的可手算分解）。这里只留两个影响 §八 读数的结论：

- **97k 参数全在第一层的输入投影**（$1433\times16$）。`oca` 与 `oca1` 的差值 8,304 里，6 个 `Merge` 只占 **1,824**，其余是两个多出来的 OCA 层（$2\times2{,}856$）与更宽的 `Classify`（768）。也就是说在 Cora 这类窄特征数据集上，多尺度融合本身几乎不带来容量差（109,346 vs 101,042，差 8%）。
- 但 `oca1` 在 `scale=n` 只有 **2** 个 OCA 层（`repeats=4` 乘 `depth=0.33` 取整为 1），而 `oca.yaml` 有 4 层 —— 所以 `single_scale` 这一行**同时动了融合与深度两个因素**，目前只能当「有没有 neck」的粗对照。要严格隔离，得补一份四行 backbone、无 `Merge` 的单尺度结构表（或在 `oca1.yaml` 里把 $n$ 的 `depth_multiple` 改成 1.0）。

------

## 七、训练协议的设计约束

配置是一棵嵌套 dataclass（[`config.py`](../config.py)），四个块与 [`cfg/train/default.yaml`](../cfg/train/default.yaml) 一一对应。当前生效的默认值、`runs/` 产物树、CLI 全集与 `OCA(...).train()` 接口都在 [`OCA_algorithm.md`](OCA_algorithm.md) §0.5、§7，且 `test/test_config.py::test_default_yaml_agrees_with_dataclass_defaults` 会断言 YAML 与 dataclass 默认逐项相等 —— 两边漂移是「文档抄 YAML、代码抄 dataclass」的开始。这里只留四条会影响读数的约定：

> ⚠️ 默认值那组数字是在**合成图**上能收敛凑效的值。真实数据集上 `lr`/`epochs` 必须按 val 重选（文献惯例 Cora 类 0.005~0.01 + 早停），不得拿现在这套去报真实数据的结果。

- **层数与宽度不在这里** —— 它们在 `model.spec` 指向的结构表里（§六）。v1 的 `hidden_dim` / `num_layers` 两个字段已删除；想改结构就换 YAML 或改 `scale`。基线 GAT/GCN/MLP 也不再由 `MODEL_PRESETS` 硬编 2 层，而是各自一份 `cfg/models/{gat,gcn,mlp}.yaml`，与 OCA 走同一条构建路径。
- **优先级**：dataclass 默认 < `cfg/train/*.yaml` < `train(**kwargs)` / CLI `--set`。`model.oca` 与结构表的 `oca:` 块是**逐键合并**（`with_oca`）而不是整块替换 —— 否则消融表里 `T4` 与 `full` 的差值就不只是 T 的差值。
- **只看 val 选模型**：`best_metric` 写 `test_*` 直接 `ValueError`（`training/trainer.py::_metric_key`，用例 `test_best_metric_guard_and_runs_artifacts` 钉住）；`val_acc`（默认）与 `val_loss` 都可用，后者靠 `evaluate` 顺带算的 val CE。
- YAML 1.1 的三个坑由 `_coerce` 兜住：`lr: 5e-4` 被读成字符串 `'5e-4'`、`save: off` / `class_weight: yes` 被读成 bool、`train_per_class: null` 与「没写这个键」必须可区分（前者 `None` ⇒ 由每类节点数推导，后者保持默认 20）。未知键**告警并忽略**（注释性字段不该炸训练），类型错则报错点名到 `block.field`。

### 运行目录与产物

`runs/<run_name>/train`、`train2…`（同 yolov8），内含 `config_used.yaml`（实际生效的合并快照，追溯以它为准）、`results.json`、`logs/train.log`、`weights/best.pt`、`checkpoints/`，`gates.npz` 在 run_dir 根。完整清单与两个坑（多种子时只有第一个种子占该目录；`best.pt` 可直接回读）见 [`OCA_algorithm.md`](OCA_algorithm.md) §7。设计层面只钉三条：所有决定训练结果的量必须进 `config_used.yaml`；选模型只看 val（写 `test_*` 直接报错）；不留「只在脚本里存在」的临时改动，否则审稿人复现不出来。**缺口：checkpoint 只落盘、不续训**（`fit` 固定传 `resume=False`、CLI 无 `--resume`），已列入 §十一。

### 训练稳定性技巧

1. ~~门控预热：前 10 epoch 固定 $\lambda_i=\alpha_i=0.5$~~ —— **v2 删掉**。把门控钉在 $0.5$ 再释放，会让"模型自动学到该用哪种模式"的故事静默失败（起点已经在中间，梯度信号弱，事后无法区分是学出来的还是初始化剩的）。改成 §2.4 的**偏置初始化**：默认不竞争，竞争要挣。
2. ~~**温度退火**：初期 $\tau$ 大（平滑），后期 $\tau$ 小（尖锐）~~ —— **未实现**。现在只有 §2.6 的逐节点可学温度，没有随 epoch 的退火schedule；要做得先加一个开关，否则与 `no_temperature` 那一行混在一起。
3. **Stop-gradient**：前 $T-1$ 轮竞争 stop-gradient（`detach_iterations=True`），最后一轮可微；省显存，代价是丢掉能量下降的完整梯度。
4. **残差 + LayerNorm**：见 §2.9（须走独立投影）。

------

## 八、消融实验设计（主张 → 实验的映射）

行与开关一一对应，由 [`training/sweep.py`](../training/sweep.py) 的 `ABLATIONS` 生成，`python train.py --ablate`（不带名字 = 全部）一次跑完。两类变体用同一个 dict 表达：值里出现 `oca` 键就是**算子消融**（合并进 `model.oca`），其余键按 `TrainConfig.patched` 的扁平名 / `block.field` 覆盖，所以「换结构表」这种**结构消融**也能写进同一张表 —— 重点是：不留「只在脚本里存在」的临时改动，否则审稿人复现不出来。**17 个变体的全集、每行覆盖哪个开关、`format_table` 的列名、合成图实跑表**全部在 [`OCA_algorithm.md`](OCA_algorithm.md) §5、§8.4，本节只留主张层面的对应关系：

| 主张（§一–§四） | 靠哪一行实验撑 | 现在的证据状态 |
| :-- | :-- | :-- |
| 横向竞争是增益来源，不是打分函数的副产品 | `full` vs `no_competition`，再 vs `gat_degenerate` / 外基线 `gat.yaml` | 未测（合成图饱和） |
| 纵向提案 $b$ 与自身嗓门 $\phi$ 各自必要 | `no_proposal` / `no_phi` | 未测 |
| 中心该不该进竞争场（B→C 是真区别） | `no_center_field` vs `center_in_field` | 机制已验（$\alpha\to0$ 时中心→邻居影响 $=0$），端到端未测 |
| 竞争迭代值不值得 $T$ 轮代价 | `T0` `T1` `T2` `T4` `T8` | Lemma 2 给了 $\rho^T$ 先验，别只报"T=2 最好" |
| 中心调温的必要性 + 统计量口径 | `no_temperature` / `raw_center_stat` | 未测 |
| 多尺度融合有增益（不是白涨参数） | `single_scale` + `scale_s`/`scale_l` | **当前不干净**：`oca1.yaml` 在 $n$ 下只有 2 个 OCA 层，同时动了融合与深度（§六 末） |
| stop-gradient 付了多少代价 | `detach_iters` | 未测 |

还没进表的三项（w/o 中心分数、核消融、**GAT + 侧向抑制插件**）逐条现状见算法文档 §5。其中 **GAT+插件** 是 §九 第一条卖点（"竞争作为原算子而非后处理"）的前提，不实现则该卖点不能写。

**基线必须补**：只 beat GAT 不够。异配图至少要有 FAGCN、GPR-GNN、GGCN、H2GCN、LINKX、ACM-GNN，同配图加 GCN/GATv2/GraphSAGE，且用各自官方 splits。

### 合成图上的实跑（只能证「开关是活的」）

命令与完整表在算法文档 §8.4。设计层面要记牢两条限制：默认合成图（`n_per_class=60, homophily=0.3`）**完全饱和**（`full` 的 acc = 1.0），所以这张表只能说明「开关确实改变了计算图、训练管线照跑、结构消融能正确换表」，**不能**当结论；`gat_degenerate` 掉到 0.5 量级是预期内的，它应当去跟外基线 GAT 比，而不是跟 `full` 比。

### 预期结果

> ⚠️ 下表是 v1 的**假设值，不是结果**，不得作为结论引用，也不要放进任何投稿材料。已核对：Chameleon/Squirrel/Actor 上 GGCN、FAGCN、ACM-GNN 公开报告的精度**高于**此处 GAT 与 OCA 的预期值——若照此表填数，"超过基线"的断言会当场不成立。请实测后覆盖，并让 OCA-full 那一列去跟上面补的基线比，而不是只跟 GAT 比。

| 数据集 | 类型 | GAT | OCA-B | OCA-C | OCA-full |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Cora | 同配 | 待测 | 待测 | 待测 | 待测 |
| Citeseer | 同配 | 待测 | 待测 | 待测 | 待测 |
| Actor | 异配 | 待测 | 待测 | 待测 | 待测 |
| Chameleon | 异配 | 待测 | 待测 | 待测 | 待测 |
| Squirrel | 异配 | 待测 | 待测 | 待测 | 待测 |

### 最该做的一张图（v1 埋没了的核心卖点）

$\lambda_i$ 与**局部同质性**的相关性散点图：对每个节点算 $\mathbb 1[y_j=y_i]$ 在 $\mathcal N(i)$ 上的均值，与训练后的 $\lambda_i$ 对点。若同配邻域 $\lambda$ 小、异配邻域 $\lambda$ 大，则"学习出的局部冗余度检测器"这个主张**一图成立**；若散点无结构，则 §一 的自适应故事不成立，需要及早知道。`OCALayer.aux['lambda']`（节点级，另有 `aux['alpha']`、`aux['s']`、`aux['tau']`）已把逐节点 $\lambda$ 暴露出来。取数已打通：`run.dump_gates: true`（默认开）会在训练结束时把逐节点的 $\lambda$/$\alpha$、局部同质性、有效位 mask、标签与预测写成 `runs/<name>/train/gates.npz`（形状与对齐由 `test/test_training.py::test_dump_gates_is_node_aligned` 钉住），$\lambda$-vs-同质性的相关性由 `training/diagnostics.py` 直接算进 `results.json`；**剩下的只是把散点图画出来**。诚实提示：合成图上 $\lambda$ 与 $h_i$ 的相关性 $\approx 0$（实测 $|r|<0.04$），这张图目前只有在真实数据上才有意义。

------

## 九、论文卖点

| 卖点 | 对应实验 | 状态 |
| :-- | :-- | :-- |
| 将横向竞争作为 GNN 聚合**原算子**（而非后处理插件） | GAT+插件 vs OCA | 需插件基线才成立 |
| 纵向 + 横向统一框架 | §八 消融 | 成立 |
| 中心参与竞争（C 方案） | OCA-C vs OCA-B | $\alpha$ 修正后才有真区别，已验（§四 表） |
| 同配/异配自适应 | §八 + $\lambda$-vs-同质性图 | **未验证**（合成图上 $\lambda$ std 仅 0.078，见 §十一） |
| 自适应粒度可细到**逐关系** $\lambda_r$（异构图上每类边一组门控） | SOY/CORN 路线 A（`OCA_algorithm.md` §9.3-5） | **未接入**；且 §9.5 实测：这张图里 record 在每条属性关系上出度恒为 1，逐关系各自的竞争场只有两个槽位（对唯一邻居 softmax 恒为 1）⇒ 该卖点在此数据上需要先把图改造成多邻居形态 |
| 可证稳定：与度数无关的一致收敛界 | Lemma 2 + §四 反向对照 | 成立，已实测 |
| 精确退化到 GAT（七条件） | `test_gat_degeneration_is_exact` | 成立，$<10^{-12}$ |
| YOLO 式多尺度图骨架 | `single_scale`（`oca1.yaml`） | 与算子零耦合，**不要作为贡献卖** |
| 能量最小化理论 | §四 | 措辞须为"投影梯度下降"，非"能量最小化" |

> novelty 风险（写作时必须正面处理，否则审稿人会先替你想到）：
> "首次将横向竞争作为聚合原算子"这个"首次"**不建议写**。相邻工作至少 FAGCN（带符号消息，自适应在同配/异配间切换）、GPR-GNN（传播阶数自适应）、entmax/$\alpha$-entmax/sparsemax（稀疏 + 竞争式归一化）、Competitive Softmax (Zhou et al., ICLR 2020)（成对竞争归一化）。OCA 可主张的差异点是：竞争作为**动力学**（迭代到能量极小）而非归一化技巧，且带 §2.3 这种可因子化、可证一致稳定的核，并严格退化回 GAT。

------

## 十、一句话总结

> **OCA = 增广边集上的一个可证稳定竞争动力学 + 两个门控。$\lambda_i$ 控制竞争强度（同配自适应），$\alpha_i$ 控制中心参与度（B→C 过渡）。核取 $\kappa_{ab}=\langle\hat q_a,\hat q_b\rangle+1$ 并 GCN 式归一化，使 $\lVert\tilde\kappa\rVert_2\le1$：迭代对任意度数线性收敛，计算 $O(\lvert\hat{\mathcal E}\rvert d)$ 无截断；七条件打开时逐元素等于 GAT。**

------

## 十一、遗留问题（v2 未处理，需决定）

| # | 问题 | 影响 | 建议 |
| :-- | :-- | :-- | :-- |
| P0-3 | 基线只有 GAT；预填数字低于已发表基线 | 主张无法成立 | §八 已换表，但实验必须重做 |
| P1-2 | "首次"撞车 FAGCN / entmax / Competitive Softmax | 审稿一击致命 | §九 已改写措辞，related work 仍需补写一节 |
| P1-3 | $\lambda,\alpha,\tau$ 三个门控同源 $h_i$，梯度纠缠；实测合成图上 $\lambda$ 的 std 只有 0.078 | 自适应故事可能是空的 | 部分缓解（$\lambda$ 负偏置初始化 + `center_stat='zscore'`）；根治要做 $\lambda$-vs-同质性图，必要时给三个门控各配独立输入投影 |
| — | 尚未在任何真实数据集上跑过：`dataset/real.py` 已接好 `torch_geometric.datasets`（Planetoid / WebKB / WikipediaNetwork 共 11 个，split 来源写进 `meta`），但本机下载 Cora 超时 —— 真实路径只跑通了接口，没跑通数据 | 全部实证结论目前只覆盖合成 BA 图 | 手动将数据放入 `data/` 后跑 `python train.py --dataset cora`（已验证失败模式：raw 文件不全时 PyG 会重拉，约 90s 后 `load_real` 抛 `DatasetUnavailable`，**不会静默拿半份脏数据**）；缺数据时 `test/test_dataset.py::test_real_dataset_optional` SKIP，不伪造通过 |
| — | SOY/CORN 农业异构图（`data/`，`HeteroData`：7 种节点 / 13 种关系 / record 145,544 与 133,422 节点）**尚未接入算法侧**：`GraphBundle` 只装单 `x`+单 `edge_index`、非 record 节点无特征、无 `val_mask`、`metrics/` 无 AUC / PR-AUC | 真实数据这条主张目前**一个数都没有**；「不做 mini-batch」的既有决定已被 145k 节点顶穿（实测单层 $T{=}2$ 就要 3.5 GiB，§9.6） | 七条缺口与两条建模路线逐条列在 [`OCA_algorithm.md`](OCA_algorithm.md) §9.2–§9.4；首个任务定为 `loss_level` 三分类 |
| **P0-4** | 已实测（`python -m dataset.hetero_probe`，详见 [`OCA_algorithm.md`](OCA_algorithm.md) §9.5–§9.7）：这张图的**结构本身几乎不带标签信息**——同组内标签一致率贴随机（除 `stage` 外）、池化邻居信息增益 $\le0.001$ AUC，而一个不用邻接矩阵的 MLP（12 特征 + 6 类属性 one-hot）已能复现生产方报告的 GNN 基线（corn 0.803 vs 0.805，soy 0.882 vs 0.894）；**单拿 `stage` 一列做 7 维 one-hot（零边、零气象）就到 0.762 / 0.858**，即基线超出 0.5 那部分的 85% / 91%，而它在源 CSV 里本来就是 one-hot 列 | 拿它当「图学习有用 / OCA 超基线」的证据，会被一个 MLP 直接驳掉 | **开工前要你拍三选一**（算法文档 §10 第 7 行）：① 先按 route B 造出真多邻居（record 间 kNN / 县级邻接 / 跨作物耦合；`graph_meta.json` 里 `knn.enabled` 本就是 `false`）；② 改卖分布漂移下的稳健性；③ 算法主张回到 Chameleon/Squirrel/Actor/Cora，本数据只当应用案例 |
| — | checkpoint 只落盘、不续训：`prepare_run_dir(resume=True)` 与 `capture_rng_state` 已就绪，但 `fit` 固定传 `resume=False`，CLI 也没有 `--resume` | 长训练断了只能从头跑（`§七` 已标注） | 接上权重 + epoch + 三套 RNG 的恢复路径，并补一个「存→读→接着跑」的用例 |
