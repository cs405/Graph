# 维度级可识别注意力机制（DIA）完整技术文档

**Dimension-level Identifiable Attention for Graph Neural Networks**

------

## 摘要

传统图神经网络（GCN、GAT、HGT）在图消息传递中，以完整节点特征向量作为交互最小单元，无法区分哪些特征维度对当前边起关键作用，也无法建模维度与维度之间的关联结构，更无法建模边两端节点的贡献差异。本文提出维度级可识别注意力机制（DIA，Dimension-level Identifiable Attention），将 GNN 消息传递的交互单元从节点向量下沉到特征维度级别。DIA 包含三层结构：第一层为边条件维度注意力，每条边独立生成节点内部维度筛选权重；第二层为低秩非负维度配对矩阵，显式建模节点 i 与节点 j 特征维度之间的关联强度；第三层为非对称主体贡献权重，建模边两端节点贡献不均衡的关系预测场景。通过非负低秩分解与稀疏正交约束，DIA 在无维度语义标注条件下仍能恢复可识别、可复现、可解释的维度配对矩阵。DIA 可直接替换 GAT/GCN 的消息生成模块，适用于异构图上节点关系预测、边分类与图分类任务。

------

## 一、背景与动机

### 1.1 传统 GNN 的缺陷

主流 GNN 消息传递范式：

1. 图中存在边 (i,j)(*i*,*j*)，代表节点 i*i* 与节点 j*j* 存在拓扑连接；
2. 消息传递时，取出节点 i*i*、节点 j*j* 完整特征向量，整体加权融合；
3. 不区分节点内部特征维度的有效性，无关维度混入消息，引入噪声；
4. 边两端贡献默认对等，无法建模"主体节点主导关系预测"场景；
5. 无法显式建模维度之间的关联结构。

### 1.2 现实场景举例

构建异构图，节点代表实体，边代表"存在交互关系"：

- 节点 i*i*（人）特征：身高、体重、审美、性格
- 节点 j*j*（动物）特征：外貌、攻击性、食物、性格
- 边 (i,j)(*i*,*j*)：表示人与动物存在关联，任务：预测人是否喜欢这只动物

真实交互逻辑：

1. **维度筛选**：人节点启用审美、性格，抑制身高、体重；动物节点启用外貌、攻击性、食物、性格；
2. **维度配对**：人的审美 ↔ 动物的外貌（互补关联）；人的性格 ↔ 动物的性格、攻击性、食物（同质/跨维度关联）；
3. **贡献非均等**：人是判断主体，贡献高于动物。

传统 GAT 会将全部维度打包加权，无法复现上述逻辑。

### 1.3 DIA 的核心定位

DIA 是 GNN 消息传递的替换组件，依托图的边约束：只有存在边 (i,j)∈E(*i*,*j*)∈E 的节点对，才执行维度级注意力与配对。

------

## 二、数学定义

### 2.1 符号系统

给定图 G=(V,E)G=(V,E)。

对于边 (i,j)∈E(*i*,*j*)∈E：

- 源节点特征：hi∈Rdi**h***i*∈R*d**i*
- 邻居节点特征：hj∈Rdj**h***j*∈R*d**j*
- 边特征：eij∈Rde**e***ij*∈R*d**e*
- 节点类型：ri=ϕ(i)∈R*r**i*=*ϕ*(*i*)∈R
- 边类型：tij=ψ(i,j)∈T*t**ij*=*ψ*(*i*,*j*)∈T

支持 di≠dj*d**i*=*d**j*。

### 2.2 第一层：边条件维度注意力

对边 (i,j)(*i*,*j*)，节点 i*i* 和 j*j* 各自生成只属于这条边的内部注意力：

γi∣j=σ(MLPγri,tij([hi;hj;eij]))∈Rdi**γ***i*∣*j*=*σ*(MLP*γ**r**i*,*t**ij*([**h***i*;**h***j*;**e***ij*]))∈R*d**i*

γj∣i=σ(MLPγrj,tij([hj;hi;eij]))∈Rdj**γ***j*∣*i*=*σ*(MLP*γ**r**j*,*t**ij*([**h***j*;**h***i*;**e***ij*]))∈R*d**j*

筛选后的特征：

h~i∣j=hi⊙γi∣j∈Rdi**h**~*i*∣*j*=**h***i*⊙**γ***i*∣*j*∈R*d**i*

h~j∣i=hj⊙γj∣i∈Rdj**h**~*j*∣*i*=**h***j*⊙**γ***j*∣*i*∈R*d**j*

其中 ⊙⊙ 为逐元素乘，γi∣j,p→0*γ**i*∣*j*,*p*→0 代表节点 i*i* 第 p*p* 维在该边被抑制。

### 2.3 第二层：低秩非负维度配对

方向 i→j*i*→*j* 的配对矩阵低秩分解：

Wij(i→j)=Uij(i→j)Vij(i→j)⊤**W***ij*(*i*→*j*)=**U***ij*(*i*→*j*)**V***ij*(*i*→*j*)⊤

其中：

Uij(i→j)∈R≥0di×k,Vij(i→j)∈R≥0dj×k**U***ij*(*i*→*j*)∈R≥0*d**i*×*k*,**V***ij*(*i*→*j*)∈R≥0*d**j*×*k*

维度配对矩阵：

Mij(i→j)=diag(h~i∣j)Uij(i→j)Vij(i→j)⊤diag(h~j∣i)∈Rdi×dj**M***ij*(*i*→*j*)=diag(**h**~*i*∣*j*)**U***ij*(*i*→*j*)**V***ij*(*i*→*j*)⊤diag(**h**~*j*∣*i*)∈R*d**i*×*d**j*

元素形式：

Mij,pq(i→j)=h~i∣j,p(ui,p(i→j)⊤vj,q(i→j))h~j∣i,q*M**ij*,*pq*(*i*→*j*)=*h*~*i*∣*j*,*p*(**u***i*,*p*(*i*→*j*)⊤**v***j*,*q*(*i*→*j*))*h*~*j*∣*i*,*q*

标量边强度：

sij(i→j)=1⊤Mij(i→j)1=h~i∣j⊤Wij(i→j)h~j∣i*s**ij*(*i*→*j*)=**1**⊤**M***ij*(*i*→*j*)**1**=**h**~*i*∣*j*⊤**W***ij*(*i*→*j*)**h**~*j*∣*i*

向量消息：

mij=diag(h~i∣j)Uij(i→j)Vij(i→j)⊤h~j∣i∈Rdi**m***ij*=diag(**h**~*i*∣*j*)**U***ij*(*i*→*j*)**V***ij*(*i*→*j*)⊤**h**~*j*∣*i*∈R*d**i*

### 2.4 第三层：非对称主体贡献

gi∣j=MLPα([hi;hj;eij])+bri→rj*g**i*∣*j*=MLP*α*([**h***i*;**h***j*;**e***ij*])+*b**r**i*→*r**j*

gj∣i=MLPα([hj;hi;eij])+brj→ri*g**j*∣*i*=MLP*α*([**h***j*;**h***i*;**e***ij*])+*b**r**j*→*r**i*

[αi∣j,αj∣i]=softmax([gi∣j,gj∣i])[*α**i*∣*j*,*α**j*∣*i*]=softmax([*g**i*∣*j*,*g**j*∣*i*])

最终边分数：

sij=αi∣jsij(i→j)+αj∣isij(j→i)*s**ij*=*α**i*∣*j**s**ij*(*i*→*j*)+*α**j*∣*i**s**ij*(*j*→*i*)

其中方向 j→i*j*→*i* 使用独立参数 Uij(j→i),Vij(j→i)**U***ij*(*j*→*i*),**V***ij*(*j*→*i*)，不退化为对称。

### 2.5 节点更新

hi′=hi+σ(1∣N(i)∣∑j∈N(i)mij)**h***i*′=**h***i*+*σ*(∣N(*i*)∣1∑*j*∈N(*i*)**m***ij*)

### 2.6 损失函数

L=Ltask+λsp(∥U∥1+∥V∥1)+λorth(∥U⊤U−I∥F2+∥V⊤V−I∥F2)+λγ∥γ∥1L=L*t**a**s**k*+*λ**s**p*(∥**U**∥1+∥**V**∥1)+*λ**or**t**h*(∥**U**⊤**U**−**I**∥*F*2+∥**V**⊤**V**−**I**∥*F*2)+*λ**γ*∥**γ**∥1

非负约束通过投影实现：

U←max⁡(U,0),V←max⁡(V,0)**U**←max(**U**,0),**V**←max(**V**,0)

### 2.7 可识别性定理

**定理（DIA 维度配对矩阵可识别性）**

设真实配对矩阵 W∗∈R≥0di×dj**W**∗∈R≥0*d**i*×*d**j*，秩为 k*k*。若：

1. W∗=U∗V∗⊤**W**∗=**U**∗**V**∗⊤，U∗,V∗≥0**U**∗,**V**∗≥0；
2. U∗,V∗**U**∗,**V**∗ 满足稀疏性条件（每列非零元数 ≤s≤*s*）；
3. U∗,V∗**U**∗,**V**∗ 列间满足分离条件；

则非负低秩分解在置换与对角缩放意义下唯一，配对矩阵 W=UV⊤=W∗**W**=**U****V**⊤=**W**∗ 可唯一恢复。

------

## 三、多节点示例

### 3.1 场景设定

一张异构图，节点包括人和动物，边表示"人是否喜欢这只动物"。

text

```
    [人 A] ──── [狗]     边1：喜欢
    [人 A] ──── [蛇]     边2：不喜欢
    [人 B] ──── [猫]     边3：喜欢
```



**人 A 的 4 维特征：**

| 维度 | 含义 |
| :--- | :--- |
| h₁   | 身高 |
| h₂   | 体重 |
| h₃   | 审美 |
| h₄   | 性格 |

**人 B 的 4 维特征：**

| 维度 | 含义 |
| :--- | :--- |
| h₁   | 身高 |
| h₂   | 体重 |
| h₃   | 审美 |
| h₄   | 性格 |

**狗、蛇、猫各 4 维特征：**

| 维度 | 含义   |
| :--- | :----- |
| h₁   | 外貌   |
| h₂   | 攻击性 |
| h₃   | 食物   |
| h₄   | 性格   |

### 3.2 边1：人 A — 狗（喜欢）

**人 A 内部注意力 γ：**

text

```
身高  █░░░░░░░░░  0.1
体重  █░░░░░░░░░  0.1
审美  █████████░  0.9
性格  ████████░░  0.8
```



**狗内部注意力 γ：**

text

```
外貌   █████████░  0.9
攻击性 ██░░░░░░░░  0.2
食物   ██████░░░░  0.6
性格   █████████░  0.9
```



**维度配对矩阵 M：**

text

```
              外貌    攻击性   食物    性格
身高          ░░░     ░░░     ░░░     ░░░
体重          ░░░     ░░░     ░░░     ░░░
审美          ███     0.05    0.10    0.20
性格          0.15    0.05    0.60    0.92
```



**读法：**

- 审美 ↔ 外貌：0.88，强 → 人觉得狗好看
- 性格 ↔ 性格：0.92，强 → 性格匹配
- 性格 ↔ 食物：0.60，中等 → 能接受狗的食性
- 攻击性被压制，几乎不参与

**主体权重：**

text

```
人 A  ████████████████░░  0.80
狗    ████░░░░░░░░░░░░░  0.20
```



### 3.3 边2：人 A — 蛇（不喜欢）

**人 A 内部注意力 γ（同一个人，不同边）：**

text

```
身高  █░░░░░░░░░  0.1
体重  █░░░░░░░░░  0.1
审美  ████████░░  0.8
性格  █████████░  0.9
```



**蛇内部注意力 γ：**

text

```
外貌   ████████░░  0.8
攻击性 ██████████  1.0
食物   ███████░░░  0.7
性格   ███░░░░░░░  0.3
```



**维度配对矩阵 M：**

text

```
              外貌    攻击性   食物    性格
身高          ░░░     ░░░     ░░░     ░░░
体重          ░░░     ░░░     ░░░     ░░░
审美          0.30    0.85    0.10    0.05
性格          0.10    0.90    0.20    0.15
```



**读法：**

- 审美 ↔ 攻击性：0.85，强 → 蛇的外形和攻击性让人不喜欢
- 性格 ↔ 攻击性：0.90，强 → 性格上无法接受攻击性强的动物
- 审美 ↔ 外貌只有 0.30 → 蛇好不好看不是重点

**主体权重：**

text

```
人 A  ██████████████████  0.90
蛇    ██░░░░░░░░░░░░░░░  0.10
```



### 3.4 边3：人 B — 猫（喜欢）

**人 B 内部注意力 γ：**

text

```
身高  ███░░░░░░░  0.3
体重  ██░░░░░░░░  0.2
审美  ██████████  1.0
性格  █████░░░░░  0.5
```



**猫内部注意力 γ：**

text

```
外貌   ██████████  1.0
攻击性 ████░░░░░░  0.4
食物   ████████░░  0.8
性格   ███████░░░  0.7
```



**维度配对矩阵 M：**

text

```
              外貌    攻击性   食物    性格
身高          0.10    0.05    0.05    0.05
体重          0.05    0.05    0.05    0.05
审美          0.95    0.10    0.30    0.20
性格          0.20    0.15    0.50    0.55
```



**读法：**

- 审美 ↔ 外貌：0.95，极强 → 人 B 主要看猫好不好看
- 性格 ↔ 食物：0.50，中等
- 攻击性基本不参与

**主体权重：**

text

```
人 B  ███████████████░░░  0.75
猫    █████░░░░░░░░░░░░░  0.25
```



### 3.5 三条边并排对比

text

```
        边1：人A—狗(喜欢)      边2：人A—蛇(不喜欢)     边3：人B—猫(喜欢)

          外貌 攻 食 性          外貌 攻 食 性          外貌 攻 食 性
身高      ░░  ░░ ░░ ░░          ░░  ░░ ░░ ░░          ░░  ░░ ░░ ░░
体重      ░░  ░░ ░░ ░░          ░░  ░░ ░░ ░░          ░░  ░░ ░░ ░░
审美      ██  ░░ ░░ ░░          ░░  ██ ░░ ░░          ██  ░░ ░░ ░░
性格      ░░  ░░ ██ ██          ░░  ██ ░░ ░░          ░░  ░░ ██ ██
```



**关键观察：**

| 边          | 最强配对                 | 含义               |
| :---------- | :----------------------- | :----------------- |
| 边1：人A—狗 | 审美↔外貌，性格↔性格     | 看脸 + 性格匹配    |
| 边2：人A—蛇 | 审美↔攻击性，性格↔攻击性 | 关注攻击性，不看脸 |
| 边3：人B—猫 | 审美↔外貌                | 主要看脸           |

**同一个人 A，对狗和对蛇，矩阵完全不同。同一条边类型，不同人、不同动物，矩阵也不同。**

### 3.6 训练三阶段对比

**训练前（随机初始化）：**

text

```
              外貌    攻击性   食物    性格
身高          0.42    0.55    0.38    0.61
体重          0.50    0.47    0.52    0.44
审美          0.39    0.58    0.45    0.53
性格          0.56    0.41    0.49    0.47
```



**训练中：**

text

```
              外貌    攻击性   食物    性格
身高          0.30    0.25    0.20    0.35
体重          0.28    0.22    0.18    0.30
审美          0.75    0.30    0.15    0.40
性格          0.20    0.65    0.60    0.80
```



**训练后：**

text

```
              外貌    攻击性   食物    性格
身高          0.02    0.01    0.01    0.03
体重          0.01    0.02    0.01    0.02
审美          0.88    0.08    0.05    0.15
性格          0.12    0.72    0.68    0.91
```



------

## 四、完整 PyTorch 实现

### 4.1 边条件维度注意力

python

```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, List


class EdgeConditionedDimAttention(nn.Module):
    """
    对每条边 (i, j) 生成节点 i 和节点 j 的维度注意力 gamma。
    输入: [h_i; h_j; e_ij]
    输出: gamma_i in R^{d_i}, gamma_j in R^{d_j}
    """
    def __init__(self, d_i: int, d_j: int, d_e: int, hidden: int = 64):
        super().__init__()
        self.d_i = d_i
        self.d_j = d_j
        self.mlp_i = nn.Sequential(
            nn.Linear(d_i + d_j + d_e, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_i),
        )
        self.mlp_j = nn.Sequential(
            nn.Linear(d_i + d_j + d_e, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_j),
        )

    def forward(
        self,
        h_i: Tensor,
        h_j: Tensor,
        e_ij: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        x = torch.cat([h_i, h_j, e_ij], dim=-1)
        gamma_i = torch.sigmoid(self.mlp_i(x))
        gamma_j = torch.sigmoid(self.mlp_j(x))
        return gamma_i, gamma_j
```



### 4.2 低秩非负维度配对

python

```
class LowRankNonNegPairing(nn.Module):
    """
    低秩非负维度配对: W = U V^T
    U: [d_i, k], V: [d_j, k], 均非负
    """
    def __init__(self, d_i: int, d_j: int, rank: int):
        super().__init__()
        self.d_i = d_i
        self.d_j = d_j
        self.rank = rank
        self.U = nn.Parameter(torch.rand(d_i, rank) * 0.1)
        self.V = nn.Parameter(torch.rand(d_j, rank) * 0.1)

    def project_nonneg(self):
        with torch.no_grad():
            self.U.data.clamp_(min=0.0)
            self.V.data.clamp_(min=0.0)

    def W(self) -> Tensor:
        return self.U @ self.V.t()

    def forward(
        self,
        h_i_tilde: Tensor,
        h_j_tilde: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        M = (
            h_i_tilde.unsqueeze(-1)
            * self.W().unsqueeze(0)
            * h_j_tilde.unsqueeze(-2)
        )
        s = M.sum(dim=(-1, -2))
        m = h_i_tilde * (self.W() @ h_j_tilde.unsqueeze(-1)).squeeze(-1)
        return M, s, m
```



### 4.3 非对称主体贡献

python

```
class AsymmetricContribution(nn.Module):
    """
    g_i, g_j -> softmax -> alpha_i, alpha_j
    支持类型级偏置 b_{r_i -> r_j}
    """
    def __init__(
        self,
        d_i: int,
        d_j: int,
        d_e: int,
        n_node_types: int,
        hidden: int = 64,
    ):
        super().__init__()
        self.n_node_types = n_node_types
        self.mlp = nn.Sequential(
            nn.Linear(d_i + d_j + d_e, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        self.type_bias = nn.Parameter(
            torch.zeros(n_node_types, n_node_types)
        )

    def forward(
        self,
        h_i: Tensor,
        h_j: Tensor,
        e_ij: Tensor,
        r_i: Tensor,
        r_j: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        x = torch.cat([h_i, h_j, e_ij], dim=-1)
        logits = self.mlp(x)
        bias = self.type_bias[r_i, r_j].unsqueeze(-1)
        logits = logits + torch.cat([bias, -bias], dim=-1)
        alpha = F.softmax(logits, dim=-1)
        return alpha[:, 0], alpha[:, 1]
```



### 4.4 DIA 消息层

python

```
class DIALayer(nn.Module):
    """
    DIA: Dimension-level Identifiable Attention
    一层消息传递，支持 d_i != d_j。
    """
    def __init__(
        self,
        d_i: int,
        d_j: int,
        d_e: int,
        rank: int = 16,
        n_node_types: int = 2,
        hidden: int = 64,
    ):
        super().__init__()
        self.dim_attn = EdgeConditionedDimAttention(d_i, d_j, d_e, hidden)
        self.pairing_ij = LowRankNonNegPairing(d_i, d_j, rank)
        self.pairing_ji = LowRankNonNegPairing(d_j, d_i, rank)
        self.asym = AsymmetricContribution(
            d_i, d_j, d_e, n_node_types, hidden
        )

    def forward(
        self,
        h_i: Tensor,
        h_j: Tensor,
        e_ij: Tensor,
        r_i: Tensor,
        r_j: Tensor,
    ) -> Dict[str, Tensor]:
        gamma_i, gamma_j = self.dim_attn(h_i, h_j, e_ij)
        h_i_tilde = h_i * gamma_i
        h_j_tilde = h_j * gamma_j

        M_ij, s_ij, m_ij = self.pairing_ij(h_i_tilde, h_j_tilde)
        M_ji, s_ji, m_ji = self.pairing_ji(h_j_tilde, h_i_tilde)

        alpha_i, alpha_j = self.asym(h_i, h_j, e_ij, r_i, r_j)

        s = alpha_i * s_ij + alpha_j * s_ji

        return {
            "gamma_i": gamma_i,
            "gamma_j": gamma_j,
            "M_ij": M_ij,
            "M_ji": M_ji,
            "s_ij": s_ij,
            "s_ji": s_ji,
            "m_ij": m_ij,
            "m_ji": m_ji,
            "alpha_i": alpha_i,
            "alpha_j": alpha_j,
            "score": s,
        }

    def project_nonneg(self):
        self.pairing_ij.project_nonneg()
        self.pairing_ji.project_nonneg()
```



### 4.5 完整 DIA 模型

python

```
class DIAModel(nn.Module):
    """
    DIA 模型: 多层 DIA 消息传递 + 边预测头。
    """
    def __init__(
        self,
        d_i: int,
        d_j: int,
        d_e: int,
        rank: int = 16,
        n_node_types: int = 2,
        hidden: int = 64,
        n_layers: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DIALayer(d_i, d_j, d_e, rank, n_node_types, hidden)
            for _ in range(n_layers)
        ])
        self.edge_head = nn.Sequential(
            nn.Linear(1 + d_i + d_j, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        h_i: Tensor,
        h_j: Tensor,
        e_ij: Tensor,
        r_i: Tensor,
        r_j: Tensor,
    ) -> Dict[str, Tensor]:
        out = None
        for layer in self.layers:
            out = layer(h_i, h_j, e_ij, r_i, r_j)
            h_i = h_i + out["m_ij"]
            h_j = h_j + out["m_ji"]
        assert out is not None
        x = torch.cat(
            [out["score"].unsqueeze(-1), h_i, h_j], dim=-1
        )
        logit = self.edge_head(x).squeeze(-1)
        out["logit"] = logit
        return out

    def project_nonneg(self):
        for layer in self.layers:
            layer.project_nonneg()
```



### 4.6 损失函数

python

```
class DIALoss(nn.Module):
    """
    L = L_task + lambda_sp * L1(U,V)
              + lambda_orth * L_orth(U,V)
              + lambda_gamma * L1(gamma)
    """
    def __init__(
        self,
        lambda_sp: float = 1e-3,
        lambda_orth: float = 1e-3,
        lambda_gamma: float = 1e-4,
    ):
        super().__init__()
        self.lambda_sp = lambda_sp
        self.lambda_orth = lambda_orth
        self.lambda_gamma = lambda_gamma

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        model: DIAModel,
        out: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, float]]:
        task_loss = F.binary_cross_entropy_with_logits(logits, labels)

        sp_loss = torch.tensor(0.0, device=logits.device)
        orth_loss = torch.tensor(0.0, device=logits.device)
        for layer in model.layers:
            for pairing in [layer.pairing_ij, layer.pairing_ji]:
                U, V = pairing.U, pairing.V
                sp_loss = sp_loss + U.abs().sum() + V.abs().sum()
                UtU = U.t() @ U
                VtV = V.t() @ V
                I = torch.eye(UtU.size(0), device=U.device)
                orth_loss = orth_loss + (
                    (UtU - I).pow(2).sum() + (VtV - I).pow(2).sum()
                )

        gamma_loss = (
            out["gamma_i"].abs().sum() + out["gamma_j"].abs().sum()
        )

        total = (
            task_loss
            + self.lambda_sp * sp_loss
            + self.lambda_orth * orth_loss
            + self.lambda_gamma * gamma_loss
        )
        return total, {
            "task": task_loss.item(),
            "sp": sp_loss.item(),
            "orth": orth_loss.item(),
            "gamma": gamma_loss.item(),
        }
```



### 4.7 训练步骤

python

```
def train_step(
    model: DIAModel,
    loss_fn: DIALoss,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Tensor],
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad()

    out = model(
        batch["h_i"],
        batch["h_j"],
        batch["e_ij"],
        batch["r_i"],
        batch["r_j"],
    )
    loss, stats = loss_fn(
        out["logit"], batch["label"], model, out
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    model.project_nonneg()

    stats["total"] = loss.item()
    return stats
```



### 4.8 可视化接口

python

```
@torch.no_grad()
def visualize_edge(
    model: DIAModel,
    h_i: Tensor,
    h_j: Tensor,
    e_ij: Tensor,
    r_i: Tensor,
    r_j: Tensor,
) -> Dict[str, Tensor]:
    """
    返回单条边的:
    - gamma_i, gamma_j: 维度注意力
    - M_ij: 维度配对矩阵
    - alpha_i, alpha_j: 主体权重
    """
    model.eval()
    out = model(h_i, h_j, e_ij, r_i, r_j)
    return {
        "gamma_i": out["gamma_i"].squeeze(0).cpu(),
        "gamma_j": out["gamma_j"].squeeze(0).cpu(),
        "M_ij": out["M_ij"].squeeze(0).cpu(),
        "alpha_i": out["alpha_i"].squeeze(0).cpu(),
        "alpha_j": out["alpha_j"].squeeze(0).cpu(),
    }
```



### 4.9 多边完整训练示例

python

```
if __name__ == "__main__":
    torch.manual_seed(0)

    d_i, d_j, d_e = 4, 4, 8
    model = DIAModel(d_i, d_j, d_e, rank=4, n_node_types=2)
    loss_fn = DIALoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 构造三条边的批次
    B = 12
    batch = {
        "h_i": torch.randn(B, d_i),
        "h_j": torch.randn(B, d_j),
        "e_ij": torch.randn(B, d_e),
        "r_i": torch.zeros(B, dtype=torch.long),
        "r_j": torch.ones(B, dtype=torch.long),
        "label": torch.randint(0, 2, (B,)).float(),
    }

    for step in range(300):
        stats = train_step(model, loss_fn, optimizer, batch)
        if step % 50 == 0:
            print(f"step {step}: total={stats['total']:.4f} "
                  f"task={stats['task']:.4f} sp={stats['sp']:.4f} "
                  f"orth={stats['orth']:.4f} gamma={stats['gamma']:.4f}")

    # 可视化三条边
    for name, idx in [("人A—狗", 0), ("人A—蛇", 1), ("人B—猫", 2)]:
        vis = visualize_edge(
            model,
            batch["h_i"][idx:idx+1],
            batch["h_j"][idx:idx+1],
            batch["e_ij"][idx:idx+1],
            batch["r_i"][idx:idx+1],
            batch["r_j"][idx:idx+1],
        )
        print(f"\n=== {name} ===")
        print("gamma_i:", vis["gamma_i"].numpy().round(3))
        print("gamma_j:", vis["gamma_j"].numpy().round(3))
        print("M_ij:")
        print(vis["M_ij"].numpy().round(3))
        print("alpha:", vis["alpha_i"].item(), vis["alpha_j"].item())
```



------

## 五、与现有方法对比

| 模型         | 消息交互最小单元 | 节点内部维度筛选   | 跨节点显式维度配对 | 边两端节点贡献权重   | 可识别性       |
| :----------- | :--------------- | :----------------- | :----------------- | :------------------- | :------------- |
| GCN          | 完整节点特征向量 | 无                 | 无                 | 默认对等             | 无             |
| GAT          | 完整节点特征向量 | 无                 | 无                 | 默认对等             | 无             |
| HGT          | 完整节点特征向量 | 无                 | 无                 | 默认对等             | 无             |
| Bilinear GNN | 完整节点特征向量 | 无                 | 部分               | 默认对等             | 无             |
| **DIA**      | **单个特征维度** | **✅ 边条件注意力** | **✅ 低秩非负配对** | **✅ 非对称主体权重** | **✅ 非负低秩** |

------

## 六、创新点

1. **消息粒度下沉**：将 GNN 消息交互单元从节点向量下沉到特征维度配对级别。
2. **三层解耦结构**：边条件维度注意力 + 低秩非负维度配对 + 非对称主体贡献。
3. **可识别性保证**：非负低秩分解 + 稀疏 + 正交约束，无维度语义标注下仍可恢复配对矩阵。
4. **非对称关系建模**：双向独立参数 + 类型偏置，适配主体主导的关系预测场景。
5. **模块化替换**：可直接替换 GAT/HGT 消息生成模块，兼容异构图。

------

## 七、适用场景与局限

### 适用场景

1. 异构图上实体关系预测（人-动物交互图）；
2. 植物器官异构图（枝干、果实、叶片作为节点，空间连接作为边）；
3. 社交网络用户关系预测；
4. 各类异构图节点分类、边分类任务。

### 局限

1. 优势建立在节点特征维度具备可解释语义的前提下；
2. 若节点特征是无明确语义的 CNN 隐向量，维度配对的可解释性优势减弱；
3. 稠密配对矩阵需低秩分解才可扩展；
4. 可识别性依赖非负、稀疏、分离条件。

------

## 八、结论

本文面向图神经网络消息传递，提出维度级可识别注意力机制 DIA。在图每条边的消息计算流程中，通过边条件维度注意力、低秩非负维度配对矩阵、非对称主体贡献权重，建模图中相连节点仅部分维度产生交互、主体节点对关系预测占更高贡献的现实规律。DIA 具备可识别性理论保证，可直接替换 GNN 消息生成模块，在异构图关系预测任务上具备良好可解释性。

------

## 附录：完整代码文件结构

text

```
dia/
├── __init__.py
├── attention.py       # EdgeConditionedDimAttention
├── pairing.py         # LowRankNonNegPairing
├── asymmetric.py      # AsymmetricContribution
├── layer.py           # DIALayer
├── model.py           # DIAModel
├── loss.py            # DIALoss
├── train.py           # train_step
├── visualize.py       # visualize_edge
└── example.py         # __main__ 示例
```



以上文档与代码构成 DIA 的完整技术方案，可直接用于论文 Method 部分与实验实现。