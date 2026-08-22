# 当前 DeepONet 模型说明

本文描述 **本仓库正在用的** DeepONet（`examples/deeponet` + `ppsci.arch.DeepONet`）：学什么、输入输出是什么、网络怎么接、这次训练到了什么程度。配置以 `conf/deeponet.yaml` 为准。

---

## 1. 在学什么

论文：[Lu et al., Nat Mach Intell 2021](https://arxiv.org/abs/1910.03193) 第 4.1.1 节，**反导数算子**（一维线性动力系统 \(g=u\)）。

算子 \(G\) 把函数 \(u\) 映成其从 0 积到 \(y\) 的原函数（初值 \(s(0)=0\)）：

\[
G(u)(y)=\int_0^y u(\tau)\,d\tau
\]

网络 **不是** 只拟合某一个 \(u\) 的积分曲线，而是拟合整个映射 \(G\)：任意一条（与训练集同族的）\(u\)，再给查询点 \(y\)，输出标量 \(G(u)(y)\)。

实现上采用论文推荐的 **Unstacked DeepONet + 偏置**（Eq. 2）：

\[
G(u)(y)\approx\sum_{k=1}^{p} b_k(u)\,t_k(y)+b_0
\]

- Branch：把传感器上的 \(u(x_1),\ldots,u(x_m)\) 编成向量 \(\mathbf{b}\in\mathbb{R}^{p}\)
- Trunk：把查询坐标 \(y\) 编成向量 \(\mathbf{t}\in\mathbb{R}^{p}\)（最后一层再过一次激活）
- 输出：内积再加可学习标量偏置 \(b_0\)

---

## 2. 输入与输出

一次前向对应 **一个样本三元组** \((u,\,y,\,G(u)(y))\)（论文 Fig. 1B，unaligned：每条 \(u\) 配一个随机 \(y\)）。

| 角色 | 键名 | 张量形状 | 含义 |
|------|------|----------|------|
| 输入 | `u` | `[B, 100]` | 输入函数在 \(m=100\) 个传感器上的取值。传感器固定为 \([0,1]\) 上均匀网格 \(x_j=j/99\)。 |
| 输入 | `y` | `[B, 1]` | 要查询的输出位置，\(y\in[0,1]\)。 |
| 输出（预测） | `G` | `[B, 1]` | 网络给出的 \(G(u)(y)\)。 |
| 标签（仅训练/验证） | `G` | `[B, 1]` | 对 \(u\) 做梯形累积积分后，在 \(y\) 处线性插值得到的参考值。 |

`B` 为 batch。本例 `IterableNPZDataset` 每个 epoch **整集一次前向**，训练时 \(B=10000\)，验证 \(B=10000\)。

数据来源：零均值 GRF，RBF 核长度 \(l=0.2\)（论文 Sec. 2.2）。npz 字段与框架键的对应：

| 文件 | npz 字段 | 模型键 |
|------|----------|--------|
| `antiderivative_unaligned_train.npz` | `X_train0`, `X_train1`, `y_train` | `u`, `y`, `G` |
| `antiderivative_unaligned_test.npz` | `X_test0`, `X_test1`, `y_test` | 同上 |

预测时只需构造 `{"u": ..., "y": ...}`，例如把同一条 \(u\) 在 \(y\) 网格上 tile 成 `[N_y, 100]`，即可画出整条 \(G(u)(\cdot)\)。

**注意**：Trunk 在 `ppsci.arch.DeepONet` 里写死 `input_dim=1`，因此当前实现只支持 **一维** 查询坐标 \(y\)。多维 \(y\) 需要改架构。

---

## 3. 模型结构

类：`ppsci.arch.DeepONet`（`ppsci/arch/deeponet.py`）。本案例实例化参数见 `MODEL`：

```yaml
u_key: u
y_key: y
G_key: G
num_loc: 100          # m，branch 输入维
num_features: 40      # p，branch/trunk 输出维（内积通道数）
branch_num_layers: 1  # branch 隐藏层数
trunk_num_layers: 2   # trunk 隐藏层数（对应论文 Table 2 trunk depth 3）
branch_hidden_size: 40
trunk_hidden_size: 40
branch_activation: relu
trunk_activation: relu
use_bias: true
```

PaddleScience 的 `MLP.num_layers` 是 **隐藏层个数**，其后还有一层无激活的 `last_fc`。再叠加 DeepONet 对 trunk 输出的额外激活，层宽如下。

### 3.1 Branch Net（编码 \(u\)）

```text
u  [B, 100]
  → Linear 100→40 + ReLU     # 1 个隐藏层
  → Linear  40→40            # last_fc，无激活
  → b  [B, 40]
```

### 3.2 Trunk Net（编码 \(y\)）

```text
y  [B, 1]
  → Linear 1→40 + ReLU       # 隐藏层 1
  → Linear 40→40 + ReLU      # 隐藏层 2
  → Linear 40→40             # last_fc，无激活
  → ReLU                     # DeepONet.forward 中 trunk_act（论文：trunk 末层有激活）
  → t  [B, 40]
```

### 3.3 合并

```text
G = sum_k b_k * t_k + b0     # einsum("bi,bi->b") 再 reshape 成 [B, 1]
```

`b0` 形状 `[1]`，初值 0，随训练更新。

数据流示意：

```text
        u[B,100]                    y[B,1]
            │                          │
            ▼                          ▼
      Branch MLP                  Trunk MLP
      100→40→40                   1→40→40→40
            │                          │
            ▼                          ▼
         b[B,40]                   ReLU → t[B,40]
            │                          │
            └──────── inner product ───┘
                          │
                          ▼
                   G = ⟨b, t⟩ + b0
                       [B, 1]
```

参数量约 **9041**（此前同配置前向检查）。无 skip connection、无 weight norm。

---

## 4. 训练设定（当前配置）

| 项 | 值 |
|----|----|
| 损失 | `MSELoss`，监督键 `G` |
| 验证指标 | `L2Rel`（相对 L2） |
| 优化器 | Adam，`lr=1e-3`，**无**学习率衰减 |
| 轮数 | 10000 epoch，每 epoch 1 次全量迭代 |
| 记录 | `log_freq=20`，`eval_freq=500`，`use_tbd=true` |

这是 **数据驱动监督学习**，损失里没有 PDE 残差；物理（积分）只体现在标签的生成方式上。

---

## 5. 这次已跑完的实验（2026-08-21/12-26-44）

| 项 | 结果 |
|----|------|
| 时长 | 约 21 分钟（12:26–12:47） |
| 训练损失 | epoch 1：0.24 → 后期约 \(10^{-5}\) |
| 最优验证 L2Rel | **0.00615**（约 epoch 9500） |
| 最后一轮验证 L2Rel | 0.0105（epoch 10000，差于最优） |
| 建议权重 | `outputs_deeponet/2026-08-21/12-26-44/checkpoints/best_model` |

后期恒定大学习率导致验证指标抖动，**不要用 last checkpoint 当最好模型**。曲线解读见 [TENSORBOARD_ANALYSIS.md](./TENSORBOARD_ANALYSIS.md)。

---

## 6. 怎么用这份模型做一次预测

输入必须同时给整条离散 \(u\) 和查询点 \(y\)：

```python
import numpy as np
import paddle
import ppsci

model = ppsci.arch.DeepONet(
    u_key="u", y_key="y", G_key="G",
    num_loc=100, num_features=40,
    branch_num_layers=1, trunk_num_layers=2,
    branch_hidden_size=40, trunk_hidden_size=40,
    branch_activation="relu", trunk_activation="relu",
    use_bias=True,
)
# solver.load 或 set_state_dict 载入 best_model 后：
x = np.linspace(0, 1, 100, dtype="float32").reshape(1, 100)
u = np.tile(np.cos(x), [200, 1])          # 同一条 u，200 个 y
y = np.linspace(0, 1, 200, dtype="float32").reshape(200, 1)
G = model({"u": paddle.to_tensor(u), "y": paddle.to_tensor(y)})["G"]
# 对 u=cos，解析目标是 sin(y)
```

训练 / 评估入口：`uv run python examples/deeponet/deeponet.py`（仓库根目录）。
