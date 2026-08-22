# 二维稳态热传导方程：FDM / FDM 数据驱动 PINNs / 纯 PINNs

## 1. 项目概述

本案例求解二维稳态热传导问题，并拆分为三种独立运行形式：

| 形式 | 入口文件 | 训练/求解方式 | 作用 |
|------|----------|---------------|------|
| FDM 算法 | `fdm.py` | 五点差分离散 Laplace 方程，直接求解线性方程组 | 生成参考温度场与训练数据 |
| FDM 数据训练 PINNs | `heat_pinn_fdm.py` | 用 FDM 生成的 `(x, y, u)` 作为监督数据训练 MLP | 学习 FDM 解的神经网络代理模型 |
| 纯 PINNs 算法 | `heat_pinn_pure.py` | 只使用 PDE 残差和边界条件约束训练，不使用 FDM 标签 | 物理约束求解器 |

`heat_pinn.py` 保留为原始纯 PINNs 实现，便于兼容旧命令；推荐新实验直接使用 `heat_pinn_pure.py`。

---

## 2. 物理问题

求解正方形区域上的稳态温度场，平板内无热源，满足 Laplace 方程：

$$\nabla^2 T = \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0$$

计算域：

- $x \in [-1, 1]$
- $y \in [-1, 1]$

Dirichlet 边界条件：

| 边界 | 位置 | 温度值 |
|------|------|--------|
| 左边界 | $x = -1$ | $T = 75$ °C |
| 右边界 | $x = +1$ | $T = 0$ °C |
| 下边界 | $y = -1$ | $T = 50$ °C |
| 上边界 | $y = +1$ | $T = 0$ °C |

---

## 3. 三种实现说明

### 3.1 FDM 算法：`fdm.py`

`fdm.py` 提供两类能力：

- `solve(n, l)`：有限差分求解温度场；
- 命令行入口：独立生成 FDM 结果文件和温度云图。

输出包括：

- `fdm_solution.npz`：保存 `x`、`y`、归一化温度 `u`、实际温度 `temperature`；
- `fdm_temperature.png`：FDM 温度场图。

### 3.2 FDM 数据训练 PINNs：`heat_pinn_fdm.py`

该形式先调用 `fdm.build_dataset()` 生成监督训练数据：

$$u = T_{FDM} / 75$$

然后使用 `ppsci.constraint.SupervisedConstraint` 训练 MLP，使网络学习 FDM 温度场映射：

$$ (x, y) \rightarrow u $$

训练完成后会在同一评估网格上与 FDM 结果计算 MSE，并输出对比图。

### 3.3 纯 PINNs 算法：`heat_pinn_pure.py`

该形式不使用 FDM 标签参与训练，只使用：

| 约束类型 | 说明 |
|----------|------|
| PDE 约束 | 内点满足 $\nabla^2 u = 0$ |
| 上边界 | $u = 0$ |
| 下边界 | $u = 50/75$ |
| 左边界 | $u = 1$ |
| 右边界 | $u = 0$ |

FDM 只在训练后作为参考解参与误差评估和可视化对比。

---

## 4. 运行方法

进入案例目录：

```bash
cd examples/heat_pinn
```

### 4.1 仅运行 FDM

```bash
python fdm.py --n 100 --output-dir outputs_heat_pinn/fdm_results
```

### 4.2 用 FDM 结果训练 PINNs

```bash
python heat_pinn_fdm.py mode=train
```

评估、导出和推理：

```bash
python heat_pinn_fdm.py mode=eval EVAL.pretrained_model_path=<模型路径>
python heat_pinn_fdm.py mode=export
python heat_pinn_fdm.py mode=infer
```

### 4.3 训练纯 PINNs

```bash
python heat_pinn_pure.py mode=train
```

评估、导出和推理：

```bash
python heat_pinn_pure.py mode=eval EVAL.pretrained_model_path=<模型路径>
python heat_pinn_pure.py mode=export
python heat_pinn_pure.py mode=infer
```

---

## 5. 配置说明

共享配置文件：`conf/heat_pinn.yaml`

关键新增配置：

```yaml
FDM:
  n: 100        # 评估/对比使用的 FDM 网格数
  length: 1.0   # 计算域半长度，对应 [-1, 1]

TRAIN:
  batch_size: 1024  # FDM 数据训练 PINNs 的监督学习 batch size
  fdm_n: 100        # FDM 数据训练 PINNs 的训练网格数
```

纯 PINNs 仍使用 `TRAIN.iters_per_epoch`、边界权重等原配置；FDM 数据训练 PINNs 会根据监督数据加载器长度自动确定每轮迭代数。

---

## 6. 输出结果

Hydra 默认输出到 `outputs_heat_pinn/<日期>/<时间>/<参数>/`。

| 文件 | 来源 | 说明 |
|------|------|------|
| `fdm_temperature.png` | `fdm.py` | FDM 温度场 |
| `fdm_solution.npz` | `fdm.py` | FDM 坐标、归一化标签和温度场 |
| `fdm_trained_pinn_comparison.png` | `heat_pinn_fdm.py` | FDM 数据训练 PINNs 与 FDM 热力图对比 |
| `fdm_trained_pinn_profiles.png` | `heat_pinn_fdm.py` | FDM 数据训练 PINNs 与 FDM 截面对比 |
| `pinn_fdm_comparison.png` | `heat_pinn_pure.py` | 纯 PINNs 与 FDM 热力图对比 |
| `profiles.png` | `heat_pinn_pure.py` | 纯 PINNs 与 FDM 截面对比 |
| `train.log` / `eval.log` | PINNs 脚本 | 训练或评估日志 |
| `checkpoints/` | PINNs 脚本 | 模型检查点 |

---

## 7. 文件结构

```text
examples/heat_pinn/
├── README.md              # 本说明文档
├── fdm.py                 # FDM 求解器 + 独立运行入口
├── heat_pinn_fdm.py       # 使用 FDM 结果监督训练 PINNs
├── heat_pinn_pure.py      # 纯物理约束 PINNs
├── heat_pinn.py           # 原始纯 PINNs 入口，保留兼容
└── conf/
    └── heat_pinn.yaml     # 共享 Hydra 配置文件
```

---

## 8. 参考文献

1. Raissi M, Perdikaris P, Karniadakis G E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational Physics, 2019, 378: 686-707.
2. 参考实现: https://github.com/314arhaam/heat-pinn
