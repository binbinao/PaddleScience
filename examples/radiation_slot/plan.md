# 雷达矩形波导辐射槽电磁仿真系统 — 项目计划

## 1. 项目概述

本项目基于 PaddleScience 框架，实现了一套完整的**雷达矩形波导辐射槽电磁仿真系统**。采用传统 FDFD（有限差分频域）方法与 PINN（物理信息神经网络）方法相结合的混合求解策略，对 Ku 波段（12–18 GHz）矩形波导辐射槽的电磁场分布、辐射方向图和 S 参数进行分析。

### 1.1 技术背景

- **物理问题**：矩形波导（WR-90 标准）中辐射槽天线的电磁场求解
- **控制方程**：含 PML 吸收边界条件的 Helmholtz 方程
- **激励模式**：TE₁₀ 主模激励
- **频段范围**：Ku 波段 12–18 GHz，中心频率 15 GHz
- **关键参数**：波导宽度 a = 1.02 cm，高度 b = 0.51 cm，槽长 2.0 cm

### 1.2 核心目标

1. 实现高精度 FDFD 数值求解器作为参考解
2. 基于 hPINNs 架构实现 PINN 求解器进行快速推理
3. 提供完整的后处理和可视化工具链
4. 实现 FDFD 与 PINN 结果的多指标对比验证

---

## 2. 项目架构

```
examples/radiation_slot/
├── __init__.py              # 包初始化，导出所有公共接口
├── main.py                  # 主入口脚本，支持 train/eval/infer/full 模式
├── geometry.py              # 几何建模模块
├── fdfd_solver.py           # FDFD 求解器
├── functions.py             # 自定义函数（PML、损失函数、输入/输出变换）
├── pinn_solver.py           # PINN 求解器（基于 hPINNs 架构）
├── postprocess.py           # 后处理模块
├── comparator.py            # 对比验证模块
├── conf/
│   └── radiation_slot.yaml  # Hydra 配置文件
└── plan.md                  # 本文档
```

---

## 3. 模块详细设计

### 3.1 geometry.py — 几何建模模块

**核心类**：`RadiationSlotGeometry`

| 功能 | 说明 |
|------|------|
| 波导几何定义 | 矩形波导 + 辐射槽 + PML 区域的完整几何域 |
| 结构化网格生成 | `generate_mesh()` — FDFD 用均匀网格 |
| PML 网格标记 | `generate_pml_mesh()` — 标记 PML 区域 |
| 随机采样点 | `generate_random_samples()` — PINN 训练用内部采样 |
| 边界采样点 | `generate_boundary_samples()` — PEC、端口等边界采样 |

**关键设计**：
- 支持 PML 厚度、网格尺寸等参数化配置
- 默认网格尺寸 λ/10（15 GHz 时约 0.02 cm）
- 采用分区域标记策略区分波导内部、槽口和自由空间

### 3.2 fdfd_solver.py — FDFD 求解器

**核心类**：`FDFDSolver`、`TE10Excitation`

| 功能 | 说明 |
|------|------|
| Helmholtz 方程离散化 | 五点差分格式 |
| PEC 边界条件 | 波导壁完美电导体边界 |
| TE₁₀ 模激励 | `TE10Excitation` 类实现端口激励 |
| PML 吸收边界 | 多项式渐变复数坐标拉伸 |
| 稀疏矩阵求解 | scipy.sparse 高效求解大规模线性系统 |

**关键参数**：
- 截止频率 f_c = c / (2a) ≈ 14.67 GHz
- PML 多项式阶数 m = 3
- PML 最大损耗 σ_max 根据反射系数自动计算

### 3.3 functions.py — 自定义函数模块

| 函数 | 说明 |
|------|------|
| `pml_sigma_x/y()` | PML x/y 方向损耗系数（多项式渐变） |
| `complex_pml_stretch()` | 复数坐标拉伸系数计算 |
| `helmholtz_residual_loss()` | Helmholtz 方程残差损失 |
| `pec_boundary_loss()` | PEC 边界条件损失 |
| `port_excitation_loss()` | TE₁₀ 端口激励匹配损失 |
| `pml_absorption_loss()` | PML 区域吸收条件损失 |
| `input_transform()` | 输入归一化变换 |
| `output_transform()` | 输出变换（施加边界约束） |

### 3.4 pinn_solver.py — PINN 求解器

**核心类**：`PINNSolver`

基于 hPINNs（Hard-constraint PINN）架构设计：
- **三网络结构**：分别预测电场实部 `e_re`、虚部 `e_im` 和介电常数 `eps`
- **硬约束嵌入**：通过 `output_transform` 自动满足 PEC 边界条件
- **训练模式**：支持 soft constraint、penalty method、augmented Lagrangian 三种模式
- **优化策略**：Adam + L-BFGS 组合优化

### 3.5 postprocess.py — 后处理模块

| 类 | 功能 |
|----|------|
| `FieldVisualizer` | 电磁场分布 VTU/PNG 可视化（幅值、相位、实部/虚部） |
| `RadiationPatternCalculator` | 远场辐射方向图计算（近场到远场变换） |
| `SParameterCalculator` | S 参数计算（S₁₁ 反射系数、VSWR） |
| `DataExporter` | 数据导出（VTU、CSV、NumPy 格式） |

**辐射方向图算法**：
- 基于近场等效面电流 → 远场积分变换
- 支持 E 面和 H 面二维方向图
- 输出增益（dBi）和归一化方向图

### 3.6 comparator.py — 对比验证模块

| 类 | 功能 |
|----|------|
| `ComparisonMetrics` | 定义对比指标数据结构 |
| `SolutionComparator` | 计算 L2 相对误差、最大误差、相关系数、SSIM |
| `ComparisonVisualizer` | 误差分布热力图、逐点对比图、散点图 |
| `ComparisonReport` | 自动生成 Markdown 格式对比报告 |
| `quick_compare()` | 一键快速对比函数 |

**对比指标**：
- L2 相对误差：`||u_pinn - u_fdfd||₂ / ||u_fdfd||₂`
- L∞ 最大误差：`max|u_pinn - u_fdfd|`
- Pearson 相关系数
- 结构相似性指数（SSIM）
- 收敛性研究：不同网格密度下的误差变化

### 3.7 main.py — 主入口脚本

支持四种运行模式：

| 模式 | 说明 |
|------|------|
| `train` | 仅训练 PINN 模型 |
| `eval` | 仅评估已训练模型 |
| `infer` | 仅运行推理 |
| `full` | 完整流程：FDFD → PINN 训练 → 对比验证 → 后处理 |

支持频段扫描功能，可自定义起止频率和采样点数。

---

## 4. 配置系统

使用 Hydra 配置管理框架，配置文件位于 `conf/radiation_slot.yaml`。

### 关键配置项

| 配置组 | 说明 |
|--------|------|
| `GEOMETRY` | 波导尺寸、槽参数、PML 厚度、网格大小 |
| `SIMULATION` | 频率设置、频段扫描范围 |
| `MODEL` | 三个神经网络（re_net、im_net、eps_net）的层数、宽度、激活函数 |
| `TRAIN` | 训练轮数、学习率、采样点数、训练模式（soft/penalty/aug_lag） |
| `EVAL` / `INFER` | 评估/推理相关设置 |
| `POSTPROCESS` | 可视化开关、导出格式、对比设置 |

---

## 5. 使用方法

### 5.1 完整仿真流程

```bash
cd examples/radiation_slot
python main.py --mode full
```

### 5.2 仅运行 FDFD 仿真

```bash
python main.py --mode fdfd
```

### 5.3 仅训练 PINN 模型

```bash
python main.py --mode train
```

### 5.4 频段扫描（12–18 GHz）

```bash
python main.py --mode full --frequency-start 12 --frequency-stop 18 --num-frequency-points 13
```

### 5.5 覆盖配置参数

```bash
python main.py TRAIN.epochs=50000 TRAIN.learning_rate=0.0005 MODEL.re_net.hidden_size=128
```

---

## 6. 开发计划与进度

### 阶段一：基础框架搭建 ✅

- [x] 几何建模模块 `geometry.py`
- [x] FDFD 求解器 `fdfd_solver.py`
- [x] 自定义函数模块 `functions.py`

### 阶段二：PINN 求解器 ✅

- [x] PINN 求解器 `pinn_solver.py`（基于 hPINNs 架构）
- [x] 硬约束输出变换
- [x] 多训练模式支持（soft/penalty/augmented Lagrangian）

### 阶段三：后处理与验证 ✅

- [x] 后处理模块 `postprocess.py`（电磁场可视化、辐射方向图、S 参数）
- [x] 对比验证模块 `comparator.py`（多指标对比、误差分析）

### 阶段四：集成与配置 ✅

- [x] 主入口脚本 `main.py`
- [x] Hydra 配置文件 `conf/radiation_slot.yaml`
- [x] 包初始化 `__init__.py`

### 阶段五：测试与优化（待进行）

- [ ] 单元测试：各模块的独立功能验证
- [ ] 集成测试：端到端完整流程测试
- [ ] 性能优化：大规模网格求解加速
- [ ] 文档完善：API 文档和使用教程

---

## 7. 依赖项

| 依赖 | 版本要求 | 用途 |
|------|----------|------|
| PaddlePaddle | >= 2.5 | 深度学习框架 |
| PaddleScience (ppsci) | latest | 科学计算 PINN 框架 |
| NumPy | >= 1.20 | 数值计算 |
| SciPy | >= 1.7 | 稀疏矩阵求解 |
| Matplotlib | >= 3.4 | 可视化绑图 |
| Hydra-core | >= 1.1 | 配置管理 |
| meshio | >= 5.0 | VTU 文件导出（可选） |

---

## 8. 参考文献

1. Lu, L., et al. "Physics-informed neural networks with hard constraints for inverse design." *SIAM Journal on Scientific Computing* (2021).
2. Raissi, M., Perdikaris, P., & Karniadakis, G.E. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics* (2019).
3. Berenger, J.P. "A perfectly matched layer for the absorption of electromagnetic waves." *Journal of Computational Physics* (1994).
4. Balanis, C.A. *Advanced Engineering Electromagnetics*, 2nd ed. Wiley (2012).
5. PaddleScience hPINNs 示例：`examples/hpinns/`
