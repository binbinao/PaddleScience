# 一维欧拉梁挠度（Euler Beam）PINN 示例

## 1. 项目概述

本案例使用 **物理信息神经网络（PINN）** 在 PaddleScience 中求解一维欧拉梁的静力挠度问题：在区间 $[0,1]$ 上满足四阶控制方程与端部边界条件，网络仅由内域 PDE 残差与边界约束训练，无监督数据参与训练；解析解用于训练后验证与可视化对比。

| 项目 | 说明 |
|------|------|
| 入口脚本 | `euler_beam.py` |
| 配置文件 | `conf/euler_beam.yaml` |
| 控制方程 | `ppsci.equation.Biharmonic`（双调和 / 梁挠度方程形式） |
| 网络结构 | `ppsci.arch.MLP` |
| 入门幻灯片 | `slides/euler_beam_intro.md`（Marp，可导出 PPTX/PDF，见 `slides/README.md`） |

更完整的公式推导与代码逐行说明见官方文档：[Euler Beam](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/euler_beam)。

解析解与 PDE 的对齐关系、PINN 中方程残差出现在何处、以及 `Biharmonic` 类的实现要点，见文末 **[附录](#附录)**。

---

## 2. 物理问题

在 $x \in [0, 1]$ 上，挠度 $u(x)$ 满足（与文档一致）：

$$
\frac{\partial^4 u}{\partial x^4} + 1 = 0
$$

边界条件（固支端在 $x=0$，自由端在 $x=1$）：

- $u(0)=0$，$u'(0)=0$
- $u''(1)=0$，$u'''(1)=0$

用于验证的闭式解（与脚本中 `u_solution_func` 一致）：

$$
u(x) = -\frac{x^4}{24} + \frac{x^3}{6} - \frac{x^2}{4}
$$

代码中 `ppsci.equation.Biharmonic` 实现为 $u'''' = q/D$（一维即 $\mathrm{d}^4 u/\mathrm{d}x^4 = q/D$）。取 `q = -1`、`D = 1` 时即为 $u'''' + 1 = 0$，与上式一致；参数见 `conf/euler_beam.yaml`。

---

## 3. 运行方法

在仓库中已正确安装 Paddle 与 PaddleScience 的前提下，进入本目录：

```bash
cd examples/euler_beam
```

### 3.1 训练

```bash
python euler_beam.py
```

默认 `mode: train`，等价于 `python euler_beam.py mode=train`。

### 3.2 评估

使用官方预训练权重示例：

```bash
python euler_beam.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams
```

也可将 `EVAL.pretrained_model_path` 换成本地训练得到的 `.pdparams` 路径。

### 3.3 导出推理模型

```bash
python euler_beam.py mode=export
```

导出路径由配置项 `INFER.export_path` 控制（默认 `./inference/euler_beam`）。

### 3.4 推理

```bash
python euler_beam.py mode=infer
```

推理脚本会读取 `INFER` 相关配置（如 `pretrained_model_path`、`batch_size` 等），并在当前工作目录生成预测与真值对比图（默认前缀 `./euler_beam_pred`）。

---

## 4. 配置说明（`conf/euler_beam.yaml`）

| 配置块 | 含义 |
|--------|------|
| `mode` | `train` / `eval` / `export` / `infer` |
| `q`, `D` | 梁方程中的载荷与抗弯刚度参数 |
| `MODEL` | MLP 的输入输出键名、层数、隐藏层宽度等 |
| `TRAIN` | 训练轮数、学习率、PDE/边界 batch、保存与评估频率等 |
| `EVAL` | 验证点数量、`pretrained_model_path`（评估用）等 |
| `INFER` | 导出路径、推理设备、batch 等 |

Hydra 输出目录默认形如：`outputs_euler_beam/<日期>/<时间>/<参数覆盖名>/`。

---

## 5. 预训练模型与指标（参考）

| 预训练模型 | 指标（参考） |
|------------|----------------|
| [euler_beam_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams) | loss(L2Rel_Metric): 0.00000；L2Rel.u(L2Rel_Metric): 0.00058 |

---

## 6. 文件结构

```text
examples/euler_beam/
├── README.md              # 本说明
├── euler_beam.py          # 训练 / 评估 / 导出 / 推理入口
├── conf/
│   └── euler_beam.yaml    # Hydra 配置
└── slides/                # 新手入门幻灯片（Marp，可导出 PPTX/PDF）
    ├── README.md          # 预览与导出说明
    └── euler_beam_intro.md
```

---

## 7. 参考文献

1. Raissi M, Perdikaris P, Karniadakis G E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. *Journal of Computational Physics*, 2019, 378: 686-707.

---

## 附录

### A. 解析解与偏微分方程的对齐

控制方程（与正文及官方文档一致）为：

$$
\frac{\mathrm{d}^4 u}{\mathrm{d}x^4} + 1 = 0
\quad\Leftrightarrow\quad
u'''' = -1.
$$

闭式解 $u(x) = -\dfrac{x^4}{24} + \dfrac{x^3}{6} - \dfrac{x^2}{4}$ 逐次求导可得 $u'''' = -1$，故在 $(0,1)$ 内满足上述方程；再代入本例在 $x=0$、$x=1$ 上的边界约束（固支端位移与转角为零，自由端弯矩与剪力为零），可与 `euler_beam.py` 中 `BoundaryConstraint` 的设定一致。因此解析解与 **同一套 PDE + 边值问题** 对齐。

`u_solution_func` 仅在 **验证器（`GeometryValidator`）与可视化** 中作为参考标签使用，**不参与** PDE 残差项的定义与训练损失。

### B. PINN 中偏微分方程体现在何处

**MLP 本身**只是映射 $x \mapsto u_\theta(x)$，网络结构里并不「内嵌」微分方程。

PDE 与边界条件体现在 **约束对应的损失** 中：对网络输出关于输入坐标做自动微分，构造残差并压到零。

1. **域内（PDE 残差）**  
   `InteriorConstraint` 使用 `Biharmonic` 注册的方程（名称为 `"biharmonic"`），目标为残差 0；框架对 $u_\theta$ 关于 $x$ 求至四阶导，与右端 $q/D$ 对齐后形成 MSE 损失。对应代码：

```39:46:examples/euler_beam/euler_beam.py
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["biharmonic"].equations,
        {"biharmonic": 0},
        geom["interval"],
        {**dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.pde},
        ppsci.loss.MSELoss(),
        random="Hammersley",
        name="EQ",
    )
```

2. **边界**  
   四阶方程在有限区间上需配足边界条件；`BoundaryConstraint` 对 $u$、$u_x$、$u_{xx}$、$u_{xxx}$ 在边界上的取值约束为 0，通过 `jacobian` / `hessian` 从 $u_\theta(x)$ 中算出再与目标做 MSE：

```48:61:examples/euler_beam/euler_beam.py
    bc = ppsci.constraint.BoundaryConstraint(
        {
            "u0": lambda d: d["u"][0:1],
            "u__x": lambda d: jacobian(d["u"], d["x"])[1:2],
            "u__x__x": lambda d: hessian(d["u"], d["x"])[2:3],
            "u__x__x__x": lambda d: jacobian(hessian(d["u"], d["x"]), d["x"])[3:4],
        },
        {"u0": 0, "u__x": 0, "u__x__x": 0, "u__x__x__x": 0},
        geom["interval"],
        {**dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc},
        ppsci.loss.MSELoss("sum"),
        evenly=True,
        name="BC",
    )
```

**小结**：PDE 在 **内点约束的损失**里通过高阶自动微分体现；解析解在 **评估与作图** 中体现为与 $u_\theta$ 的对比，而非用监督数据替代物理方程。

### C. `ppsci.equation.Biharmonic` 实现了什么

`Biharmonic` 继承 `base.PDE`，用 **SymPy** 构造一条名为 **`"biharmonic"`** 的标量方程（残差表达式），供求解器在内点上要求其逼近 0。

文档中的连续形式为：

$$
\nabla^4 \varphi = \frac{q}{D},
$$

实现上等价于残差

$$
\text{biharmonic} = \nabla^4 u - \frac{q}{D} = 0
$$

（未知函数在板壳问题中常记为 $\varphi$，本例 Euler 梁中为挠度 $u$。）

源码中先置 $-\,q/D$，再对所有坐标对 $(x_i, x_j)$ 累加 $\partial^4 u / (\partial x_i^2\,\partial x_j^2)$：

```73:78:ppsci/equation/pde/biharmonic.py
        biharmonic = -self.q / self.D
        for invar_i in invars:
            for invar_j in invars:
                biharmonic += u.diff(invar_i, 2).diff(invar_j, 2)

        self.add_equation("biharmonic", biharmonic)
```

- **一维**（`dim=1`）：双重循环仅一项，得到 $\dfrac{\mathrm{d}^4 u}{\mathrm{d}x^4} - \dfrac{q}{D}$，即 $u'''' = q/D$。本例 `q = -1`、`D = 1` 时与 $u'''' + 1 = 0$ 一致。  
- **二维**（`dim=2`）：得到 $u_{xxxx} + 2u_{xxyy} + u_{yyyy} - q/D$，即笛卡尔坐标下标量 **双调和算子** $\nabla^4 u$ 减右端。更高维同理，为各坐标二阶导两两组合之和。

**参数 `q`、`D` 的类型**：

- 数值：常载荷、常刚度；  
- 字符串 `"q"` / `"D"`：将 $q$、$D$ 建成依赖坐标的 SymPy 函数；  
- 其他字符串：经 `sympy.sympify` 解析为符号表达式。

**小结**：`Biharmonic` 不负责训练网络，只 **注册方程名与 SymPy 残差**；训练时框架对网络输出的 $u$ 自动求导，使 `"biharmonic"` 在内点约束中逼近 0，从而实现 $\nabla^4 u = q/D$（一维即 $u'''' = q/D$）。
