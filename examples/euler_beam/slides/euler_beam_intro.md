---
marp: true
theme: gaia
size: 16:9
paginate: true
style: |
  section { font-size: 26px; }
  h1 { font-size: 1.6em; }
  h2 { font-size: 1.25em; }
  code { font-size: 0.85em; }
  table { font-size: 0.9em; }
header: "Euler Beam PINN · PaddleScience"
footer: "examples/euler_beam"
---

<!-- _class: lead -->
# 一维欧拉梁 PINN 入门
## 从零跑通到理解在学什么

PaddleScience · `examples/euler_beam`

---

## 学完本页你能做什么

| 能力 | 说明 |
|------|------|
| **跑起来** | 进入目录、一条命令开始训练或评估 |
| **找对文件** | 知道脚本、配置、日志、图各自在哪 |
| **建立直觉** | 梁在求什么、PINN 和传统「差分」差在哪 |
| **继续深入** | 知道官方文档与 README 附录该读哪 |

*不要求*事先会有限元或 Paddle 细节——按步骤复制命令即可。

---

## 先建立直觉：我们在算什么？

- 想象一根 **水平梁**，长度归一化后坐标 $x \in [0,1]$。
- **左端 $x=0$ 固定**（不能动、不能转）；**右端 $x=1$ 自由**。
- 在分布载荷下，梁会 **向下弯**，每一点的 **竖向位移**记为 $u(x)$（挠度）。
- **本案例目标**：用神经网络近似整条曲线 $u(x)$，并满足 **力学方程 + 边界条件**。

---

## 传统方法 vs PINN（一句话）

| 思路 | 含义 |
|------|------|
| **差分 / 有限元** | 把区间剖分成网格，在格点上解线性/非线性方程组 |
| **PINN** | 用一个可微网络 $u_\theta(x)$，在 **随机或规则采样点**上让 **方程残差** 和 **边界误差** 尽量为 0，用梯度下降训练 $\theta$ |

本例 **训练阶段不用**「真解上的 $(x,u)$ 数据」；真解只用来 **训练后** 算误差、画图。

---

## 本目录里有什么？（项目地图）

```
examples/euler_beam/
├── euler_beam.py      ← 入口：train / eval / export / infer
├── README.md          ← 运行说明 + 原理附录
├── conf/
│   └── euler_beam.yaml ← Hydra：超参、模式、路径
└── slides/
    └── euler_beam_intro.md  ← 本幻灯片（Marp）
```

---

## 数学上：一条四阶方程 + 四个边界条件

**域内（$0<x<1$）** 控制方程（与代码默认 `q,D` 一致）：

$$\frac{\mathrm{d}^4 u}{\mathrm{d}x^4} + 1 = 0$$

**边界（合起来唯一确定解）**：

- $x=0$：$u=0,\; u'=0$（固定端）
- $x=1$：$u''=0,\; u'''=0$（自由端）

有 **闭式解**（用于验证，不参与训练损失）：

$$u(x)=-\frac{x^4}{24}+\frac{x^3}{6}-\frac{x^2}{4}$$

---

## PINN 里「方程」出现在哪里？（核心一张图）

```
  输入 x ──►  MLP  ──►  u_θ(x)
                 │
                 ├── 自动求导 ──► u', u'', u'''', …
                 │
                 ▼
        内点：让 u'''' + 1 ≈ 0   (PDE 残差 MSE)
        边界：让 u,u',u'',u''' 满足上述值 (BC MSE)
```

**网络**只负责拟合函数；**物理**全部在 **损失函数** 里体现。

---

## `euler_beam.py` 里训练在干什么（模块视角）

1. **`MLP`**：`x → u`，结构由 `conf/euler_beam.yaml` 里 `MODEL` 决定。
2. **`Biharmonic`**：注册 PDE 残差「$u'''' - q/D$」对应方程名 `biharmonic`。
3. **`InteriorConstraint`**：在 $(0,1)$ 内采样点，最小化 PDE 残差。
4. **`BoundaryConstraint`**：在端点用 `jacobian`/`hessian` 施加四个边界条件。
5. **`GeometryValidator`**：用闭式解算 **L2 相对误差** 等指标。
6. **`VisualizerScatter1D`**：画 **预测 vs 真解** 曲线。

---

## 运行前：你需要什么环境？

1. 已安装 **Python 3** 与 **飞桨 PaddlePaddle**（与机器 CUDA 版本匹配更佳）。
2. 已在 PaddleScience **仓库根目录**按官方说明完成 **可编辑安装**，例如：

```bash
cd /path/to/PaddleScience
pip install -e .
```

*具体版本以仓库 README / 文档为准。*

---

## 第一步：进入案例目录

```bash
cd examples/euler_beam
```

确认当前目录下有 `euler_beam.py` 与 `conf/euler_beam.yaml`。

---

## 第二步：训练（默认模式）

```bash
python euler_beam.py
```

等价于：

```bash
python euler_beam.py mode=train
```

日志与结果默认在 **`outputs_euler_beam/日期/时间/...`**（由 Hydra 配置）。

---

## 想快速看效果：只评估（不下训练）

使用官方预训练权重：

```bash
python euler_beam.py mode=eval \
  EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams
```

会跑验证指标并生成可视化；适合 **先验证环境** 再自己训练。

---

## 导出与推理（了解即可）

| 模式 | 命令 | 作用 |
|------|------|------|
| `export` | `python euler_beam.py mode=export` | 导出静态图推理模型 |
| `infer` | `python euler_beam.py mode=infer` | 用部署预测器跑一批点并画图 |

推理结果图默认前缀：`./euler_beam_pred`（见脚本 `inference` 函数）。

---

## 想改行为？先动配置文件

打开 **`conf/euler_beam.yaml`**，新手最常改：

| 键 | 含义 |
|----|------|
| `TRAIN.epochs` | 训练轮数 |
| `TRAIN.learning_rate` | 学习率 |
| `MODEL.hidden_size` / `num_layers` | 网络大小 |
| `q` / `D` | 载荷与刚度（与方程 $u''''=q/D$ 对应） |

命令行可 **覆盖** 配置，例如：`python euler_beam.py TRAIN.epochs=2000`

---

## 学习路径建议（从 0 到 1）

1. **跑通** `mode=eval` → 再 **训练** 一小轮观察 loss。
2. 打开 **`euler_beam.py`**，对照本页「模块视角」搜 `Constraint` / `Solver`。
3. 阅读 **`README.md` 附录**：解析解对齐、`Biharmonic` 源码含义。
4. 阅读 **官方教程**（逐步对应公式与代码）：  
   [Euler Beam 文档](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/euler_beam)

---

## 结果长什么样？

训练结束后可在输出目录查看 **日志、检查点、一维散点可视化**（预测与 `u_label`）。

文档示意图（需联网加载）：

![预测与真解对比](https://paddle-org.bj.bcebos.com/paddlescience/docs/euler_beam/euler_beam.png)

---

## 小结：记住三句话

1. **PINN = 网络 + 方程/边界残差的损失**，不是单纯拟合数据表。
2. 本例 **PDE** 由 **`Biharmonic` + `InteriorConstraint`** 实现；**边界**由 **`BoundaryConstraint`** 实现。
3. **闭式解**只用于 **验算与画图**；真正学物理的是 **内点 + 边界的损失**。

---

## 如何把本文件变成 PPT / PDF？

本文件为 **[Marp](https://marp.app/)** Markdown。

- **VS Code**：安装「Marp for VS Code」插件 → 打开本文件 → 右上角导出 **HTML / PDF / PPTX**。
- **命令行**（需 Node.js）：

```bash
npx @marp-team/marp-cli euler_beam_intro.md --html euler_beam_intro.html
npx @marp-team/marp-cli euler_beam_intro.md --pdf euler_beam_intro.pdf
# PPTX 需本机安装 LibreOffice 时通常可用：
npx @marp-team/marp-cli euler_beam_intro.md --pptx euler_beam_intro.pptx
```

更多说明见同目录 **`README.md`**。

---

<!-- _class: lead -->
# 谢谢
## 问题建议：先查 `examples/euler_beam/README.md` 附录与官方文档
