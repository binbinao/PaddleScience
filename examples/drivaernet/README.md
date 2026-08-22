# DrivAerNet：汽车阻力预测

从零跟练：打开本目录 [学习文档.md](学习文档.md)（课表、动手实验、评价指标、YAML 逐项说明）。文档站版本：[流体代理模型入门](../../docs/zh/tutorials/fluid_surrogate_drivaernet.md)。

本案例在 PaddleScience 中用 **DrivAerNet** 点云数据训练代理模型，预测汽车阻力系数 \(C_d\)。目录里提供两条数据驱动路径，都是把「车身几何」映射到标量阻力，**损失里不加 PDE 残差**；物理信息来自 CFD 算好的 \(C_d\) 标签。

| 配置 | 模型 | 几何表示 | 算子类型 |
|------|------|----------|----------|
| `conf/drivaernet.yaml` | **RegDGCNN**（论文原模型） | 不规则点云 `vertices` | 动态图卷积（EdgeConv） |
| `conf/drivaernet_fno.yaml` | **FNOCdNet** | 三视图占有格 `occupancy` | 傅里叶神经算子（TFNO2d） |
| `conf/drivaernet_fno3d.yaml` | **FNOCdNet** + TFNO3d | 三维体素占有格 | 三维傅里叶神经算子（TFNO3d） |

论文与官方文档：[DrivAerNet](https://paddlescience-docs.readthedocs.io/zh-cn/latest/examples/drivaernet)。官方 RegDGCNN 预训练权重测试集 \(R^2 \approx 88.22\%\)。

阻力系数定义为

$$
C_d = \frac{F_d}{\tfrac{1}{2}\rho u_\infty^2 A_{\mathrm{ref}}},
$$

其中 \(F_d\) 为阻力，\(u_\infty\) 为来流速度，\(A_{\mathrm{ref}}\) 为参考迎风面积。代理模型直接从几何预测 \(C_d\)，避免对每个新外形再跑一遍 CFD。

---

## 1. 三种算子方法

三条路径解决同一映射 \(G:\text{几何}\mapsto C_d\)，差别在于如何把三维车身变成网络能处理的函数/图。

### 1.1 RegDGCNN：点云上的动态图算子

论文把 DGCNN（Dynamic Graph CNN）从分类改成回归，得到 **RegDGCNN**：输入是车身表面点云（默认 \(N=5000\) 个点、坐标 \((x,y,z)\)），输出一个标量 \(C_d\)。点云本身没有固定网格，因此用 **k 近邻图** 在特征空间里做局部卷积。

```
点云 (N, 3)
    → 每层按特征空间 k-NN 动态建图（默认 k=40）
    → EdgeConv：对边 (x_i, x_j − x_i) 做共享 MLP，邻域 max 池化
    → 四层 EdgeConv 特征拼接 → 1D 卷积嵌入（emb_dims=512）
    → 全局 max + avg 池化
    → MLP（128→64→32→16→1）→ Cd
```

**EdgeConv。** 对每个点 \(x_i\)，在当前特征空间找 \(k\) 个近邻 \(\mathcal{N}(i)\)，边特征为

$$
h_{ij}=\mathrm{MLP}\big([x_i,\; x_j-x_i]\big),\qquad
x_i'=\max_{j\in\mathcal{N}(i)} h_{ij}.
$$

相对坐标 \(x_j-x_i\) 编码局部形状；图在 **每一层之后重建**（动态图），与静态 GCN 不同。实现见 `ppsci/arch/regdgcnn.py`：`knn` / `get_graph_feature` 拼出形状 `[B, 2C, N, k]` 的边特征，再经 `Conv2D`（通道 256→512→512→1024）沿邻域维 max 池化。四层输出拼接后 `conv5` 得到嵌入，再 `adaptive_max_pool1d` 与 `adaptive_avg_pool1d` 合成全局描述符，最后线性头回归 \(C_d\)。

**为何算「算子」。** 输入是定义在车身流形上的离散函数（点集），输出是该几何对应的气动泛函 \(C_d\)。RegDGCNN 用置换近似不变的图卷积逼近这个几何→标量映射，不必把外形先栅格化或渲染成 2D 图。

**数据与训练（与代码一致）。**

- 数据集：`DrivAerNetDataset` 读 `.paddle_tensor` 点云与 CSV 中的 `Average Cd`；训练时可平移/抖动增强。
- 约束：`SupervisedConstraint` + `MSELoss`。
- 优化：Adam（`lr=0.001`，`weight_decay=1e-4`）+ `ReduceOnPlateau`（patience 20，factor 0.1）。
- 默认：100 epoch，`batch_size=1`，`num_points=5000`，`dropout=0.4`。

入口：

```python
model = ppsci.arch.RegDGCNN(
    input_keys=cfg.MODEL.input_keys,   # ["vertices"]
    label_keys=cfg.MODEL.output_keys,  # ["cd_value"]
    weight_keys=cfg.MODEL.weight_keys,
    args=cfg.MODEL,                    # k / emb_dims / dropout
)
```

`conf/drivaernet.yaml` 中与算子相关的键：

| 键 | 含义 |
|----|------|
| `MODEL.k` | 每层 k-NN 邻域大小，默认 40 |
| `MODEL.emb_dims` | 全局嵌入维数，默认 512 |
| `MODEL.dropout` | MLP 头 dropout，默认 0.4 |
| `TRAIN.num_points` | 每辆车采样点数，默认 5000 |
| `TRAIN.train_fractions` | 使用训练集的比例 |

论文在未见测试集上报告 \(R^2\approx 0.9\)；仓库预训练权重指标为 \(R^2:88.22\%\)。推理约秒级，同规模 CFD 往往数小时。

### 1.2 FNO：规则网格上的傅里叶算子

不规则点云不能直接做 FFT。本案例把点云投成三张二维占有密度图（俯视 \(x\)–\(y\)、侧视 \(x\)–\(z\)、前视 \(y\)–\(z\)），拼成 `[3, H, W]`，再用 `TFNO2dNet` 在频域做全局卷积。每辆车先缩放到单位立方体，让 \(128\times 128\) 网格用来编码外形（约 4 cm/像素）；绝对尺寸作为 3 维特征拼进回归头。傅里叶块使用 GroupNorm，每轴保留 32 个模态；空间上做 mean+max 池化。头网络预测相对训练集 \(C_d\) 均值的残差。\(64\times 64\) 对照的验证 \(R^2\) 峰值略高，但训练末期和最大误差是 128 格更好。

```
点云 (N, 3)
    → 单车对齐三视图占有格 (3, 128, 128) + 相对尺寸 (3,)
    → TFNO2d（lifting → 4 层 Fourier + GroupNorm → projection）
    → mean ⊕ max 池化 ⊕ 尺寸
    → MLP 残差 + Cd 均值
```

与 RegDGCNN 对比：FNO 需要先栅格化，换来规则域上的谱卷积与分辨率外推潜力；RegDGCNN 直接吃点云，保留三维邻接，不必选投影方向。

### 1.3 TFNO3d：三维占有格上的傅里叶算子

三视图会把底盘通道、轮腔叠成一层密度。`drivaernet_fno3d.yaml` 改成把点云投到 \(48^3\) 体素（约 11 cm），再用 `TFNO3dNet` 做三维谱卷积。读出头与 2D 相同：mean⊕max 池化、相对尺寸、残差 \(C_d\)。

```
点云 (N, 3)
    → 单车对齐体素占有格 (1, 48, 48, 48) + 相对尺寸 (3,)
    → TFNO3d（lifting → 4 层 3D Fourier + GroupNorm → projection）
    → mean ⊕ max 池化 ⊕ 尺寸
    → MLP 残差 + Cd 均值
```

```bash
uv run python examples/drivaernet/drivaernet.py --config-name drivaernet_fno3d
```

输出在 `outputs_drivaernet_fno3d/`。T4 上默认 `batch_size=4`、hidden 24、每轴 12 个模态。

---

## 2. 数据准备

官方压缩包与文档一致：

```bash
mkdir -p examples/drivaernet/data
cd examples/drivaernet/data
wget -c https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/data.tar
tar -xvf data.tar
```

解压后应出现：

- `DrivAerNetPlusPlus_Processed_Point_Clouds_100k_paddle/`（`.paddle_tensor` 点云）
- `AeroCoefficients_DrivAerNet_FilteredCorrected.csv`（含 `Average Cd`）
- `subset_dir/{train,val,test}_design_ids.txt`

若只放了 `data.tar`、尚未解压，**FNO 训练脚本**会自动解压（必要时下载）。路径相对 **案例目录** 或当前工作目录均可。RegDGCNN 默认按 YAML 里的 `ARGS.*` 相对路径读同一套 `data/`。

划分与论文一致：约 70% 训练、15% 验证、15% 测试（`subset_dir` 中的 id 列表）。

---

## 3. 运行方法

在仓库根目录、已安装 PaddleScience 的环境中执行（可用 `uv run python` 或已激活的 `.venv`）。

### 3.1 训练 RegDGCNN（论文路径，默认配置）

```bash
uv run python examples/drivaernet/drivaernet.py
```

等价于 `--config-name drivaernet`。输出在 `outputs_drivaernet/<日期>/<时间>/`。TensorBoardX 默认打开：

```bash
uv run tensorboard --logdir ./outputs_drivaernet --port 6006
```

### 3.2 评估 RegDGCNN

默认使用官方预训练权重：

```bash
uv run python examples/drivaernet/drivaernet.py mode=eval
```

或换成自己的检查点：

```bash
uv run python examples/drivaernet/drivaernet.py mode=eval \
  EVAL.pretrained_model_path=outputs_drivaernet/<run>/checkpoints/best_model.pdparams
```

评估指标含 MSE、MAE、Max AE、\(R^2\)。

### 3.3 训练 FNO

```bash
uv run python examples/drivaernet/drivaernet.py --config-name drivaernet_fno
```

日志写到 `outputs_drivaernet_fno/<日期>/<时间>/`。终端出现 `TensorboardX is enabled` 后：

```bash
uv run tensorboard --logdir ./outputs_drivaernet_fno --port 6006
```

关闭监控：`use_tbd=false`。结束后 `visual/` 下有占有格预览与 Cd 散点图。

### 3.4 评估 FNO

```bash
uv run python examples/drivaernet/drivaernet.py --config-name drivaernet_fno \
  mode=eval EVAL.pretrained_model_path=outputs_drivaernet_fno/<run>/checkpoints/best_model.pdparams
```

---

## 4. TensorBoardX 监控（FNO / RegDGCNN 共用）

两条 YAML 均为 **`use_tbd: true`**。`Solver` 把标量写到本次 run 的 `tensorboard/`；每隔 `TRAIN.eval_freq` 再写直方图、最优指标和 Cd 散点。

| 标签 | 含义 | 时机 |
|------|------|------|
| `train/loss`、`train/DrivAerNet_constraint` | 训练 MSE | 每 `log_freq` |
| `train/lr`、`train/ips`、`train/batch_cost` | 学习率、吞吐、耗时 | 每 `log_freq` |
| `eval/DrivAerNet_eval/*` | 验证 loss / MSE / MAE / R2 等 | 每 `eval_freq` |
| `eval/best_metric` | 迄今最优验证指标 | 每 `eval_freq` |
| `params/*` | 权重直方图 | 每 `eval_freq` |
| `pred/cd_scatter` | 真值 vs 预测 Cd | 每 `eval_freq` |
| `pred/occupancy_views` | FNO 三视图占有格（仅 FNO） | 每 `eval_freq` |

依赖：`uv pip install tensorboardX tensorboard`。远程请自行做端口转发。

---

## 5. 常用配置

**RegDGCNN**（`conf/drivaernet.yaml`）：

```bash
uv run python examples/drivaernet/drivaernet.py TRAIN.epochs=20 TRAIN.num_points=2048 MODEL.k=20
```

**FNO**（`conf/drivaernet_fno.yaml`）：

| 键 | 含义 |
|----|------|
| `MODEL.grid_size` | 占有格分辨率，默认 \(128\times 128\) |
| `MODEL.n_modes_height / n_modes_width` | 保留的傅里叶模态数，默认 32 |
| `MODEL.hidden_channels` | FNO 通道宽度，默认 32 |
| `MODEL.norm` | FNO 块归一化，默认 `group_norm` |
| `TRAIN.epochs` | 训练轮数，默认 80 |
| `TRAIN.batch_size` | 批大小 |
| `TRAIN.num_points` | 从点云中抽样的点数（再栅格化），默认 65536 |
| `optimizer.eta_min` | Cosine 下限学习率，默认 \(10^{-4}\) |
| `TRAIN.train_fractions` | 使用训练集的比例 |
| `use_tbd` | TensorBoardX 监控，**两条路径默认 `true`**；关闭用 `use_tbd=false` |

```bash
uv run python examples/drivaernet/drivaernet.py --config-name drivaernet_fno \
  TRAIN.epochs=20 TRAIN.batch_size=4 MODEL.grid_size=32
```

### 3.5 训练 TFNO3d

```bash
uv run python examples/drivaernet/drivaernet.py --config-name drivaernet_fno3d
```

```bash
uv run tensorboard --logdir ./outputs_drivaernet_fno3d --port 6007
```

---

## 6. 文件结构

```text
examples/drivaernet/
├── README.md
├── drivaernet.py          # train / eval；默认 RegDGCNN，--config-name drivaernet_fno 为 FNO
├── fno_model.py           # 占有格转换、OccupancyDataset、FNOCdNet
└── conf/
    ├── drivaernet.yaml       # RegDGCNN
    ├── drivaernet_fno.yaml   # TFNO2d 三视图
    └── drivaernet_fno3d.yaml # TFNO3d 体素
```

核心实现：`ppsci/arch/regdgcnn.py`（EdgeConv / RegDGCNN）、`ppsci/data/dataset/drivaernet_dataset.py`（点云与 \(C_d\) 标签）。

---

## 7. 参考文献

1. Elrefaie M, Dai A, Ahmed F. DrivAerNet: A parametric car dataset for data-driven aerodynamic design and graph-based drag prediction[C]//ASME IDETC, 2024.
2. Wang Y, Sun Y, Liu Z, et al. Dynamic graph CNN for learning on point clouds[J]. ACM TOG, 2019, 38(5): 1-12.
3. Li Z, Kovachki N, Azizzadenesheli K, et al. Fourier Neural Operator for parametric partial differential equations[C]//ICLR, 2021.
