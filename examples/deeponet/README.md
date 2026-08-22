# DeepONet TensorBoardX 使用说明

本说明对应分支 `feat/deeponet-tensorboard`，介绍两个可视化功能：

1. **训练过程监控**：训练时由 TensorBoardX 实时记录 loss、指标、吞吐、权重直方图与预测图。
2. **历史曲线回放**：从已有 `train.log` 解析指标，再写入 TensorBoardX，给「当时没开 TensorBoard」的实验补画曲线。

案例入口：`examples/deeponet/deeponet.py`，配置：`examples/deeponet/conf/deeponet.yaml`。

当前已训练模型的**输入 / 输出 / 结构 / 权重**见 **[MODEL.md](./MODEL.md)**。对 **http://localhost:6006/** 上本次历史 run 的逐图解读见 **[TENSORBOARD_ANALYSIS.md](./TENSORBOARD_ANALYSIS.md)**。

---

## 0. 环境准备

在仓库根目录使用 uv 创建的 `.venv`（已含 PaddleScience）。若尚未安装可视化依赖：

```bash
cd /path/to/PaddleScience
uv pip install tensorboardX tensorboard
```

下文命令均在 **仓库根目录** 执行（与 Hydra `chdir: false`、默认输出目录 `./outputs_deeponet` 一致）。

```bash
uv run python examples/deeponet/deeponet.py ...
```

若已 `source .venv/bin/activate`，可将 `uv run python` 换成 `python`。

---

## 1. 功能一：训练过程监控（实时 TensorBoard）

### 1.1 做什么

训练时 `Solver` 打开 TensorBoardX，把每个 `log_freq` 步的标量和每隔 `eval_freq` 的评估/直方图/预测图写到本次 run 的 `tensorboard/` 目录。浏览器里即可看曲线是否下降、评估是否变好、GPU 是否打满。

默认已在配置中打开：

```yaml
use_tbd: true
log_freq: 20          # 每 20 个 epoch 写一次训练标量（本例 iters_per_epoch=1）
TRAIN:
  eval_freq: 500      # 每 500 epoch 评估，并写直方图与预测图
```

关闭监控：`use_tbd=false`。

### 1.2 启动训练

```bash
uv run python examples/deeponet/deeponet.py
```

等价于 `mode=train`。无现成 npz 时会按论文用 GRF（$l=0.2$，$m=100$）自动生成反导数数据。

终端出现类似提示即表示 TensorBoardX 已启用：

```text
TensorboardX is enabled for logging, you can view it by
running:
tensorboard --logdir outputs_deeponet/<日期>/<时间>/.../tensorboard
```

另开一个终端启动可视化（可盯整个输出树，便于对比多次实验）：

```bash
uv run tensorboard --logdir ./outputs_deeponet --port 6006
```

浏览器打开终端给出的地址（一般为 `http://localhost:6006`）。远程机器请自行做端口转发。

### 1.3 能看到哪些曲线

| TensorBoard 标签 | 含义 | 写入时机 |
|------------------|------|----------|
| `train/loss`、`train/Sup` | 训练 MSE / 约束损失 | 每 `log_freq` |
| `train/lr` | 学习率 | 每 `log_freq` |
| `train/ips`、`train/batch_cost`、`train/eta_sec` | 吞吐、单步耗时、剩余秒数 | 每 `log_freq` |
| `train/gpu_mem_*_mb` | GPU 显存（若可得） | 每 `log_freq` |
| `train/num_params` | 模型参数量 | 每 `eval_freq` |
| `eval/G_eval/loss`、`eval/G_eval/L2Rel.G` | 验证集 MSE、相对 L2 | 每 `eval_freq` |
| `eval/best_metric`、`eval/best_metric_epoch` | 迄今最优指标及对应 epoch | 每 `eval_freq` |
| `eval/sample_cos_L2Rel` | 样本 $u=\cos x$ 的 L2Rel | 每 `eval_freq` |
| `params/*` | 权重直方图 | 每 `eval_freq` |
| `pred/cos_antiderivative` | $\int\cos=\sin$ 预测对比图 | 每 `eval_freq` |
| `config` | 本次 Hydra 配置文本 | 训练开始 |

SCALARS 页看曲线，IMAGES 页看预测图，HISTOGRAMS 页看权重分布。

### 1.4 输出落在哪

```text
outputs_deeponet/<日期>/<时间>/<覆盖项>/
├── train.log
├── tensorboard/          ← 实时监控，用 --logdir 指到这里或上一级
├── checkpoints/
└── visual/               ← 训练结束后 9 组解析函数对比 PNG
```

---

## 2. 功能二：历史训练曲线回放（`mode=plot_history`）

### 2.1 做什么

早期实验若 **没有** 开 `use_tbd`，磁盘上通常只有 `train.log`，TensorBoard 是空的。本功能解析 PaddleScience 的 `train.log`，用 TensorBoardX 补写标量和汇总图，效果与实时监控的 SCALARS 对齐。

实现文件：`examples/deeponet/history_tb.py`。

### 2.2 何时使用

- 某次训练忘了开 TensorBoard，只留下 `train.log`。
- 训练仍在跑，想先根据已有日志看曲线（可反复执行，会覆盖同名 run 的事件文件）。
- 同一 `outputs_deeponet/` 下有多次 Hydra run，想在一个 TensorBoard 里对比。

### 2.3 命令

默认扫描 `./outputs_deeponet` 下所有 `train.log`：

```bash
uv run python examples/deeponet/deeponet.py mode=plot_history
uv run tensorboard --logdir ./outputs_deeponet/tensorboard_history --port 6006
```

只回放某一次 run，或改输出目录：

```bash
uv run python examples/deeponet/deeponet.py mode=plot_history \
  HISTORY.log_path=./outputs_deeponet/2026-08-21/12-26-44/train.log \
  HISTORY.tbd_dir=./outputs_deeponet/tensorboard_history
```

`HISTORY.log_path` 可以是：

- 目录：递归查找其中全部 `train.log`；
- 单个 `train.log` 文件。

### 2.4 配置项（`conf/deeponet.yaml`）

```yaml
HISTORY:
  log_path: ./outputs_deeponet              # 历史日志根目录或文件
  tbd_dir: ./outputs_deeponet/tensorboard_history
```

路径相对 **当前工作目录**。从仓库根目录启动时，与默认 Hydra 输出 `./outputs_deeponet` 一致；若先 `cd examples/deeponet` 再跑，请改成指向实际日志位置，例如 `HISTORY.log_path=../../outputs_deeponet`。

### 2.5 回放后的文件

```text
outputs_deeponet/tensorboard_history/
└── 2026-08-21_12-26-44/     # 由 Hydra run 路径生成的 run 名
    └── events.out.tfevents.*

outputs_deeponet/<日期>/<时间>/.../visual_history/
├── curves_loss.png          # 训练 / 验证 loss（对数纵轴）
├── curves_l2rel.png         # L2Rel 与 best metric
├── curves_lr_ips.png        # 学习率与吞吐
└── curves_time.png          # batch / reader 耗时
```

不打开浏览器时，可直接看 `visual_history/*.png`。

### 2.6 从日志里解析出的曲线

| 标签 | 来源日志 |
|------|----------|
| `train/lr`、`train/loss`、`train/Sup` | `[Train][Epoch i/N] ...` |
| `train/batch_cost`、`train/reader_cost`、`train/ips` | 同上 |
| `eval/G_eval/loss`、`eval/G_eval/L2Rel.G` | `[Eval][Epoch i][Avg] ...` |
| `eval/best_metric` | `[Eval][Epoch i][best metric: ...]` |

横轴为 **epoch**（本案例 `iters_per_epoch=1`，与训练步一致）。日志继续增长时，再跑一次 `plot_history` 即可更新。

---

## 3. 两个功能怎么配合

| 场景 | 建议 |
|------|------|
| 新开训练 | 保持 `use_tbd: true`，用功能一实时盯盘：`--logdir ./outputs_deeponet` |
| 旧实验只有 `train.log` | 用功能二 `mode=plot_history`，再 `--logdir ./outputs_deeponet/tensorboard_history` |
| 训练还在跑、当时没开 TB | 可随时 `plot_history` 看已有进度；**不必停训**。新代码不会热更新到已启动的进程 |
| 对比多次实验 | TensorBoard 的 `--logdir` 指到共同父目录，左侧按 run 勾选 |

实时监控写在各次 run 的 `tensorboard/`；历史回放写在统一的 `tensorboard_history/`。两者目录不同，可用两个 TensorBoard 分别看，或把 `--logdir` 指到 `outputs_deeponet` 一次性都加载。

---

## 4. 常见问题

**Q: 启动 TensorBoard 后 SCALARS 是空的？**  
A: 先确认打开的 `--logdir` 下确有 `events.out.tfevents.*`。实时监控在 `outputs_deeponet/<日期>/.../tensorboard/`；回放在 `outputs_deeponet/tensorboard_history/`。旧进程在改代码前启动则不会写 `tensorboard/`，请用功能二回放。

**Q: `plot_history` 报 `No train.log found`？**  
A: 当前工作目录下没有 `HISTORY.log_path` 指向的日志。在仓库根目录执行，或显式传入实际路径。

**Q: 远程服务器打不开 `localhost:6006`？**  
A: 在本地做 SSH 转发，例如 `ssh -L 6006:127.0.0.1:6006 user@host`，再访问本机 `http://localhost:6006`。

**Q: 回放后曲线点数比 epoch 少？**  
A: 训练日志按 `log_freq`（默认 20）打印，验证按 `eval_freq`（默认 500）打印，点数会少于总 epoch，这是预期行为。

---

## 5. 相关文件

```text
examples/deeponet/
├── README.md              # 本说明
├── deeponet.py            # train / eval / plot_history 入口
├── history_tb.py          # 解析 train.log、写入 TensorBoardX
├── conf/deeponet.yaml     # use_tbd、HISTORY、训练超参
└── paper/1910.03193.pdf   # DeepONet 论文
```
