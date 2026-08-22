# 本次 DeepONet 历史训练解读（TensorBoard :6006）

数据来源：`outputs_deeponet/2026-08-21/12-26-44`（`uv run python examples/deeponet/deeponet.py`，未开实时 TB，后用 `mode=plot_history` 回放）。TensorBoard 启动命令为：

```bash
uv run tensorboard --logdir ./outputs_deeponet/tensorboard_history --port 6006
```

浏览器：<http://localhost:6006/>。左侧 run 名一般为 `2026-08-21_12-26-44`。本页按 **IMAGES 四张汇总图 + SCALARS 各曲线** 说明该看什么、说明什么、不必纠结什么。

实验设定（便于对照曲线）：Unstacked DeepONet，反导数算子，Adam **恒定** `lr=1e-3`，10000 epoch × 每 epoch 1 个全量 batch（10000 样本），`log_freq=20`，`eval_freq=500`。墙钟约 **12:26–12:47（约 21 分钟）**。

---

## 0. 先看这三张（结论可以先记）

| 优先级 | TensorBoard 位置 | 为什么必须看 |
|--------|------------------|--------------|
| **P0** | IMAGES → `curves/l2rel`，或 SCALARS → `eval/G_eval/L2Rel.G` 与 `eval/best_metric` | 这是本任务的**对错标准**（相对 L2）。决定用哪份权重，而不是看最后一个 epoch。 |
| **P0** | IMAGES → `curves/loss`，或 SCALARS → `train/loss` 与 `eval/G_eval/loss` | 看有没有学进去、训练/验证是否同步、后期有没有抖起来。 |
| **P1** | SCALARS → `eval/best_metric` | 橙色「历史最优」与蓝色「当前验证」一旦分开，说明**最后一轮不是最好的**。 |

其余图（学习率、ips、耗时、`train/Sup`）是**工程与超参诊断**，不改变「算子学得好不好」的判断。

**一句话结论**：前 8000 epoch 健康下降；8000 之后验证指标来回跳；**最优不在 epoch 10000，而在 epoch 9500，L2Rel ≈ 0.00615**。部署请用 `checkpoints/best_model`，不要用最后一轮。

---

## 1. IMAGES：四张汇总图分别在说什么

回放脚本把同一组标量画成四张图，写在 Images 页（标签 `curves/*`）。本地 PNG 也在该次 run 的 `visual_history/`。

### 1.1 `curves/loss`（Train / Eval Loss）——必须看

**图上有什么**：蓝线 `train/loss`（每 20 epoch 一个点），橙点 `eval/G_eval/loss`（每 500 epoch）。纵轴对数。

**阶段解读**

| 阶段 | epoch | 现象 | 含义 |
|------|-------|------|------|
| 陡降 | 1 → ~2000 | 训练损失从 **0.24 → ~7×10⁻⁵** | 网络很快抓住了「积分算子」的主结构，优化正常。 |
| 缓降 | 2000 → ~4000 | 训练/验证损失几乎贴在一起，到 ~3×10⁻⁵ | **泛化良好**：验证集与训练集同一分布（同款 GRF），没有「只背训练集」。 |
| 进入噪声地板 | 4000 → 8000 | 训练损失在 10⁻⁵ 量级上下毛刺 | 已接近数值地板；恒定 `lr=1e-3` 相对当前损失偏大，Adam 仍在大步更新。 |
| 后期不稳 | 8000 → 10000 | 验证损失先抬再落再抬 | 与 L2Rel 图同一件事：后期在最优点附近抖动，**不能把 last epoch 当最优**。 |

**该盯的细节**

- 蓝橙是否同趋势：本 run 前 8000 epoch **同趋势**，这是好信号。
- 不要被蓝线后期尖刺吓到：全量 batch 仍会因优化器噪声、日志精度（日志里 loss 常打成 `0.00001`）看起来「跳」。以橙点（验证）为准。
- 若蓝线继续降、橙线持续升，才是典型过拟合；本 run **不是那种形态**，更像 **学习率未衰减导致的后期振荡**。

### 1.2 `curves/l2rel`（Evaluation L2 Relative Error）——必须看，且比 loss 更重要

**图上有什么**：蓝线当前验证 L2Rel，橙线 `best_metric`（只降不升的历史最优）。

L2Rel 是 \(\|G_{\mathrm{pred}}-G_{\mathrm{true}}\|_2 / \|G_{\mathrm{true}}\|_2\)，比 MSE 更接近「这条函数曲线像不像」。官方 DeepONet 文档参考指标约 **0.018**；本 run 最优 **0.00615**，优于该参考（数据/结构略有差别，只能作量级对照）。

**关键数字**

| epoch | 验证 L2Rel | 当时 best |
|-------|------------|-----------|
| 500 | 0.0428 | 0.0428 |
| 2000 | 0.0199 | 0.0199 |
| 5000 | 0.0108 | 0.0108 |
| 8000 | 0.00718 | 0.00718 |
| 8500 | 0.0100 | 0.00718（橙线不再跟蓝线） |
| 9000 | 0.0121 | 0.00718 |
| **9500** | **0.00615** | **0.00615（全程最优）** |
| 10000 | 0.0105 | 0.00615 |

**洞察**

- 0–8000：蓝橙重合，每次评估都在创新高（误差更低）。
- 8000 之后蓝橙分离：当前模型变差，但 `best_model` 仍锁在更好的点。
- 9500 出现一次更好的点（0.00615），10000 又回到 0.0105。说明后期 **没有稳定收敛**，只是在最优点附近随机游走。
- **实操**：推理、画 `visual/func_*.png` 若用的是**最后一轮权重**，曲线会对得上但不一定是最好的那份；要复现最优指标，加载 `best_model`。

### 1.3 `curves/lr_ips`（Learning Rate / Throughput）——辅助，不是精度图

**上：lr**  
整条水平线 **0.001**。没有 scheduler。这能解释 1.1 / 1.2 的后期抖动：损失已经到 10⁻⁵，步长仍按训练初期来。

**下：ips**  
首步约 2.2 万（编译 / 初始化），随后 8–10 万样本/秒，再缓慢掉到约 7 万。这是机器与 DataLoader 计时，**与算子学得好不好无关**。缓慢下降常与 checkpoint、评估、后台负载有关，不必为了 ips 去改网络。

**什么时候才需要回头看这张**：精度曲线异常时，先确认 lr 是否意外变成 0 或突然放大。本 run 没有这个问题。

### 1.4 `curves/time`（Time Cost）——工程图，可扫一眼

`batch_cost` 与 `reader_cost` 几乎重合（约 0.11–0.14 s）。本案例用 `IterableNPZDataset`，每个 epoch 只吐 **一个** 全量 batch，计时上「读数据」占了逐步时间的绝大部分，前向+反传相对很短。

首步 ~0.45 s 是正常热身。若 `reader_cost` 突然到数秒，再查磁盘 / 数据路径；本 run 平稳。

---

## 2. SCALARS：和汇总图重复的曲线怎么点

TensorBoard 左侧 SCALARS 是「一条曲线一个图」，和 IMAGES 是同一份数。建议：

| 标签 | 建议 | 原因 |
|------|------|------|
| `eval/G_eval/L2Rel.G` | **打开** | 主指标 |
| `eval/best_metric` | **打开，并和上一条叠在一起对比** | 看是否该用 last 还是 best |
| `eval/G_eval/loss` | 打开 | 与 L2Rel 交叉验证（二者应同向） |
| `train/loss` | 打开 | 看下降与后期毛刺 |
| `train/Sup` | 可关 | 本例只有一个监督约束，与 `train/loss` **数值相同** |
| `train/lr` | 扫一眼即可 | 已确认恒为 1e-3 |
| `train/ips`、`train/batch_cost`、`train/reader_cost` | 默认折叠 | 运维信息 |

Smoothing：先把 smoothing 滑到 0，看真实毛刺；再开一点 smoothing 看趋势。对数纵轴对 `train/loss` 更合适（IMAGES 里已经用了 semilogy）。

---

## 3. 这次 run 没有出现在 :6006 上的图

当前 TensorBoard 只加载了 **`tensorboard_history`**（历史回放），因此：

- **没有** `params/*` 直方图（那是 `use_tbd=true` 的实时训练才会写）。
- **没有** `pred/cos_antiderivative` 训练中预测图。
- 训练结束时的 9 张解析对照图在磁盘：`outputs_deeponet/2026-08-21/12-26-44/visual/func_0_result.png` 等，不在这次 TB 里。

`func_0`（\(u=\cos x,\ G=\sin x\)）上 pred 与 ref 几乎重合，与 L2Rel ~1% 量级一致：算子已经学到，残差是细偏差而不是学错形状。若某张 `func_*` 系统性偏离，应回头看验证 L2Rel 是否也差，而不是先改 ips。

下次从训练开始就开 `use_tbd: true` 时，:6006 的 `--logdir` 可改为 `./outputs_deeponet`，才能看到直方图和训练中预测图。

---

## 4. 对后续实验的建议（从图里直接读出）

1. **用 `best_model`（约 epoch 9500）** 做评估/展示，不要默认 last。
2. 若还要再训：在 ~4000–6000 之后给 Adam **降学习率**（或 cosine），专门压后期振荡。
3. 8000 以后增益不稳定，再加 2000 epoch 性价比低，除非改 lr / 早停（例如连续 2–3 次 eval 不创新高就停）。
4. `train/Sup` 可从看板里去掉心理负担，本配置下它不是第二条损失。

---

## 5. 和论文设定的对应关系（避免误读）

- 论文 4.1.1 用 50000 次迭代；本配置 10000 epoch × 1 iter，更短。即便如此，验证 L2Rel 已到 10⁻²～10⁻³ 量级，说明 **当前数据与网络宽度对该反导数任务足够**。
- 训练/验证都来自同一 GRF（\(l=0.2\)），「train≈eval」**不能**推广到「任意光滑函数都同样准」；`visual/func_*` 里的多项式 / 指数才是分布外的冒烟检查。
- 日志里 loss 打印到 5 位小数，后期 `0.00001` 会量化成台阶，SCALARS 上的细台阶有一部分是**打印精度**，不是物理上的平台。

---

## 6. 关键文件

```text
outputs_deeponet/2026-08-21/12-26-44/
├── train.log
├── checkpoints/best_model.*     ← 对应 eval/best_metric 最低点
├── visual/func_*.png            ← 训练结束、最后一轮权重的解析对照
└── visual_history/curves_*.png  ← 与 :6006 Images 相同的四张汇总图

outputs_deeponet/tensorboard_history/2026-08-21_12-26-44/
└── events.out.tfevents.*        ← 当前 localhost:6006 读的就是这份
```
