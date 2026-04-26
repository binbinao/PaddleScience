"""
I25a 工字梁弹塑性 PINNs 训练

载荷 10~250 kN, 覆盖线弹性 + 屈服平台 + 应变强化。
双输出: 挠度 + 应力比。网络规模适度增大以拟合非线性段。
"""

import os
import numpy as np
import paddle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

paddle.set_default_dtype("float64")

E = 206000.0; Ix = 50170000.0; Wx = 398200.0; L = 3000.0
SIGMA_Y = 235.0; SIGMA_U = 420.0; EI = E * Ix
P_MAX = 250000.0; P_MIN = 10000.0
X_REF = L; P_REF = P_MAX
# W_REF 取弹塑性最大载荷下的线弹性参考 (实际塑性更大)
W_REF = P_MAX * L**3 / (48.0 * EI)


def load_fem_data(npz_path):
    data = np.load(npz_path)
    x_nodes = data["x_nodes"]
    loads = data["loads"]
    deflections = data["deflections"]
    stress_ratios = data["stress_ratios"]
    n_cases, n_nodes = deflections.shape

    x_train = np.zeros((n_cases * n_nodes, 2), dtype="float64")
    w_train = np.zeros((n_cases * n_nodes, 1), dtype="float64")
    sr_train = np.zeros((n_cases * n_nodes, 1), dtype="float64")

    for i in range(n_cases):
        for j in range(n_nodes):
            idx = i * n_nodes + j
            x_train[idx, 0] = x_nodes[j] / X_REF
            x_train[idx, 1] = loads[i] / P_REF
            w_train[idx, 0] = deflections[i, j] / W_REF
            sr_train[idx, 0] = stress_ratios[i, j]

    return x_train, w_train, sr_train, x_nodes, loads, deflections, stress_ratios


class BucklingPINN(paddle.nn.Layer):
    """双输出 PINNs: 挠度 + 应力比, 5层×64 适配非线性"""
    def __init__(self, num_layers=5, hidden_size=64):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(num_layers):
            layers.append(paddle.nn.Linear(in_dim, hidden_size))
            layers.append(paddle.nn.Tanh())
            in_dim = hidden_size
        self.backbone = paddle.nn.Sequential(*layers)
        self.head_w = paddle.nn.Linear(hidden_size, 1)
        self.head_sr = paddle.nn.Linear(hidden_size, 1)

    def forward(self, x_in):
        feat = self.backbone(x_in)
        x_norm = x_in[:, 0:1]
        P_norm = x_in[:, 1:2]
        # 挠度: 硬边界, 不再乘 P_norm (非线性下 w 不与 P 成正比)
        w_raw = self.head_w(feat)
        w = w_raw * x_norm * (1.0 - x_norm)
        # 应力比: softplus 保证非负, 上限不设硬约束
        sr_raw = self.head_sr(feat)
        sr = paddle.nn.functional.softplus(sr_raw)
        return w, sr


def train(fem_path, out_dir, cfg):
    os.makedirs(out_dir, exist_ok=True)
    x_train, w_train, sr_train, x_nodes, loads, deflections, stress_ratios = load_fem_data(fem_path)
    n_total = x_train.shape[0]

    print("=" * 70)
    print("  I25a 弹塑性 PINNs 训练 (10~250 kN)")
    print("=" * 70)
    print(f"  数据点: {n_total}, W_REF={W_REF:.4f} mm")

    data_input = paddle.to_tensor(x_train)
    w_data = paddle.to_tensor(w_train)
    sr_data = paddle.to_tensor(sr_train)

    model = BucklingPINN(num_layers=cfg["num_layers"], hidden_size=cfg["hidden_size"])
    n_params = sum(p.numpy().size for p in model.parameters())
    print(f"  模型: {cfg['num_layers']}层×{cfg['hidden_size']}, 参数量: {n_params}")

    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=cfg["lr"], T_max=cfg["epochs"], eta_min=cfg["lr"]*0.01)
    optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
    print(f"  训练轮数: {cfg['epochs']}")
    print("=" * 70)

    history = {"epoch": [], "loss": [], "l_w": [], "l_sr": []}
    best_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        w_pred, sr_pred = model(data_input)
        loss_w = paddle.mean((w_pred - w_data)**2)
        loss_sr = paddle.mean((sr_pred - sr_data)**2)
        loss = cfg["wt_w"] * loss_w + cfg["wt_sr"] * loss_sr

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        scheduler.step()

        lv = loss.numpy().item()
        if epoch % cfg["log_freq"] == 0 or epoch == 1:
            print(f"  Epoch {epoch:5d}/{cfg['epochs']} | Loss={lv:.2e} | "
                  f"W={loss_w.numpy().item():.2e} | SR={loss_sr.numpy().item():.2e}")
        if epoch % 200 == 0:
            history["epoch"].append(epoch)
            history["loss"].append(lv)
            history["l_w"].append(loss_w.numpy().item())
            history["l_sr"].append(loss_sr.numpy().item())
        if lv < best_loss:
            best_loss = lv
            paddle.save(model.state_dict(), os.path.join(out_dir, "best_model.pdparams"))

    paddle.save(model.state_dict(), os.path.join(out_dir, "final_model.pdparams"))
    print(f"\n  训练完成! 最佳损失: {best_loss:.2e}")

    # ============ 评估 ============
    model.set_state_dict(paddle.load(os.path.join(out_dir, "best_model.pdparams")))
    model.eval()

    print(f"\n  {'工况':<7} {'P(kN)':>6} {'FEM δ':>10} {'PINN δ':>10} {'δ误差':>8} "
          f"{'FEM σ/σy':>9} {'PINN σ/σy':>10} {'σ误差':>8} {'区域':>6}")
    print("  " + "-" * 80)

    all_pw = []; all_psr = []
    for i, P in enumerate(loads):
        inp = np.hstack([x_nodes.reshape(-1,1)/X_REF, np.full((len(x_nodes),1), P/P_REF)])
        with paddle.no_grad():
            wp, srp = model(paddle.to_tensor(inp))
        wp = wp.numpy().flatten() * W_REF
        srp = srp.numpy().flatten()
        all_pw.append(wp); all_psr.append(srp)

        fd = np.max(np.abs(deflections[i]))
        pd_ = np.max(np.abs(wp))
        fsr = np.max(stress_ratios[i])
        psr = np.max(srp)
        ed = abs(pd_ - fd)/fd*100 if fd > 0 else 0
        esr = abs(psr - fsr)/fsr*100 if fsr > 0 else 0
        zone = "弹性" if fsr < 1.0 else "塑性"
        print(f"  LC-{i+1:02d} {P/1000:>6.0f} {fd:>10.4f} {pd_:>10.4f} {ed:>7.2f}% "
              f"{fsr*100:>8.1f}% {psr*100:>9.1f}% {esr:>7.2f}% {zone:>6}")

    all_pw = np.array(all_pw); all_psr = np.array(all_psr)

    # ============ 可视化 ============
    print("\n  生成图表...")
    P_kN = loads / 1000
    colors = plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, len(loads)))

    # 图1: 载荷-挠度 (FEM vs PINNs, 含非线性分叉)
    fig, ax = plt.subplots(figsize=(12, 7))
    fem_maxd = np.max(np.abs(all_pw), axis=1)  # PINNs
    fem_maxd_real = np.max(np.abs(deflections), axis=1)  # FEM
    elastic_ref = loads * L**3 / (48*E*Ix)

    ax.plot(P_kN, elastic_ref, "b--", lw=2, label="Linear Elastic (ref)", zorder=2)
    ax.plot(P_kN, fem_maxd_real, "ko-", lw=2.5, ms=5, label="FEM (Elastic-Plastic)", zorder=3)
    ax.plot(P_kN, fem_maxd, "r^--", lw=2, ms=5, label="PINNs prediction", zorder=4)

    P_yield_theory = 4 * SIGMA_Y * Wx / L
    ax.axvline(x=P_yield_theory/1000, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.annotate(f"Yield: {P_yield_theory/1000:.1f} kN", xy=(P_yield_theory/1000, 5),
                fontsize=10, color="gray", rotation=90, va="bottom")
    ax.set_xlabel("Load (kN)", fontsize=13)
    ax.set_ylabel("Max Deflection (mm)", fontsize=13)
    ax.set_title("Load-Deflection: FEM vs PINNs (Elastic + Plastic)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "load_deflection_comparison.png"), dpi=150)
    plt.close(fig)

    # 图2: 应力比对比
    fig, ax = plt.subplots(figsize=(12, 7))
    fem_maxsr = np.max(stress_ratios, axis=1)*100
    pinn_maxsr = np.max(all_psr, axis=1)*100
    ax.plot(P_kN, fem_maxsr, "ko-", lw=2.5, ms=5, label="FEM")
    ax.plot(P_kN, pinn_maxsr, "r^--", lw=2, ms=5, label="PINNs")
    ax.axhline(y=100, color="red", ls="--", lw=2, alpha=0.5, label="Yield (100%)")
    ax.axhline(y=SIGMA_U/SIGMA_Y*100, color="darkred", ls=":", lw=1.5, alpha=0.5,
               label=f"Ultimate ({SIGMA_U/SIGMA_Y*100:.0f}%)")
    ax.set_xlabel("Load (kN)", fontsize=13)
    ax.set_ylabel("Max Stress Ratio σ/σ_y (%)", fontsize=13)
    ax.set_title("Stress Ratio: FEM vs PINNs", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stress_ratio_comparison.png"), dpi=150)
    plt.close(fig)

    # 图3: 选代表工况挠度曲线对比
    rep = [0, len(loads)//4, len(loads)//2, 3*len(loads)//4, len(loads)-1]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    for idx, ci in enumerate(rep):
        ax = axes[idx]
        x_m = x_nodes / 1000
        ax.plot(x_m, -deflections[ci], "b-", lw=2, label="FEM")
        ax.plot(x_m, -all_pw[ci], "r--", lw=2, label="PINNs")
        sr = stress_ratios[ci].max()
        zone = "Elastic" if sr < 1.0 else "Plastic"
        ax.set_title(f"P={loads[ci]/1000:.0f}kN\nσ/σy={sr*100:.0f}% [{zone}]", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("m")
    fig.suptitle("FEM vs PINNs: Representative Load Cases", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "representative_cases.png"), dpi=150)
    plt.close(fig)

    # 图4: 训练损失
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(history["epoch"], history["loss"], "k-", lw=2, label="Total")
    ax.semilogy(history["epoch"], history["l_w"], "b--", lw=1.5, label="Deflection")
    ax.semilogy(history["epoch"], history["l_sr"], "r--", lw=1.5, label="Stress Ratio")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_loss.png"), dpi=150)
    plt.close(fig)

    np.savez(os.path.join(out_dir, "pinn_predictions.npz"),
             x_nodes=x_nodes, loads=loads,
             fem_deflections=deflections, pinn_deflections=all_pw,
             fem_stress_ratios=stress_ratios, pinn_stress_ratios=all_psr)

    print("  [PNG] load_deflection_comparison.png")
    print("  [PNG] stress_ratio_comparison.png")
    print("  [PNG] representative_cases.png")
    print("  [PNG] training_loss.png")
    print("=" * 70)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fem_path = os.path.join(script_dir, "outputs/fem_results/fem_training_data.npz")
    out_dir = os.path.join(script_dir, "outputs/pinn_results")

    config = {
        "num_layers": 5, "hidden_size": 64,
        "epochs": 20000, "lr": 1e-3, "log_freq": 2000,
        "wt_w": 1.0, "wt_sr": 0.5,
    }
    train(fem_path, out_dir, config)
