"""
I25a 工字梁 PINNs 训练程序

基于 FEM 数据 + Euler-Bernoulli 梁物理约束的混合训练。
输入: (x_norm, P_norm) — 归一化位置和载荷
输出: w_norm — 归一化挠度

物理约束: EI * d⁴w/dx⁴ = q(x)  (简支梁跨中集中力)
数据约束: w_pred ≈ w_fem (FEM计算结果)
边界约束: w(0) = 0, w(L) = 0

模型规模: 3层 × 32神经元, tanh激活, ~2241 参数
"""

import os
import sys
import numpy as np
import paddle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

paddle.set_default_dtype("float64")


# =============================================================================
# 1. 物理常数与归一化
# =============================================================================
# I25a 截面 + Q235B 材料
E = 206000.0        # MPa
Ix = 50170000.0     # mm^4
Wx = 398200.0       # mm^3
L = 3000.0          # mm
EI = E * Ix         # N*mm^2

P_MIN = 1000.0      # N
P_MAX = 5000.0      # N

# 归一化参考值
X_REF = L           # x / L -> [0, 1]
P_REF = P_MAX       # P / P_MAX -> [0.2, 1.0]
# 最大挠度参考: P_MAX * L^3 / (48*EI)
W_REF = P_MAX * L**3 / (48.0 * EI)  # ~0.2721 mm


def load_fem_data(npz_path):
    """加载FEM数据并归一化"""
    data = np.load(npz_path)
    x_nodes = data["x_nodes"]     # (101,)
    loads = data["loads"]         # (9,)
    deflections = data["deflections"]  # (9, 101)

    # 构建训练对 (x, P) -> w
    n_cases, n_nodes = deflections.shape
    x_train = np.zeros((n_cases * n_nodes, 2), dtype="float64")
    w_train = np.zeros((n_cases * n_nodes, 1), dtype="float64")

    for i in range(n_cases):
        for j in range(n_nodes):
            idx = i * n_nodes + j
            x_train[idx, 0] = x_nodes[j] / X_REF       # x_norm
            x_train[idx, 1] = loads[i] / P_REF          # P_norm
            w_train[idx, 0] = deflections[i, j] / W_REF  # w_norm (负值)

    return x_train, w_train, x_nodes, loads, deflections


# =============================================================================
# 2. 网络模型
# =============================================================================
class IBeamPINN(paddle.nn.Layer):
    """
    工字梁PINNs模型: MLP(3层×32, tanh)
    输入: (x_norm, P_norm)  dim=2
    输出: w_raw  dim=1
    通过 output_transform 强制满足边界条件: w(0)=0, w(1)=0
    """

    def __init__(self, num_layers=3, hidden_size=32):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(num_layers):
            layers.append(paddle.nn.Linear(in_dim, hidden_size))
            layers.append(paddle.nn.Tanh())
            in_dim = hidden_size
        layers.append(paddle.nn.Linear(in_dim, 1))
        self.net = paddle.nn.Sequential(*layers)

    def forward(self, x_in):
        """
        x_in: (N, 2) — [x_norm, P_norm]
        """
        w_raw = self.net(x_in)
        x_norm = x_in[:, 0:1]
        P_norm = x_in[:, 1:2]
        # 硬边界: w(0)=0, w(1)=0 => 乘以 x*(1-x)
        # 物理先验: w 与 P 成正比 => 乘以 P_norm
        w = w_raw * x_norm * (1.0 - x_norm) * P_norm
        return w


# =============================================================================
# 3. 物理残差计算
# =============================================================================
def compute_physics_residual(model, x_pde_input):
    """
    计算 Euler-Bernoulli 梁 ODE 残差 (归一化):
      d⁴w_norm / dx_norm⁴ = 0  (在非加载点处)

    x_pde_input: (N, 2) — [x_norm, P_norm], requires_grad
    """
    x_pde_input = x_pde_input.clone()
    x_pde_input.stop_gradient = False

    w = model(x_pde_input)

    ones = paddle.ones_like(w)

    # dw/d(input) -> (N, 2), 取第0列为 dw/dx_norm
    dw_dinp = paddle.grad(w, x_pde_input, grad_outputs=ones,
                          create_graph=True, retain_graph=True)[0]
    dw_dx = dw_dinp[:, 0:1]

    d2w_dinp = paddle.grad(dw_dx, x_pde_input, grad_outputs=paddle.ones_like(dw_dx),
                           create_graph=True, retain_graph=True)[0]
    d2w_dx2 = d2w_dinp[:, 0:1]

    d3w_dinp = paddle.grad(d2w_dx2, x_pde_input, grad_outputs=paddle.ones_like(d2w_dx2),
                           create_graph=True, retain_graph=True)[0]
    d3w_dx3 = d3w_dinp[:, 0:1]

    d4w_dinp = paddle.grad(d3w_dx3, x_pde_input, grad_outputs=paddle.ones_like(d3w_dx3),
                           create_graph=True, retain_graph=True)[0]
    d4w_dx4 = d4w_dinp[:, 0:1]

    return d4w_dx4


# =============================================================================
# 4. 训练循环
# =============================================================================
def train(fem_data_path, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)

    # 加载FEM数据
    x_train, w_train, x_nodes, loads, deflections = load_fem_data(fem_data_path)
    n_total = x_train.shape[0]

    print("=" * 65)
    print("  I25a 工字梁 PINNs 训练 (推荐方案: 3层×32)")
    print("=" * 65)
    print(f"  FEM数据点: {n_total}")
    print(f"  归一化: x/L=[0,1], P/P_max=[{P_MIN/P_REF:.2f},{P_MAX/P_REF:.2f}], w/w_ref~O(1)")
    print(f"  W_REF = {W_REF:.6f} mm")

    # 转 tensor
    data_input = paddle.to_tensor(x_train)  # (N, 2) [x_norm, P_norm]
    w_data = paddle.to_tensor(w_train)

    # 创建模型
    model = IBeamPINN(
        num_layers=config["num_layers"],
        hidden_size=config["hidden_size"],
    )

    # 统计参数量
    n_params = sum(p.numpy().size for p in model.parameters())
    print(f"  模型参数量: {n_params}")
    print(f"  参数/数据比: 1:{n_total / n_params:.1f}")

    # 优化器: Adam + 学习率衰减
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=config["lr"],
        T_max=config["epochs"],
        eta_min=config["lr"] * 0.01,
    )
    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler,
        parameters=model.parameters(),
    )

    # PDE 配点: 在 [0,1]×[P_min/P_max, 1] 域内均匀采样，排除加载点附近
    n_pde = config["n_pde_points"]
    x_pde_np = np.random.rand(n_pde, 1).astype("float64")
    P_pde_np = np.random.uniform(P_MIN / P_REF, 1.0, (n_pde, 1)).astype("float64")
    # 排除 x_norm=0.5 附近 (|x-0.5| < 0.02 的区域，集中力不连续点)
    mask = np.abs(x_pde_np[:, 0] - 0.5) > 0.02
    x_pde_np = x_pde_np[mask]
    P_pde_np = P_pde_np[mask]
    pde_input = paddle.to_tensor(np.hstack([x_pde_np, P_pde_np]))  # (M, 2)

    print(f"  PDE配点数: {len(x_pde_np)} (排除集中力点)")
    print(f"  训练轮数: {config['epochs']}")
    print("=" * 65)

    # 损失权重
    w_data_weight = config["w_data"]
    w_pde_weight = config["w_pde"]

    history = {"epoch": [], "loss": [], "loss_data": [], "loss_pde": []}
    best_loss = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        model.train()

        # --- 数据损失 ---
        w_pred = model(data_input)
        loss_data = paddle.mean((w_pred - w_data) ** 2)

        # --- PDE 残差损失 ---
        residual = compute_physics_residual(model, pde_input)
        loss_pde = paddle.mean(residual ** 2)

        # --- 总损失 ---
        loss = w_data_weight * loss_data + w_pde_weight * loss_pde

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        scheduler.step()

        # 记录
        loss_val = loss.numpy().item()
        loss_data_val = loss_data.numpy().item()
        loss_pde_val = loss_pde.numpy().item()

        if epoch % config["log_freq"] == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:5d}/{config['epochs']} | "
                f"Loss={loss_val:.2e} | "
                f"Data={loss_data_val:.2e} | "
                f"PDE={loss_pde_val:.2e} | "
                f"LR={scheduler.get_lr():.1e}"
            )

        if epoch % 100 == 0:
            history["epoch"].append(epoch)
            history["loss"].append(loss_val)
            history["loss_data"].append(loss_data_val)
            history["loss_pde"].append(loss_pde_val)

        if loss_val < best_loss:
            best_loss = loss_val
            paddle.save(model.state_dict(), os.path.join(output_dir, "best_model.pdparams"))

    # 保存最终模型
    paddle.save(model.state_dict(), os.path.join(output_dir, "final_model.pdparams"))
    print(f"\n  训练完成! 最佳损失: {best_loss:.2e}")

    # =================================================================
    # 5. 评估与可视化
    # =================================================================
    print("\n[评估] 加载最佳模型...")
    model.set_state_dict(paddle.load(os.path.join(output_dir, "best_model.pdparams")))
    model.eval()

    # --- 5a. 在训练载荷上评估 ---
    print("\n  训练工况评估:")
    print(f"  {'工况':<8} {'P(N)':>7} {'FEM δ_max':>12} {'PINN δ_max':>12} {'相对误差':>10}")
    print("  " + "-" * 55)

    all_pinn_deflections = []
    for i, P in enumerate(loads):
        eval_inp = np.hstack([
            x_nodes.reshape(-1, 1) / X_REF,
            np.full((len(x_nodes), 1), P / P_REF),
        ])
        eval_tensor = paddle.to_tensor(eval_inp)
        with paddle.no_grad():
            w_pred_norm = model(eval_tensor).numpy().flatten()
        w_pred_mm = w_pred_norm * W_REF  # 还原为 mm

        all_pinn_deflections.append(w_pred_mm)

        fem_max = np.max(np.abs(deflections[i]))
        pinn_max = np.max(np.abs(w_pred_mm))
        rel_err = abs(pinn_max - fem_max) / fem_max * 100

        print(f"  LC-{i+1:<4} {P:>7.0f} {fem_max:>12.6f} {pinn_max:>12.6f} {rel_err:>9.4f}%")

    all_pinn_deflections = np.array(all_pinn_deflections)

    # --- 5b. 在未见载荷上插值测试 (泛化能力) ---
    print("\n  插值泛化测试 (未见载荷):")
    print(f"  {'P(N)':>7} {'解析 δ_max':>12} {'PINN δ_max':>12} {'相对误差':>10}")
    print("  " + "-" * 45)

    test_loads = [1250, 1750, 2750, 3250, 4250, 4750]
    for P_test in test_loads:
        eval_inp = np.hstack([
            x_nodes.reshape(-1, 1) / X_REF,
            np.full((len(x_nodes), 1), P_test / P_REF),
        ])
        eval_tensor = paddle.to_tensor(eval_inp)
        with paddle.no_grad():
            w_pred_norm = model(eval_tensor).numpy().flatten()
        w_pred_mm = w_pred_norm * W_REF

        analytical_max = P_test * L**3 / (48.0 * EI)
        pinn_max = np.max(np.abs(w_pred_mm))
        rel_err = abs(pinn_max - analytical_max) / analytical_max * 100

        print(f"  {P_test:>7} {analytical_max:>12.6f} {pinn_max:>12.6f} {rel_err:>9.4f}%")

    # =================================================================
    # 6. 可视化
    # =================================================================
    print("\n[可视化] 生成图表...")
    x_m = x_nodes / 1000.0
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(loads)))

    # --- 图1: PINNs vs FEM 挠度对比 ---
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for i, P in enumerate(loads):
        ax = axes[i // 3][i % 3]
        ax.plot(x_m, -deflections[i], "b-", linewidth=2, label="FEM")
        ax.plot(x_m, -all_pinn_deflections[i], "r--", linewidth=2, label="PINNs")
        ax.set_title(f"P = {P:.0f} N", fontsize=11)
        ax.set_xlabel("Position (m)", fontsize=9)
        ax.set_ylabel("Deflection (mm)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("PINNs vs FEM: Deflection Comparison (9 Load Cases)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pinn_vs_fem_all.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] pinn_vs_fem_all.png")

    # --- 图2: 训练损失曲线 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(history["epoch"], history["loss"], "k-", linewidth=2, label="Total Loss")
    ax.semilogy(history["epoch"], history["loss_data"], "b--", linewidth=1.5, label="Data Loss")
    ax.semilogy(history["epoch"], history["loss_pde"], "r--", linewidth=1.5, label="PDE Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("PINNs Training Loss History", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_loss.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] training_loss.png")

    # --- 图3: 插值预测热力图 ---
    n_x = 200
    n_p = 50
    x_grid = np.linspace(0, 1, n_x)
    p_grid = np.linspace(P_MIN / P_REF, 1.0, n_p)
    XX, PP = np.meshgrid(x_grid, p_grid)

    x_flat = paddle.to_tensor(np.hstack([XX.reshape(-1, 1), PP.reshape(-1, 1)]))
    with paddle.no_grad():
        w_flat = model(x_flat).numpy().flatten()
    W_grid = (w_flat * W_REF).reshape(n_p, n_x)

    fig, ax = plt.subplots(figsize=(12, 6))
    c = ax.pcolormesh(x_grid * L / 1000, p_grid * P_REF, -W_grid,
                      shading="auto", cmap="RdYlBu_r")
    plt.colorbar(c, ax=ax, label="Deflection (mm)")
    # 标注训练载荷
    for P in loads:
        ax.axhline(y=P, color="white", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Position along beam (m)", fontsize=12)
    ax.set_ylabel("Applied Load P (N)", fontsize=12)
    ax.set_title("PINNs Predicted Deflection Field w(x, P)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "deflection_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] deflection_heatmap.png")

    # --- 保存预测数据供后续使用 ---
    np.savez(
        os.path.join(output_dir, "pinn_predictions.npz"),
        x_nodes=x_nodes,
        loads=loads,
        fem_deflections=deflections,
        pinn_deflections=all_pinn_deflections,
        W_REF=W_REF,
        X_REF=X_REF,
        P_REF=P_REF,
    )
    print(f"  [NPZ] pinn_predictions.npz")

    print("\n" + "=" * 65)
    print(f"  输出目录: {os.path.abspath(output_dir)}")
    print("=" * 65)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fem_data_path = os.path.join(script_dir, "outputs", "fem_results", "fem_training_data.npz")
    output_dir = os.path.join(script_dir, "outputs", "pinn_results")

    config = {
        # 模型
        "num_layers": 3,
        "hidden_size": 32,
        # 训练
        "epochs": 15000,
        "lr": 1e-3,
        "log_freq": 1500,
        # 配点
        "n_pde_points": 2000,
        # 损失权重
        "w_data": 1.0,
        "w_pde": 0.01,
    }

    train(fem_data_path, output_dir, config)
