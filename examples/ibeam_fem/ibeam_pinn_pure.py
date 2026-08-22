"""
I25a 工字梁 纯物理驱动 PINNs 训练程序（无 FEM 数据依赖）

与 ibeam_pinn_train.py 的关键区别：
1. 无数据损失 —— 不使用任何 FEM 计算结果作为训练标签
2. 载荷编码   —— 通过高斯光滑化将跨中集中力引入 PDE 约束
3. 弯矩边界   —— 显式约束支座处弯矩为零 (d²w/dx² = 0)

物理约束：
  PDE:    d⁴w_norm/dξ⁴ = 48 · P_norm · G(ξ; σ)   (含高斯载荷)
  弯矩BC: d²w_norm/dξ² = 0   at ξ = 0, 1          (简支梁支座无弯矩)
  位移BC: w(0) = w(1) = 0                          (硬约束 via output_transform)

其中 G(ξ; σ) = 1/(σ√2π) · exp(-(ξ-0.5)²/(2σ²)) 是跨中集中力的高斯光滑近似，
积分 ∫G dξ = 1，当 σ→0 时趋近 Dirac-δ 分布。

输入: (x_norm, P_norm)  输出: w_norm
模型规模: 4层 × 64神经元, tanh激活, ~8385 参数

参考验证: Euler-Bernoulli 梁解析解
"""

import os
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
E = 206000.0  # MPa
Ix = 50170000.0  # mm^4
Wx = 398200.0  # mm^3
L = 3000.0  # mm
EI = E * Ix  # N*mm^2

P_MIN = 1000.0  # N
P_MAX = 5000.0  # N

# 归一化参考值
X_REF = L  # x / L -> [0, 1]
P_REF = P_MAX  # P / P_MAX -> [0.2, 1.0]
W_REF = P_MAX * L**3 / (48.0 * EI)  # ~0.2721 mm

# 高斯光滑化参数 (σ/L)
# σ = 60mm ≈ 2% 梁长，足够窄以近似点载荷，足够宽以避免数值困难
SIGMA_NORM = 0.02


# =============================================================================
# 2. 网络模型
# =============================================================================
class IBeamPINNPure(paddle.nn.Layer):
    """
    纯物理驱动 PINN: MLP + output_transform

    输入: (x_norm, P_norm)  dim=2
    输出: w_norm  dim=1

    output_transform: w = w_raw * x_norm * (1 - x_norm) * P_norm
      - 硬约束 w(0) = 0, w(1) = 0  (位移边界条件自动满足)
      - 物理先验: w 与 P 成正比    (线性弹性范围内)
    """

    def __init__(self, num_layers=4, hidden_size=64):
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
# 3. PDE 残差计算 (含高斯载荷)
# =============================================================================
def compute_pde_residual(model, x_pde_input, sigma_norm):
    """
    计算 PDE 残差: d⁴w_norm/dξ⁴ - 48 · P_norm · G(ξ) = 0

    物理含义:
      EI · d⁴w/dx⁴ = q(x)  (高斯近似跨中集中力)

    归一化推导:
      d⁴w/dx⁴ = (W_REF/L⁴) · d⁴w_norm/dξ⁴
      q(x) = (P/(L·σ̂·√2π)) · exp(-(ξ-0.5)²/(2σ̂²))
      代入: d⁴w_norm/dξ⁴ = (P·L³)/(EI·W_REF) · G(ξ)
                = 48 · P_norm · G(ξ)    [因为 W_REF = P_MAX·L³/(48·EI)]

    Args:
        model: PINN 模型
        x_pde_input: (N, 2) — [x_norm, P_norm], requires_grad
        sigma_norm: σ/L, 高斯宽度

    Returns:
        residual: (N, 1) PDE 残差
    """
    x_pde_input = x_pde_input.clone()
    x_pde_input.stop_gradient = False

    w = model(x_pde_input)
    ones = paddle.ones_like(w)

    # 1st derivative
    dw_dinp = paddle.grad(
        w, x_pde_input, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]
    dw_dx = dw_dinp[:, 0:1]

    # 2nd derivative
    d2w_dinp = paddle.grad(
        dw_dx,
        x_pde_input,
        grad_outputs=paddle.ones_like(dw_dx),
        create_graph=True,
        retain_graph=True,
    )[0]
    d2w_dx2 = d2w_dinp[:, 0:1]

    # 3rd derivative
    d3w_dinp = paddle.grad(
        d2w_dx2,
        x_pde_input,
        grad_outputs=paddle.ones_like(d2w_dx2),
        create_graph=True,
        retain_graph=True,
    )[0]
    d3w_dx3 = d3w_dinp[:, 0:1]

    # 4th derivative
    d4w_dinp = paddle.grad(
        d3w_dx3,
        x_pde_input,
        grad_outputs=paddle.ones_like(d3w_dx3),
        create_graph=True,
        retain_graph=True,
    )[0]
    d4w_dx4 = d4w_dinp[:, 0:1]

    # 高斯载荷: G(ξ) = 1/(σ√2π) · exp(-(ξ-0.5)²/(2σ²))
    xi = x_pde_input[:, 0:1]
    P_n = x_pde_input[:, 1:2]

    G = (1.0 / (sigma_norm * np.sqrt(2.0 * np.pi))) * paddle.exp(
        -paddle.square(xi - 0.5) / (2.0 * sigma_norm**2)
    )

    # PDE residual: d⁴w/dξ⁴ - 48·P_norm·G(ξ) = 0
    residual = d4w_dx4 - 48.0 * P_n * G

    return residual


def compute_moment_bc_residual(model, bc_input):
    """
    计算弯矩边界条件残差: d²w_norm/dξ² = 0  at ξ = 0 and ξ = 1

    物理含义:
      简支梁支座处弯矩为零: M = EI · d²w/dx² = 0
      归一化后等价于: d²w_norm/dξ² = 0

    Args:
        model: PINN 模型
        bc_input: (N, 2) — [x_norm, P_norm] at boundary points

    Returns:
        residual: (N, 1) 弯矩残差
    """
    bc_input = bc_input.clone()
    bc_input.stop_gradient = False

    w = model(bc_input)
    ones = paddle.ones_like(w)

    dw_dinp = paddle.grad(
        w, bc_input, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]
    dw_dx = dw_dinp[:, 0:1]

    d2w_dinp = paddle.grad(
        dw_dx,
        bc_input,
        grad_outputs=paddle.ones_like(dw_dx),
        create_graph=True,
        retain_graph=True,
    )[0]
    d2w_dx2 = d2w_dinp[:, 0:1]

    return d2w_dx2


# =============================================================================
# 4. 解析解 (用于验证)
# =============================================================================
def analytical_deflection(x, P, L_val, E_val, I_val):
    """
    简支梁跨中集中力解析解 (挠度, 正值向下)
    """
    w = np.zeros_like(x)
    half_L = L_val / 2.0
    coeff = P / (48.0 * E_val * I_val)
    mask_left = x <= half_L
    mask_right = ~mask_left
    w[mask_left] = coeff * x[mask_left] * (3 * L_val**2 - 4 * x[mask_left] ** 2)
    xr = L_val - x[mask_right]
    w[mask_right] = coeff * xr * (3 * L_val**2 - 4 * xr**2)
    return w


# =============================================================================
# 5. 训练
# =============================================================================
def train(output_dir, config):
    os.makedirs(output_dir, exist_ok=True)

    sigma_norm = config.get("sigma_norm", SIGMA_NORM)

    print("=" * 65)
    print("  I25a 工字梁 纯物理驱动 PINNs (无 FEM 数据)")
    print("=" * 65)
    print(f"  训练方式: 纯物理约束 (PDE + 弯矩BC)")
    print(f"  载荷编码: 高斯光滑化 σ/L = {sigma_norm} (σ = {sigma_norm * L:.0f} mm)")
    print(f"  归一化: x/L=[0,1], P/P_max=[{P_MIN / P_REF:.2f},{P_MAX / P_REF:.2f}]")
    print(f"  W_REF = {W_REF:.6f} mm")

    # --- 创建模型 ---
    model = IBeamPINNPure(
        num_layers=config["num_layers"],
        hidden_size=config["hidden_size"],
    )
    n_params = sum(p.numpy().size for p in model.parameters())
    print(f"  模型参数量: {n_params}")

    # --- PDE 配点 (域内随机采样，含载荷区域) ---
    n_pde = config["n_pde_points"]
    x_pde_np = np.random.rand(n_pde, 1).astype("float64")
    P_pde_np = np.random.uniform(P_MIN / P_REF, 1.0, (n_pde, 1)).astype("float64")
    # 额外在载荷附近加密采样 (ξ ∈ [0.35, 0.65])
    n_pde_focus = config.get("n_pde_focus", 1000)
    x_focus = np.random.uniform(0.35, 0.65, (n_pde_focus, 1)).astype("float64")
    P_focus = np.random.uniform(P_MIN / P_REF, 1.0, (n_pde_focus, 1)).astype(
        "float64"
    )
    x_pde_np = np.vstack([x_pde_np, x_focus])
    P_pde_np = np.vstack([P_pde_np, P_focus])
    pde_input = paddle.to_tensor(np.hstack([x_pde_np, P_pde_np]))
    print(f"  PDE配点数: {n_pde + n_pde_focus} (含载荷区加密 {n_pde_focus})")

    # --- 弯矩BC 配点 (支座 ξ=0 和 ξ=1) ---
    n_bc = config["n_bc_points"]
    P_bc_np = np.random.uniform(P_MIN / P_REF, 1.0, (n_bc, 1)).astype("float64")
    bc_left = np.hstack([np.zeros((n_bc, 1)), P_bc_np])
    bc_right = np.hstack([np.ones((n_bc, 1)), P_bc_np.copy()])
    bc_input = paddle.to_tensor(np.vstack([bc_left, bc_right]))
    print(f"  弯矩BC点数: {n_bc * 2} (左右支座各 {n_bc})")

    print(f"  训练轮数: {config['epochs']}")
    print(f"  损失权重: w_pde={config['w_pde']}, w_moment={config['w_moment']}")
    print("=" * 65)

    # --- 优化器 ---
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=config["lr"],
        T_max=config["epochs"],
        eta_min=config["lr"] * 0.01,
    )
    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler,
        parameters=model.parameters(),
    )

    # 损失权重
    w_pde = config["w_pde"]
    w_moment = config["w_moment"]

    history = {"epoch": [], "loss": [], "loss_pde": [], "loss_moment": []}
    best_loss = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        model.train()

        # --- PDE 残差损失 ---
        residual = compute_pde_residual(model, pde_input, sigma_norm)
        loss_pde = paddle.mean(residual**2)

        # --- 弯矩 BC 损失 ---
        moment_residual = compute_moment_bc_residual(model, bc_input)
        loss_moment = paddle.mean(moment_residual**2)

        # --- 总损失 ---
        loss = w_pde * loss_pde + w_moment * loss_moment

        loss.backward()
        # 梯度裁剪 (防止高斯载荷区域梯度爆炸)
        paddle.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.clear_grad()
        scheduler.step()

        loss_val = loss.numpy().item()
        loss_pde_val = loss_pde.numpy().item()
        loss_moment_val = loss_moment.numpy().item()

        if epoch % config["log_freq"] == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:5d}/{config['epochs']} | "
                f"Loss={loss_val:.2e} | "
                f"PDE={loss_pde_val:.2e} | "
                f"Moment={loss_moment_val:.2e} | "
                f"LR={scheduler.get_lr():.1e}"
            )

        if epoch % 100 == 0:
            history["epoch"].append(epoch)
            history["loss"].append(loss_val)
            history["loss_pde"].append(loss_pde_val)
            history["loss_moment"].append(loss_moment_val)

        if loss_val < best_loss:
            best_loss = loss_val
            paddle.save(
                model.state_dict(), os.path.join(output_dir, "best_model.pdparams")
            )

    # 保存最终模型
    paddle.save(model.state_dict(), os.path.join(output_dir, "final_model.pdparams"))
    print(f"\n  训练完成! 最佳损失: {best_loss:.2e}")

    # =================================================================
    # 6. 评估
    # =================================================================
    print("\n[评估] 加载最佳模型...")
    model.set_state_dict(
        paddle.load(os.path.join(output_dir, "best_model.pdparams"))
    )
    model.eval()

    x_nodes = np.linspace(0, L, 101)
    loads = np.arange(P_MIN, P_MAX + 250, 500)  # 9 载荷工况

    print("\n  训练载荷评估 (vs 解析解):")
    print(
        f"  {'工况':<8} {'P(N)':>7} {'解析 δ_max':>12} {'PINN δ_max':>12} {'相对误差':>10}"
    )
    print("  " + "-" * 55)

    all_pinn_deflections = []
    for i, P in enumerate(loads):
        eval_inp = np.hstack(
            [
                x_nodes.reshape(-1, 1) / X_REF,
                np.full((len(x_nodes), 1), P / P_REF),
            ]
        )
        eval_tensor = paddle.to_tensor(eval_inp)
        with paddle.no_grad():
            w_pred_norm = model(eval_tensor).numpy().flatten()
        w_pred_mm = w_pred_norm * W_REF
        all_pinn_deflections.append(w_pred_mm)

        w_analytical = analytical_deflection(x_nodes, P, L, E, Ix)
        ana_max = np.max(np.abs(w_analytical))
        pinn_max = np.max(np.abs(w_pred_mm))
        rel_err = abs(pinn_max - ana_max) / ana_max * 100

        print(
            f"  LC-{i + 1:<4} {P:>7.0f} {ana_max:>12.6f} {pinn_max:>12.6f} {rel_err:>9.4f}%"
        )

    all_pinn_deflections = np.array(all_pinn_deflections)

    # --- 插值泛化测试 ---
    print("\n  插值泛化测试 (未见载荷):")
    print(f"  {'P(N)':>7} {'解析 δ_max':>12} {'PINN δ_max':>12} {'相对误差':>10}")
    print("  " + "-" * 45)

    test_loads = [1250, 1750, 2750, 3250, 4250, 4750]
    for P_test in test_loads:
        eval_inp = np.hstack(
            [
                x_nodes.reshape(-1, 1) / X_REF,
                np.full((len(x_nodes), 1), P_test / P_REF),
            ]
        )
        eval_tensor = paddle.to_tensor(eval_inp)
        with paddle.no_grad():
            w_pred_norm = model(eval_tensor).numpy().flatten()
        w_pred_mm = w_pred_norm * W_REF

        analytical_max = P_test * L**3 / (48.0 * EI)
        pinn_max = np.max(np.abs(w_pred_mm))
        rel_err = abs(pinn_max - analytical_max) / analytical_max * 100

        print(
            f"  {P_test:>7} {analytical_max:>12.6f} {pinn_max:>12.6f} {rel_err:>9.4f}%"
        )

    # =================================================================
    # 7. 可视化
    # =================================================================
    print("\n[可视化] 生成图表...")
    x_m = x_nodes / 1000.0

    # --- 图1: PINNs vs 解析解 挠度对比 ---
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for i, P in enumerate(loads):
        ax = axes[i // 3][i % 3]
        w_ana = analytical_deflection(x_nodes, P, L, E, Ix)
        ax.plot(x_m, w_ana, "b-", linewidth=2, label="Analytical")
        ax.plot(x_m, -all_pinn_deflections[i], "r--", linewidth=2, label="PINN (pure)")
        ax.set_title(f"P = {P:.0f} N", fontsize=11)
        ax.set_xlabel("Position (m)", fontsize=9)
        ax.set_ylabel("Deflection (mm)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Pure Physics PINN vs Analytical: Deflection Comparison", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "pinn_pure_vs_analytical.png"), dpi=150
    )
    plt.close(fig)
    print("  [PNG] pinn_pure_vs_analytical.png")

    # --- 图2: 训练损失曲线 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(
        history["epoch"], history["loss"], "k-", linewidth=2, label="Total Loss"
    )
    ax.semilogy(
        history["epoch"],
        history["loss_pde"],
        "r--",
        linewidth=1.5,
        label="PDE Loss",
    )
    ax.semilogy(
        history["epoch"],
        history["loss_moment"],
        "b--",
        linewidth=1.5,
        label="Moment BC Loss",
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Pure Physics PINN Training Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_loss.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] training_loss.png")

    # --- 图3: 挠度场热力图 ---
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
    c = ax.pcolormesh(
        x_grid * L / 1000,
        p_grid * P_REF,
        -W_grid,
        shading="auto",
        cmap="RdYlBu_r",
    )
    plt.colorbar(c, ax=ax, label="Deflection (mm)")
    for P in loads:
        ax.axhline(y=P, color="white", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Position along beam (m)", fontsize=12)
    ax.set_ylabel("Applied Load P (N)", fontsize=12)
    ax.set_title(
        "Pure Physics PINN: Predicted Deflection Field w(x, P)", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "deflection_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] deflection_heatmap.png")

    # --- 图4: 误差分布 (沿梁长度) ---
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for i, P in enumerate(loads):
        ax = axes[i // 3][i % 3]
        w_ana = analytical_deflection(x_nodes, P, L, E, Ix)
        w_pinn = -all_pinn_deflections[i]
        # 相对误差
        rel_err = np.where(
            np.abs(w_ana) > 1e-10,
            np.abs(w_pinn - w_ana) / np.abs(w_ana) * 100,
            0,
        )
        ax.plot(x_m, rel_err, "r-", linewidth=1.5)
        ax.set_title(f"P = {P:.0f} N", fontsize=11)
        ax.set_xlabel("Position (m)", fontsize=9)
        ax.set_ylabel("Relative Error (%)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    fig.suptitle("Pure Physics PINN: Pointwise Relative Error vs Analytical", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "error_distribution.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] error_distribution.png")

    # --- 保存预测数据 ---
    np.savez(
        os.path.join(output_dir, "pinn_pure_predictions.npz"),
        x_nodes=x_nodes,
        loads=loads,
        pinn_deflections=all_pinn_deflections,
        W_REF=W_REF,
        X_REF=X_REF,
        P_REF=P_REF,
        sigma_norm=sigma_norm,
    )
    print("  [NPZ] pinn_pure_predictions.npz")

    print(f"\n  输出目录: {os.path.abspath(output_dir)}")
    print("=" * 65)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs", "pinn_pure_results")

    config = {
        # 模型 (比混合PINN稍大，因为无数据引导)
        "num_layers": 4,
        "hidden_size": 64,
        # 训练
        "epochs": 20000,
        "lr": 5e-4,
        "log_freq": 2000,
        # 配点
        "n_pde_points": 5000,
        "n_pde_focus": 1000,  # 载荷区域加密点
        "n_bc_points": 50,
        # 损失权重
        "w_pde": 1.0,
        "w_moment": 10.0,
        # 高斯载荷宽度
        "sigma_norm": 0.02,
    }

    train(output_dir, config)
