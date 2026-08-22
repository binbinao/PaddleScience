"""
I25a 工字梁弹塑性 纯物理驱动 PINNs 训练程序（无 FEM 数据依赖）

═══════════════════════════════════════════════════════════════════════════════
核心设计 (v2 — 修正版)
═══════════════════════════════════════════════════════════════════════════════

v1 的三大缺陷:
  1. 4阶 PDE (d⁴w/dξ⁴=48·P·G) 是线弹性方程 → 无法产生塑性变形
  2. σ_ratio 作为独立输出与 w 完全解耦 → 退化为常数≈1
  3. 本构约束太弱 → 无法区分弹塑性

v2 的修正:
  1. 2阶 PDE + 已知弯矩分布 (静定梁, 弯矩由平衡唯一确定)
  2. 应力比从曲率推导, 不作为独立网络输出
  3. 双线性 M-κ 本构模型嵌入 PDE, 通过刚度折减因子 α(κ) 实现弹塑性

═══════════════════════════════════════════════════════════════════════════════
物理约束体系
═══════════════════════════════════════════════════════════════════════════════

简支梁跨中集中力的弯矩分布 (静定结构, 由平衡唯一确定):
  M(x,P) = P·min(x, L-x)/2

归一化弯矩:  g(ξ) = min(ξ, 1-ξ)
归一化 PDE:  d²w_norm/dξ² = -24·P_norm·g(ξ)/α(κ)

其中 α(κ) = EI_eff/EI 是刚度折减因子:
  ┌───────────────────────────────────────────────────────────────────┐
  │  弹性区 (|κ| ≤ κ_y):   α = 1                                    │
  │  塑性区 (|κ| > κ_y):   α = p + (1-p)·κ_y/|κ|                   │
  │                                                                   │
  │  p = E_st/E ≈ 0.005 (Q235B 应变强化比)                           │
  │  κ_y = M_y/EI (屈服曲率)                                         │
  │                                                                   │
  │  本质: 双线性 M-κ 模型                                            │
  │    弹性:  M = EI·κ          → κ = M/EI                           │
  │    塑性:  M = M_y + p·EI·(κ-κ_y) → κ = (M - M_y)/(p·EI) + κ_y  │
  └───────────────────────────────────────────────────────────────────┘

边界条件:
  位移 BC: w(0) = w(1) = 0   (硬约束 via output_transform: w = û·ξ(1-ξ))
  弯矩 BC: d²w/dξ² = 0 at ξ=0,1  (自动满足, 因为 g(0)=g(1)=0)

为什么2阶比4阶好:
  - 4阶 PDE d⁴w/dx⁴ = q/EI 只对线弹性成立, 塑性时 EI→EI_eff(x) 非常难处理
  - 2阶 PDE d²w/dx² = -M/EI_eff 天然包含变刚度, 且仅需2次自动微分
  - 静定梁弯矩分布已知, 不需要从4阶方程求解弯矩

输入: (ξ, P_norm)   输出: w_norm (挠度唯一输出, 应力比从曲率推导)
网络: 5层 × 64神经元, tanh激活

参考验证: FEM 弹塑性解 (仅用于最终对比, 不参与训练)
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
E = 206000.0  # MPa
Ix = 50170000.0  # mm^4
Wx = 398200.0  # mm^3
L = 3000.0  # mm
h = 252.0  # mm (截面高度)
EI = E * Ix  # N*mm^2

# Q235B 弹塑性本构参数
SIGMA_Y = 235.0  # MPa
SIGMA_U = 420.0  # MPa
EPS_Y = SIGMA_Y / E  # 0.00114
EPS_ST = 0.015  # 强化起始应变
JC_B = 230.2  # MPa (Johnson-Cook)
JC_N = 0.578

# 载荷范围
P_MIN = 10000.0  # N (10 kN)
P_MAX = 250000.0  # N (250 kN)

# 归一化参考值
X_REF = L  # x / L -> [0, 1]
P_REF = P_MAX  # P / P_MAX -> [0.04, 1.0]
W_REF = P_MAX * L**3 / (48.0 * EI)  # ~13.607 mm

# 屈服参数
M_YIELD = SIGMA_Y * Wx  # 93,577,000 N*mm
P_YIELD = 4 * M_YIELD / L  # ~124,769 N = 124.8 kN
KAPPA_YIELD = M_YIELD / EI  # ~9.054e-6 /mm (屈服曲率)

# 归一化屈服曲率: |d²w_norm/dξ²|_yield
# 推导: κ_y = M_y/EI, d²w/dx² = -M/(EI_eff) = -κ
#        d²w_norm/dξ² = d²w/dx² · L²/W_REF
#        |d²w_norm/dξ²|_y = κ_y · L²/W_REF
KAPPA_Y_NORM = KAPPA_YIELD * L**2 / W_REF  # ~5.989

# 应变强化比 (Q235B) — 仅作为参考
P_HARDENING_BILINEAR = (SIGMA_U - SIGMA_Y) / ((0.20 - EPS_ST) * E)

# 幂律 M-κ 本构模型参数 (由纤维法拟合)
# α(κ) = A / (κ/κ_y)^B for κ > κ_y; α = 1 for κ ≤ κ_y
# 拟合自 I25a 截面纤维法 M-κ 曲线, 误差 < 5%
MK_ALPHA_A = 1.1323
MK_ALPHA_B = 0.9433


# =============================================================================
# 2. Q235B 弹塑性本构 (NumPy, 用于后处理对比)
# =============================================================================
def stress_from_strain(eps):
    """Q235B 弹塑性本构: 工程应变 → 工程应力 (数组)"""
    eps = np.asarray(eps, dtype=float)
    sig = np.zeros_like(eps)
    ae = np.abs(eps)
    sign = np.sign(eps)

    m1 = ae <= EPS_Y
    sig[m1] = E * eps[m1]

    m2 = (ae > EPS_Y) & (ae <= EPS_ST)
    sig[m2] = sign[m2] * SIGMA_Y

    m3 = (ae > EPS_ST) & (ae <= 0.20)
    eps_pl = ae[m3] - SIGMA_Y / E
    sig[m3] = sign[m3] * (SIGMA_Y + JC_B * np.power(np.maximum(eps_pl, 1e-12), JC_N))

    m4 = ae > 0.20
    sig[m4] = sign[m4] * SIGMA_U

    return sig


# =============================================================================
# 3. 网络模型 — 单输出 w_norm
# =============================================================================
class BucklingPINNPure(paddle.nn.Layer):
    """
    纯物理驱动弹塑性 PINN: MLP + output_transform

    输入: (ξ, P_norm)  dim=2
    输出: w_norm  dim=1

    output_transform: w = w_raw * ξ * (1-ξ)
      - 硬约束 w(0) = 0, w(1) = 0
      - 弹塑性下 w 不与 P 成正比, 不乘 P_norm
    """

    def __init__(self, num_layers=5, hidden_size=64):
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
        w_raw = self.net(x_in)
        xi = x_in[:, 0:1]
        w = w_raw * xi * (1.0 - xi)
        return w


# =============================================================================
# 4. 刚度折减因子 α(κ) — 双线性 M-κ 本构模型
# =============================================================================
def compute_stiffness_reduction(kappa_norm):
    """
    从归一化曲率计算刚度折减因子 α = EI_eff/EI

    幂律模型 (由 I25a 截面纤维法 M-κ 曲线拟合):
      弹性: α = 1                              for κ ≤ κ_y
      塑性: α = A / (κ/κ_y)^B                  for κ > κ_y

    拟合参数: A=1.1323, B=0.9433 (误差 < 5%)

    对比双线性模型 α = p + (1-p)·κ_y/κ (p=0.005):
      - 双线性: κ/κ_y=10 → α=0.105 (过刚)
      - 幂律:   κ/κ_y=10 → α=0.124 (接近纤维法0.124)

    Args:
        kappa_norm: |d²w_norm/dξ²| (归一化曲率绝对值)
    Returns:
        alpha: EI_eff/EI ∈ (0, 1]
    """
    ky = KAPPA_Y_NORM
    A = MK_ALPHA_A
    B = MK_ALPHA_B

    # κ/κ_y
    ratio = kappa_norm / ky

    # 弹性区: α = 1; 塑性区: α = A / (κ/κ_y)^B
    # 使用 smooth transition: α = 1 - (1 - A/ratio^B) * relu(1 - 1/ratio)
    inv_ratio = ky / paddle.maximum(kappa_norm, paddle.to_tensor(1e-10))
    plastic_mask = paddle.nn.functional.relu(1.0 - inv_ratio)  # 0 if κ≤κ_y, >0 if κ>κ_y

    alpha_plastic = A / paddle.pow(paddle.maximum(ratio, paddle.to_tensor(1.0)), B)
    alpha = 1.0 - (1.0 - alpha_plastic) * plastic_mask

    return alpha


# =============================================================================
# 5. PDE 残差计算 — 2阶 PDE + 弹塑性刚度折减
# =============================================================================
def compute_pde_residual(model, x_pde_input):
    """
    计算 2阶弹塑性 PDE 残差:

    d²w_norm/dξ² + 24·P_norm·g(ξ)/α(κ) = 0

    物理推导:
      对于简支梁跨中集中力, 弯矩分布由静力平衡唯一确定:
        M(x,P) = P·min(x, L-x)/2

      曲率-弯矩关系:
        d²w/dx² = -M/EI_eff = -M/(α·EI)

      归一化 (ξ=x/L, w_norm=w/W_REF):
        d²w_norm/dξ² = -M(ξ,P)·L²/(EI·W_REF) / α

      其中 M(ξ,P)·L²/(EI·W_REF) = 24·P_norm·min(ξ,1-ξ)

      所以: d²w_norm/dξ² + 24·P_norm·g(ξ)/α(κ) = 0

      弹性区 (α=1): d²w_norm/dξ² = -24·P_norm·g(ξ)
      塑性区 (α<1): |d²w_norm/dξ²| > 24·P_norm·g(ξ) (曲率增大)

    Returns:
        residual: PDE 残差
        alpha: 刚度折减因子
        kappa_norm: 归一化曲率绝对值
    """
    x_pde_input = x_pde_input.clone()
    x_pde_input.stop_gradient = False

    w = model(x_pde_input)
    ones = paddle.ones_like(w)

    # 1st derivative
    dw_dinp = paddle.grad(
        w, x_pde_input, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]
    dw_dxi = dw_dinp[:, 0:1]

    # 2nd derivative
    d2w_dinp = paddle.grad(
        dw_dxi,
        x_pde_input,
        grad_outputs=paddle.ones_like(dw_dxi),
        create_graph=True,
        retain_graph=True,
    )[0]
    d2w_dxi2 = d2w_dinp[:, 0:1]

    # 归一化曲率绝对值
    kappa_norm = paddle.abs(d2w_dxi2)

    # 刚度折减因子
    alpha = compute_stiffness_reduction(kappa_norm)

    # 归一化弯矩分布: g(ξ) = min(ξ, 1-ξ)
    xi = x_pde_input[:, 0:1]
    P_n = x_pde_input[:, 1:2]
    g = paddle.minimum(xi, 1.0 - xi)

    # PDE residual: d²w/dξ² + 24·P_norm·g(ξ)/α = 0
    residual = d2w_dxi2 + 24.0 * P_n * g / alpha

    return residual, alpha, kappa_norm


# =============================================================================
# 6. 解析解 (用于评估)
# =============================================================================
def analytical_deflection(x, P, L_val, E_val, I_val):
    """简支梁跨中集中力解析解 (挠度, 正值向下, 仅弹性区准确)"""
    w = np.zeros_like(x)
    half_L = L_val / 2.0
    coeff = P / (48.0 * E_val * I_val)
    mask_left = x <= half_L
    mask_right = ~mask_left
    w[mask_left] = coeff * x[mask_left] * (3 * L_val**2 - 4 * x[mask_left] ** 2)
    xr = L_val - x[mask_right]
    w[mask_right] = coeff * xr * (3 * L_val**2 - 4 * xr**2)
    return w


def compute_stress_ratio_from_curvature(kappa_norm_array):
    """
    从归一化曲率计算应力比 σ/σ_y

    对于幂律 M-κ 模型:
      弹性: M = EI·κ → M/M_y = κ/κ_y
      塑性: M = α(κ)·EI·κ → M/M_y = α(κ)·κ/κ_y
    """
    kappa_ratio = np.abs(kappa_norm_array) / KAPPA_Y_NORM

    A = MK_ALPHA_A
    B = MK_ALPHA_B
    alpha = np.where(
        kappa_ratio <= 1.0,
        1.0,
        A / np.power(np.maximum(kappa_ratio, 1.0), B),
    )
    sr = alpha * kappa_ratio  # M/M_y = α·κ/κ_y
    return sr


# =============================================================================
# 7. 训练
# =============================================================================
def train(output_dir, config):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  I25a 工字梁 弹塑性 纯物理驱动 PINNs v2 (无 FEM 数据)")
    print("=" * 70)
    print(f"  核心改进: 2阶PDE + 已知弯矩分布 + 幂律M-κ本构")
    print(f"  PDE: d²w/dξ² + 24·P_norm·g(ξ)/α(κ) = 0")
    print(f"  弯矩分布: g(ξ) = min(ξ, 1-ξ)  [静定梁, 平衡唯一确定]")
    print(f"  刚度折减: α=1(弹性), α=A/(κ/κ_y)^B(塑性)")
    print(f"  M-κ模型: 幂律 α = {MK_ALPHA_A:.4f}/(κ/κ_y)^{MK_ALPHA_B:.4f}")
    print(f"  κ_y_norm = {KAPPA_Y_NORM:.4f}")
    print(f"  W_REF = {W_REF:.6f} mm")
    print(f"  P_YIELD = {P_YIELD/1000:.1f} kN")

    # --- 创建模型 ---
    model = BucklingPINNPure(
        num_layers=config["num_layers"],
        hidden_size=config["hidden_size"],
    )
    n_params = sum(p.numpy().size for p in model.parameters())
    print(f"  模型: {config['num_layers']}层×{config['hidden_size']}, 参数量: {n_params}")

    # --- 配点生成 ---
    n_pde = config["n_pde_points"]
    x_pde_np = np.random.rand(n_pde, 1).astype("float64")
    P_pde_np = np.random.uniform(P_MIN / P_REF, 1.0, (n_pde, 1)).astype("float64")

    # 跨中区域加密 (弯矩最大, 最先屈服)
    n_mid = config.get("n_mid_points", 3000)
    x_mid = np.random.uniform(0.3, 0.7, (n_mid, 1)).astype("float64")
    P_mid = np.random.uniform(P_MIN / P_REF, 1.0, (n_mid, 1)).astype("float64")
    x_pde_np = np.vstack([x_pde_np, x_mid])
    P_pde_np = np.vstack([P_pde_np, P_mid])

    # 大载荷加密 (塑性区 P > P_YIELD)
    n_plastic = config.get("n_plastic_points", 3000)
    x_plastic = np.random.uniform(0.2, 0.8, (n_plastic, 1)).astype("float64")
    P_plastic = np.random.uniform(P_YIELD / P_REF, 1.0, (n_plastic, 1)).astype("float64")
    x_pde_np = np.vstack([x_pde_np, x_plastic])
    P_pde_np = np.vstack([P_pde_np, P_plastic])

    # 屈服边界加密 (ξ ≈ P_y/(2P) 处是弹塑性边界)
    n_ybd = config.get("n_yield_boundary", 2000)
    P_ybd = np.random.uniform(P_YIELD / P_REF * 0.8, 1.0, (n_ybd, 1)).astype("float64")
    # 弹塑性边界 ξ_b = P_y / (2P) (for P > P_y)
    xi_b = np.clip(P_YIELD / (2.0 * P_ybd * P_REF / L * L), 0.1, 0.45)
    x_ybd = xi_b + np.random.randn(n_ybd, 1).astype("float64") * 0.03
    x_ybd = np.clip(x_ybd, 0.05, 0.95)
    x_pde_np = np.vstack([x_pde_np, x_ybd])
    P_pde_np = np.vstack([P_pde_np, P_ybd])

    pde_input = paddle.to_tensor(np.hstack([x_pde_np, P_pde_np]))
    total_pts = len(x_pde_np)
    print(f"  PDE配点数: {total_pts}")

    # --- Continuation method 训练 ---
    # β 从 0 渐增到 1:
    #   β=0: α_eff = 1 (纯弹性 PDE)
    #   β=1: α_eff = α(κ) (完整弹塑性 PDE)
    # α_eff = 1 - β·(1 - α(κ))
    epochs_warmup = config.get("epochs_warmup", config["epochs"] // 4)

    print(f"  训练策略: Continuation method")
    print(f"  预热期: {epochs_warmup} epochs (β: 0→1)")
    print(f"  总轮数: {config['epochs']}")
    print("=" * 70)

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

    history = {"epoch": [], "loss": [], "loss_pde": [], "beta": []}
    best_loss = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        model.train()

        # --- Continuation parameter β ---
        if epoch <= epochs_warmup:
            beta = epoch / epochs_warmup  # 0 → 1
        else:
            beta = 1.0

        # --- 计算 PDE 残差 ---
        # 需要重新计算以获得 d2w_dxi2
        x_cg = pde_input.clone()
        x_cg.stop_gradient = False

        w = model(x_cg)
        ones = paddle.ones_like(w)

        dw_dinp = paddle.grad(
            w, x_cg, grad_outputs=ones, create_graph=True, retain_graph=True
        )[0]
        dw_dxi = dw_dinp[:, 0:1]

        d2w_dinp = paddle.grad(
            dw_dxi, x_cg,
            grad_outputs=paddle.ones_like(dw_dxi),
            create_graph=True, retain_graph=True,
        )[0]
        d2w_dxi2 = d2w_dinp[:, 0:1]

        # 归一化曲率 & 刚度折减
        kappa_norm = paddle.abs(d2w_dxi2)
        alpha_constitutive = compute_stiffness_reduction(kappa_norm)

        # Continuation: α_eff = 1 - β·(1 - α(κ))
        alpha_eff = 1.0 - beta * (1.0 - alpha_constitutive)

        # PDE residual: d²w/dξ² + 24·P_norm·g(ξ)/α_eff = 0
        xi = x_cg[:, 0:1]
        P_n = x_cg[:, 1:2]
        g = paddle.minimum(xi, 1.0 - xi)

        residual = d2w_dxi2 + 24.0 * P_n * g / alpha_eff
        loss_pde = paddle.mean(residual**2)

        loss = loss_pde

        loss.backward()
        paddle.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.clear_grad()
        scheduler.step()

        loss_val = loss.numpy().item()
        loss_pde_val = loss_pde.numpy().item()

        if epoch % config["log_freq"] == 0 or epoch == 1:
            alpha_np = alpha_constitutive.numpy().flatten()
            alpha_min = alpha_np.min()
            alpha_elastic_pct = (alpha_np > 0.99).mean() * 100
            print(
                f"  Epoch {epoch:5d}/{config['epochs']} | "
                f"Loss={loss_val:.2e} | "
                f"β={beta:.3f} | "
                f"α_min={alpha_min:.4f} | "
                f"α>0.99: {alpha_elastic_pct:.1f}% | "
                f"LR={scheduler.get_lr():.1e}"
            )

        if epoch % 200 == 0:
            history["epoch"].append(epoch)
            history["loss"].append(loss_val)
            history["loss_pde"].append(loss_pde_val)
            history["beta"].append(beta)

        if loss_val < best_loss:
            best_loss = loss_val
            paddle.save(
                model.state_dict(), os.path.join(output_dir, "best_model.pdparams")
            )

    paddle.save(model.state_dict(), os.path.join(output_dir, "final_model.pdparams"))
    print(f"\n  训练完成! 最佳损失: {best_loss:.2e}")

    # =================================================================
    # 8. 评估
    # =================================================================
    print("\n  [评估] 加载最佳模型...")
    model.set_state_dict(
        paddle.load(os.path.join(output_dir, "best_model.pdparams"))
    )
    model.eval()

    x_nodes = np.linspace(0, L, 201)
    loads = np.arange(P_MIN, P_MAX + 10000, 10000)

    print(f"\n  理论屈服载荷: {P_YIELD/1000:.1f} kN")
    print(f"\n  {'工况':<7} {'P(kN)':>6} {'弹性δ_max':>10} {'PINNδ_max':>10} {'δ比':>7} "
          f"{'PINNσ/σy':>9} {'α_min':>7} {'区域':>6}")
    print("  " + "-" * 75)

    all_pinn_deflections = []
    all_pinn_stress_ratios = []
    all_pinn_alpha = []

    for i, P in enumerate(loads):
        eval_inp = np.hstack(
            [
                x_nodes.reshape(-1, 1) / X_REF,
                np.full((len(x_nodes), 1), P / P_REF),
            ]
        )
        eval_tensor = paddle.to_tensor(eval_inp)
        with paddle.no_grad():
            w_pred = model(eval_tensor)
        w_pred_mm = w_pred.numpy().flatten() * W_REF

        # 从曲率计算应力比和刚度折减
        eval_tensor_cg = paddle.to_tensor(eval_inp)
        eval_tensor_cg.stop_gradient = False
        w_cg = model(eval_tensor_cg)
        dw = paddle.grad(w_cg, eval_tensor_cg,
                         grad_outputs=paddle.ones_like(w_cg),
                         create_graph=True, retain_graph=True)[0]
        dw_dxi = dw[:, 0:1]
        d2w = paddle.grad(dw_dxi, eval_tensor_cg,
                          grad_outputs=paddle.ones_like(dw_dxi),
                          create_graph=True, retain_graph=True)[0]
        d2w_dxi2 = d2w[:, 0:1].numpy().flatten()

        kappa_norm_arr = np.abs(d2w_dxi2)
        sr_arr = compute_stress_ratio_from_curvature(kappa_norm_arr)
        alpha_arr = compute_stiffness_reduction(paddle.to_tensor(kappa_norm_arr)).numpy()

        all_pinn_deflections.append(w_pred_mm)
        all_pinn_stress_ratios.append(sr_arr)
        all_pinn_alpha.append(alpha_arr)

        w_ana = analytical_deflection(x_nodes, P, L, E, Ix)
        ana_max = np.max(np.abs(w_ana))
        pinn_max = np.max(np.abs(w_pred_mm))
        max_sr = np.max(sr_arr)
        min_alpha = np.min(alpha_arr[alpha_arr > 0.01])  # 排除支座附近
        ratio = pinn_max / ana_max if ana_max > 0 else 0
        zone = "弹性" if max_sr < 1.0 else "塑性"

        print(f"  LC-{i+1:02d} {P/1000:>6.0f} {ana_max:>10.4f} {pinn_max:>10.4f} {ratio:>6.3f} "
              f"{max_sr*100:>8.1f}% {min_alpha:>7.4f} {zone:>6}")

    all_pinn_deflections = np.array(all_pinn_deflections)
    all_pinn_stress_ratios = np.array(all_pinn_stress_ratios)
    all_pinn_alpha = np.array(all_pinn_alpha)

    # =================================================================
    # 9. 可视化
    # =================================================================
    print("\n[可视化] 生成图表...")
    x_m = x_nodes / 1000.0
    P_kN = loads / 1000

    # --- 图1: 载荷-挠度曲线 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    pinn_maxd = np.max(np.abs(all_pinn_deflections), axis=1)
    elastic_ref = loads * L**3 / (48 * EI)

    ax.plot(P_kN, elastic_ref, "b--", lw=2, label="Linear Elastic (analytical)", zorder=2)
    ax.plot(P_kN, pinn_maxd, "r^--", lw=2, ms=5, label="Pure PINN (elastic-plastic)", zorder=4)

    ax.axvline(x=P_YIELD / 1000, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.annotate(f"Yield: {P_YIELD/1000:.1f} kN", xy=(P_YIELD / 1000, 5),
                fontsize=10, color="gray", rotation=90, va="bottom")

    ax.fill_between(P_kN, elastic_ref, pinn_maxd,
                    where=pinn_maxd > elastic_ref * 1.01,
                    alpha=0.15, color="red", label="Plastic additional deflection")

    ax.set_xlabel("Load (kN)", fontsize=13)
    ax.set_ylabel("Max Deflection (mm)", fontsize=13)
    ax.set_title("Pure PINN v2: Load-Deflection (2nd-order PDE + M-κ constitutive)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "load_deflection_pure_pinn.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] load_deflection_pure_pinn.png")

    # --- 图2: 应力比 vs 载荷 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    pinn_maxsr = np.max(all_pinn_stress_ratios, axis=1) * 100
    # 弹性参考线: σ/σ_y = M_max/M_y = P·L/(4·M_y)
    elastic_sr = loads * L / (4 * M_YIELD) * 100
    ax.plot(P_kN, elastic_sr, "b--", lw=1.5, alpha=0.5, label="Elastic σ/σ_y (reference)")
    ax.plot(P_kN, pinn_maxsr, "r^--", lw=2, ms=5, label="Pure PINN (from curvature)")
    ax.axhline(y=100, color="red", ls="--", lw=2, alpha=0.5, label="Yield (100%)")
    ax.axhline(y=SIGMA_U / SIGMA_Y * 100, color="darkred", ls=":", lw=1.5, alpha=0.5,
               label=f"Ultimate ({SIGMA_U / SIGMA_Y * 100:.0f}%)")
    ax.axvline(x=P_YIELD / 1000, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.set_xlabel("Load (kN)", fontsize=13)
    ax.set_ylabel("Max Stress Ratio σ/σ_y (%)", fontsize=13)
    ax.set_title("Stress Ratio: PINN (from curvature) vs Elastic reference", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "stress_ratio_pure_pinn.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] stress_ratio_pure_pinn.png")

    # --- 图3: 刚度折减因子 vs 载荷 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    pinn_min_alpha = np.array([np.min(a[a > 0.01]) for a in all_pinn_alpha])
    ax.plot(P_kN, pinn_min_alpha, "rs--", lw=2, ms=5, label="min α (midspan)")
    ax.axhline(y=1.0, color="blue", ls="--", lw=1, alpha=0.5, label="Elastic (α=1)")
    ax.axhline(y=MK_ALPHA_A / 100**MK_ALPHA_B, color="red", ls=":", lw=1, alpha=0.5,
               label=f"Min (~{MK_ALPHA_A / 100**MK_ALPHA_B:.4f})")
    ax.axvline(x=P_YIELD / 1000, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.set_xlabel("Load (kN)", fontsize=13)
    ax.set_ylabel("Stiffness Reduction Factor α = EI_eff/EI", fontsize=13)
    ax.set_title("Stiffness Degradation vs Load", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "stiffness_reduction.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] stiffness_reduction.png")

    # --- 图4: 选代表工况挠度曲线 ---
    rep = [0, len(loads) // 4, len(loads) // 2, 3 * len(loads) // 4, len(loads) - 1]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    for idx, ci in enumerate(rep):
        ax = axes[idx]
        w_ana = analytical_deflection(x_nodes, loads[ci], L, E, Ix)
        ax.plot(x_m, w_ana, "b-", lw=2, label="Elastic (analytical)")
        ax.plot(x_m, all_pinn_deflections[ci], "r--", lw=2, label="Pure PINN (ep)")
        sr = np.max(all_pinn_stress_ratios[ci])
        zone = "Elastic" if sr < 1.0 else "Plastic"
        ax.set_title(f"P={loads[ci]/1000:.0f}kN\nσ/σy={sr*100:.0f}% [{zone}]", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("m")
        ax.set_ylabel("w (mm)")
    fig.suptitle("Pure PINN v2: Representative Load Cases", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "representative_cases_pure.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] representative_cases_pure.png")

    # --- 图5: 训练损失 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(history["epoch"], history["loss"], "k-", lw=2, label="Total Loss")
    ax.semilogy(history["epoch"], history["loss_pde"], "r--", lw=1.5, label="PDE Loss")
    epochs_warmup = config.get("epochs_warmup", config["epochs"] // 4)
    if epochs_warmup > 0:
        ax.axvline(x=epochs_warmup, color="green", ls="--", lw=1.5, alpha=0.7)
        ax.annotate("β=1 (full constitutive)", xy=(epochs_warmup, ax.get_ylim()[1]*0.1),
                    fontsize=9, color="green")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Pure PINN v2 Training Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_loss.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] training_loss.png")

    # --- 图6: 挠度场 + 应力比场 热力图 ---
    n_x = 200
    n_p = 50
    x_grid = np.linspace(0.01, 0.99, n_x)
    p_grid = np.linspace(P_MIN / P_REF, 1.0, n_p)
    XX, PP = np.meshgrid(x_grid, p_grid)

    x_flat = paddle.to_tensor(np.hstack([XX.reshape(-1, 1), PP.reshape(-1, 1)]))
    with paddle.no_grad():
        w_flat = model(x_flat).numpy().flatten()
    W_grid = (w_flat * W_REF).reshape(n_p, n_x)

    # 从曲率计算应力比
    x_flat_cg = paddle.to_tensor(np.hstack([XX.reshape(-1, 1), PP.reshape(-1, 1)]))
    x_flat_cg.stop_gradient = False
    w_cg = model(x_flat_cg)
    dw_cg = paddle.grad(w_cg, x_flat_cg, grad_outputs=paddle.ones_like(w_cg),
                        create_graph=True, retain_graph=True)[0]
    dw_dxi_cg = dw_cg[:, 0:1]
    d2w_cg = paddle.grad(dw_dxi_cg, x_flat_cg, grad_outputs=paddle.ones_like(dw_dxi_cg),
                         create_graph=True, retain_graph=True)[0]
    d2w_dxi2_flat = d2w_cg[:, 0:1].numpy().flatten()
    kappa_norm_flat = np.abs(d2w_dxi2_flat)
    sr_flat = compute_stress_ratio_from_curvature(kappa_norm_flat)
    SR_grid = sr_flat.reshape(n_p, n_x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    c1 = ax1.pcolormesh(x_grid * L / 1000, p_grid * P_REF, W_grid,
                        shading="auto", cmap="RdYlBu_r")
    plt.colorbar(c1, ax=ax1, label="Deflection w (mm)")
    ax1.axhline(y=P_YIELD, color="white", lw=1.5, ls="--", alpha=0.7)
    ax1.set_xlabel("Position along beam (m)", fontsize=12)
    ax1.set_ylabel("Applied Load P (N)", fontsize=12)
    ax1.set_title("Deflection Field w(x, P)", fontsize=14)

    c2 = ax2.pcolormesh(x_grid * L / 1000, p_grid * P_REF, SR_grid * 100,
                        shading="auto", cmap="hot_r", vmin=0)
    plt.colorbar(c2, ax=ax2, label="σ/σ_y (%)")
    ax2.axhline(y=P_YIELD, color="cyan", lw=1.5, ls="--", alpha=0.7)
    ax2.contour(x_grid * L / 1000, p_grid * P_REF, SR_grid, levels=[1.0],
                colors="cyan", linewidths=2)
    ax2.set_xlabel("Position along beam (m)", fontsize=12)
    ax2.set_ylabel("Applied Load P (N)", fontsize=12)
    ax2.set_title("Stress Ratio σ/σ_y(x, P)", fontsize=14)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "deflection_stress_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] deflection_stress_heatmap.png")

    # --- 保存预测数据 ---
    np.savez(
        os.path.join(output_dir, "pinn_pure_predictions.npz"),
        x_nodes=x_nodes, loads=loads,
        pinn_deflections=all_pinn_deflections,
        pinn_stress_ratios=all_pinn_stress_ratios,
        pinn_alpha=all_pinn_alpha,
        W_REF=W_REF, X_REF=X_REF, P_REF=P_REF,
        KAPPA_Y_NORM=KAPPA_Y_NORM, MK_ALPHA_A=MK_ALPHA_A, MK_ALPHA_B=MK_ALPHA_B,
    )
    print("  [NPZ] pinn_pure_predictions.npz")

    # =================================================================
    # 10. 与 FEM 结果对比
    # =================================================================
    fem_path = os.path.join(os.path.dirname(output_dir), "fem_results", "fem_training_data.npz")
    if os.path.exists(fem_path):
        print("\n[对比] 与 FEM 结果对比:")
        fem_data = np.load(fem_path)
        fem_loads = fem_data["loads"]
        fem_defl = fem_data["deflections"]
        fem_sr = fem_data["stress_ratios"]

        print(f"\n  {'工况':<7} {'P(kN)':>6} {'FEM δ':>10} {'PINN δ':>10} {'δ误差':>8} "
              f"{'FEM σ/σy':>9} {'PINN σ/σy':>10} {'区域':>6}")
        print("  " + "-" * 70)

        for i, P in enumerate(fem_loads):
            idx = np.argmin(np.abs(loads - P))
            fd = np.max(np.abs(fem_defl[i]))
            pd_ = np.max(np.abs(all_pinn_deflections[idx]))
            fsr = np.max(fem_sr[i])
            psr = np.max(all_pinn_stress_ratios[idx])
            ed = abs(pd_ - fd) / fd * 100 if fd > 0 else 0
            zone = "弹性" if fsr < 1.0 else "塑性"
            print(f"  LC-{i+1:02d} {P/1000:>6.0f} {fd:>10.4f} {pd_:>10.4f} {ed:>7.2f}% "
                  f"{fsr*100:>8.1f}% {psr*100:>9.1f}% {zone:>6}")

        # 对比图
        fig, ax = plt.subplots(figsize=(12, 7))
        fem_maxd = np.max(np.abs(fem_defl), axis=1)
        pinn_maxd = np.max(np.abs(all_pinn_deflections), axis=1)
        pinn_sel = [np.argmin(np.abs(loads - P)) for P in fem_loads]
        pinn_maxd_sel = np.array([pinn_maxd[j] for j in pinn_sel])
        elastic_ref = fem_loads * L**3 / (48 * EI)

        ax.plot(fem_loads / 1000, elastic_ref, "b--", lw=2, label="Linear Elastic", zorder=2)
        ax.plot(fem_loads / 1000, fem_maxd, "ko-", lw=2.5, ms=5, label="FEM", zorder=3)
        ax.plot(fem_loads / 1000, pinn_maxd_sel, "r^--", lw=2, ms=5, label="Pure PINN v2", zorder=4)
        ax.axvline(x=P_YIELD / 1000, color="gray", ls="--", lw=1.5, alpha=0.7)
        ax.set_xlabel("Load (kN)", fontsize=13)
        ax.set_ylabel("Max Deflection (mm)", fontsize=13)
        ax.set_title("Load-Deflection: FEM vs Pure PINN v2", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fem_vs_pure_pinn.png"), dpi=150)
        plt.close(fig)
        print("  [PNG] fem_vs_pure_pinn.png")
    else:
        print("\n  [提示] 未找到 FEM 数据, 跳过对比")

    print(f"\n  输出目录: {os.path.abspath(output_dir)}")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs", "pinn_pure_results")

    config = {
        # 模型
        "num_layers": 5,
        "hidden_size": 64,
        # 训练
        "epochs": 30000,
        "epochs_warmup": 7500,  # β: 0→1 的预热期
        "lr": 5e-4,
        "log_freq": 2000,
        # 配点
        "n_pde_points": 8000,
        "n_mid_points": 3000,  # 跨中加密
        "n_plastic_points": 3000,  # 塑性区加密
        "n_yield_boundary": 2000,  # 屈服边界加密
    }

    train(output_dir, config)
