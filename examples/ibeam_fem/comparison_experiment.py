"""
FEM vs PINNs 对比实验
针对 P=1350N 和 P=3720N 两个未见载荷，对比两种方法的精度和速度。
"""
import os, sys, time
import numpy as np
import paddle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

paddle.set_default_dtype("float64")

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# ============ 物理常数 ============
E = 206000.0;  Ix = 50170000.0;  Wx = 398200.0;  L = 3000.0
EI = E * Ix;   P_REF = 5000.0;   X_REF = L
W_REF = P_REF * L**3 / (48.0 * EI)

# ============ FEM 求解函数 (从 ibeam_fem_analysis 提取核心) ============
def fem_solve_single(P, num_elem=100):
    """对单个载荷 P 做完整 FEM 求解，返回节点挠度和用时"""
    le = L / num_elem
    n_nodes = num_elem + 1
    n_dof = 2 * n_nodes

    t0 = time.perf_counter()

    # 单元刚度矩阵
    coeff = EI / (le ** 3)
    ke = coeff * np.array([
        [12, 6*le, -12, 6*le],
        [6*le, 4*le**2, -6*le, 2*le**2],
        [-12, -6*le, 12, -6*le],
        [6*le, 2*le**2, -6*le, 4*le**2],
    ])
    # 组装整体刚度矩阵
    K = np.zeros((n_dof, n_dof))
    for i in range(num_elem):
        dof = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        for a in range(4):
            for b in range(4):
                K[dof[a], dof[b]] += ke[a, b]
    # 力向量
    F = np.zeros(n_dof)
    load_node = num_elem // 2
    F[2 * load_node] = -P
    # 边界条件 (大数法)
    bc_dofs = [0, 2*(n_nodes-1)]
    big = 1e20
    for d in bc_dofs:
        K[d, :] = 0; K[:, d] = 0; K[d, d] = big; F[d] = 0
    # 求解
    U = np.linalg.solve(K, F)
    deflections = U[0::2]

    # 弯矩
    moments = np.zeros(n_nodes)
    for i in range(num_elem):
        dof = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        u_e = U[dof]
        f_e = ke @ u_e
        M_right = -f_e[3]
        if i == 0:
            moments[i] = f_e[1]
        moments[i+1] = M_right

    stresses = np.abs(moments) / Wx
    t_fem = time.perf_counter() - t0
    x_nodes = np.linspace(0, L, n_nodes)
    return x_nodes, deflections, moments, stresses, t_fem

# ============ PINNs 预测函数 ============
from ibeam_pinn_train import IBeamPINN

def pinn_predict_single(model, P, x_nodes):
    """用 PINNs 模型预测单个载荷，返回预测结果和用时"""
    inp = np.hstack([
        x_nodes.reshape(-1,1) / X_REF,
        np.full((len(x_nodes),1), P / P_REF),
    ])
    t0 = time.perf_counter()
    with paddle.no_grad():
        w_norm = model(paddle.to_tensor(inp)).numpy().flatten()
    t_pinn = time.perf_counter() - t0
    w_mm = w_norm * W_REF
    return w_mm, t_pinn

# ============ 解析解 ============
def analytical_deflection(x, P):
    w = np.zeros_like(x)
    half = L / 2.0
    c = P / (48.0 * EI)
    left = x <= half
    w[left] = c * x[left] * (3*L**2 - 4*x[left]**2)
    xr = L - x[~left]
    w[~left] = c * xr * (3*L**2 - 4*xr**2)
    return w

# ============ 主程序 ============
if __name__ == "__main__":
    test_loads = [1350.0, 3720.0]

    # 加载PINNs模型
    model = IBeamPINN(num_layers=3, hidden_size=32)
    model.set_state_dict(paddle.load(os.path.join(script_dir, "outputs/pinn_results/best_model.pdparams")))
    model.eval()

    # 预热 GPU
    dummy = paddle.to_tensor(np.zeros((10,2)))
    with paddle.no_grad(): _ = model(dummy)

    results = []
    for P in test_loads:
        print(f"\n{'='*70}")
        print(f"  载荷 P = {P:.0f} N  对比实验")
        print(f"{'='*70}")

        # --- FEM ---
        # 多次运行取平均
        fem_times = []
        for _ in range(20):
            x_n, defl, mom, stress, t = fem_solve_single(P)
            fem_times.append(t)
        x_nodes, defl_fem, moments_fem, stress_fem, _ = fem_solve_single(P)
        t_fem_avg = np.mean(fem_times)
        t_fem_std = np.std(fem_times)

        # --- PINNs ---
        pinn_times = []
        for _ in range(100):
            _, t = pinn_predict_single(model, P, x_nodes)
            pinn_times.append(t)
        w_pinn, _ = pinn_predict_single(model, P, x_nodes)
        t_pinn_avg = np.mean(pinn_times)
        t_pinn_std = np.std(pinn_times)

        # --- 解析解 ---
        w_analytical = analytical_deflection(x_nodes, P)

        # 关键指标
        fem_max = np.max(np.abs(defl_fem))
        pinn_max = np.max(np.abs(w_pinn))
        ana_max = P * L**3 / (48*EI)
        stress_max = np.max(stress_fem)
        safety_factor = 235.0 / stress_max if stress_max > 0 else float('inf')

        fem_vs_ana = abs(fem_max - ana_max) / ana_max * 100
        pinn_vs_ana = abs(pinn_max - ana_max) / ana_max * 100
        pinn_vs_fem = abs(pinn_max - fem_max) / fem_max * 100
        speedup = t_fem_avg / t_pinn_avg

        print(f"\n  解析解 δ_max = {ana_max:.6f} mm")
        print(f"  FEM    δ_max = {fem_max:.6f} mm  (vs 解析: {fem_vs_ana:.4f}%)")
        print(f"  PINNs  δ_max = {pinn_max:.6f} mm (vs 解析: {pinn_vs_ana:.4f}%)")
        print(f"  PINNs vs FEM 误差: {pinn_vs_fem:.4f}%")
        print(f"  最大应力: {stress_max:.4f} MPa, 安全系数: {safety_factor:.1f}")
        print(f"\n  FEM 用时:   {t_fem_avg*1000:.3f} ± {t_fem_std*1000:.3f} ms (20次平均)")
        print(f"  PINNs 用时: {t_pinn_avg*1000:.3f} ± {t_pinn_std*1000:.3f} ms (100次平均)")
        print(f"  加速比: PINNs 比 FEM 快 {speedup:.1f}x")

        results.append({
            "P": P, "ana_max": ana_max, "fem_max": fem_max, "pinn_max": pinn_max,
            "fem_vs_ana": fem_vs_ana, "pinn_vs_ana": pinn_vs_ana, "pinn_vs_fem": pinn_vs_fem,
            "t_fem": t_fem_avg, "t_pinn": t_pinn_avg, "speedup": speedup,
            "stress_max": stress_max, "safety_factor": safety_factor,
            "x_nodes": x_nodes, "defl_fem": defl_fem, "w_pinn": w_pinn, "w_analytical": w_analytical,
        })

    # ============ 对比可视化 ============
    out_dir = os.path.join(script_dir, "outputs", "comparison")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, r in enumerate(results):
        ax = axes[idx]
        x_m = r["x_nodes"] / 1000
        ax.plot(x_m, r["w_analytical"], "g-", linewidth=2.5, label="Analytical", zorder=3)
        ax.plot(x_m, -r["defl_fem"], "b--", linewidth=2, label=f"FEM (err={r['fem_vs_ana']:.2f}%)")
        ax.plot(x_m, -r["w_pinn"], "r:", linewidth=2.5, label=f"PINNs (err={r['pinn_vs_ana']:.2f}%)")
        ax.set_xlabel("Position (m)", fontsize=12)
        ax.set_ylabel("Deflection (mm)", fontsize=12)
        ax.set_title(f"P = {r['P']:.0f} N  |  FEM: {r['t_fem']*1000:.1f}ms  PINNs: {r['t_pinn']*1000:.2f}ms  ({r['speedup']:.0f}x faster)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("FEM vs PINNs vs Analytical: Deflection Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fem_vs_pinn_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [PNG] {out_dir}/fem_vs_pinn_comparison.png")

    # 速度对比柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"P={r['P']:.0f}N" for r in results]
    fem_t = [r["t_fem"]*1000 for r in results]
    pinn_t = [r["t_pinn"]*1000 for r in results]
    x_pos = np.arange(len(labels))
    w = 0.35
    ax.bar(x_pos - w/2, fem_t, w, label="FEM", color="#4477AA")
    ax.bar(x_pos + w/2, pinn_t, w, label="PINNs", color="#EE6677")
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Computation Time: FEM vs PINNs", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    for i, (f, p) in enumerate(zip(fem_t, pinn_t)):
        ax.annotate(f"{f:.1f}ms", (i-w/2, f), ha="center", va="bottom", fontsize=9)
        ax.annotate(f"{p:.2f}ms", (i+w/2, p), ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "time_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  [PNG] {out_dir}/time_comparison.png")

    print("\n  对比实验完成!")
