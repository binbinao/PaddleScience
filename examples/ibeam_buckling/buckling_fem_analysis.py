"""
3米工字梁弹塑性分析 FEM (Euler-Bernoulli 梁 + 材料非线性)

载荷范围: 10 kN ~ 250 kN (步长 10 kN, 25 工况)
覆盖: 线弹性 → 屈服平台 → 应变强化 → 深度塑性

方法: 对每个载荷，用截面纤维法 (fiber method) 计算弹塑性弯矩-曲率关系，
      迭代求解非线性挠度。
"""

import os
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# Q235B 弹塑性本构 (四阶段)
# =============================================================================
E_MAT = 206000.0    # MPa
SIGMA_Y = 235.0     # MPa
SIGMA_U = 420.0     # MPa
EPS_Y = SIGMA_Y / E_MAT   # 0.00114
EPS_ST = 0.015             # 强化起始
EPS_U = 0.20               # 抗拉应变
JC_B = 230.2               # MPa
JC_N = 0.578


def stress_from_strain(eps):
    """Q235B 弹塑性本构: 工程应变 → 工程应力 (标量或数组)"""
    eps = np.asarray(eps, dtype=float)
    sig = np.zeros_like(eps)
    ae = np.abs(eps)
    sign = np.sign(eps)

    # 阶段1: 弹性
    m1 = ae <= EPS_Y
    sig[m1] = E_MAT * eps[m1]

    # 阶段2: 屈服平台
    m2 = (ae > EPS_Y) & (ae <= EPS_ST)
    sig[m2] = sign[m2] * SIGMA_Y

    # 阶段3: 应变强化 (J-C)
    m3 = (ae > EPS_ST) & (ae <= EPS_U)
    eps_pl = ae[m3] - SIGMA_Y / E_MAT
    sig[m3] = sign[m3] * (SIGMA_Y + JC_B * np.power(np.maximum(eps_pl, 1e-12), JC_N))

    # 阶段4: 超过抗拉 → 限制在 σ_u
    m4 = ae > EPS_U
    sig[m4] = sign[m4] * SIGMA_U

    return sig


def tangent_modulus(eps):
    """切线模量 dσ/dε"""
    eps = np.asarray(eps, dtype=float)
    Et = np.zeros_like(eps)
    ae = np.abs(eps)

    m1 = ae <= EPS_Y
    Et[m1] = E_MAT

    m2 = (ae > EPS_Y) & (ae <= EPS_ST)
    Et[m2] = 0.0  # 屈服平台

    m3 = (ae > EPS_ST) & (ae <= EPS_U)
    eps_pl = ae[m3] - SIGMA_Y / E_MAT
    Et[m3] = JC_B * JC_N * np.power(np.maximum(eps_pl, 1e-12), JC_N - 1)

    m4 = ae > EPS_U
    Et[m4] = 0.0  # 后屈服平台

    return Et


# =============================================================================
# 截面纤维法: 计算弯矩-曲率关系 M(κ)
# =============================================================================
def compute_moment_curvature(kappa, h, b, t_f, t_w, n_fibers=40):
    """
    给定曲率 κ, 用纤维法计算 I25a 截面的弯矩 M。
    假设平截面假设: ε(y) = κ * y, 中性轴在截面中心。
    
    I25a 截面: 上翼缘 + 腹板 + 下翼缘
    """
    h_w = h - 2 * t_f  # 腹板净高

    # 纤维位置 (y: 从底到顶, 0 = 中性轴)
    fibers_y = []
    fibers_A = []

    # 下翼缘
    n_flange = max(4, n_fibers // 5)
    y_bot = np.linspace(-h/2 + t_f/(2*n_flange), -h/2 + t_f - t_f/(2*n_flange), n_flange)
    for y in y_bot:
        fibers_y.append(y)
        fibers_A.append(b * t_f / n_flange)

    # 腹板
    n_web = n_fibers - 2 * n_flange
    y_web = np.linspace(-h_w/2 + h_w/(2*n_web), h_w/2 - h_w/(2*n_web), n_web)
    for y in y_web:
        fibers_y.append(y)
        fibers_A.append(t_w * h_w / n_web)

    # 上翼缘
    y_top = np.linspace(h/2 - t_f + t_f/(2*n_flange), h/2 - t_f/(2*n_flange), n_flange)
    for y in y_top:
        fibers_y.append(y)
        fibers_A.append(b * t_f / n_flange)

    fibers_y = np.array(fibers_y)
    fibers_A = np.array(fibers_A)

    # 应变分布
    strains = kappa * fibers_y
    stresses = stress_from_strain(strains)

    M = np.sum(stresses * fibers_y * fibers_A)
    return M


def compute_EI_effective(kappa, h, b, t_f, t_w, n_fibers=40):
    """计算等效弯曲刚度 EI_eff = M/κ"""
    if abs(kappa) < 1e-15:
        # 纯弹性
        h_w = h - 2*t_f
        Ix = (b*h**3 - (b-t_w)*h_w**3) / 12.0
        return E_MAT * Ix
    M = compute_moment_curvature(kappa, h, b, t_f, t_w, n_fibers)
    return M / kappa


# =============================================================================
# 非线性 FEM 求解器 (Newton-Raphson 迭代)
# =============================================================================
def fem_solve_nonlinear(P, L, h, b, t_f, t_w, num_elem=100, tol=1e-6, max_iter=50):
    """弹塑性梁 FEM 求解: 迭代更新每个单元的 EI"""
    Ix_elastic = (b*h**3 - (b-t_w)*(h-2*t_f)**3) / 12.0
    EI_elastic = E_MAT * Ix_elastic
    le = L / num_elem
    n_nodes = num_elem + 1
    n_dof = 2 * n_nodes
    load_node = num_elem // 2
    x_nodes = np.linspace(0, L, n_nodes)

    # 初始 EI 为弹性值
    EI_elem = np.full(num_elem, EI_elastic)

    for iteration in range(max_iter):
        # 组装刚度矩阵 (每个单元用各自的 EI)
        K = np.zeros((n_dof, n_dof))
        for i in range(num_elem):
            c = EI_elem[i] / le**3
            ke = c * np.array([
                [12, 6*le, -12, 6*le],
                [6*le, 4*le**2, -6*le, 2*le**2],
                [-12, -6*le, 12, -6*le],
                [6*le, 2*le**2, -6*le, 4*le**2],
            ])
            d = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            for a in range(4):
                for b_ in range(4):
                    K[d[a], d[b_]] += ke[a, b_]

        # 力向量
        F = np.zeros(n_dof)
        F[2 * load_node] = -P

        # 边界条件
        big = 1e20
        for bc in [0, 2*(n_nodes-1)]:
            K[bc, :] = 0; K[:, bc] = 0; K[bc, bc] = big; F[bc] = 0

        # 求解
        U = np.linalg.solve(K, F)
        deflections = U[0::2]

        # 更新每个单元的 EI: 计算单元中点曲率, 再查 M-κ 关系
        EI_new = np.zeros(num_elem)
        curvatures = np.zeros(num_elem)
        moments_elem = np.zeros(num_elem)

        for i in range(num_elem):
            d = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            u_e = U[d]
            # 单元中点曲率: κ = d²w/dx² ≈ (6/le²)(-w_i - le*θ_i/2 + w_j - le*θ_j/2)
            # 更简单: 用 Hermite 插值在 x=le/2 处的二阶导
            kappa_mid = (1.0/le**2) * np.dot(np.array([-6, -2*le, 6, -2*le]), u_e)
            # 按照上一步计算: 在x=le/2处, N''(le/2) 的系数
            # 实际上用两端转角近似: κ ≈ (θ_j - θ_i) / le
            kappa_approx = (u_e[3] - u_e[1]) / le
            kappa = kappa_approx

            curvatures[i] = kappa
            M = compute_moment_curvature(kappa, h, b, t_f, t_w)
            moments_elem[i] = M

            if abs(kappa) > 1e-15:
                EI_new[i] = abs(M / kappa)
            else:
                EI_new[i] = EI_elastic

            # 限制 EI 不超过弹性值, 不低于某个最小值 (数值稳定)
            EI_new[i] = min(EI_new[i], EI_elastic)
            EI_new[i] = max(EI_new[i], EI_elastic * 0.001)

        # 收敛检查
        if iteration > 0:
            rel_change = np.max(np.abs(EI_new - EI_elem) / (EI_elastic + 1e-10))
            if rel_change < tol:
                break

        # 松弛更新 (防止振荡)
        alpha = 0.5
        EI_elem = alpha * EI_new + (1 - alpha) * EI_elem

    # 后处理: 节点弯矩和应力
    moments = np.zeros(n_nodes)
    stresses = np.zeros(n_nodes)
    kappas = np.zeros(n_nodes)

    for i in range(num_elem):
        # 节点弯矩取相邻单元平均
        if i == 0:
            moments[i] = moments_elem[i]
            kappas[i] = curvatures[i]
        moments[i+1] = moments_elem[i]
        kappas[i+1] = curvatures[i]
        if i > 0:
            moments[i] = 0.5*(moments_elem[i-1] + moments_elem[i])
            kappas[i] = 0.5*(curvatures[i-1] + curvatures[i])

    # 应力: 截面边缘最大应力
    for i in range(n_nodes):
        edge_strain = kappas[i] * h / 2.0
        stresses[i] = abs(stress_from_strain(edge_strain))

    stress_ratios = stresses / SIGMA_Y

    return x_nodes, deflections, moments, stresses, stress_ratios, iteration+1


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(script_dir, "conf", "buckling_config.yaml"))

    L = cfg["geometry"]["beam_length"]
    h = cfg["section"]["h"]
    b_ = cfg["section"]["b"]
    t_w = cfg["section"]["t_w"]
    t_f = cfg["section"]["t_f"]
    Wx = cfg["section"]["Wx"]
    Ix = cfg["section"]["Ix"]
    num_elem = cfg["fem"]["num_elements"]

    P_min = cfg["loads"]["P_min"]
    P_max = cfg["loads"]["P_max"]
    P_step = cfg["loads"]["P_step"]
    loads = np.arange(P_min, P_max + P_step/2, P_step)

    out_dir = os.path.join(script_dir, cfg["output"]["dir"])
    os.makedirs(out_dir, exist_ok=True)

    P_yield_theory = 4 * SIGMA_Y * Wx / L

    print("=" * 70)
    print("  I25a 工字梁弹塑性分析 FEM (10~250 kN)")
    print("=" * 70)
    print(f"  材料: Q235B 弹塑性 (σ_y={SIGMA_Y}, σ_u={SIGMA_U} MPa)")
    print(f"  理论屈服载荷: {P_yield_theory:.0f} N = {P_yield_theory/1000:.2f} kN")
    print(f"  工况数: {len(loads)}, 载荷: {loads[0]/1000:.0f}~{loads[-1]/1000:.0f} kN")
    print(f"  求解方法: 纤维截面法 + Newton-Raphson 迭代")
    print("=" * 70)

    all_defl = []
    all_mom = []
    all_stress = []
    all_ratio = []
    summary = []

    # 也计算线弹性参考
    elastic_max_defls = []

    for idx, P in enumerate(loads):
        x, defl, mom, stress, ratio, n_iter = fem_solve_nonlinear(
            P, L, h, b_, t_f, t_w, num_elem)
        all_defl.append(defl)
        all_mom.append(mom)
        all_stress.append(stress)
        all_ratio.append(ratio)

        max_defl = np.max(np.abs(defl))
        max_stress = np.max(stress)
        max_ratio = np.max(ratio)
        elastic_defl = P * L**3 / (48 * E_MAT * Ix)
        elastic_max_defls.append(elastic_defl)
        nonlin_ratio = max_defl / elastic_defl if elastic_defl > 0 else 1.0

        status = "弹性" if max_ratio < 1.0 else "塑性"
        print(f"  LC-{idx+1:02d}: P={P/1000:6.0f} kN | "
              f"δ={max_defl:.4f} mm (弹性参考:{elastic_defl:.4f}) | "
              f"非线性比={nonlin_ratio:.3f} | "
              f"σ_max={max_stress:.1f} MPa | σ/σy={max_ratio*100:.1f}% | "
              f"iter={n_iter} | {status}")

        summary.append({
            "P_kN": P/1000, "P_N": P,
            "max_defl_mm": max_defl, "elastic_defl_mm": elastic_defl,
            "nonlinear_ratio": nonlin_ratio,
            "max_stress_MPa": max_stress, "stress_ratio": max_ratio,
            "n_iter": n_iter,
        })

    x_nodes = np.linspace(0, L, num_elem + 1)
    all_defl = np.array(all_defl)
    all_mom = np.array(all_mom)
    all_stress = np.array(all_stress)
    all_ratio = np.array(all_ratio)

    # 保存
    np.savez(
        os.path.join(out_dir, "fem_training_data.npz"),
        x_nodes=x_nodes, loads=loads,
        deflections=all_defl, moments=all_mom,
        stresses=all_stress, stress_ratios=all_ratio,
        E=E_MAT, Ix=Ix, Wx=Wx, L=L, sigma_y=SIGMA_Y, sigma_u=SIGMA_U,
    )

    with open(os.path.join(out_dir, "results_summary.csv"), "w") as f:
        f.write("load_case,P_kN,P_N,max_defl_mm,elastic_defl_mm,nonlinear_ratio,"
                "max_stress_MPa,stress_ratio,n_iter\n")
        for i, s in enumerate(summary):
            f.write(f"LC-{i+1:02d},{s['P_kN']:.0f},{s['P_N']:.0f},"
                    f"{s['max_defl_mm']:.6f},{s['elastic_defl_mm']:.6f},"
                    f"{s['nonlinear_ratio']:.6f},{s['max_stress_MPa']:.4f},"
                    f"{s['stress_ratio']:.4f},{s['n_iter']}\n")

    print(f"\n  [NPZ] {out_dir}/fem_training_data.npz")
    print(f"  [CSV] {out_dir}/results_summary.csv")

    # ============ 可视化 ============
    print("\n  生成图表...")
    x_m = x_nodes / 1000
    P_kN = loads / 1000

    # 颜色: 弹性阶段绿色 → 屈服区黄色 → 塑性区红色
    colors = plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, len(loads)))

    # 图1: 挠度曲线
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, P in enumerate(loads):
        lbl = f"{P/1000:.0f}kN" if i % 4 == 0 or i == len(loads)-1 else None
        ax.plot(x_m, -all_defl[i], color=colors[i], linewidth=1.2, label=lbl)
    ax.set_xlabel("Position (m)", fontsize=12)
    ax.set_ylabel("Deflection (mm)", fontsize=12)
    ax.set_title("I25a Beam: Elastic-Plastic Deflection (10~250 kN)", fontsize=14)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "deflection_curves.png"), dpi=150)
    plt.close(fig)

    # 图2: 载荷-挠度 (关键图: 线弹性 vs 弹塑性)
    fig, ax = plt.subplots(figsize=(12, 7))
    max_defls = np.max(np.abs(all_defl), axis=1)
    elastic_refs = np.array(elastic_max_defls)

    ax.plot(P_kN, elastic_refs, "b--", linewidth=2, label="Linear Elastic (reference)", zorder=2)
    ax.plot(P_kN, max_defls, "ro-", linewidth=2.5, markersize=6, label="Elastic-Plastic (FEM)", zorder=3)

    # 标注屈服点
    ax.axvline(x=P_yield_theory/1000, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.annotate(f"Yield onset\nP={P_yield_theory/1000:.1f} kN",
                xy=(P_yield_theory/1000, elastic_refs[int(P_yield_theory/P_step)-1]),
                fontsize=10, ha="right", xytext=(-15, 20), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="gray"),
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

    # 非线性区域填充
    ax.fill_between(P_kN, elastic_refs, max_defls,
                    where=max_defls > elastic_refs*1.01,
                    alpha=0.15, color="red", label="Plastic additional deflection")

    ax.set_xlabel("Applied Load (kN)", fontsize=13)
    ax.set_ylabel("Max Deflection (mm)", fontsize=13)
    ax.set_title("Load-Deflection: Linear vs Elastic-Plastic Response", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "load_deflection_nonlinear.png"), dpi=150)
    plt.close(fig)

    # 图3: 应力比
    fig, ax = plt.subplots(figsize=(12, 7))
    max_ratios = np.max(all_ratio, axis=1) * 100
    ax.plot(P_kN, max_ratios, "rs-", linewidth=2, markersize=6)
    ax.axhline(y=100, color="red", ls="--", lw=2, alpha=0.7, label="Yield (100%)")
    ax.axhline(y=SIGMA_U/SIGMA_Y*100, color="darkred", ls=":", lw=1.5, alpha=0.7,
               label=f"Ultimate ({SIGMA_U/SIGMA_Y*100:.0f}%)")
    ax.fill_between(P_kN, 0, max_ratios, where=np.array(max_ratios)<100,
                    alpha=0.1, color="green", label="Elastic region")
    ax.fill_between(P_kN, 0, max_ratios, where=np.array(max_ratios)>=100,
                    alpha=0.1, color="red", label="Plastic region")
    ax.set_xlabel("Applied Load (kN)", fontsize=13)
    ax.set_ylabel("Max Stress Ratio σ/σ_y (%)", fontsize=13)
    ax.set_title("Stress Utilization vs Load", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stress_ratio_vs_load.png"), dpi=150)
    plt.close(fig)

    print("  [PNG] deflection_curves.png")
    print("  [PNG] load_deflection_nonlinear.png")
    print("  [PNG] stress_ratio_vs_load.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
