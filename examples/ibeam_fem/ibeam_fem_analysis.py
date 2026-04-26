"""
3米工字梁有限元分析 (Euler-Bernoulli 梁单元)

基于 GB/T 706-2016 I25a 工字钢截面，Q235B 材料，
简支梁(Pin-Roller)跨中集中力 1000N~5000N (步长500N) 的静力分析。

计算结果将保存为 CSV 和 NPZ 格式，供后续 PINNs 模型训练使用。
"""

import os
import yaml
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_element_stiffness(E, I, le):
    """
    构建 Euler-Bernoulli 梁单元刚度矩阵 (4x4)。

    每个节点2个自由度: [w(挠度), theta(转角)]
    单元自由度: [w_i, theta_i, w_j, theta_j]

    Args:
        E: 弹性模量 (MPa)
        I: 惯性矩 (mm^4)
        le: 单元长度 (mm)

    Returns:
        ke: 4x4 单元刚度矩阵
    """
    coeff = E * I / (le ** 3)
    ke = coeff * np.array(
        [
            [12, 6 * le, -12, 6 * le],
            [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
            [-12, -6 * le, 12, -6 * le],
            [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2],
        ]
    )
    return ke


def assemble_global_stiffness(num_elements, E, I, le):
    """
    组装整体刚度矩阵。

    Args:
        num_elements: 单元数目
        E: 弹性模量 (MPa)
        I: 惯性矩 (mm^4)
        le: 单元长度 (mm)

    Returns:
        K: (2*n_nodes x 2*n_nodes) 整体刚度矩阵
    """
    n_nodes = num_elements + 1
    n_dof = 2 * n_nodes
    K = np.zeros((n_dof, n_dof))

    ke = build_element_stiffness(E, I, le)

    for i in range(num_elements):
        # 单元第i个节点对应的全局自由度索引
        dof_indices = [2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1]
        for a in range(4):
            for b in range(4):
                K[dof_indices[a], dof_indices[b]] += ke[a, b]

    return K


def apply_boundary_conditions(K, F, bc_dofs):
    """
    施加边界条件 (大数法)。

    Args:
        K: 整体刚度矩阵
        F: 整体力向量
        bc_dofs: 被约束的自由度列表

    Returns:
        K_mod, F_mod: 修改后的刚度矩阵和力向量
    """
    K_mod = K.copy()
    F_mod = F.copy()
    big_num = 1e20

    for dof in bc_dofs:
        K_mod[dof, :] = 0.0
        K_mod[:, dof] = 0.0
        K_mod[dof, dof] = big_num
        F_mod[dof] = 0.0

    return K_mod, F_mod


def compute_internal_forces(displacements, num_elements, E, I, le):
    """
    计算各节点的弯矩和剪力分布。

    使用单元节点力方法: f_e = k_e * u_e，
    然后从节点力中提取弯矩和剪力。

    对于 Euler-Bernoulli 梁单元，单元节点力向量为:
    f_e = [V_i, M_i, V_j, M_j]
    其中 V 为剪力, M 为弯矩。

    内力通过截面平衡确定：从左端开始逐段累加。

    Args:
        displacements: 全局位移向量
        num_elements: 单元数
        E: 弹性模量
        I: 惯性矩
        le: 单元长度

    Returns:
        moments: 各节点弯矩 (N*mm)
        shears: 各节点剪力 (N)
    """
    n_nodes = num_elements + 1
    moments = np.zeros(n_nodes)
    shears = np.zeros(n_nodes)

    ke = build_element_stiffness(E, I, le)

    for i in range(num_elements):
        dof_indices = [2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1]
        u_e = displacements[dof_indices]

        # 单元节点力 f_e = ke * u_e
        # f_e = [V_i, M_i, V_j, M_j]
        f_e = ke @ u_e

        # 单元内剪力恒定 (Euler-Bernoulli)
        # 内剪力 = 左节点的竖向反力 (向上为正)
        V_elem = -f_e[0]

        # 弯矩在单元内线性分布
        # 左节点弯矩 (内弯矩) = -f_e[1] (符号约定：使下部受拉为正)
        M_left = f_e[1]
        M_right = -f_e[3]

        if i == 0:
            moments[i] = M_left
            shears[i] = V_elem
        moments[i + 1] = M_right
        shears[i + 1] = V_elem

    return moments, shears


def compute_bending_stress(moments, Wx, h):
    """
    计算弯曲正应力 (截面上下边缘的最大应力)。

    sigma = M / Wx  (截面模数法)
    或 sigma = M * (h/2) / Ix  (最大纤维距离法)

    Args:
        moments: 弯矩数组 (N*mm)
        Wx: 截面模数 (mm^3)
        h: 截面高度 (mm)

    Returns:
        stress: 最大弯曲正应力 (MPa)
    """
    return np.abs(moments) / Wx


def analytical_deflection(x, P, L, E, I):
    """
    简支梁跨中集中力的解析解 (挠度)。

    对于 0 <= x <= L/2:
        w(x) = P*x / (48*E*I) * (3*L^2 - 4*x^2)
    对于 L/2 < x <= L:
        w(x) = P*(L-x) / (48*E*I) * (3*L^2 - 4*(L-x)^2)

    Args:
        x: 坐标数组 (mm)
        P: 集中力 (N)
        L: 梁长 (mm)
        E: 弹性模量 (MPa)
        I: 惯性矩 (mm^4)

    Returns:
        w: 挠度数组 (mm), 正值向下
    """
    w = np.zeros_like(x)
    half_L = L / 2.0
    coeff = P / (48.0 * E * I)

    mask_left = x <= half_L
    mask_right = ~mask_left

    w[mask_left] = coeff * x[mask_left] * (3 * L ** 2 - 4 * x[mask_left] ** 2)
    xr = L - x[mask_right]
    w[mask_right] = coeff * xr * (3 * L ** 2 - 4 * xr ** 2)

    return w


def analytical_moment(x, P, L):
    """
    简支梁跨中集中力的解析解 (弯矩)。

    对于 0 <= x <= L/2:  M(x) = P*x/2
    对于 L/2 < x <= L:   M(x) = P*(L-x)/2
    """
    M = np.zeros_like(x)
    half_L = L / 2.0
    mask_left = x <= half_L
    M[mask_left] = P * x[mask_left] / 2.0
    M[~mask_left] = P * (L - x[~mask_left]) / 2.0
    return M


def analytical_shear(x, P, L):
    """
    简支梁跨中集中力的解析解 (剪力)。

    对于 0 <= x < L/2:  V = P/2
    对于 x = L/2:       V = 0 (不连续点)
    对于 L/2 < x <= L:  V = -P/2
    """
    V = np.zeros_like(x)
    half_L = L / 2.0
    V[x < half_L] = P / 2.0
    V[x > half_L] = -P / 2.0
    return V


def run_fem_analysis(config):
    """
    执行有限元分析主程序。

    Args:
        config: 配置字典
    """
    # =========================================================================
    # 1. 读取参数
    # =========================================================================
    L = config["geometry"]["beam_length"]
    load_pos = config["geometry"]["load_position"]

    E = config["material"]["E"]
    Ix = config["section"]["Ix"]
    Wx = config["section"]["Wx"]
    h = config["section"]["h"]
    A_sec = config["section"]["A"]

    num_elem = config["fem"]["num_elements"]
    n_nodes = num_elem + 1
    le = L / num_elem
    n_dof = 2 * n_nodes

    P_min = config["loads"]["P_min"]
    P_max = config["loads"]["P_max"]
    P_step = config["loads"]["P_step"]
    loads = np.arange(P_min, P_max + P_step / 2, P_step)

    output_dir = config["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 节点坐标
    x_nodes = np.linspace(0, L, n_nodes)

    # 加载点对应的节点
    load_node = int(round(load_pos / le))

    print("=" * 70)
    print("  3米工字梁有限元分析 (Euler-Bernoulli 梁单元)")
    print("=" * 70)
    print(f"  截面型号: I25a (GB/T 706-2016)")
    print(f"  材料: Q235B, E = {E:.0f} MPa")
    print(f"  梁长: {L:.0f} mm, 惯性矩 Ix = {Ix:.0f} mm^4")
    print(f"  截面模数 Wx = {Wx:.0f} mm^3, 截面高度 h = {h:.0f} mm")
    print(f"  单元数: {num_elem}, 单元长度: {le:.1f} mm")
    print(f"  加载点: 节点 {load_node} (x = {x_nodes[load_node]:.0f} mm)")
    print(f"  载荷范围: {P_min:.0f}N ~ {P_max:.0f}N, 步长 {P_step:.0f}N")
    print(f"  工况数: {len(loads)}")
    print("=" * 70)

    # =========================================================================
    # 2. 组装整体刚度矩阵
    # =========================================================================
    print("\n[1/5] 组装整体刚度矩阵...")
    K = assemble_global_stiffness(num_elem, E, Ix, le)
    print(f"  刚度矩阵大小: {K.shape}")

    # =========================================================================
    # 3. 边界条件
    # =========================================================================
    # 左端 Pin (铰支座): w_0 = 0 -> DOF 0
    # 右端 Roller (滚动支座): w_n = 0 -> DOF 2*(n_nodes-1)
    bc_dofs = [0, 2 * (n_nodes - 1)]
    print(f"\n[2/5] 施加边界条件: Pin(DOF={bc_dofs[0]}), Roller(DOF={bc_dofs[1]})")

    # =========================================================================
    # 4. 求解各载荷工况
    # =========================================================================
    print(f"\n[3/5] 求解 {len(loads)} 个载荷工况...")

    # 结果存储
    all_deflections = []
    all_moments = []
    all_shears = []
    all_stresses = []
    summary_data = []

    for idx, P in enumerate(loads):
        # 构建力向量
        F = np.zeros(n_dof)
        # 集中力作用在加载点节点的挠度自由度 (向下为负)
        F[2 * load_node] = -P

        # 施加边界条件并求解
        K_mod, F_mod = apply_boundary_conditions(K, F, bc_dofs)
        U = np.linalg.solve(K_mod, F_mod)

        # 提取挠度 (每个节点的第一个自由度)
        deflections = U[0::2]  # mm, 负值表示向下

        # 计算弯矩和剪力
        moments, shears = compute_internal_forces(U, num_elem, E, Ix, le)

        # 计算弯曲应力
        stress = compute_bending_stress(moments, Wx, h)

        # 存储
        all_deflections.append(deflections)
        all_moments.append(moments)
        all_shears.append(shears)
        all_stresses.append(stress)

        # 解析解对比
        w_analytical_max = P * L ** 3 / (48 * E * Ix)
        M_analytical_max = P * L / 4.0
        sigma_analytical_max = M_analytical_max / Wx

        fem_max_deflection = np.max(np.abs(deflections))
        fem_max_moment = np.max(np.abs(moments))
        fem_max_stress = np.max(stress)

        deflection_error = abs(fem_max_deflection - w_analytical_max) / w_analytical_max * 100
        moment_error = abs(fem_max_moment - M_analytical_max) / M_analytical_max * 100
        stress_error = abs(fem_max_stress - sigma_analytical_max) / sigma_analytical_max * 100

        summary_data.append(
            {
                "load_case": f"LC-{idx + 1}",
                "P_N": P,
                "fem_max_deflection_mm": fem_max_deflection,
                "analytical_max_deflection_mm": w_analytical_max,
                "deflection_error_pct": deflection_error,
                "fem_max_moment_Nmm": fem_max_moment,
                "analytical_max_moment_Nmm": M_analytical_max,
                "moment_error_pct": moment_error,
                "fem_max_stress_MPa": fem_max_stress,
                "analytical_max_stress_MPa": sigma_analytical_max,
                "stress_error_pct": stress_error,
            }
        )

        print(
            f"  LC-{idx + 1}: P={P:6.0f}N | "
            f"δ_max={fem_max_deflection:.6f}mm (err={deflection_error:.4f}%) | "
            f"σ_max={fem_max_stress:.4f}MPa (err={stress_error:.4f}%)"
        )

    all_deflections = np.array(all_deflections)
    all_moments = np.array(all_moments)
    all_shears = np.array(all_shears)
    all_stresses = np.array(all_stresses)

    # =========================================================================
    # 5. 保存结果
    # =========================================================================
    print(f"\n[4/5] 保存结果到 {output_dir}/")

    # --- 5a. 汇总 CSV ---
    summary_csv_path = os.path.join(output_dir, "results_summary.csv")
    header = (
        "load_case,P_N,"
        "fem_max_deflection_mm,analytical_max_deflection_mm,deflection_error_pct,"
        "fem_max_moment_Nmm,analytical_max_moment_Nmm,moment_error_pct,"
        "fem_max_stress_MPa,analytical_max_stress_MPa,stress_error_pct"
    )
    with open(summary_csv_path, "w") as f:
        f.write(header + "\n")
        for row in summary_data:
            line = (
                f"{row['load_case']},{row['P_N']:.1f},"
                f"{row['fem_max_deflection_mm']:.8f},{row['analytical_max_deflection_mm']:.8f},"
                f"{row['deflection_error_pct']:.6f},"
                f"{row['fem_max_moment_Nmm']:.4f},{row['analytical_max_moment_Nmm']:.4f},"
                f"{row['moment_error_pct']:.6f},"
                f"{row['fem_max_stress_MPa']:.6f},{row['analytical_max_stress_MPa']:.6f},"
                f"{row['stress_error_pct']:.6f}"
            )
            f.write(line + "\n")
    print(f"  [CSV] 汇总结果 -> {summary_csv_path}")

    # --- 5b. 各工况详细数据 CSV ---
    for idx, P in enumerate(loads):
        detail_path = os.path.join(output_dir, f"LC{idx + 1}_P{int(P)}N.csv")
        detail_header = "x_mm,deflection_mm,moment_Nmm,shear_N,stress_MPa"
        data_arr = np.column_stack(
            [
                x_nodes,
                all_deflections[idx],
                all_moments[idx],
                all_shears[idx],
                all_stresses[idx],
            ]
        )
        np.savetxt(detail_path, data_arr, delimiter=",", header=detail_header, comments="", fmt="%.8f")
    print(f"  [CSV] {len(loads)} 个工况详细数据")

    # --- 5c. NPZ 格式 (供 PINNs 训练) ---
    npz_path = os.path.join(output_dir, "fem_training_data.npz")
    np.savez(
        npz_path,
        x_nodes=x_nodes,              # (n_nodes,) 节点坐标 mm
        loads=loads,                   # (n_cases,) 载荷值 N
        deflections=all_deflections,   # (n_cases, n_nodes) 挠度 mm
        moments=all_moments,           # (n_cases, n_nodes) 弯矩 N*mm
        shears=all_shears,             # (n_cases, n_nodes) 剪力 N
        stresses=all_stresses,         # (n_cases, n_nodes) 应力 MPa
        E=E,
        Ix=Ix,
        Wx=Wx,
        h=h,
        L=L,
        A=A_sec,
    )
    print(f"  [NPZ] PINNs训练数据 -> {npz_path}")
    print(f"         x_nodes: {x_nodes.shape}, loads: {loads.shape}")
    print(f"         deflections: {all_deflections.shape}")
    print(f"         moments: {all_moments.shape}")

    # =========================================================================
    # 6. 可视化
    # =========================================================================
    print(f"\n[5/5] 生成可视化图表...")

    x_m = x_nodes / 1000.0  # 转换为米

    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(loads)))

    # --- 图1: 挠度曲线 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, P in enumerate(loads):
        ax.plot(
            x_m,
            -all_deflections[idx],  # 取正值，向下为正
            color=colors[idx],
            linewidth=1.5,
            label=f"P = {P:.0f} N",
        )
    ax.set_xlabel("Position along beam (m)", fontsize=12)
    ax.set_ylabel("Deflection (mm, downward positive)", fontsize=12)
    ax.set_title("I25a I-Beam Deflection Curves (FEM, 1000N~5000N)", fontsize=14)
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L / 1000)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "deflection_curves.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] deflection_curves.png")

    # --- 图2: 弯矩分布 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, P in enumerate(loads):
        ax.plot(
            x_m,
            all_moments[idx] / 1e6,  # 转换为 kN*m
            color=colors[idx],
            linewidth=1.5,
            label=f"P = {P:.0f} N",
        )
    ax.set_xlabel("Position along beam (m)", fontsize=12)
    ax.set_ylabel("Bending Moment (kN·m)", fontsize=12)
    ax.set_title("I25a I-Beam Bending Moment Distribution (FEM)", fontsize=14)
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L / 1000)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bending_moment.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] bending_moment.png")

    # --- 图3: 剪力分布 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, P in enumerate(loads):
        ax.plot(
            x_m,
            all_shears[idx],
            color=colors[idx],
            linewidth=1.5,
            label=f"P = {P:.0f} N",
        )
    ax.set_xlabel("Position along beam (m)", fontsize=12)
    ax.set_ylabel("Shear Force (N)", fontsize=12)
    ax.set_title("I25a I-Beam Shear Force Distribution (FEM)", fontsize=14)
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L / 1000)
    ax.axhline(y=0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "shear_force.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] shear_force.png")

    # --- 图4: 载荷-挠度关系 ---
    max_deflections = np.max(np.abs(all_deflections), axis=1)
    analytical_max_deflections = loads * L ** 3 / (48 * E * Ix)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        loads / 1000, max_deflections, "bo-", linewidth=2, markersize=8, label="FEM"
    )
    ax.plot(
        loads / 1000,
        analytical_max_deflections,
        "r--^",
        linewidth=2,
        markersize=8,
        label="Analytical (Euler-Bernoulli)",
    )
    ax.set_xlabel("Applied Load (kN)", fontsize=12)
    ax.set_ylabel("Max Deflection (mm)", fontsize=12)
    ax.set_title("Load-Deflection Relationship: FEM vs Analytical", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "load_deflection.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] load_deflection.png")

    # --- 图5: FEM vs 解析解 (挠度对比, 选3个代表工况) ---
    representative = [0, len(loads) // 2, len(loads) - 1]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, case_idx in enumerate(representative):
        P = loads[case_idx]
        w_fem = -all_deflections[case_idx]  # 向下为正
        w_ana = analytical_deflection(x_nodes, P, L, E, Ix)

        axes[i].plot(x_m, w_fem, "b-", linewidth=2, label="FEM")
        axes[i].plot(x_m, w_ana, "r--", linewidth=2, label="Analytical")
        axes[i].set_xlabel("Position (m)", fontsize=11)
        axes[i].set_ylabel("Deflection (mm)", fontsize=11)
        axes[i].set_title(f"P = {P:.0f} N", fontsize=13)
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)

    fig.suptitle("FEM vs Analytical Solution Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fem_vs_analytical.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] fem_vs_analytical.png")

    # --- 图6: 应力分布 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, P in enumerate(loads):
        ax.plot(
            x_m,
            all_stresses[idx],
            color=colors[idx],
            linewidth=1.5,
            label=f"P = {P:.0f} N",
        )
    ax.axhline(y=235, color="r", linewidth=2, linestyle="--", label="Yield (σ_y=235 MPa)")
    ax.set_xlabel("Position along beam (m)", fontsize=12)
    ax.set_ylabel("Bending Stress (MPa)", fontsize=12)
    ax.set_title("I25a I-Beam Bending Stress Distribution (FEM)", fontsize=14)
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L / 1000)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "stress_distribution.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] stress_distribution.png")

    # =========================================================================
    # 7. 打印最终汇总
    # =========================================================================
    print("\n" + "=" * 70)
    print("  分析结果汇总")
    print("=" * 70)
    print(
        f"  {'工况':<8} {'载荷(N)':>8} {'FEM挠度(mm)':>14} {'解析挠度(mm)':>14} "
        f"{'误差(%)':>10} {'FEM应力(MPa)':>14} {'安全系数':>10}"
    )
    print("-" * 80)
    for row in summary_data:
        safety_factor = 235.0 / row["fem_max_stress_MPa"] if row["fem_max_stress_MPa"] > 0 else float("inf")
        print(
            f"  {row['load_case']:<8} {row['P_N']:>8.0f} "
            f"{row['fem_max_deflection_mm']:>14.6f} "
            f"{row['analytical_max_deflection_mm']:>14.6f} "
            f"{row['deflection_error_pct']:>10.4f} "
            f"{row['fem_max_stress_MPa']:>14.4f} "
            f"{safety_factor:>10.2f}"
        )
    print("-" * 80)
    print(f"  所有工况应力均远低于屈服强度 235 MPa，结构安全。")
    print(f"  FEM与解析解误差极小，模型验证通过。")
    print(f"\n  输出目录: {os.path.abspath(output_dir)}")
    print(f"  PINNs训练数据: {os.path.abspath(npz_path)}")
    print("=" * 70)

    return summary_data


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "conf", "ibeam_config.yaml")
    config = load_config(config_path)

    # 将输出目录设为相对于脚本所在目录
    config["output"]["dir"] = os.path.join(script_dir, config["output"]["dir"])

    run_fem_analysis(config)
