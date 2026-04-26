"""
Q235B 钢材完整应力-应变曲线 (从弹性到断裂)

基于多源交叉验证的材料本构模型：
- 来源1: steelcalculator.app (A36/S235等效)
- 来源2: 爆炸与冲击期刊 J-C模型参数 (郭子涛等, 2016)
- 来源3: GB/T 700 标准 + 材料力学教科书
- 来源4: ScienceDirect全系列应变硬化研究

四阶段模型: 弹性 → 屈服平台 → 应变强化 → 颈缩断裂
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# Q235B 材料属性（多源交叉验证）
# =============================================================================
MATERIAL = {
    "name": "Q235B",
    "E": 206000.0,           # MPa 弹性模量 (GB/T 700: 200~210, 工程取206)
    "nu": 0.3,               # 泊松比
    "rho": 7850.0,           # kg/m³ 密度
    "sigma_y": 235.0,        # MPa 下屈服强度 (GB/T 700: ≥235 for t≤16mm)
    "sigma_yu": 245.0,       # MPa 上屈服强度 (典型值, 约1.04×σ_y)
    "sigma_u": 420.0,        # MPa 抗拉强度 (GB/T 700: 370~500, 取中值)
    "elongation": 0.26,      # 断后延伸率 (GB/T 700: ≥26%)
    "area_reduction": 0.55,  # 断面收缩率 (典型值 50~60%)

    # 应变关键点
    "eps_y": None,           # 屈服应变 = σ_y/E (自动计算)
    "eps_st": 0.015,         # 屈服平台结束/强化起始应变 (10~15倍εy, 取1.5%)
    "eps_u": 0.20,           # 抗拉强度对应应变 (均匀延伸率, 15~25%)
    "eps_f": 0.30,           # 断裂工程应变 (与延伸率对应)

    # Johnson-Cook 参数 (准静态, 郭子涛等 2016)
    "JC_A": 293.8,           # MPa (J-C屈服强度, 略高于标称因试件差异)
    "JC_B": 230.2,           # MPa (硬化模量, 动态实验拟合值)
    "JC_n": 0.578,           # 硬化指数
}

MATERIAL["eps_y"] = MATERIAL["sigma_y"] / MATERIAL["E"]


def build_engineering_curve(mat, n_points=500):
    """
    构建 Q235B 完整工程应力-应变曲线（四阶段模型）

    阶段1: 弹性段      0 ≤ ε ≤ ε_y        σ = E·ε
    阶段2: 屈服平台    ε_y < ε ≤ ε_st     σ = σ_y (常数, Lüders带)
    阶段3: 应变强化    ε_st < ε ≤ ε_u     σ 幂律上升至 σ_u
    阶段4: 颈缩断裂    ε_u < ε ≤ ε_f      σ 下降 (颈缩导致工程应力降低)
    """
    E = mat["E"]
    sig_y = mat["sigma_y"]
    sig_u = mat["sigma_u"]
    eps_y = mat["eps_y"]
    eps_st = mat["eps_st"]
    eps_u = mat["eps_u"]
    eps_f = mat["eps_f"]

    # 阶段1: 弹性
    n1 = int(n_points * 0.08)
    eps1 = np.linspace(0, eps_y, n1)
    sig1 = E * eps1

    # 阶段2: 屈服平台 (含上屈服点尖峰)
    n2 = int(n_points * 0.12)
    eps2 = np.linspace(eps_y, eps_st, n2)
    # 模拟上屈服点: 先升后降到下屈服
    t2 = np.linspace(0, 1, n2)
    spike = mat["sigma_yu"] * np.exp(-20*t2) + sig_y * (1 - np.exp(-20*t2))
    # 平台主体保持 σ_y, 头部有尖峰
    sig2 = np.where(t2 < 0.08, spike, sig_y * np.ones_like(t2))

    # 阶段3: 应变强化 (幂律硬化)
    n3 = int(n_points * 0.50)
    eps3 = np.linspace(eps_st, eps_u, n3)
    # σ = σ_y + (σ_u - σ_y) * ((ε - ε_st)/(ε_u - ε_st))^n_hard
    n_hard = 0.45  # 强化指数 (控制曲线形状)
    t3 = (eps3 - eps_st) / (eps_u - eps_st)
    sig3 = sig_y + (sig_u - sig_y) * t3**n_hard

    # 阶段4: 颈缩断裂 (工程应力下降)
    n4 = n_points - n1 - n2 - n3
    eps4 = np.linspace(eps_u, eps_f, n4)
    # 工程应力从 σ_u 下降到约 0.75*σ_u (断裂时)
    sig_fracture = 0.72 * sig_u
    t4 = (eps4 - eps_u) / (eps_f - eps_u)
    sig4 = sig_u - (sig_u - sig_fracture) * t4**1.5

    eps = np.concatenate([eps1, eps2, eps3, eps4])
    sig = np.concatenate([sig1, sig2, sig3, sig4])
    return eps, sig


def build_true_curve(eps_eng, sig_eng):
    """从工程应力-应变转换为真实应力-应变 (颈缩前有效)"""
    eps_true = np.log(1 + eps_eng)
    sig_true = sig_eng * (1 + eps_eng)
    return eps_true, sig_true


def build_abaqus_plasticity(mat, n_points=50):
    """
    生成 ABAQUS/有限元塑性输入数据
    格式: (真实应力, 等效塑性应变)
    """
    sig_y = mat["sigma_y"]
    sig_u = mat["sigma_u"]
    eps_st = mat["eps_st"]
    eps_u = mat["eps_u"]
    E = mat["E"]

    # 强化段等效塑性应变
    eps_pl = np.linspace(0, eps_u - sig_y/E, n_points)
    # J-C 模型: σ = A + B·ε_p^n
    A = sig_y
    B = mat["JC_B"]
    n = mat["JC_n"]
    sig = A + B * np.power(np.maximum(eps_pl, 1e-10), n)
    # 限制不超过真实抗拉强度
    sig_u_true = sig_u * (1 + eps_u)
    sig = np.minimum(sig, sig_u_true)

    return eps_pl, sig


def main():
    mat = MATERIAL
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "outputs", "material_data")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print("  Q235B 钢材完整应力-应变曲线")
    print("=" * 65)

    # 关键参数汇总
    print(f"\n  {'参数':<20} {'数值':<15} {'来源/验证'}")
    print("  " + "-" * 60)
    params = [
        ("弹性模量 E", f"{mat['E']:.0f} MPa", "GB/T 700 + 4源一致"),
        ("屈服强度 σ_y", f"{mat['sigma_y']:.0f} MPa", "GB/T 700 (t≤16mm)"),
        ("上屈服强度 σ_yu", f"{mat['sigma_yu']:.0f} MPa", "典型值 ~1.04σ_y"),
        ("抗拉强度 σ_u", f"{mat['sigma_u']:.0f} MPa", "GB/T 700: 370~500"),
        ("屈服应变 ε_y", f"{mat['eps_y']:.5f}", "σ_y/E"),
        ("屈服平台终点 ε_st", f"{mat['eps_st']:.3f} (1.5%)", "10~15×ε_y"),
        ("抗拉应变 ε_u", f"{mat['eps_u']:.2f} (20%)", "文献: 15~25%"),
        ("断裂应变 ε_f", f"{mat['eps_f']:.2f} (30%)", "GB: δ≥26%, ψ~55%"),
        ("J-C A (屈服)", f"{mat['JC_A']:.1f} MPa", "郭子涛等 2016"),
        ("J-C B (硬化)", f"{mat['JC_B']:.1f} MPa", "动态实验拟合"),
        ("J-C n (指数)", f"{mat['JC_n']:.3f}", "郭子涛等 2016"),
    ]
    for name, val, src in params:
        print(f"  {name:<20} {val:<15} {src}")

    # 构建曲线
    eps_eng, sig_eng = build_engineering_curve(mat)
    eps_true, sig_true = build_true_curve(eps_eng, sig_eng)
    eps_pl, sig_pl = build_abaqus_plasticity(mat)

    # ============ 可视化 ============
    print(f"\n  生成图表...")

    # 图1: 完整工程应力-应变曲线（带阶段标注）
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(eps_eng * 100, sig_eng, "b-", linewidth=2.5, label="Engineering σ-ε", zorder=3)
    ax.plot(eps_true * 100, sig_true, "r--", linewidth=2, label="True σ-ε", zorder=2)

    # 标注关键点
    kp = [
        (mat["eps_y"]*100, mat["sigma_y"], f"Yield\nσ_y={mat['sigma_y']:.0f} MPa\nε_y={mat['eps_y']*100:.3f}%", "ko", 10),
        (mat["eps_st"]*100, mat["sigma_y"], f"Hardening onset\nε_st={mat['eps_st']*100:.1f}%", "gs", 10),
        (mat["eps_u"]*100, mat["sigma_u"], f"Ultimate\nσ_u={mat['sigma_u']:.0f} MPa\nε_u={mat['eps_u']*100:.0f}%", "r^", 12),
        (mat["eps_f"]*100, sig_eng[-1], f"Fracture\nε_f={mat['eps_f']*100:.0f}%", "rx", 14),
    ]
    for x, y, txt, marker, ms in kp:
        ax.plot(x, y, marker, markersize=ms, zorder=5)
        ax.annotate(txt, xy=(x, y), fontsize=9,
                    xytext=(15, 10), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", color="gray"))

    # 阶段背景色
    ax.axvspan(0, mat["eps_y"]*100, alpha=0.08, color="blue", label="① Elastic")
    ax.axvspan(mat["eps_y"]*100, mat["eps_st"]*100, alpha=0.08, color="green", label="② Yield plateau")
    ax.axvspan(mat["eps_st"]*100, mat["eps_u"]*100, alpha=0.08, color="orange", label="③ Strain hardening")
    ax.axvspan(mat["eps_u"]*100, mat["eps_f"]*100, alpha=0.08, color="red", label="④ Necking → Fracture")

    ax.axhline(y=mat["sigma_y"], color="gray", ls=":", alpha=0.5)
    ax.axhline(y=mat["sigma_u"], color="gray", ls=":", alpha=0.5)

    ax.set_xlabel("Strain ε (%)", fontsize=13)
    ax.set_ylabel("Stress σ (MPa)", fontsize=13)
    ax.set_title("Q235B Steel Complete Stress-Strain Curve", fontsize=15)
    ax.legend(loc="center right", fontsize=9)
    ax.set_xlim(-0.5, 33)
    ax.set_ylim(0, 550)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stress_strain_full.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] stress_strain_full.png")

    # 图2: 弹性+屈服平台放大图
    fig, ax = plt.subplots(figsize=(10, 6))
    mask = eps_eng <= 0.025
    ax.plot(eps_eng[mask]*100, sig_eng[mask], "b-", linewidth=2.5, label="Engineering")
    ax.plot(eps_true[mask]*100, sig_true[mask], "r--", linewidth=2, label="True")
    ax.axhline(y=mat["sigma_y"], color="gray", ls=":", alpha=0.5, label=f"σ_y = {mat['sigma_y']} MPa")
    ax.axhline(y=mat["sigma_yu"], color="orange", ls=":", alpha=0.5, label=f"σ_yu = {mat['sigma_yu']} MPa")
    ax.axvline(x=mat["eps_y"]*100, color="green", ls="--", alpha=0.5, label=f"ε_y = {mat['eps_y']*100:.3f}%")
    ax.axvline(x=mat["eps_st"]*100, color="purple", ls="--", alpha=0.5, label=f"ε_st = {mat['eps_st']*100:.1f}%")
    ax.set_xlabel("Strain ε (%)", fontsize=12)
    ax.set_ylabel("Stress σ (MPa)", fontsize=12)
    ax.set_title("Q235B: Elastic + Yield Plateau (Zoomed)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 2.5)
    ax.set_ylim(0, 280)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stress_strain_yield_zoom.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] stress_strain_yield_zoom.png")

    # 图3: ABAQUS 塑性输入曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eps_pl * 100, sig_pl, "b-o", linewidth=2, markersize=4, label="True stress vs Plastic strain")
    ax.axhline(y=mat["sigma_y"], color="gray", ls=":", alpha=0.5, label=f"σ_y = {mat['sigma_y']} MPa")
    ax.set_xlabel("Equivalent Plastic Strain ε_pl (%)", fontsize=12)
    ax.set_ylabel("True Stress σ (MPa)", fontsize=12)
    ax.set_title("Q235B: Plasticity Input Data (for FEM)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plasticity_input.png"), dpi=150)
    plt.close(fig)
    print("  [PNG] plasticity_input.png")

    # ============ 保存数据 ============
    np.savez(
        os.path.join(out_dir, "q235b_material_curves.npz"),
        # 工程曲线
        eps_eng=eps_eng, sig_eng=sig_eng,
        # 真实曲线
        eps_true=eps_true, sig_true=sig_true,
        # ABAQUS 塑性输入
        eps_plastic=eps_pl, sig_plastic=sig_pl,
        # 关键参数
        E=mat["E"], sigma_y=mat["sigma_y"], sigma_u=mat["sigma_u"],
        eps_y=mat["eps_y"], eps_st=mat["eps_st"],
        eps_u=mat["eps_u"], eps_f=mat["eps_f"],
        JC_A=mat["JC_A"], JC_B=mat["JC_B"], JC_n=mat["JC_n"],
    )
    print(f"  [NPZ] q235b_material_curves.npz")

    # CSV: ABAQUS 塑性数据表
    csv_path = os.path.join(out_dir, "plasticity_table.csv")
    with open(csv_path, "w") as f:
        f.write("true_stress_MPa,plastic_strain\n")
        for s, e in zip(sig_pl, eps_pl):
            f.write(f"{s:.4f},{e:.8f}\n")
    print(f"  [CSV] plasticity_table.csv")

    print(f"\n  输出目录: {os.path.abspath(out_dir)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
