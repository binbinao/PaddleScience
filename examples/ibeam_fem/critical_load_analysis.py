"""
I25a 工字梁临界载荷反推分析
计算三种失效模式下的临界载荷：
1. 材料屈服（弯曲应力达到屈服强度）
2. 整体侧向扭转屈曲（Lateral-Torsional Buckling）
3. 腹板局部屈曲（Web Crippling）
"""
import numpy as np

# ============ I25a 截面参数 + Q235B 材料 ============
E = 206000.0       # MPa
G = 79230.77       # MPa (剪切模量)
sigma_y = 235.0    # MPa (屈服强度)
L = 3000.0         # mm (梁长)

# I25a 截面
h = 252.0          # mm 截面高度
b = 118.0          # mm 翼缘宽度
t_w = 8.5          # mm 腹板厚度
t_f = 13.7         # mm 翼缘厚度
Ix = 50170000.0    # mm^4 强轴惯性矩
Iy = 2811000.0     # mm^4 弱轴惯性矩
Wx = 398200.0      # mm^3 强轴截面模数
A = 4853.0         # mm^2 截面面积

# 扭转惯性矩近似（开口薄壁截面）
# J ≈ (1/3) * (2*b*t_f^3 + (h-2*t_f)*t_w^3)
h_w = h - 2 * t_f  # 腹板净高
J = (1.0/3.0) * (2 * b * t_f**3 + h_w * t_w**3)

# 翘曲常数近似
# Cw ≈ Iy * (h - t_f)^2 / 4
Cw = Iy * (h - t_f)**2 / 4.0

print("=" * 65)
print("  I25a 工字梁临界载荷反推分析")
print("=" * 65)
print(f"  截面: I25a, h={h}mm, b={b}mm, t_w={t_w}mm, t_f={t_f}mm")
print(f"  材料: Q235B, E={E}MPa, σ_y={sigma_y}MPa")
print(f"  梁长: L={L}mm")
print(f"  扭转惯性矩 J = {J:.0f} mm⁴")
print(f"  翘曲常数 Cw = {Cw:.3e} mm⁶")
print()

# ============================================================
# 1. 材料屈服临界载荷
# ============================================================
# 简支梁跨中集中力: M_max = P*L/4, σ_max = M/Wx
# 令 σ_max = σ_y => P_yield = 4 * σ_y * Wx / L
P_yield = 4.0 * sigma_y * Wx / L
M_yield = P_yield * L / 4.0
delta_yield = P_yield * L**3 / (48 * E * Ix)

print("─" * 65)
print("  1. 材料屈服 (弯曲应力 = 屈服强度)")
print("─" * 65)
print(f"  临界条件: σ_max = M_max / Wx = P·L / (4·Wx) = σ_y")
print(f"  P_yield = 4 × σ_y × Wx / L")
print(f"          = 4 × {sigma_y} × {Wx} / {L}")
print(f"          = {P_yield:.1f} N")
print(f"          = {P_yield/1000:.2f} kN")
print(f"  对应弯矩: M_yield = {M_yield:.0f} N·mm = {M_yield/1e6:.2f} kN·m")
print(f"  对应挠度: δ_yield = {delta_yield:.4f} mm")
print()

# ============================================================
# 2. 侧向扭转屈曲 (Lateral-Torsional Buckling)
# ============================================================
# 简支梁跨中集中力的临界弯矩（Timoshenko公式）:
# M_cr = C1 * (π/L) * sqrt(E*Iy*G*J) * sqrt(1 + (π²*E*Cw)/(G*J*L²))
# C1 = 1.365 (跨中集中力的等效弯矩系数)
C1 = 1.365

term1 = E * Iy * G * J
term2 = (np.pi**2 * E * Cw) / (G * J * L**2)

M_cr_ltb = C1 * (np.pi / L) * np.sqrt(term1) * np.sqrt(1 + term2)
P_cr_ltb = 4.0 * M_cr_ltb / L  # M_max = P*L/4 => P = 4*M/L
sigma_cr_ltb = M_cr_ltb / Wx

print("─" * 65)
print("  2. 侧向扭转屈曲 (Lateral-Torsional Buckling)")
print("─" * 65)
print(f"  Timoshenko 公式 (C1={C1}, 跨中集中力):")
print(f"  M_cr = C1·(π/L)·√(E·Iy·G·J)·√(1 + π²·E·Cw/(G·J·L²))")
print(f"  E·Iy·G·J = {term1:.3e}")
print(f"  π²·E·Cw/(G·J·L²) = {term2:.4f}")
print(f"  M_cr = {M_cr_ltb:.0f} N·mm = {M_cr_ltb/1e6:.2f} kN·m")
print(f"  P_cr_LTB = 4·M_cr/L = {P_cr_ltb:.0f} N = {P_cr_ltb/1000:.2f} kN")
print(f"  对应名义应力: σ_cr = {sigma_cr_ltb:.1f} MPa")
print()
if sigma_cr_ltb > sigma_y:
    print(f"  ⚠️  σ_cr ({sigma_cr_ltb:.1f} MPa) > σ_y ({sigma_y} MPa)")
    print(f"      说明梁在侧扭屈曲前就已经材料屈服，侧扭屈曲不是控制因素")
else:
    print(f"  ⚠️  σ_cr ({sigma_cr_ltb:.1f} MPa) < σ_y ({sigma_y} MPa)")
    print(f"      侧扭屈曲先于材料屈服发生，是控制因素！")
print()

# ============================================================
# 3. 腹板局部屈曲 (集中力作用下的腹板压屈)
# ============================================================
# 简支梁支座处腹板抗剪屈曲 (近似)
# τ_cr = k * π² * E / (12*(1-ν²)) * (t_w/h_w)²
# k ≈ 5.34 (简支板剪切屈曲系数, a/h > 1)
nu = 0.3
k_shear = 5.34
tau_cr = k_shear * np.pi**2 * E / (12*(1-nu**2)) * (t_w/h_w)**2
# 最大剪力 V = P/2, 剪应力 τ ≈ V/(h_w*t_w)
# P_cr_web = 2 * τ_cr * h_w * t_w
P_cr_web_shear = 2 * tau_cr * h_w * t_w

# 集中力下腹板局部压屈 (bearing)
# σ_cr_bearing = k_b * π² * E / (12*(1-ν²)) * (t_w/h_w)²
# k_b ≈ 2.0~4.0 取 2.0 (保守)
k_bearing = 2.0
sigma_cr_bearing = k_bearing * np.pi**2 * E / (12*(1-nu**2)) * (t_w/h_w)**2
# 承压长度近似 n ≈ t_f + 2.5*t_f = 3.5*t_f (45度扩散)
n_bearing = 3.5 * t_f
P_cr_web_bearing = sigma_cr_bearing * n_bearing * t_w

print("─" * 65)
print("  3. 腹板局部屈曲")
print("─" * 65)
print(f"  (a) 腹板剪切屈曲:")
print(f"      τ_cr = {tau_cr:.1f} MPa")
print(f"      P_cr = 2·τ_cr·h_w·t_w = {P_cr_web_shear:.0f} N = {P_cr_web_shear/1000:.1f} kN")
print(f"  (b) 集中力下腹板承压屈曲:")
print(f"      σ_cr = {sigma_cr_bearing:.1f} MPa")
print(f"      承压长度 n = {n_bearing:.1f} mm")
print(f"      P_cr = {P_cr_web_bearing:.0f} N = {P_cr_web_bearing/1000:.1f} kN")
print()

# ============================================================
# 汇总
# ============================================================
print("=" * 65)
print("  汇总：各失效模式临界载荷")
print("=" * 65)

modes = [
    ("材料屈服 (弯曲)", P_yield),
    ("侧向扭转屈曲 (LTB)", P_cr_ltb),
    ("腹板剪切屈曲", P_cr_web_shear),
    ("腹板承压屈曲", P_cr_web_bearing),
]
modes_sorted = sorted(modes, key=lambda x: x[1])

for i, (name, P) in enumerate(modes_sorted):
    marker = "◀ 控制因素" if i == 0 else ""
    print(f"  {name:<22} P_cr = {P:>12.0f} N = {P/1000:>8.2f} kN  {marker}")

P_critical = modes_sorted[0][1]
mode_name = modes_sorted[0][0]
safety_5000 = P_critical / 5000

print()
print(f"  ➤ 控制失效模式: {mode_name}")
print(f"  ➤ 临界载荷: {P_critical:.0f} N = {P_critical/1000:.2f} kN")
print(f"  ➤ 相对于实验最大载荷 5000N 的安全系数: {safety_5000:.1f}")
print(f"  ➤ 临界载荷下的挠度: {P_critical * L**3 / (48*E*Ix):.2f} mm")
print("=" * 65)
