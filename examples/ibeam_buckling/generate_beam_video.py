"""
生成 I25a 工字梁从 10kN 到 250kN 的真实物理变形视频
展示工字钢截面的实际弯曲变形过程（侧视图 + 截面图）
"""
import os
import subprocess
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe

script_dir = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(script_dir, "outputs/fem_results/fem_training_data.npz"))

x_nodes = data["x_nodes"]          # (101,) mm
loads = data["loads"]               # (25,)
deflections = data["deflections"]   # (25, 101) mm
stress_ratios = data["stress_ratios"]  # (25, 101)

SIGMA_Y = 235.0
L = 3000.0
E = 206000.0
Ix = data["Ix"].item()

# I25a 截面参数 (mm)
H = 252.0
B = 118.0
TF = 13.7
TW = 8.5

# ============================================================
# 构建插值帧序列
# ============================================================
n_interp = 6
pause_key = 12
pause_normal = 2

all_P = []
all_defl = []
all_sr = []

for i in range(len(loads)):
    if i == 0:
        for t in np.linspace(0, 1, n_interp):
            all_P.append(loads[i] * t)
            all_defl.append(deflections[i] * t)
            all_sr.append(stress_ratios[i] * t)
    else:
        for t in np.linspace(0, 1, n_interp, endpoint=False):
            all_P.append(loads[i-1] + (loads[i] - loads[i-1]) * t)
            all_defl.append(deflections[i-1] + (deflections[i] - deflections[i-1]) * t)
            all_sr.append(stress_ratios[i-1] + (stress_ratios[i] - stress_ratios[i-1]) * t)

    is_key = (130000 <= loads[i] <= 150000) or loads[i] == loads[-1]
    extra = pause_key if is_key else pause_normal
    for _ in range(extra):
        all_P.append(loads[i])
        all_defl.append(deflections[i])
        all_sr.append(stress_ratios[i])

all_P = np.array(all_P)
all_defl = np.array(all_defl)
all_sr = np.array(all_sr)

total = len(all_P)
print(f"总帧数: {total}, 载荷范围: {all_P.min()/1000:.0f}~{all_P.max()/1000:.0f} kN")


def draw_ibeam_cross_section(ax, cy, stress_ratio, h=H, b=B, tf=TF, tw=TW, scale=1.0):
    """在给定位置绘制 I 型截面, 颜色按应力比着色"""
    h_ = h * scale
    b_ = b * scale
    tf_ = tf * scale
    tw_ = tw * scale

    sr = min(stress_ratio, 1.8)
    if sr < 0.5:
        color = plt.cm.RdYlGn(1.0 - sr)
    elif sr < 1.0:
        color = plt.cm.RdYlGn(1.0 - sr)
    else:
        color = plt.cm.Reds(min((sr - 0.8) / 1.2, 1.0))

    # 上翼缘
    ax.add_patch(Rectangle((-b_/2, cy + h_/2 - tf_), b_, tf_,
                           facecolor=color, edgecolor="black", linewidth=0.8))
    # 下翼缘
    ax.add_patch(Rectangle((-b_/2, cy - h_/2), b_, tf_,
                           facecolor=color, edgecolor="black", linewidth=0.8))
    # 腹板
    ax.add_patch(Rectangle((-tw_/2, cy - h_/2 + tf_), tw_, h_ - 2*tf_,
                           facecolor=color, edgecolor="black", linewidth=0.8))


def get_deform_scale(max_d):
    """动态放大系数: 让变形始终可见但不失真"""
    if max_d < 0.5:
        return 80.0
    elif max_d < 5:
        return 40.0
    elif max_d < 20:
        return 10.0
    elif max_d < 100:
        return 2.0
    elif max_d < 400:
        return 0.6
    else:
        return 0.3


# ============================================================
# 渲染帧
# ============================================================
frames_dir = os.path.join(script_dir, "outputs", "_beam_frames")
os.makedirs(frames_dir, exist_ok=True)

print("开始渲染帧...")

for fi in range(total):
    P = all_P[fi]
    defl = all_defl[fi]
    sr = all_sr[fi]
    max_d = np.max(np.abs(defl))
    max_sr = np.max(sr)

    # 状态
    if max_sr < 0.95:
        state, state_c = "ELASTIC", "#27ae60"
    elif max_sr < 1.05:
        state, state_c = "YIELD ONSET", "#f39c12"
    elif max_sr < 1.5:
        state, state_c = "PLASTIC", "#e74c3c"
    else:
        state, state_c = "DEEP PLASTIC", "#8e44ad"

    fig = plt.figure(figsize=(18, 10), facecolor="#1a1a2e")

    # 上: 侧视图 (工字梁实体)
    ax_side = fig.add_axes([0.04, 0.38, 0.64, 0.58])
    ax_side.set_facecolor("#16213e")

    # 下: 截面图
    ax_sec = fig.add_axes([0.72, 0.38, 0.25, 0.55])
    ax_sec.set_facecolor("#16213e")

    # 底部信息
    ax_info = fig.add_axes([0.04, 0.03, 0.92, 0.30])
    ax_info.set_facecolor("#0f3460")

    # ===== 侧视图: 绘制变形的工字梁 =====
    scale = get_deform_scale(max_d)
    x_m = x_nodes / 1000.0  # 转为 m

    # 梁的上下边缘 (未变形)
    beam_h_m = H / 1000.0  # 0.252 m

    # 变形后的中性轴 (mm → m, 放大)
    y_mid = defl * scale / 1000.0  # 变形后的中性轴位置 (m)

    # 绘制变形梁的上下翼缘轮廓
    y_top = y_mid + beam_h_m / 2
    y_bot = y_mid - beam_h_m / 2

    # 填充梁体 — 按应力比着色分段
    n_seg = len(x_m) - 1
    for j in range(n_seg):
        seg_sr = (sr[j] + sr[j+1]) / 2
        if seg_sr < 0.3:
            fc = "#4a90d9"  # 蓝色 (低应力)
        elif seg_sr < 0.7:
            fc = "#5dade2"  # 浅蓝
        elif seg_sr < 0.95:
            fc = "#f4d03f"  # 黄色 (临近屈服)
        elif seg_sr < 1.05:
            fc = "#e67e22"  # 橙色 (屈服)
        elif seg_sr < 1.5:
            fc = "#e74c3c"  # 红色 (塑性)
        else:
            fc = "#8e44ad"  # 紫色 (深度塑性)

        verts = [
            (x_m[j], y_bot[j]),
            (x_m[j+1], y_bot[j+1]),
            (x_m[j+1], y_top[j+1]),
            (x_m[j], y_top[j]),
        ]
        ax_side.add_patch(Polygon(verts, closed=True, facecolor=fc,
                                  edgecolor="none", alpha=0.95))

    # 上下边缘线
    ax_side.plot(x_m, y_top, color="white", lw=1.5, zorder=3)
    ax_side.plot(x_m, y_bot, color="white", lw=1.5, zorder=3)
    # 中性轴虚线
    ax_side.plot(x_m, y_mid, color="white", lw=0.5, ls="--", alpha=0.4, zorder=2)

    # 未变形梁 (灰色虚线)
    ax_side.plot(x_m, np.full_like(x_m, beam_h_m/2), color="gray", lw=0.8,
                 ls=":", alpha=0.3)
    ax_side.plot(x_m, np.full_like(x_m, -beam_h_m/2), color="gray", lw=0.8,
                 ls=":", alpha=0.3)

    # 支座
    tri_size = 0.06
    # 左: Pin 支座 (三角形)
    pin_tri = Polygon([(0, y_bot[0]),
                       (-tri_size, y_bot[0] - tri_size*1.5),
                       (tri_size, y_bot[0] - tri_size*1.5)],
                      closed=True, facecolor="#95a5a6", edgecolor="white", lw=1.5, zorder=5)
    ax_side.add_patch(pin_tri)
    # 右: Roller 支座 (三角形 + 圆)
    roller_tri = Polygon([(3, y_bot[-1]),
                          (3 - tri_size, y_bot[-1] - tri_size*1.5),
                          (3 + tri_size, y_bot[-1] - tri_size*1.5)],
                         closed=True, facecolor="#95a5a6", edgecolor="white", lw=1.5, zorder=5)
    ax_side.add_patch(roller_tri)
    roller_circle = plt.Circle((3, y_bot[-1] - tri_size*1.5 - 0.015), 0.015,
                               color="#95a5a6", ec="white", lw=1, zorder=5)
    ax_side.add_patch(roller_circle)

    # 载荷箭头
    if P > 0:
        arrow_y_start = y_top[50] + 0.15 + P / 250000 * 0.2
        ax_side.annotate("",
                         xy=(1.5, y_top[50] + 0.01),
                         xytext=(1.5, arrow_y_start),
                         arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.05",
                                        color="#e74c3c", lw=3),
                         zorder=10)
        ax_side.text(1.5, arrow_y_start + 0.02, f"P = {P/1000:.0f} kN",
                     ha="center", va="bottom", fontsize=16, fontweight="bold",
                     color="#e74c3c",
                     path_effects=[pe.withStroke(linewidth=3, foreground="#1a1a2e")])

    # 挠度标注
    if max_d > 0.05:
        mid_y = y_mid[50]
        ax_side.annotate(f"δ = {max_d:.1f} mm",
                         xy=(1.5, mid_y), xytext=(2.2, mid_y - 0.08),
                         fontsize=13, color="#3498db", fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color="#3498db", lw=1.5),
                         path_effects=[pe.withStroke(linewidth=2, foreground="#1a1a2e")])

    # 放大系数标注
    ax_side.text(0.98, 0.02, f"Deformation scale: {scale:.0f}×" if scale >= 1 else f"Deformation scale: {scale:.1f}×",
                 transform=ax_side.transAxes, fontsize=10, color="#7f8c8d",
                 ha="right", va="bottom")

    # 状态标签
    ax_side.text(0.02, 0.95, state, transform=ax_side.transAxes,
                 fontsize=22, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=state_c, alpha=0.95),
                 va="top",
                 path_effects=[pe.withStroke(linewidth=1, foreground="black")])

    # 轴设置
    y_range_min = min(np.min(y_bot) - 0.15, -0.3)
    y_range_max = max(np.max(y_top) + 0.25 + P/250000*0.3, 0.6)
    ax_side.set_xlim(-0.15, 3.15)
    ax_side.set_ylim(y_range_min, y_range_max)
    ax_side.set_xlabel("Position (m)", fontsize=12, color="white")
    ax_side.set_title("I25a Steel I-Beam — Elastic-Plastic Deformation",
                      fontsize=16, color="white", fontweight="bold", pad=10)
    ax_side.tick_params(colors="white")
    for spine in ax_side.spines.values():
        spine.set_color("#34495e")
    ax_side.grid(True, alpha=0.1, color="white")

    # ===== 截面图: 跨中截面 =====
    mid_sr = sr[50]
    draw_ibeam_cross_section(ax_sec, 0, mid_sr, scale=1.0)

    # 标注尺寸
    ax_sec.text(0, H/2 + 12, f"I25a", ha="center", va="bottom",
                fontsize=14, color="white", fontweight="bold")
    ax_sec.text(0, -H/2 - 12, f"σ/σ_y = {mid_sr*100:.0f}%", ha="center", va="top",
                fontsize=13, color="#f39c12" if mid_sr < 1.0 else "#e74c3c",
                fontweight="bold")

    # 截面尺寸线
    ax_sec.annotate("", xy=(B/2+5, -H/2), xytext=(B/2+5, H/2),
                    arrowprops=dict(arrowstyle="<->", color="white", lw=1))
    ax_sec.text(B/2+12, 0, f"{H:.0f}", ha="left", va="center",
                fontsize=9, color="white")
    ax_sec.annotate("", xy=(-B/2, -H/2-8), xytext=(B/2, -H/2-8),
                    arrowprops=dict(arrowstyle="<->", color="white", lw=1))
    ax_sec.text(0, -H/2-18, f"{B:.0f}", ha="center", va="top",
                fontsize=9, color="white")

    ax_sec.set_xlim(-B*0.9, B*0.9)
    ax_sec.set_ylim(-H*0.7, H*0.7)
    ax_sec.set_aspect("equal")
    ax_sec.set_title("Mid-span Section", fontsize=13, color="white", pad=8)
    ax_sec.tick_params(colors="#34495e", labelsize=0)
    for spine in ax_sec.spines.values():
        spine.set_color("#34495e")

    # ===== 底部信息面板 =====
    ax_info.axis("off")

    elastic_d = P * L**3 / (48 * E * Ix) if P > 0 else 0
    nl_ratio = max_d / elastic_d if elastic_d > 0.01 else 1.0
    safety = 1.0 / max_sr if max_sr > 0.01 else 99.9

    # 载荷进度条
    bar_y = 0.82
    bar_h = 0.12
    ax_info.add_patch(Rectangle((0.02, bar_y), 0.96, bar_h,
                                facecolor="#34495e", edgecolor="#7f8c8d", lw=1,
                                transform=ax_info.transAxes, zorder=1))
    progress = P / 250000
    bar_color = state_c
    ax_info.add_patch(Rectangle((0.02, bar_y), 0.96 * progress, bar_h,
                                facecolor=bar_color, alpha=0.8,
                                transform=ax_info.transAxes, zorder=2))
    ax_info.text(0.5, bar_y + bar_h/2,
                 f"Loading: {P/1000:.0f} / 250 kN",
                 transform=ax_info.transAxes, ha="center", va="center",
                 fontsize=14, color="white", fontweight="bold", zorder=3)

    # 数据卡片
    cards = [
        ("Max Deflection", f"{max_d:.2f} mm", "#3498db"),
        ("Elastic Reference", f"{elastic_d:.2f} mm", "#2ecc71"),
        ("Nonlinear Ratio", f"{nl_ratio:.1f}×", "#e67e22" if nl_ratio > 1.5 else "#95a5a6"),
        ("Max σ/σ_y", f"{max_sr*100:.1f}%", "#e74c3c" if max_sr >= 1.0 else "#27ae60"),
        ("Safety Factor", f"{safety:.1f}", "#e74c3c" if safety < 1.5 else "#27ae60"),
    ]

    for ci, (label, val, color) in enumerate(cards):
        cx = 0.02 + ci * 0.196
        cy_ = 0.12
        cw = 0.185
        ch = 0.55
        ax_info.add_patch(Rectangle((cx, cy_), cw, ch,
                                    facecolor="#1a1a2e", edgecolor=color, lw=2,
                                    transform=ax_info.transAxes, zorder=1,
                                    clip_on=False))
        ax_info.text(cx + cw/2, cy_ + ch*0.68, val,
                     transform=ax_info.transAxes, ha="center", va="center",
                     fontsize=16, color=color, fontweight="bold", zorder=2)
        ax_info.text(cx + cw/2, cy_ + ch*0.22, label,
                     transform=ax_info.transAxes, ha="center", va="center",
                     fontsize=9, color="#95a5a6", zorder=2)

    # 保存帧
    fig.savefig(os.path.join(frames_dir, f"frame_{fi:04d}.png"),
                dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)

    if (fi + 1) % 50 == 0 or fi == total - 1:
        print(f"  渲染帧: {fi+1}/{total}")

# ============================================================
# ffmpeg 合成 MP4
# ============================================================
out_mp4 = os.path.join(script_dir, "outputs", "ibeam_deformation_realistic.mp4")
fps = 20

cmd = [
    "ffmpeg", "-y",
    "-framerate", str(fps),
    "-i", os.path.join(frames_dir, "frame_%04d.png"),
    "-c:v", "mpeg4",
    "-q:v", "2",
    "-pix_fmt", "yuv420p",
    out_mp4,
]
print(f"\n合成视频: ffmpeg mpeg4 @ {fps}fps ...")
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"ffmpeg error: {result.stderr[:500]}")
else:
    fsize = os.path.getsize(out_mp4) / (1024 * 1024)
    duration = total / fps
    print(f"\n视频已保存: {out_mp4}")
    print(f"时长: {duration:.1f} 秒, {total} 帧, {fps} fps, {fsize:.1f} MB")

# 清理
shutil.rmtree(frames_dir, ignore_errors=True)
print("临时文件已清理")
