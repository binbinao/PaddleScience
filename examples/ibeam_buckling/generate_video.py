"""
生成 I25a 工字梁从 10kN 到 250kN 弹塑性变形动画视频
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch

script_dir = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(script_dir, "outputs/fem_results/fem_training_data.npz"))

x_nodes = data["x_nodes"]      # (101,)
loads = data["loads"]           # (25,)
deflections = data["deflections"]  # (25, 101)
stress_ratios = data["stress_ratios"]  # (25, 101)

SIGMA_Y = 235.0
L = 3000.0
E = 206000.0
Ix = data["Ix"].item()

x_m = x_nodes / 1000.0
n_frames_per_case = 8   # 每个工况插值帧数
pause_frames = 15       # 关键帧停留

# 构建完整帧序列: 0→LC1(插值)→LC2(插值)→...→LC25
all_P = []
all_defl = []
all_sr = []

for i in range(len(loads)):
    if i == 0:
        # 从零载荷插值到第一个
        for t in np.linspace(0, 1, n_frames_per_case):
            all_P.append(loads[i] * t)
            all_defl.append(deflections[i] * t)
            all_sr.append(stress_ratios[i] * t)
    else:
        for t in np.linspace(0, 1, n_frames_per_case, endpoint=False):
            all_P.append(loads[i-1] + (loads[i] - loads[i-1]) * t)
            all_defl.append(deflections[i-1] + (deflections[i] - deflections[i-1]) * t)
            all_sr.append(stress_ratios[i-1] + (stress_ratios[i] - stress_ratios[i-1]) * t)

    # 在关键点停留
    is_key = (loads[i] >= 130000 and loads[i] <= 150000) or loads[i] == loads[-1]
    extra = pause_frames if is_key else 3
    for _ in range(extra):
        all_P.append(loads[i])
        all_defl.append(deflections[i])
        all_sr.append(stress_ratios[i])

all_P = np.array(all_P)
all_defl = np.array(all_defl)
all_sr = np.array(all_sr)

print(f"总帧数: {len(all_P)}, 载荷范围: {all_P.min()/1000:.0f}~{all_P.max()/1000:.0f} kN")

# ============ 创建动画 ============
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.5, 1.5], hspace=0.35, wspace=0.3)

ax_beam = fig.add_subplot(gs[0, :])     # 梁变形
ax_ld = fig.add_subplot(gs[1, 0])       # 载荷-挠度
ax_sr = fig.add_subplot(gs[1, 1])       # 应力比分布
ax_info = fig.add_subplot(gs[2, :])     # 信息面板

# 预计算载荷-挠度历史
max_defls_hist = np.max(np.abs(deflections), axis=1)
elastic_refs = loads * L**3 / (48 * E * Ix)

# 变形放大系数 (让小变形也可见)
def get_scale(max_d):
    if max_d < 1:
        return 300
    elif max_d < 10:
        return 30
    elif max_d < 100:
        return 3
    else:
        return 0.5

def animate(frame):
    P = all_P[frame]
    defl = all_defl[frame]
    sr = all_sr[frame]
    max_d = np.max(np.abs(defl))
    max_sr = np.max(sr)

    # 状态判断
    if max_sr < 0.95:
        state = "ELASTIC"
        state_color = "#2ecc71"
    elif max_sr < 1.05:
        state = "YIELD ONSET"
        state_color = "#f39c12"
    elif max_sr < 1.5:
        state = "PLASTIC"
        state_color = "#e74c3c"
    else:
        state = "DEEP PLASTIC"
        state_color = "#8e44ad"

    # === 梁变形图 ===
    ax_beam.clear()
    scale = get_scale(max_d)
    y_defl = -defl * scale  # 放大显示

    # 未变形梁 (灰色虚线)
    ax_beam.plot(x_m, np.zeros_like(x_m), "k--", lw=1, alpha=0.3, label="Undeformed")

    # 变形梁 (颜色编码应力比)
    for j in range(len(x_m)-1):
        seg_sr = (sr[j] + sr[j+1]) / 2
        if seg_sr < 0.5:
            c = plt.cm.RdYlGn_r(seg_sr / 2)
        elif seg_sr < 1.0:
            c = plt.cm.RdYlGn_r(0.25 + seg_sr * 0.5)
        else:
            c = plt.cm.RdYlGn_r(min(seg_sr / 1.8, 1.0))
        ax_beam.plot(x_m[j:j+2], y_defl[j:j+2], color=c, lw=4, solid_capstyle="round")

    # 支座
    ax_beam.plot(0, 0, "k^", ms=15, zorder=5)
    ax_beam.plot(3, 0, "ko", ms=12, zorder=5)

    # 载荷箭头
    if P > 0:
        arrow_len = min(max(P/250000 * 150, 20), 200)
        ax_beam.annotate("", xy=(1.5, y_defl[50]), xytext=(1.5, y_defl[50] + arrow_len),
                        arrowprops=dict(arrowstyle="->, head_width=0.4, head_length=0.2",
                                       color="red", lw=2.5))
        ax_beam.text(1.5, y_defl[50] + arrow_len + 10, f"P = {P/1000:.0f} kN",
                    ha="center", fontsize=14, fontweight="bold", color="red")

    # 标注最大挠度
    mid_idx = 50
    if max_d > 0.01:
        ax_beam.annotate(f"δ = {max_d:.2f} mm", xy=(1.5, y_defl[mid_idx]),
                        xytext=(2.2, y_defl[mid_idx] - 30),
                        fontsize=11, color="blue",
                        arrowprops=dict(arrowstyle="->", color="blue", lw=1))

    ax_beam.set_xlim(-0.2, 3.2)
    y_min = min(np.min(y_defl) - 80, -50)
    ax_beam.set_ylim(y_min, 250)
    ax_beam.set_xlabel("Position (m)", fontsize=12)
    ax_beam.set_title(f"I25a I-Beam Deformation  |  Scale: {scale:.0f}×", fontsize=13)

    # 状态标签
    ax_beam.text(0.02, 0.95, state, transform=ax_beam.transAxes,
                fontsize=18, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=state_color, alpha=0.9),
                va="top")

    ax_beam.set_aspect("equal")
    ax_beam.grid(True, alpha=0.2)

    # === 载荷-挠度图 ===
    ax_ld.clear()
    ax_ld.plot(loads/1000, elastic_refs, "b--", lw=1.5, alpha=0.5, label="Linear Elastic")
    ax_ld.plot(loads/1000, max_defls_hist, "k-", lw=1.5, alpha=0.5, label="FEM Elastic-Plastic")
    # 当前点
    ax_ld.plot(P/1000, max_d, "ro", ms=10, zorder=5)
    ax_ld.axvline(x=124.77, color="gray", ls=":", lw=1, alpha=0.5)
    ax_ld.set_xlabel("Load (kN)", fontsize=10)
    ax_ld.set_ylabel("Max δ (mm)", fontsize=10)
    ax_ld.set_title("Load-Deflection", fontsize=11)
    ax_ld.legend(fontsize=8)
    ax_ld.grid(True, alpha=0.2)
    ax_ld.set_xlim(0, 260)

    # === 应力比分布 ===
    ax_sr.clear()
    sr_colors = plt.cm.RdYlGn_r(np.clip(sr / 1.8, 0, 1))
    ax_sr.bar(x_m, sr * 100, width=0.032, color=sr_colors, edgecolor="none")
    ax_sr.axhline(y=100, color="red", ls="--", lw=2, alpha=0.7, label="Yield (100%)")
    ax_sr.set_xlabel("Position (m)", fontsize=10)
    ax_sr.set_ylabel("σ/σ_y (%)", fontsize=10)
    ax_sr.set_title("Stress Ratio Distribution", fontsize=11)
    ax_sr.set_xlim(0, 3)
    ax_sr.set_ylim(0, max(max_sr * 100 * 1.15, 110))
    ax_sr.legend(fontsize=8)
    ax_sr.grid(True, alpha=0.2)

    # === 信息面板 ===
    ax_info.clear()
    ax_info.axis("off")
    elastic_d = P * L**3 / (48 * E * Ix) if P > 0 else 0
    nl_ratio = max_d / elastic_d if elastic_d > 0.001 else 1.0
    safety = SIGMA_Y / (max_sr * SIGMA_Y) if max_sr > 0.01 else 99.9

    info_text = (
        f"Load: {P/1000:.1f} kN    │    "
        f"Max Deflection: {max_d:.3f} mm    │    "
        f"Nonlinear Ratio: {nl_ratio:.2f}×    │    "
        f"Max σ/σy: {max_sr*100:.1f}%    │    "
        f"Safety Factor: {safety:.1f}"
    )
    ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes,
                fontsize=12, ha="center", va="center", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))

    return []

print("开始生成 MP4 视频...")

out_mp4 = os.path.join(script_dir, "outputs", "beam_deformation_10kN_to_250kN.mp4")
frames_dir = os.path.join(script_dir, "outputs", "_frames")
os.makedirs(frames_dir, exist_ok=True)

fps = 20
total = len(all_P)

# 逐帧渲染为 PNG
for i in range(total):
    animate(i)
    fig.savefig(os.path.join(frames_dir, f"frame_{i:04d}.png"), dpi=100, bbox_inches="tight")
    if (i + 1) % 50 == 0 or i == total - 1:
        print(f"  渲染帧: {i+1}/{total}")

plt.close(fig)

# 用 ffmpeg 合成 MP4（mpeg4 编码器，所有 ffmpeg 都内置）
import subprocess
cmd = [
    "ffmpeg", "-y",
    "-framerate", str(fps),
    "-i", os.path.join(frames_dir, "frame_%04d.png"),
    "-c:v", "mpeg4",
    "-q:v", "3",
    "-pix_fmt", "yuv420p",
    out_mp4,
]
print(f"\n  合成视频: ffmpeg mpeg4 @ {fps}fps ...")
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"  ffmpeg stderr: {result.stderr[:500]}")
else:
    fsize = os.path.getsize(out_mp4) / (1024 * 1024)
    duration = total / fps
    print(f"\n视频已保存: {out_mp4}")
    print(f"时长: {duration:.1f} 秒, {total} 帧, {fps} fps, {fsize:.1f} MB")

# 清理临时帧文件
import shutil
shutil.rmtree(frames_dir, ignore_errors=True)
print("  临时帧文件已清理")
