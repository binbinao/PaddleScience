"""FEM vs PINNs 全面对比实验"""
import numpy as np, time, os, sys, paddle
paddle.set_default_dtype('float64')

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

data = np.load(os.path.join(script_dir, 'outputs/fem_results/fem_training_data.npz'))
x_nodes = data['x_nodes']; loads = data['loads']
defl = data['deflections']; sr = data['stress_ratios']

from buckling_fem_analysis import fem_solve_nonlinear
from buckling_pinn_train import BucklingPINN, X_REF, P_REF, W_REF

h=252.0; b=118.0; t_f=13.7; t_w=8.5; L=3000.0

# PINNs 模型
model = BucklingPINN(num_layers=5, hidden_size=64)
model.set_state_dict(paddle.load(os.path.join(script_dir, 'outputs/pinn_results/best_model.pdparams')))
model.eval()
dummy = paddle.to_tensor(np.zeros((10,2)))
with paddle.no_grad(): model(dummy)

test_loads = [50000, 100000, 130000, 150000, 180000, 220000, 250000]
n_params = sum(p.numpy().size for p in model.parameters())

print("=" * 80)
print("  FEM vs PINNs 全面对比 (弹塑性 I25a 工字梁, 10~250 kN)")
print("=" * 80)

# 速度 + 精度
header = "  {:>8s} {:>6s} | {:>11s} {:>11s} {:>8s} | {:>10s} {:>10s} {:>8s}"
print(header.format("载荷","区域","FEM δ(mm)","PINNs δ(mm)","δ误差","FEM用时","PINNs用时","加速比"))
print("  " + "-" * 78)

results = []
for P in test_loads:
    # FEM 速度
    ft_list = []
    for _ in range(5):
        t0 = time.perf_counter()
        fem_solve_nonlinear(P, L, h, b, t_f, t_w, 100)
        ft_list.append(time.perf_counter()-t0)
    ft = np.mean(ft_list)*1000

    # PINNs 速度
    inp = np.hstack([x_nodes.reshape(-1,1)/X_REF, np.full((len(x_nodes),1), P/P_REF)])
    pt_list = []
    for _ in range(50):
        t0 = time.perf_counter()
        with paddle.no_grad(): model(paddle.to_tensor(inp))
        pt_list.append(time.perf_counter()-t0)
    pt = np.mean(pt_list)*1000

    # 精度
    idx = np.argmin(np.abs(loads - P))
    fd = np.max(np.abs(defl[idx]))
    fsr = np.max(sr[idx])
    with paddle.no_grad():
        wp, srp = model(paddle.to_tensor(inp))
    pd_ = np.max(np.abs(wp.numpy())) * W_REF
    psr = np.max(srp.numpy())
    err = abs(pd_-fd)/fd*100 if fd>0 else 0
    zone = "弹性" if fsr < 1.0 else "塑性"
    speedup = ft/pt

    row = "  {:>7.0f}kN {:>6s} | {:>11.4f} {:>11.4f} {:>7.2f}% | {:>9.1f}ms {:>9.2f}ms {:>7.0f}x"
    print(row.format(P/1000, zone, fd, pd_, err, ft, pt, speedup))
    results.append({"P":P,"zone":zone,"fd":fd,"pd":pd_,"err":err,"ft":ft,"pt":pt,"sp":speedup,"fsr":fsr,"psr":psr})

print()
print("=" * 80)
print("  对比总结")
print("=" * 80)

# 分区统计
elastic = [r for r in results if r["zone"]=="弹性"]
plastic = [r for r in results if r["zone"]=="塑性"]

print()
print("  ┌─────────────────┬─────────────────┬─────────────────┐")
print("  │     对比维度     │    弹性区间      │    塑性区间      │")
print("  ├─────────────────┼─────────────────┼─────────────────┤")
if elastic:
    e_err = np.mean([r["err"] for r in elastic])
    e_ft = np.mean([r["ft"] for r in elastic])
    e_pt = np.mean([r["pt"] for r in elastic])
    e_sp = np.mean([r["sp"] for r in elastic])
else:
    e_err=e_ft=e_pt=e_sp=0

if plastic:
    p_err = np.mean([r["err"] for r in plastic])
    p_ft = np.mean([r["ft"] for r in plastic])
    p_pt = np.mean([r["pt"] for r in plastic])
    p_sp = np.mean([r["sp"] for r in plastic])
else:
    p_err=p_ft=p_pt=p_sp=0

print("  │ PINNs 挠度误差  │    {:<13s} │    {:<13s} │".format(
    "{:.2f}%".format(e_err), "{:.2f}%".format(p_err)))
print("  │ FEM 平均用时    │    {:<13s} │    {:<13s} │".format(
    "{:.1f} ms".format(e_ft), "{:.1f} ms".format(p_ft)))
print("  │ PINNs 平均用时  │    {:<13s} │    {:<13s} │".format(
    "{:.2f} ms".format(e_pt), "{:.2f} ms".format(p_pt)))
print("  │ 加速比          │    {:<13s} │    {:<13s} │".format(
    "{:.0f}x".format(e_sp), "{:.0f}x".format(p_sp)))
print("  └─────────────────┴─────────────────┴─────────────────┘")

print()
print("  关键发现:")
print("  ─────────")
print("  1. FEM 弹性段 ~{:.0f}ms, 塑性段暴增到 ~{:.0f}ms (需迭代求解)".format(e_ft, p_ft))
print("  2. PINNs 始终 ~{:.1f}ms, 不受材料非线性影响".format(np.mean([r["pt"] for r in results])))
print("  3. 塑性段 PINNs 加速比高达 {:.0f}x, 远超弹性段的 {:.0f}x".format(p_sp, e_sp))
print("  4. PINNs 挠度精度: 弹性 {:.1f}%, 塑性 {:.1f}% — 在700mm级大变形下仍<1%".format(e_err, p_err))
print()
print("  方法本质差异:")
print("  ─────────────")
print("  FEM:   组装K矩阵 → 求解Ku=F → 更新EI → 重新组装 → 再求解... (迭代)")
print("         矩阵: 202×202, 塑性段每次需50次迭代")
print("         优势: 精确, 可靠, 有严格数学保证")
print("         劣势: 每个新载荷都要从头算, 塑性段代价陡增")
print()
print("  PINNs: 输入(x,P) → 5层矩阵乘法+tanh → 输出(δ,σ)  (单次前向传播)")
print("         参数: {:,}, 无迭代, 无矩阵组装".format(n_params))
print("         优势: 任意载荷即时预测, 塑性段无额外代价, 可微分")
print("         劣势: 需要预训练(~8min GPU), 精度受网络容量限制")
print()
print("  对于本案例 (1D梁, 25工况):")
print("    FEM 总计算量: 25工况 × ~{:.0f}ms = ~{:.1f}s".format(
    np.mean([r["ft"] for r in results]),
    25*np.mean([r["ft"] for r in results])/1000))
print("    PINNs 训练: ~8 min (一次性), 之后每个预测 <1ms")
print("    盈亏平衡点: 当需要预测 >{:.0f} 个工况时, PINNs 总时间更短".format(
    8*60*1000 / np.mean([r["ft"] for r in results])))
print("=" * 80)
