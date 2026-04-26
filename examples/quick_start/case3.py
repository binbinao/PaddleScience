# =============================================================================
# 基于 PaddleScience 的薄板弯曲问题 PINN 求解
# =============================================================================
# 本案例使用 PINN (Physics-informed Neural Network) 方法求解薄板小挠度弯曲问题
# 薄板控制方程为双调和方程（Kirchhoff 板理论）：
#   ∂⁴w/∂x⁴ + 2∂⁴w/∂x²∂y² + ∂⁴w/∂y⁴ = q/D
# 其中 w(x,y) 为挠度，q 为均布载荷，D 为弯曲刚度
#
# 边界条件：
#   - 左右边界 (x = ±1): 简支边界条件 (w = 0, ∂²w/∂x² = 0)
#   - 上下边界 (y = ±0.5): 自由边界条件
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt
import sympy as sp

import ppsci


def main():
    # =========================================================================
    # 1. 设置计算域和物理参数
    # =========================================================================
    # 薄板几何参数
    Lx = 2.0  # 薄板 x 方向长度 (m)
    Ly = 1.0  # 薄板 y 方向宽度 (m)

    # 材料和载荷参数
    E = 210000.0e6  # 弹性模量 (Pa)，钢材典型值
    mu = 0.28  # 泊松比 (无量纲)
    h = 0.01  # 薄板厚度 (m)，即 10mm
    D = E * (h**3) / (12 * (1 - mu**2))  # 弯曲刚度 (N·m)
    q = 1000.0  # 均布载荷 (N/m²)

    # 创建矩形计算域
    rectangle = ppsci.geometry.Rectangle([-Lx / 2, -Ly / 2], [Lx / 2, Ly / 2])

    # =========================================================================
    # 2. 使用 SymPy 构建控制方程
    # =========================================================================
    # 定义符号变量
    x, y = sp.symbols("x y")
    w = sp.Function("w")(x, y)  # 挠度函数 w(x, y)

    # 构建双调和方程（Kirchhoff 板方程）
    # 左侧: ∂⁴w/∂x⁴ + 2∂⁴w/∂x²∂y² + ∂⁴w/∂y⁴
    left = w.diff(x, 4) + 2 * w.diff(x, 2).diff(y, 2) + w.diff(y, 4)
    # 右侧: q/D
    right = q / D
    # 残差表达式
    res = left - right

    print("控制方程残差表达式:")
    print(res)

    # =========================================================================
    # 3. 初始化神经网络模型
    # =========================================================================
    # 使用 4 层隐藏层，每层 50 个神经元的 MLP 网络
    # 输入: (x, y) 坐标，输出: 挠度 w
    model = ppsci.arch.MLP(["x", "y"], ["w"], 4, 50)
    print("\n模型结构:")
    print(model)

    # =========================================================================
    # 4. 设置约束条件
    # =========================================================================

    # -------------------------------------------------------------------------
    # 4.1 内部区域约束 (PDE 约束)
    # -------------------------------------------------------------------------
    # 在矩形内部采样配点，满足控制方程
    pde_constraint = ppsci.constraint.InteriorConstraint(
        {"kirchhoff_res": res},  # 残差表达式
        {"kirchhoff_res": 0.0},  # 残差目标值为 0
        rectangle,
        {
            "dataset": "IterableNamedArrayDataset",
            "iters_per_epoch": 1,
            "batch_size": 20000,  # 采样 20000 个配点用于训练
        },
        random="Halton",  # 使用 Halton 序列采样
        loss=ppsci.loss.MSELoss(),
    )

    # -------------------------------------------------------------------------
    # 4.2 简支边界条件 (左右边界 x = ±1)
    # -------------------------------------------------------------------------
    # 简支边界: w = 0, ∂²w/∂x² = 0
    constraint_left_right = ppsci.constraint.BoundaryConstraint(
        {"w": w, "ddw_dxx": w.diff(x, 2)},  # 挠度和 x 方向二阶导数
        {"w": 0, "ddw_dxx": 0},  # 目标值均为 0
        rectangle,
        {
            "dataset": "IterableNamedArrayDataset",
            "iters_per_epoch": 1,
            "batch_size": 10000,
        },
        criteria=lambda x, y: np.isclose(x, -Lx / 2) | np.isclose(x, Lx / 2),
        loss=ppsci.loss.MSELoss(),
    )

    # -------------------------------------------------------------------------
    # 4.3 自由边界条件 (上下边界 y = ±0.5)
    # -------------------------------------------------------------------------
    # 自由边界:
    #   ∂²w/∂y² + μ·∂²w/∂x² = 0 (弯矩为零)
    #   ∂³w/∂y³ + (2-μ)·∂³w/∂x²∂y = 0 (等效剪力为零)
    constraint_up_down = ppsci.constraint.BoundaryConstraint(
        {
            "item1": w.diff(y, 2) + mu * w.diff(x, 2),
            "item2": w.diff(y, 3) + (2 - mu) * w.diff(x, 2).diff(y),
        },
        {"item1": 0.0, "item2": 0.0},
        rectangle,
        {
            "dataset": "IterableNamedArrayDataset",
            "iters_per_epoch": 1,
            "batch_size": 10000,
        },
        criteria=lambda x, y: np.isclose(y, -Ly / 2) | np.isclose(y, Ly / 2),
        loss=ppsci.loss.MSELoss(),
    )

    # =========================================================================
    # 5. 初始化求解器并训练
    # =========================================================================
    # 使用 L-BFGS 优化器，适合小批量高精度优化问题
    opt = ppsci.optimizer.LBFGS(max_iter=1000)(model)

    solver = ppsci.solver.Solver(
        model,
        {
            "pde_constraint": pde_constraint,
            "constraint_left_right": constraint_left_right,
            "constraint_up_down": constraint_up_down,
        },
        output_dir="./output_kirchhoff",
        optimizer=opt,
        epochs=400,
        iters_per_epoch=1,
        log_freq=100,
        # pretrained_model_path="./output_kirchhoff/checkpoints/latest"  # 加载预训练模型
    )

    # 开始训练
    solver.train()

    # =========================================================================
    # 6. 结果可视化
    # =========================================================================
    # 生成预测网格
    num_cord0 = 101
    num_cord1 = 101
    num_cords = num_cord0 * num_cord1
    print(f"\n预测点数: {num_cords}")

    x_grid, y_grid = np.meshgrid(
        np.linspace(-Lx / 2, Lx / 2, num_cord0, dtype="float32"),
        np.linspace(-Ly / 2, Ly / 2, num_cord1, dtype="float32"),
    )
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()

    # 预测挠度场
    w_pred = solver.predict(
        {"x": x_flat[:, None], "y": y_flat[:, None]}, return_numpy=True
    )["w"]

    # 绘制挠度云图
    fig = plt.figure(figsize=(8, 6))
    w_min = w_pred.min()
    w_max = w_pred.max()

    plt.tricontourf(x_flat, y_flat, w_pred[:, 0], levels=30, cmap="rainbow")
    plt.colorbar(label="Deflection w (m)")
    plt.axis("equal")
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.title(f"Thin Plate Deflection Field\nRange: [{w_min:.6f}, {w_max:.6f}] m")

    # 保存结果图
    plt.savefig("./output_kirchhoff/result.png", dpi=150, bbox_inches="tight")
    print("结果图已保存至: ./output_kirchhoff/result.png")
    plt.show()

    print(f"\n计算完成！")
    print(f"最大挠度: {w_max * 1000:.2f} mm")
    print(f"最小挠度: {w_min * 1000:.2f} mm")


if __name__ == "__main__":
    main()
