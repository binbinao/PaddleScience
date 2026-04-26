## 一、实证案例：雷达矩形波导辐射槽仿真

我们选择的问题是**Ku波段（12-18 GHz）矩形波导辐射槽天线的电磁场仿真**。这是雷达和卫星通信中的经典问题：一根标准WR-90波导（宽1.02 cm，高0.51 cm），壁面开设辐射槽，TE₁₀主模激励，需要求解槽口附近的电场分布、辐射方向图和S参数。

控制方程是带PML（完美匹配层）吸收边界的二维Helmholtz方程：

```
∇²E + k²εE = J（源项）
```

其中 k 是波数，ε 是相对介电常数，J 是TE₁₀模激励源。PML在计算域边界模拟无限空间，吸收出射波。

我们用 **PaddleScience** 框架搭建了完整的仿真系统，包含8个Python模块：

| 模块 | 功能 |
|------|------|
| `geometry.py` | 波导几何建模、网格生成 |
| `fdfd_solver.py` | 传统FDFD数值求解器 |
| `functions.py` | PML实现、Fourier特征、损失函数 |
| `pinn_solver.py` | PINN神经网络求解器 |
| `postprocess.py` | 场可视化、辐射方向图、S参数 |
| `comparator.py` | FDFD vs PINN多指标对比 |

硬件环境：NVIDIA Tesla T4 GPU（16GB显存），PaddlePaddle深度学习框架。

接下来，我们分别用FDFD和PINN两种方法求解同一个问题，然后逐项对比。
