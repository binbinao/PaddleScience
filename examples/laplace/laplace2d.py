# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
拉普拉斯方程求解案例 - 使用物理信息神经网络(PINN)

物理背景：求解二维矩形区域上的拉普拉斯方程 ∇²u = 0
数学方程：∂²u/∂x² + ∂²u/∂y² = 0
解析解：u(x,y) = cos(x) * cosh(y)

机械工程应用：
- 稳态热传导问题（温度场分布）
- 静电场的电势分布
- 不可压缩无旋流动的速度势
- 平面应力问题的应力函数
"""

import hydra
import numpy as np
from omegaconf import DictConfig

import ppsci
import matplotlib.pyplot as plt
from ppsci.utils import logger


def train(cfg: DictConfig):
    """
    训练函数：使用PINN方法训练拉普拉斯方程求解模型
    
    参数:
        cfg: 配置文件对象，包含所有训练参数和模型配置
    """
    # 设置神经网络模型 - 多层感知机(MLP)
    # 输入：坐标(x,y)，输出：物理场u
    model = ppsci.arch.MLP(**cfg.MODEL)

    # 设置物理方程 - 二维拉普拉斯方程
    # 描述物理场在空间中的二阶变化率
    equation = {"laplace": ppsci.equation.Laplace(dim=2)}

    # 设置几何区域 - 矩形计算域
    # 定义求解的物理空间范围
    geom = {
        "rect": ppsci.geometry.Rectangle(
            cfg.DIAGONAL_COORD.xmin,  # 左下角x坐标
            cfg.DIAGONAL_COORD.xmax   # 右上角x坐标
        )
    }

    # 计算解析解函数 - 用于验证和边界条件
    def u_solution_func(out):
        """计算拉普拉斯方程的解析解作为标签数据"""
        x, y = out["x"], out["y"]
        return np.cos(x) * np.cosh(y)

    # 设置训练数据加载器配置
    # 使用迭代数据集避免内存爆炸
    train_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
    }

    # 计算总采样点数
    NPOINT_TOTAL = cfg.NPOINT_INTERIOR + cfg.NPOINT_BC

    # 设置PDE约束（内点约束）
    # 在计算域内部强制满足拉普拉斯方程
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["laplace"].equations,  # 拉普拉斯方程表达式
        {"laplace": 0},                # 残差目标值（应为0）
        geom["rect"],                   # 应用几何区域
        {**train_dataloader_cfg, "batch_size": NPOINT_TOTAL},  # 数据加载配置
        ppsci.loss.MSELoss("sum"),      # 均方误差损失函数
        evenly=True,                    # 均匀采样策略
        name="EQ"                       # 约束名称
    )
    
    # 设置边界约束
    # 在边界上强制满足Dirichlet边界条件
    bc = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},   # 边界上的预测值
        {"u": u_solution_func},        # 边界上的真实值（解析解）
        geom["rect"],                   # 应用几何边界
        {**train_dataloader_cfg, "batch_size": cfg.NPOINT_BC},  # 边界采样配置
        ppsci.loss.MSELoss("sum"),      # 均方误差损失函数
        name="BC"                       # 边界约束名称
    )
    
    # 将所有约束包装在一起
    constraint = {
        pde_constraint.name: pde_constraint,  # PDE方程约束
        bc.name: bc,                          # 边界条件约束
    }

    # 设置优化器 - Adam优化算法
    optimizer = ppsci.optimizer.Adam(learning_rate=cfg.TRAIN.learning_rate)(model)

    # 设置验证器 - 用于评估模型精度
    mse_metric = ppsci.validate.GeometryValidator(
        {"u": lambda out: out["u"]},       # 预测输出
        {"u": u_solution_func},            # 真实标签（解析解）
        geom["rect"],                       # 验证区域
        {
            "dataset": "IterableNamedArrayDataset",
            "total_size": NPOINT_TOTAL,    # 验证点总数
        },
        ppsci.loss.MSELoss(),               # 验证损失函数
        evenly=True,                        # 均匀采样
        metric={"MSE": ppsci.metric.MSE()}, # 评估指标
        with_initial=True,                  # 包含初始条件
        name="MSE_Metric"                   # 验证器名称
    )
    validator = {mse_metric.name: mse_metric}

    # 设置可视化器（可选）- 用于结果可视化
    vis_points = geom["rect"].sample_interior(NPOINT_TOTAL, evenly=True)
    visualizer = {
        "visualize_u": ppsci.visualize.VisualizerVtu(
            vis_points,                     # 可视化点
            {"u": lambda d: d["u"]},       # 可视化字段
            num_timestamps=1,               # 时间戳数量（稳态问题）
            prefix="result_u",              # 输出文件前缀
        )
    }

    # === 训练监控看板功能 ===
    # 创建训练监控回调函数
    class TrainingMonitor:
        """训练监控器：实时监控训练损失和残差收敛情况"""
        
        def __init__(self, log_freq=cfg.log_freq):
            self.log_freq = log_freq
            self.epochs = []
            self.total_losses = []
            self.pde_losses = []
            self.bc_losses = []
            self.mse_scores = []
            self.current_epoch = 0  # 添加全局计数器
            
        def on_epoch_end(self, solver):
            """每个epoch结束时调用的回调函数"""
            self.current_epoch += 1  # 每次回调增加计数器
            epoch = self.current_epoch
            
            if epoch % self.log_freq == 0 or epoch == cfg.TRAIN.epochs:
                # 获取当前epoch的损失值
                loss_dict = solver.train_loss_info
                
                # 记录当前epoch的损失值
                self.epochs.append(epoch)
                total_loss_meter = loss_dict.get("total_loss")
                pde_loss_meter = loss_dict.get("EQ")
                bc_loss_meter = loss_dict.get("BC")
                
                # 获取损失值的平均值
                total_loss = total_loss_meter.avg if total_loss_meter else 0
                pde_loss = pde_loss_meter.avg if pde_loss_meter else 0
                bc_loss = bc_loss_meter.avg if bc_loss_meter else 0
                
                self.total_losses.append(total_loss)
                self.pde_losses.append(pde_loss)
                self.bc_losses.append(bc_loss)
                
                # 记录验证集MSE（仅在评估周期记录）
                if epoch % cfg.TRAIN.eval_freq == 0 or epoch == cfg.TRAIN.epochs:
                    # 使用Solver内置的验证结果
                    if hasattr(solver, 'eval_result') and solver.eval_result:
                        if "MSE_Metric" in solver.eval_result:
                            mse_value = solver.eval_result["MSE_Metric"]["MSE"]
                            self.mse_scores.append(mse_value)
                            logger.info(f"Epoch {epoch}: MSE = {mse_value:.6e}")
                
                # 打印训练进度
                logger.info(f"Epoch {epoch}: Total Loss = {total_loss:.6e}, "
                           f"PDE Loss = {pde_loss:.6e}, "
                           f"BC Loss = {bc_loss:.6e}")
        
        def plot_training_curves(self, save_path="training_curves.png"):
            """绘制训练曲线图"""
            if len(self.epochs) == 0:
                logger.warning("No training data to plot")
                return
                
            plt.figure(figsize=(12, 8))
            
            # 绘制总损失曲线
            plt.subplot(2, 2, 1)
            plt.semilogy(self.epochs, self.total_losses, 'b-', linewidth=2, label='Total Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Total Loss')
            plt.title('Total Training Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 绘制PDE和边界损失曲线
            plt.subplot(2, 2, 2)
            plt.semilogy(self.epochs, self.pde_losses, 'r-', linewidth=2, label='PDE Loss')
            plt.semilogy(self.epochs, self.bc_losses, 'g-', linewidth=2, label='BC Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Component Losses')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 绘制MSE收敛曲线
            if len(self.mse_scores) > 0:
                mse_epochs = [self.epochs[i] for i in range(len(self.epochs)) 
                             if i % cfg.TRAIN.eval_freq == 0 or i == len(self.epochs)-1]
                mse_epochs = mse_epochs[:len(self.mse_scores)]
                
                plt.subplot(2, 2, 3)
                plt.semilogy(mse_epochs, self.mse_scores, 'm-', linewidth=2, label='Validation MSE')
                plt.xlabel('Epoch')
                plt.ylabel('MSE')
                plt.title('Validation MSE Convergence')
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            # 绘制损失比例图
            plt.subplot(2, 2, 4)
            if len(self.pde_losses) > 0 and len(self.bc_losses) > 0:
                pde_ratio = [pde/(pde+bc+1e-10) for pde, bc in zip(self.pde_losses, self.bc_losses)]
                bc_ratio = [bc/(pde+bc+1e-10) for pde, bc in zip(self.pde_losses, self.bc_losses)]
                
                plt.plot(self.epochs, pde_ratio, 'r-', linewidth=2, label='PDE Ratio')
                plt.plot(self.epochs, bc_ratio, 'g-', linewidth=2, label='BC Ratio')
                plt.xlabel('Epoch')
                plt.ylabel('Loss Ratio')
                plt.title('Loss Component Ratios')
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training curves saved to {save_path}")
    
    # 创建监控器实例
    monitor = TrainingMonitor(log_freq=cfg.log_freq)

    # 初始化求解器 - 整合所有组件
    solver = ppsci.solver.Solver(
        model,                              # 神经网络模型
        constraint,                         # 物理约束
        cfg.output_dir,                     # 输出目录
        optimizer,                          # 优化器
        epochs=cfg.TRAIN.epochs,            # 训练轮数
        iters_per_epoch=cfg.TRAIN.iters_per_epoch,  # 每轮迭代次数
        eval_during_train=cfg.TRAIN.eval_during_train,  # 训练中评估
        eval_freq=cfg.TRAIN.eval_freq,      # 评估频率
        equation=equation,                  # 物理方程
        geom=geom,                          # 几何区域
        validator=validator,                # 验证器
        visualizer=visualizer,              # 可视化器
    )
    
    # 注册训练监控回调函数
    solver.register_callback_on_epoch_end(monitor.on_epoch_end)
    
    # 训练模型
    logger.info("开始训练，启动训练监控看板...")
    solver.train()
    
    # 训练完成后绘制监控曲线
    monitor.plot_training_curves(f"{cfg.output_dir}/training_curves.png")
    
    # 训练完成后进行评估
    solver.eval()
    # 训练完成后进行可视化
    solver.visualize()


def evaluate(cfg: DictConfig):
    """
    评估函数：加载预训练模型并进行评估
    
    参数:
        cfg: 配置文件对象，包含模型路径和评估参数
    """
    # 设置神经网络模型
    model = ppsci.arch.MLP(**cfg.MODEL)

    # 设置物理方程
    equation = {"laplace": ppsci.equation.Laplace(dim=2)}

    # 设置几何区域
    geom = {
        "rect": ppsci.geometry.Rectangle(
            cfg.DIAGONAL_COORD.xmin,
            cfg.DIAGONAL_COORD.xmax
        )
    }

    # 计算解析解函数
    def u_solution_func(out):
        """计算拉普拉斯方程的解析解"""
        x, y = out["x"], out["y"]
        return np.cos(x) * np.cosh(y)

    # 计算总采样点数
    NPOINT_TOTAL = cfg.NPOINT_INTERIOR + cfg.NPOINT_BC

    # 设置验证器
    mse_metric = ppsci.validate.GeometryValidator(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["rect"],
        {
            "dataset": "IterableNamedArrayDataset",
            "total_size": NPOINT_TOTAL,
        },
        ppsci.loss.MSELoss(),
        evenly=True,
        metric={"MSE": ppsci.metric.MSE()},
        with_initial=True,
        name="MSE_Metric",
    )
    validator = {mse_metric.name: mse_metric}

    # 设置可视化器
    vis_points = geom["rect"].sample_interior(NPOINT_TOTAL, evenly=True)
    visualizer = {
        "visualize_u": ppsci.visualize.VisualizerVtu(
            vis_points,
            {"u": lambda d: d["u"]},
            num_timestamps=1,
            prefix="result_u",
        )
    }

    # 初始化求解器（仅用于评估）
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,  # 预训练模型路径
    )
    
    # 进行评估
    solver.eval()
    # 进行可视化
    solver.visualize()


def export(cfg: DictConfig):
    """
    模型导出函数：将训练好的模型导出为推理格式
    
    参数:
        cfg: 配置文件对象，包含导出路径和模型配置
    """
    # 设置神经网络模型
    model = ppsci.arch.MLP(**cfg.MODEL)

    # 初始化求解器
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,  # 预训练模型路径
    )
    
    # 导出模型为推理格式
    from paddle.static import InputSpec

    # 定义输入规范
    input_spec = [
        {key: InputSpec([None, 1], "float32", name=key) for key in model.input_keys},
    ]
    
    # 执行模型导出
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    """
    推理函数：使用导出的模型进行预测
    
    参数:
        cfg: 配置文件对象，包含推理参数和模型路径
    """
    from deploy.python_infer import pinn_predictor

    # 创建预测器实例
    predictor = pinn_predictor.PINNPredictor(cfg)

    # 设置几何区域
    geom = {
        "rect": ppsci.geometry.Rectangle(
            cfg.DIAGONAL_COORD.xmin,
            cfg.DIAGONAL_COORD.xmax
        )
    }
    
    # 计算总采样点数
    NPOINT_TOTAL = cfg.NPOINT_INTERIOR + cfg.NPOINT_BC
    
    # 在几何区域内采样点
    input_dict = geom["rect"].sample_interior(NPOINT_TOTAL, evenly=True)

    # 使用预测器进行预测
    output_dict = predictor.predict(
        {key: input_dict[key] for key in cfg.MODEL.input_keys},  # 输入数据
        cfg.INFER.batch_size                                     # 批处理大小
    )

    # 映射输出数据到配置文件中定义的输出键
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict.keys())
    }

    # 保存预测结果到VTU文件（用于可视化）
    ppsci.visualize.save_vtu_from_dict(
        "./laplace2d_pred.vtu",           # 输出文件名
        {**input_dict, **output_dict},     # 合并输入和输出数据
        input_dict.keys(),                 # 输入字段名
        cfg.MODEL.output_keys,             # 输出字段名
    )


@hydra.main(version_base=None, config_path="./conf", config_name="laplace2d.yaml")
def main(cfg: DictConfig):
    """
    主函数：根据配置模式执行相应操作
    
    参数:
        cfg: Hydra配置对象，包含运行模式和所有参数
    """
    if cfg.mode == "train":
        train(cfg)      # 训练模式
    elif cfg.mode == "eval":
        evaluate(cfg)   # 评估模式
    elif cfg.mode == "export":
        export(cfg)     # 导出模式
    elif cfg.mode == "infer":
        inference(cfg)  # 推理模式
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'export', 'infer'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    """程序入口点"""
    main()
