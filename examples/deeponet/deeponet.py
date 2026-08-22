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

"""Unstacked DeepONet for the antiderivative operator.

Paper: Lu et al. Learning nonlinear operators via DeepONet based on the
universal approximation theorem of operators. Nat Mach Intell, 2021.
(arXiv:1910.03193, Sec. 4.1.1)

    G: u(x) |-> s(x) = ∫_0^x u(τ) dτ,    s(0) = 0

Branch net encodes u at m sensors; trunk net encodes query location y;
the operator value is the inner product plus a bias (paper Eq. 2):

    G(u)(y) ≈ Σ_k b_k(u) t_k(y) + b_0
"""

import os
from os import path as osp
from typing import Callable
from typing import Dict
from typing import Tuple

import hydra
import matplotlib
import numpy as np
import paddle
from omegaconf import DictConfig

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import ppsci
from history_tb import export_history
from ppsci.utils import logger


def _rbf_kernel(sensors: np.ndarray, length_scale: float) -> np.ndarray:
    """RBF covariance used by the paper's mean-zero GRF (Sec. 2.2)."""
    dist_sq = (sensors[:, None] - sensors[None, :]) ** 2
    cov = np.exp(-dist_sq / (2.0 * length_scale**2))
    cov = cov + 1e-12 * np.eye(sensors.shape[0])
    return cov


def _sample_grf(
    num_samples: int,
    sensors: np.ndarray,
    length_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw `num_samples` GRF paths evaluated at the same sensors."""
    cov = _rbf_kernel(sensors, length_scale)
    chol = np.linalg.cholesky(cov)
    noise = rng.standard_normal((num_samples, sensors.shape[0]))
    return noise @ chol.T


def _antiderivative_on_grid(u: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of u from sensors[0] to each sensor."""
    dx = np.diff(sensors)
    pieces = 0.5 * (u[:, 1:] + u[:, :-1]) * dx
    zeros = np.zeros((u.shape[0], 1), dtype=u.dtype)
    return np.concatenate([zeros, np.cumsum(pieces, axis=1)], axis=1)


def _interp_rows(query: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Linear interpolation of each row of `fp` at the corresponding `query`."""
    idx = np.searchsorted(xp, query, side="left")
    idx = np.clip(idx, 1, xp.shape[0] - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    rows = np.arange(fp.shape[0])
    f0, f1 = fp[rows, idx - 1], fp[rows, idx]
    weight = (query - x0) / np.maximum(x1 - x0, 1e-12)
    return f0 + weight * (f1 - f0)


def generate_unaligned_dataset(
    num_samples: int,
    num_loc: int,
    length_scale: float,
    rng: np.random.Generator,
    dtype: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build unaligned triplets (u, y, G(u)(y)) as in the paper (Fig. 1B)."""
    sensors = np.linspace(0.0, 1.0, num_loc)
    u = _sample_grf(num_samples, sensors, length_scale, rng).astype(dtype)
    y = rng.random((num_samples, 1)).astype(dtype)
    g_grid = _antiderivative_on_grid(u, sensors)
    g = _interp_rows(y[:, 0], sensors, g_grid).reshape(-1, 1).astype(dtype)
    return u, y, g


def load_or_generate_npz(file_path: str, split: str, cfg: DictConfig) -> None:
    """Reuse an existing npz, otherwise generate GRF antiderivative data."""
    if osp.exists(file_path):
        logger.message(f"Use existing dataset: {file_path}")
        return

    dtype = paddle.get_default_dtype()
    rng = np.random.default_rng(cfg.seed if split == "train" else cfg.seed + 1)
    n_samples = cfg.DATA.n_train if split == "train" else cfg.DATA.n_test
    u, y, g = generate_unaligned_dataset(
        n_samples,
        cfg.MODEL.num_loc,
        cfg.DATA.length_scale,
        rng,
        dtype,
    )
    os.makedirs(osp.dirname(osp.abspath(file_path)), exist_ok=True)
    if split == "train":
        np.savez(file_path, X_train0=u, X_train1=y, y_train=g)
    else:
        np.savez(file_path, X_test0=u, X_test1=y, y_test=g)
    logger.message(
        f"Generated {split} GRF antiderivative data "
        f"(N={n_samples}, m={cfg.MODEL.num_loc}, l={cfg.DATA.length_scale}) "
        f"-> {file_path}"
    )


def _alias_dict(split: str) -> Dict[str, str]:
    if split == "train":
        return {"u": "X_train0", "y": "X_train1", "G": "y_train"}
    return {"u": "X_test0", "y": "X_test1", "G": "y_test"}


def build_npz_dataloader_cfg(file_path: str, split: str) -> dict:
    return {
        "dataset": {
            "name": "IterableNPZDataset",
            "file_path": file_path,
            "input_keys": ("u", "y"),
            "label_keys": ("G",),
            "alias_dict": _alias_dict(split),
        },
    }


def _log_tensorboard_extras(solver: ppsci.solver.Solver, cfg: DictConfig) -> None:
    """Log histograms, best metric, param count and a sample prediction figure."""
    writer = getattr(solver, "tbd_writer", None)
    if writer is None:
        return

    epoch_id = solver.global_step // max(solver.iters_per_epoch, 1)
    if epoch_id % max(solver.eval_freq, 1) != 0:
        return

    writer.add_scalar(
        "train/num_params",
        float(solver.model.num_params),
        solver.global_step,
    )
    writer.add_scalar(
        "eval/best_metric",
        float(solver.best_metric.get("metric", float("nan"))),
        solver.global_step,
    )
    writer.add_scalar(
        "eval/best_metric_epoch",
        float(solver.best_metric.get("epoch", 0)),
        solver.global_step,
    )

    for name, param in solver.model.named_parameters():
        writer.add_histogram(f"params/{name}", param.numpy(), solver.global_step)

    dtype = paddle.get_default_dtype()
    x = np.linspace(0, 1, cfg.MODEL.num_loc, dtype=dtype).reshape([1, cfg.MODEL.num_loc])
    u = np.tile(np.cos(x), [min(int(cfg.NUM_Y), 256), 1])
    y = np.linspace(0, 1, u.shape[0], dtype=dtype).reshape([-1, 1])
    g_ref = np.sin(y)
    g_pred = solver.predict({"u": u, "y": y}, return_numpy=True)[cfg.MODEL.G_key]
    l2rel = np.linalg.norm(g_pred - g_ref) / np.maximum(np.linalg.norm(g_ref), 1e-12)
    writer.add_scalar("eval/sample_cos_L2Rel", float(l2rel), solver.global_step)

    fig, ax = plt.subplots()
    ax.plot(y, g_ref, label=r"$G(u)_{\mathrm{ref}}$")
    ax.plot(y, g_pred, label=r"$G(u)_{\mathrm{pred}}$")
    ax.set_title(r"$u=\cos(x),\ G(u)=\sin(x)$")
    ax.legend()
    writer.add_figure("pred/cos_antiderivative", fig, global_step=solver.global_step)
    plt.close(fig)
    writer.flush()


def train(cfg: DictConfig):
    ppsci.utils.misc.set_random_seed(cfg.seed)
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    load_or_generate_npz(cfg.TRAIN_FILE_PATH, "train", cfg)
    load_or_generate_npz(cfg.VALID_FILE_PATH, "test", cfg)

    # Unstacked DeepONet: branch(u) inner-product trunk(y) + bias
    model = ppsci.arch.DeepONet(**cfg.MODEL)

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        build_npz_dataloader_cfg(cfg.TRAIN_FILE_PATH, "train"),
        ppsci.loss.MSELoss(),
        {"G": lambda out: out["G"]},
    )
    constraint = {sup_constraint.name: sup_constraint}

    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    sup_validator = ppsci.validate.SupervisedValidator(
        build_npz_dataloader_cfg(cfg.VALID_FILE_PATH, "test"),
        ppsci.loss.MSELoss(),
        {"G": lambda out: out["G"]},
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="G_eval",
    )
    validator = {sup_validator.name: sup_validator}

    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        validator=validator,
        cfg=cfg,
    )
    solver.register_callback_on_epoch_end(lambda s: _log_tensorboard_extras(s, cfg))
    solver.train()
    solver.eval()
    if solver.tbd_writer is not None:
        solver.tbd_writer.flush()
        solver.tbd_writer.close()

    def predict_func(input_dict):
        return solver.predict(input_dict, return_numpy=True)[cfg.MODEL.G_key]

    plot(cfg, predict_func)


def evaluate(cfg: DictConfig):
    ppsci.utils.misc.set_random_seed(cfg.seed)
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    load_or_generate_npz(cfg.VALID_FILE_PATH, "test", cfg)
    model = ppsci.arch.DeepONet(**cfg.MODEL)

    sup_validator = ppsci.validate.SupervisedValidator(
        build_npz_dataloader_cfg(cfg.VALID_FILE_PATH, "test"),
        ppsci.loss.MSELoss(),
        {"G": lambda out: out["G"]},
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="G_eval",
    )
    validator = {sup_validator.name: sup_validator}

    solver = ppsci.solver.Solver(
        model,
        validator=validator,
        cfg=cfg,
    )
    solver.eval()

    def predict_func(input_dict):
        return solver.predict(input_dict, return_numpy=True)[cfg.MODEL.G_key]

    plot(cfg, predict_func)


def export(cfg: DictConfig):
    model = ppsci.arch.DeepONet(**cfg.MODEL)
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )
    from paddle.static import InputSpec

    input_spec = [
        {
            model.input_keys[0]: InputSpec(
                [None, cfg.MODEL.num_loc], "float32", name=model.input_keys[0]
            ),
            model.input_keys[1]: InputSpec(
                [None, 1], "float32", name=model.input_keys[1]
            ),
        }
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    from deploy import python_infer

    predictor = python_infer.GeneralPredictor(cfg)

    def predict_func(input_dict):
        return next(iter(predictor.predict(input_dict).values()))

    plot(cfg, predict_func)


def plot_history(cfg: DictConfig):
    """Replay historical train.log into TensorBoardX scalars and curve figures."""
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")
    log_path = cfg.HISTORY.log_path
    tbd_dir = cfg.HISTORY.tbd_dir
    written = export_history(log_path, tbd_dir)
    for logdir in written:
        logger.message(f"Wrote TensorBoardX history run: {logdir}")
    logger.message(
        "View historical curves with:\n"
        f"tensorboard --logdir {tbd_dir} --port 6006"
    )


def plot(cfg: DictConfig, predict_func: Callable):
    dtype = paddle.get_default_dtype()

    def generate_y_u_G_ref(
        u_func: Callable, G_u_func: Callable
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(0, 1, cfg.MODEL.num_loc, dtype=dtype).reshape(
            [1, cfg.MODEL.num_loc]
        )
        u = np.tile(u_func(x), [cfg.NUM_Y, 1])
        y = np.linspace(0, 1, cfg.NUM_Y, dtype=dtype).reshape([cfg.NUM_Y, 1])
        G_ref = G_u_func(y)
        return u, y, G_ref

    func_u_G_pair = [
        (r"$u=\cos(x), G(u)=\sin(x)$", lambda x: np.cos(x), lambda y: np.sin(y)),
        (
            r"$u=\sec^2(x), G(u)=\tan(x)$",
            lambda x: (1 / np.cos(x)) ** 2,
            lambda y: np.tan(y),
        ),
        (
            r"$u=\sec(x)\tan(x), G(u)=\sec(x)-1$",
            lambda x: (1 / np.cos(x) * np.tan(x)),
            lambda y: 1 / np.cos(y) - 1,
        ),
        (
            r"$u=1.5^x\ln{1.5}, G(u)=1.5^x-1$",
            lambda x: 1.5**x * np.log(1.5),
            lambda y: 1.5**y - 1,
        ),
        (r"$u=3x^2, G(u)=x^3$", lambda x: 3 * x**2, lambda y: y**3),
        (r"$u=4x^3, G(u)=x^4$", lambda x: 4 * x**3, lambda y: y**4),
        (r"$u=5x^4, G(u)=x^5$", lambda x: 5 * x**4, lambda y: y**5),
        (r"$u=6x^5, G(u)=x^6$", lambda x: 6 * x**5, lambda y: y**6),
        (r"$u=e^x, G(u)=e^x-1$", lambda x: np.exp(x), lambda y: np.exp(y) - 1),
    ]

    os.makedirs(os.path.join(cfg.output_dir, "visual"), exist_ok=True)
    for i, (title, u_func, G_func) in enumerate(func_u_G_pair):
        u, y, G_ref = generate_y_u_G_ref(u_func, G_func)
        G_pred = predict_func({"u": u, "y": y})
        plt.plot(y, G_ref, label=r"$G(u)(y)_{\mathrm{ref}}$")
        plt.plot(y, G_pred, label=r"$G(u)(y)_{\mathrm{pred}}$")
        plt.legend()
        plt.title(title)
        plt.savefig(os.path.join(cfg.output_dir, "visual", f"func_{i}_result.png"))
        logger.message(
            f"Saved result of function {i} to {cfg.output_dir}/visual/func_{i}_result.png"
        )
        plt.clf()
    plt.close()


@hydra.main(version_base=None, config_path="./conf", config_name="deeponet.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "infer":
        inference(cfg)
    elif cfg.mode == "plot_history":
        plot_history(cfg)
    else:
        raise ValueError(
            "cfg.mode should in ['train', 'eval', 'export', 'infer', "
            f"'plot_history'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
