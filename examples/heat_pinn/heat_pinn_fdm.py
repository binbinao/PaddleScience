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

from os import path as osp

import fdm
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def plot(input_data, n_eval, pinn_output, fdm_output, cfg):
    x = input_data["x"].reshape(n_eval, n_eval)
    y = input_data["y"].reshape(n_eval, n_eval)

    plt.subplot(2, 1, 1)
    plt.pcolormesh(x, y, pinn_output * fdm.TEMP_SCALE, cmap="magma")
    plt.colorbar()
    plt.title("PINNs trained by FDM data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(2, 1, 2)
    plt.pcolormesh(x, y, fdm_output, cmap="magma")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("FDM")
    plt.tight_layout()
    plt.axis("square")
    plt.savefig(osp.join(cfg.output_dir, "fdm_trained_pinn_comparison.png"))
    plt.close()

    frames_val = np.array([-0.75, -0.5, -0.25, 0.0, +0.25, +0.5, +0.75])
    frames = [*map(int, (frames_val + 1) / 2 * (n_eval - 1))]
    height = 3
    plt.figure("", figsize=(len(frames) * height, 2 * height))

    for i, var_index in enumerate(frames):
        plt.subplot(2, len(frames), i + 1)
        plt.title(f"y = {frames_val[i]:.2f}")
        plt.plot(
            x[:, var_index],
            pinn_output[:, var_index] * fdm.TEMP_SCALE,
            "r--",
            lw=4.0,
            label="PINNs(FDM data)",
        )
        plt.plot(x[:, var_index], fdm_output[:, var_index], "b", lw=2.0, label="FDM")
        plt.ylim(0.0, 100.0)
        plt.xlim(-1.0, +1.0)
        plt.xlabel("x")
        plt.ylabel("T")
        plt.tight_layout()
        plt.legend()

    for i, var_index in enumerate(frames):
        plt.subplot(2, len(frames), len(frames) + i + 1)
        plt.title(f"x = {frames_val[i]:.2f}")
        plt.plot(
            y[var_index, :],
            pinn_output[var_index, :] * fdm.TEMP_SCALE,
            "r--",
            lw=4.0,
            label="PINNs(FDM data)",
        )
        plt.plot(y[var_index, :], fdm_output[var_index, :], "b", lw=2.0, label="FDM")
        plt.ylim(0.0, 100.0)
        plt.xlim(-1.0, +1.0)
        plt.xlabel("y")
        plt.ylabel("T")
        plt.tight_layout()
        plt.legend()

    plt.savefig(osp.join(cfg.output_dir, "fdm_trained_pinn_profiles.png"))
    plt.close()


def eval_with_solver(solver, cfg, no_grad=True):
    n_eval = cfg.FDM.n
    input_data, _, fdm_output = fdm.build_dataset(n_eval, cfg.FDM.length)
    pinn_output = solver.predict(
        input_data, no_grad=no_grad, return_numpy=True
    )["u"].reshape(n_eval, n_eval)
    mse_loss = np.mean(np.square(pinn_output - (fdm_output / fdm.TEMP_SCALE)))
    logger.info(f"The norm MSE loss between the FDM and PINNs(FDM data) is {mse_loss:.5e}")
    plot(input_data, n_eval, pinn_output, fdm_output, cfg)


def train(cfg: DictConfig):
    ppsci.utils.misc.set_random_seed(cfg.seed)
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

    model = ppsci.arch.MLP(**cfg.MODEL)
    input_data, label_data, _ = fdm.build_dataset(cfg.TRAIN.fdm_n, cfg.FDM.length)

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_data,
                "label": label_data,
            },
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": True,
            },
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={"u": lambda out: out["u"]},
        name="FDM_data",
    )
    constraint = {sup_constraint.name: sup_constraint}
    iters_per_epoch = len(sup_constraint.data_loader)

    optimizer = ppsci.optimizer.Adam(learning_rate=cfg.TRAIN.learning_rate)(model)

    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=iters_per_epoch,
        save_freq=cfg.TRAIN.save_freq,
        log_freq=cfg.log_freq,
        seed=cfg.seed,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )
    solver.train()
    eval_with_solver(solver, cfg, no_grad=False)


def evaluate(cfg: DictConfig):
    ppsci.utils.misc.set_random_seed(cfg.seed)
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    model = ppsci.arch.MLP(**cfg.MODEL)
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        log_freq=cfg.log_freq,
        seed=cfg.seed,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )
    eval_with_solver(solver, cfg)


def export(cfg: DictConfig):
    model = ppsci.arch.MLP(**cfg.MODEL)
    solver = ppsci.solver.Solver(model, cfg=cfg)

    from paddle.static import InputSpec

    input_spec = [
        {key: InputSpec([None, 1], "float32", name=key) for key in model.input_keys},
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)
    n_eval = cfg.FDM.n
    input_data, _, fdm_output = fdm.build_dataset(n_eval, cfg.FDM.length)
    output_data = predictor.predict(
        {key: input_data[key] for key in cfg.MODEL.input_keys}, cfg.INFER.batch_size
    )
    output_data = {
        store_key: output_data[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_data.keys())
    }["u"].reshape(n_eval, n_eval)
    mse_loss = np.mean(np.square(output_data - (fdm_output / fdm.TEMP_SCALE)))
    logger.info(
        "The norm MSE loss between the FDM and PINNs(FDM data) "
        f"is {mse_loss:.5e}"
    )
    plot(input_data, n_eval, output_data, fdm_output, cfg)


@hydra.main(version_base=None, config_path="./conf", config_name="heat_pinn.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'export', 'infer'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
