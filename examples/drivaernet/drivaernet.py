# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""DrivAerNet drag prediction.

Two models share the same CFD car dataset:

* ``RegDGCNN`` (default yaml): graph network on raw point clouds.
* ``FNO`` (``conf/drivaernet_fno.yaml``): TFNO2d on three occupancy views.
"""

from __future__ import annotations

import os
import sys
import tarfile
import urllib.request
import warnings
from functools import partial
from os import path as osp
from typing import Dict
from typing import Tuple

import hydra
import matplotlib
import numpy as np
import paddle
from omegaconf import DictConfig
from omegaconf import open_dict

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import ppsci
from ppsci.utils import logger

EXAMPLE_DIR = osp.dirname(osp.abspath(__file__))
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from fno_model import build_fno_model  # noqa: E402  (script dir must be on sys.path)
from fno_model import prepare_fno_geometry_stats  # noqa: E402
DEFAULT_DATA_URL = (
    "https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/data.tar"
)


def _is_fno(cfg: DictConfig) -> bool:
    arch = str(cfg.MODEL.get("arch", "RegDGCNN")).upper()
    return arch in {"FNO", "TFNO", "FNO2D", "FNO3D", "TFNO3D", "TFNO3DNET"}


def resolve_path(path: str) -> str:
    """Resolve a data path against cwd first, then the example directory."""
    if osp.isabs(path):
        return path
    cwd_path = osp.abspath(path)
    if osp.exists(cwd_path):
        return cwd_path
    alt = osp.abspath(osp.join(EXAMPLE_DIR, path))
    if osp.exists(alt):
        return alt
    return cwd_path


def ensure_drivaernet_data(cfg: DictConfig) -> Tuple[str, str, str]:
    """Return ``(dataset_path, aero_coeff, subset_dir)``, extracting data.tar if needed."""
    dataset_path = resolve_path(cfg.ARGS.dataset_path)
    aero_coeff = resolve_path(cfg.ARGS.aero_coeff)
    subset_dir = resolve_path(cfg.ARGS.subset_dir)
    if osp.isdir(dataset_path) and osp.isfile(aero_coeff) and osp.isdir(subset_dir):
        return dataset_path, aero_coeff, subset_dir

    data_dir = osp.join(EXAMPLE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    tar_path = osp.join(data_dir, "data.tar")
    if not osp.isfile(tar_path):
        url = str(cfg.ARGS.get("data_url", DEFAULT_DATA_URL))
        logger.message(f"Downloading DrivAerNet archive from {url}")
        urllib.request.urlretrieve(url, tar_path)

    logger.message(f"Extracting {tar_path}")
    with tarfile.open(tar_path, "r") as archive:
        names = archive.getnames()
        prefix = os.path.commonprefix(names)
        dest = EXAMPLE_DIR if prefix.startswith("data") else data_dir
        archive.extractall(dest)

    dataset_path = resolve_path(cfg.ARGS.dataset_path)
    aero_coeff = resolve_path(cfg.ARGS.aero_coeff)
    subset_dir = resolve_path(cfg.ARGS.subset_dir)
    missing = [
        name
        for name, path in (
            ("dataset_path", dataset_path),
            ("aero_coeff", aero_coeff),
            ("subset_dir", subset_dir),
        )
        if not osp.exists(path)
    ]
    if missing:
        raise FileNotFoundError(
            "DrivAerNet files missing after extract: "
            + ", ".join(missing)
            + f" (dataset={dataset_path}, csv={aero_coeff}, subset={subset_dir})"
        )
    return dataset_path, aero_coeff, subset_dir


def _fno_dataloader_cfg(
    cfg: DictConfig,
    dataset_path: str,
    aero_coeff: str,
    subset_dir: str,
    ids_file: str,
    num_points: int,
    batch_size: int,
    num_workers: int,
    mode: str,
    train_fractions: float = 1.0,
) -> Dict:
    dataset_cfg = {
        "name": "DrivAerNetOccupancyDataset",
        "root_dir": dataset_path,
        "input_keys": list(cfg.MODEL.input_keys),
        "label_keys": list(cfg.MODEL.output_keys),
        "weight_keys": list(cfg.MODEL.weight_keys),
        "subset_dir": subset_dir,
        "ids_file": ids_file,
        "csv_file": aero_coeff,
        "num_points": num_points,
        "grid_size": int(cfg.MODEL.grid_size),
        "occupancy_layout": str(cfg.MODEL.get("occupancy_layout", "views")),
        "mode": mode,
    }
    bbox_min = cfg.MODEL.get("bbox_min", None)
    bbox_span = cfg.MODEL.get("bbox_span", None)
    if bbox_min is not None and bbox_span is not None:
        dataset_cfg["bbox_min"] = [float(x) for x in bbox_min]
        dataset_cfg["bbox_span"] = [float(x) for x in bbox_span]
    if mode == "train":
        dataset_cfg["train_fractions"] = train_fractions
    return {
        "dataset": dataset_cfg,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": mode == "train",
            "shuffle": mode == "train",
        },
        "batch_size": batch_size,
        "num_workers": num_workers,
    }


def _cd_scatter_figure(pred: np.ndarray, label: np.ndarray, title: str):
    pred = np.asarray(pred).reshape(-1)
    label = np.asarray(label).reshape(-1)
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(label, pred, s=12, alpha=0.7)
    lo = float(min(label.min(), pred.min()))
    hi = float(max(label.max(), pred.max()))
    axis.plot([lo, hi], [lo, hi], color="crimson", linestyle="--", linewidth=1)
    axis.set_xlabel("True Cd")
    axis.set_ylabel("Predicted Cd")
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def _occupancy_preview_views(occupancy: np.ndarray) -> np.ndarray:
    """Return three 2D slices for TensorBoard / PNG, from views or a 3D volume."""
    arr = np.asarray(occupancy)
    if arr.ndim == 3:
        return arr[:3]
    if arr.ndim == 4:
        volume = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
        depth, height, width = volume.shape
        return np.stack(
            [
                volume[depth // 2],
                volume[:, height // 2, :],
                volume[:, :, width // 2],
            ]
        )
    raise ValueError(f"Unsupported occupancy shape {arr.shape}")


def _occupancy_figure(occupancy: np.ndarray):
    views = _occupancy_preview_views(occupancy)
    titles = ["top (x-y)", "side (x-z)", "front (y-z)"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, image, title in zip(axes, views, titles):
        axis.imshow(image, origin="lower", cmap="magma")
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    return fig


def _plot_fno_results(
    occupancy: np.ndarray,
    pred: np.ndarray,
    label: np.ndarray,
    output_dir: str,
) -> None:
    visual_dir = osp.join(output_dir, "visual")
    os.makedirs(visual_dir, exist_ok=True)
    fig = _occupancy_figure(occupancy)
    occ_path = osp.join(visual_dir, "occupancy_views.png")
    fig.savefig(occ_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = _cd_scatter_figure(pred, label, "FNO drag coefficient")
    scatter_path = osp.join(visual_dir, "cd_scatter.png")
    fig.savefig(scatter_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.message(f"Saved occupancy preview to {occ_path}")
    logger.message(f"Saved Cd scatter to {scatter_path}")


def _sample_cd_batch(solver: ppsci.solver.Solver, validator, max_batches: int = 4):
    preds = []
    labels = []
    first_input = None
    cd_key = solver.model.output_keys[0]
    for batch_id, (input_dict, label_dict, _) in enumerate(validator.data_loader):
        out = solver.predict(input_dict, batch_size=None, return_numpy=True)
        preds.append(np.asarray(out[cd_key]).reshape(-1))
        labels.append(np.asarray(label_dict[cd_key]).reshape(-1))
        if first_input is None:
            first_input = input_dict
        if batch_id + 1 >= max_batches:
            break
    if not preds:
        return None, None, None
    return np.concatenate(preds), np.concatenate(labels), first_input


def _log_tensorboard_extras(
    solver: ppsci.solver.Solver, validator, cfg: DictConfig
) -> None:
    writer = getattr(solver, "tbd_writer", None)
    if writer is None:
        return
    try:
        _log_tensorboard_extras_impl(solver, validator, cfg, writer)
    except Exception as exc:
        logger.warning(f"TensorBoard extra logging skipped: {exc}")


def _log_tensorboard_extras_impl(
    solver: ppsci.solver.Solver, validator, cfg: DictConfig, writer
) -> None:
    epoch_id = solver.global_step // max(solver.iters_per_epoch, 1)
    if epoch_id % max(solver.eval_freq, 1) != 0:
        return

    num_params = getattr(solver.model, "num_params", None)
    if num_params is None:
        num_params = int(sum(p.numel() for p in solver.model.parameters()))
    writer.add_scalar(
        "train/num_params", float(num_params), solver.global_step
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
        try:
            writer.add_histogram(
                f"params/{name}", param.numpy().reshape(-1), solver.global_step
            )
        except Exception:
            continue

    pred, label, first_input = _sample_cd_batch(solver, validator)
    if pred is not None:
        arch = str(cfg.MODEL.get("arch", "FNO")).upper()
        fig = _cd_scatter_figure(pred, label, f"{arch} Cd")
        writer.add_figure("pred/cd_scatter", fig, global_step=solver.global_step)
        plt.close(fig)
        if _is_fno(cfg) and first_input is not None:
            occ = first_input[solver.model.input_keys[0]]
            if isinstance(occ, paddle.Tensor):
                occ = occ.numpy()
            fig = _occupancy_figure(np.asarray(occ)[0])
            writer.add_figure(
                "pred/occupancy_views", fig, global_step=solver.global_step
            )
            plt.close(fig)
    writer.flush()


def _attach_tensorboard(
    solver: ppsci.solver.Solver, validator, cfg: DictConfig
) -> None:
    if getattr(solver, "tbd_writer", None) is None:
        return
    solver.register_callback_on_epoch_end(
        lambda s: _log_tensorboard_extras(s, validator, cfg)
    )


def _close_tensorboard(solver: ppsci.solver.Solver) -> None:
    if solver.tbd_writer is not None:
        solver.tbd_writer.flush()
        solver.tbd_writer.close()


def _collect_cd(solver: ppsci.solver.Solver, validator) -> Tuple[np.ndarray, np.ndarray]:
    preds = []
    labels = []
    occupancy = None
    for input_dict, label_dict, _ in validator.data_loader:
        out = solver.predict(input_dict, batch_size=None, return_numpy=True)
        preds.append(np.asarray(out[solver.model.output_keys[0]]))
        labels.append(np.asarray(label_dict[solver.model.output_keys[0]]))
        if occupancy is None:
            occ = input_dict[solver.model.input_keys[0]]
            if isinstance(occ, paddle.Tensor):
                occ = occ.numpy()
            occupancy = np.asarray(occ)[0]
    return occupancy, np.concatenate(preds, axis=0), np.concatenate(labels, axis=0)


def _prepare_fno_runtime(cfg: DictConfig):
    dataset_path, aero_coeff, subset_dir = ensure_drivaernet_data(cfg)
    bbox_min, bbox_span, cd_mean, cd_std = prepare_fno_geometry_stats(
        dataset_path,
        aero_coeff,
        subset_dir,
        cfg.TRAIN.train_ids_file,
        cache_path=osp.join(EXAMPLE_DIR, "data", "fno_global_bbox.npz"),
    )
    with open_dict(cfg):
        cfg.MODEL.bbox_min = [float(x) for x in bbox_min]
        cfg.MODEL.bbox_span = [float(x) for x in bbox_span]
    model = build_fno_model(cfg, cd_mean=cd_mean, cd_std=cd_std)
    return dataset_path, aero_coeff, subset_dir, model


def train_fno(cfg: DictConfig):
    dataset_path, aero_coeff, subset_dir, model = _prepare_fno_runtime(cfg)

    train_loader_cfg = _fno_dataloader_cfg(
        cfg,
        dataset_path,
        aero_coeff,
        subset_dir,
        cfg.TRAIN.train_ids_file,
        cfg.TRAIN.num_points,
        cfg.TRAIN.batch_size,
        cfg.TRAIN.num_workers,
        mode="train",
        train_fractions=cfg.TRAIN.train_fractions,
    )
    constraint = {
        "DrivAerNet_constraint": ppsci.constraint.SupervisedConstraint(
            train_loader_cfg,
            ppsci.loss.MSELoss("mean"),
            name="DrivAerNet_constraint",
        )
    }

    valid_loader_cfg = _fno_dataloader_cfg(
        cfg,
        dataset_path,
        aero_coeff,
        subset_dir,
        cfg.TRAIN.eval_ids_file,
        cfg.TRAIN.num_points,
        cfg.EVAL.batch_size,
        cfg.EVAL.num_workers,
        mode="eval",
    )
    validator = {
        "DrivAerNet_eval": ppsci.validate.SupervisedValidator(
            valid_loader_cfg,
            loss=ppsci.loss.MSELoss("mean"),
            metric={
                "MSE": ppsci.metric.MSE(),
                "MAE": ppsci.metric.MAE(),
                "Max AE": ppsci.metric.MaxAE(),
                "R2": ppsci.metric.R2Score(),
            },
            name="DrivAerNet_eval",
        )
    }

    iters_per_epoch = len(next(iter(constraint.values())).data_loader)
    lr_scheduler = ppsci.optimizer.lr_scheduler.Cosine(
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=iters_per_epoch,
        learning_rate=cfg.optimizer.lr,
        eta_min=float(cfg.optimizer.get("eta_min", 1e-4)),
        warmup_epoch=int(cfg.optimizer.get("warmup_epoch", 0)),
        by_epoch=True,
    )()
    optimizer = ppsci.optimizer.Adam(
        lr_scheduler, weight_decay=cfg.optimizer.weight_decay
    )(model)

    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        validator=validator,
        cfg=cfg,
    )
    fno_validator = next(iter(validator.values()))
    _attach_tensorboard(solver, fno_validator, cfg)
    solver.train()
    solver.eval()
    solver.plot_loss_history(by_epoch=True, smooth_step=1)
    occupancy, pred, label = _collect_cd(solver, fno_validator)
    _plot_fno_results(occupancy, pred, label, cfg.output_dir)
    _close_tensorboard(solver)


def evaluate_fno(cfg: DictConfig):
    dataset_path, aero_coeff, subset_dir, model = _prepare_fno_runtime(cfg)
    valid_loader_cfg = _fno_dataloader_cfg(
        cfg,
        dataset_path,
        aero_coeff,
        subset_dir,
        cfg.EVAL.ids_file,
        cfg.EVAL.num_points,
        cfg.EVAL.batch_size,
        cfg.EVAL.num_workers,
        mode="eval",
    )
    validator = {
        "DrivAerNet_eval": ppsci.validate.SupervisedValidator(
            valid_loader_cfg,
            loss=ppsci.loss.MSELoss("mean"),
            metric={
                "MSE": ppsci.metric.MSE(),
                "MAE": ppsci.metric.MAE(),
                "Max AE": ppsci.metric.MaxAE(),
                "R2": ppsci.metric.R2Score(),
            },
            name="DrivAerNet_eval",
        )
    }
    solver = ppsci.solver.Solver(
        model,
        validator=validator,
        cfg=cfg,
    )
    solver.eval()
    occupancy, pred, label = _collect_cd(
        solver, next(iter(validator.values()))
    )
    _plot_fno_results(occupancy, pred, label, cfg.output_dir)


def train(cfg: DictConfig):
    if _is_fno(cfg):
        train_fno(cfg)
        return

    dataset_path, aero_coeff, subset_dir = ensure_drivaernet_data(cfg)

    # set model
    model = ppsci.arch.RegDGCNN(
        input_keys=cfg.MODEL.input_keys,
        label_keys=cfg.MODEL.output_keys,
        weight_keys=cfg.MODEL.weight_keys,
        args=cfg.MODEL,
    )

    train_dataloader_cfg = {
        "dataset": {
            "name": "DrivAerNetDataset",
            "root_dir": dataset_path,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "weight_keys": cfg.MODEL.weight_keys,
            "subset_dir": subset_dir,
            "ids_file": cfg.TRAIN.train_ids_file,
            "csv_file": aero_coeff,
            "num_points": cfg.TRAIN.num_points,
            "train_fractions": cfg.TRAIN.train_fractions,
            "mode": cfg.mode,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "num_workers": cfg.TRAIN.num_workers,
    }

    drivaernet_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        name="DrivAerNet_constraint",
    )

    constraint = {drivaernet_constraint.name: drivaernet_constraint}

    valid_dataloader_cfg = {
        "dataset": {
            "name": "DrivAerNetDataset",
            "root_dir": dataset_path,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "weight_keys": cfg.MODEL.weight_keys,
            "subset_dir": subset_dir,
            "ids_file": cfg.TRAIN.eval_ids_file,
            "csv_file": aero_coeff,
            "num_points": cfg.TRAIN.num_points,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "num_workers": cfg.TRAIN.num_workers,
    }

    drivaernet_eval = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        loss=ppsci.loss.MSELoss("mean"),
        metric={
            "MSE": ppsci.metric.MSE(),
            "MAE": ppsci.metric.MAE(),
            "Max AE": ppsci.metric.MaxAE(),
            "R2": ppsci.metric.R2Score(),
        },
        name="DrivAerNet_eval",
    )

    validator = {drivaernet_eval.name: drivaernet_eval}

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ReduceOnPlateau(
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=(
            cfg.TRAIN.iters_per_epoch
            * cfg.TRAIN.train_fractions
            // (paddle.distributed.get_world_size() * cfg.TRAIN.batch_size)
            + 1
        ),
        learning_rate=cfg.optimizer.lr,
        mode=cfg.TRAIN.scheduler.mode,
        patience=cfg.TRAIN.scheduler.patience,
        factor=cfg.TRAIN.scheduler.factor,
        verbose=cfg.TRAIN.scheduler.verbose,
    )()

    optimizer = (
        ppsci.optimizer.Adam(lr_scheduler, weight_decay=cfg.optimizer.weight_decay)(
            model
        )
        if cfg.optimizer.optimizer == "adam"
        else ppsci.optimizer.SGD(lr_scheduler, weight_decay=cfg.optimizer.weight_decay)(
            model
        )
    )

    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        iters_per_epoch=(
            cfg.TRAIN.iters_per_epoch
            * cfg.TRAIN.train_fractions
            // (paddle.distributed.get_world_size() * cfg.TRAIN.batch_size)
            + 1
        ),
        constraint=constraint,
        output_dir=cfg.output_dir,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=cfg.TRAIN.epochs,
        validator=validator,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        use_tbd=bool(cfg.get("use_tbd", True)),
        log_freq=cfg.log_freq,
        eval_freq=int(cfg.TRAIN.get("eval_freq", 1)),
        use_amp=True,
        amp_level="O1",
    )

    lr_scheduler.step = partial(lr_scheduler.step, metrics=solver.cur_metric)
    solver.lr_scheduler = lr_scheduler

    _attach_tensorboard(solver, drivaernet_eval, cfg)
    solver.train()
    solver.eval()
    _close_tensorboard(solver)


def evaluate(cfg: DictConfig):
    if _is_fno(cfg):
        evaluate_fno(cfg)
        return

    dataset_path, aero_coeff, subset_dir = ensure_drivaernet_data(cfg)

    # set model
    model = ppsci.arch.RegDGCNN(
        input_keys=cfg.MODEL.input_keys,
        label_keys=cfg.MODEL.output_keys,
        weight_keys=cfg.MODEL.weight_keys,
        args=cfg.MODEL,
    )

    valid_dataloader_cfg = {
        "dataset": {
            "name": "DrivAerNetDataset",
            "root_dir": dataset_path,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "weight_keys": cfg.MODEL.weight_keys,
            "subset_dir": subset_dir,
            "ids_file": cfg.EVAL.ids_file,
            "csv_file": aero_coeff,
            "num_points": cfg.EVAL.num_points,
            "mode": cfg.mode,
        },
        "batch_size": cfg.EVAL.batch_size,
        "num_workers": cfg.EVAL.num_workers,
    }

    drivaernet_eval = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        loss=ppsci.loss.MSELoss("mean"),
        metric={
            "MSE": ppsci.metric.MSE(),
            "MAE": ppsci.metric.MAE(),
            "Max AE": ppsci.metric.MaxAE(),
            "R²": ppsci.metric.R2Score(),
        },
        name="DrivAerNet_eval",
    )

    validator = {drivaernet_eval.name: drivaernet_eval}

    solver = ppsci.solver.Solver(
        model=model,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # evaluate model
    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="drivaernet.yaml")
def main(cfg: DictConfig):
    warnings.filterwarnings("ignore")
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
