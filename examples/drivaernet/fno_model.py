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

"""FNO drag-coefficient head and occupancy dataset for DrivAerNet.

Point clouds are rasterized onto three orthogonal occupancy views so that
``TFNO2dNet`` can run on a regular grid; a small MLP then maps pooled
Fourier features (plus raw vehicle extents) to the scalar drag coefficient Cd.
"""

from __future__ import annotations

import os
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import pandas as pd
from paddle import nn

from ppsci.arch import base
from ppsci.arch.tfnonet import TFNO2dNet
from ppsci.arch.tfnonet import TFNO3dNet
from ppsci.data import register_to_dataset
from ppsci.data.dataset.drivaernet_dataset import DrivAerNetDataset
from ppsci.utils import logger

GEOM_SIZE_KEY = "geom_size"


def points_to_multiview_occupancy(
    points: Union[np.ndarray, paddle.Tensor],
    grid_size: int,
    bbox_min: Optional[np.ndarray] = None,
    bbox_span: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin a point cloud into three log-density occupancy maps of shape ``(3, H, W)``.

    Channel 0 is the top view (x–y), channel 1 the side view (x–z), and
    channel 2 the front view (y–z). Each car is fitted to the unit cube so the
    64×64 / 128×128 grid is spent on shape, not empty margin. Absolute size is returned
    separately as ``sample_span / bbox_span`` (or ones if no dataset box).
    """
    if isinstance(points, paddle.Tensor):
        points = points.numpy()
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    sample_min = pts.min(axis=0)
    sample_span = np.maximum(pts.max(axis=0) - sample_min, 1e-6)
    uvw = (pts - sample_min) / sample_span
    idx = np.clip((uvw * (grid_size - 1e-4)).astype(np.int32), 0, grid_size - 1)

    occ = np.zeros((3, grid_size, grid_size), dtype=np.float32)
    np.add.at(occ[0], (idx[:, 1], idx[:, 0]), 1.0)
    np.add.at(occ[1], (idx[:, 2], idx[:, 0]), 1.0)
    np.add.at(occ[2], (idx[:, 2], idx[:, 1]), 1.0)
    occ = np.log1p(occ)
    occ = _smooth_occupancy(occ)
    peak = float(occ.max())
    if peak > 0.0:
        occ = occ / peak
    if bbox_span is None:
        geom_size = np.ones(3, dtype=np.float32)
    else:
        ref = np.maximum(np.asarray(bbox_span, dtype=np.float32).reshape(3), 1e-6)
        geom_size = (sample_span / ref).astype(np.float32)
    return occ, geom_size


def points_to_voxel_occupancy(
    points: Union[np.ndarray, paddle.Tensor],
    grid_size: int,
    bbox_min: Optional[np.ndarray] = None,
    bbox_span: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin a point cloud into a log-density volume of shape ``(1, D, H, W)``.

    Axes are ``(z, y, x)`` so ``TFNO3dNet`` sees a channel-first 3D field.
    Each car is fitted to the unit cube; absolute size is returned separately.
    """
    if isinstance(points, paddle.Tensor):
        points = points.numpy()
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    sample_min = pts.min(axis=0)
    sample_span = np.maximum(pts.max(axis=0) - sample_min, 1e-6)
    uvw = (pts - sample_min) / sample_span
    idx = np.clip((uvw * (grid_size - 1e-4)).astype(np.int32), 0, grid_size - 1)
    occ = np.zeros((1, grid_size, grid_size, grid_size), dtype=np.float32)
    np.add.at(occ[0], (idx[:, 2], idx[:, 1], idx[:, 0]), 1.0)
    occ = np.log1p(occ)
    occ = _smooth_occupancy_3d(occ)
    peak = float(occ.max())
    if peak > 0.0:
        occ = occ / peak
    if bbox_span is None:
        geom_size = np.ones(3, dtype=np.float32)
    else:
        ref = np.maximum(np.asarray(bbox_span, dtype=np.float32).reshape(3), 1e-6)
        geom_size = (sample_span / ref).astype(np.float32)
    return occ, geom_size


def _smooth_occupancy(occ: np.ndarray) -> np.ndarray:
    """3×3 box filter so sparse bins are less salt-and-pepper for the FNO."""
    padded = np.pad(occ, ((0, 0), (1, 1), (1, 1)), mode="constant")
    smoothed = (
        padded[:, 0:-2, 0:-2]
        + padded[:, 0:-2, 1:-1]
        + padded[:, 0:-2, 2:]
        + padded[:, 1:-1, 0:-2]
        + padded[:, 1:-1, 1:-1]
        + padded[:, 1:-1, 2:]
        + padded[:, 2:, 0:-2]
        + padded[:, 2:, 1:-1]
        + padded[:, 2:, 2:]
    ) / 9.0
    return smoothed.astype(np.float32)


def _smooth_occupancy_3d(occ: np.ndarray) -> np.ndarray:
    """3×3×3 box filter on a ``(C, D, H, W)`` occupancy volume."""
    padded = np.pad(occ, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="constant")
    smoothed = np.zeros_like(occ)
    for dz in range(3):
        for dy in range(3):
            for dx in range(3):
                smoothed += padded[:, dz : dz + occ.shape[1], dy : dy + occ.shape[2], dx : dx + occ.shape[3]]
    return (smoothed / 27.0).astype(np.float32)


def _read_design_ids(subset_dir: str, ids_file: str) -> list:
    path = os.path.join(subset_dir, ids_file)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().split()


def compute_cd_stats(csv_file: str, subset_dir: str, ids_file: str) -> Tuple[float, float]:
    """Mean / std of ``Average Cd`` on the given design-id split."""
    frame = pd.read_csv(csv_file)
    ids = set(_read_design_ids(subset_dir, ids_file))
    values = frame.loc[frame["Design"].isin(ids), "Average Cd"].to_numpy(dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"No Cd values found for ids in {ids_file}")
    mean = float(values.mean())
    std = float(max(values.std(ddof=0), 1e-8))
    return mean, std


def compute_global_bbox(
    root_dir: str,
    design_ids: Sequence[str],
    cache_path: Optional[str] = None,
    pad: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box covering the listed point clouds, with optional cache."""
    if cache_path and os.path.isfile(cache_path):
        cached = np.load(cache_path)
        return cached["mins"].astype(np.float32), cached["span"].astype(np.float32)

    mins = np.full(3, np.inf, dtype=np.float64)
    maxs = np.full(3, -np.inf, dtype=np.float64)
    used = 0
    for design_id in design_ids:
        load_path = os.path.join(root_dir, f"{design_id}.paddle_tensor")
        if not os.path.isfile(load_path):
            continue
        vertices = paddle.load(path=str(load_path))
        pts = vertices.numpy() if isinstance(vertices, paddle.Tensor) else np.asarray(vertices)
        mins = np.minimum(mins, pts.min(axis=0))
        maxs = np.maximum(maxs, pts.max(axis=0))
        used += 1
        del vertices, pts

    if used == 0 or not np.isfinite(mins).all():
        raise FileNotFoundError(f"No point clouds found under {root_dir} for global bbox")

    span = np.maximum(maxs - mins, 1e-6)
    mins = mins - pad * span
    span = span * (1.0 + 2.0 * pad)
    mins_f = mins.astype(np.float32)
    span_f = span.astype(np.float32)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, mins=mins_f, span=span_f)
        logger.message(f"Cached FNO global bbox ({used} cars) to {cache_path}")
    else:
        logger.message(f"Computed FNO global bbox from {used} cars")
    return mins_f, span_f


def prepare_fno_geometry_stats(
    dataset_path: str,
    aero_coeff: str,
    subset_dir: str,
    train_ids_file: str,
    cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Train-split global occupancy box and Cd mean/std."""
    ids = _read_design_ids(subset_dir, train_ids_file)
    bbox_min, bbox_span = compute_global_bbox(dataset_path, ids, cache_path=cache_path)
    cd_mean, cd_std = compute_cd_stats(aero_coeff, subset_dir, train_ids_file)
    logger.message(
        f"FNO stats: bbox_min={bbox_min.tolist()}, bbox_span={bbox_span.tolist()}, "
        f"Cd mean={cd_mean:.5f}, std={cd_std:.5f}"
    )
    return bbox_min, bbox_span, cd_mean, cd_std


@register_to_dataset
class DrivAerNetOccupancyDataset(paddle.io.Dataset):
    """Wrap ``DrivAerNetDataset`` and convert vertices to multi-view occupancy.

    Scale/translate augmentations from the point-cloud dataset are disabled:
    they would erase the absolute size encoded by a shared world box.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        weight_keys: Tuple[str, ...],
        subset_dir: str,
        ids_file: str,
        root_dir: str,
        csv_file: str,
        num_points: int,
        grid_size: int = 64,
        occupancy_layout: str = "views",
        bbox_min: Optional[Sequence[float]] = None,
        bbox_span: Optional[Sequence[float]] = None,
        transform=None,
        pointcloud_exist: bool = True,
        train_fractions: float = 1.0,
        mode: str = "eval",
    ):
        super().__init__()
        self.input_keys = tuple(input_keys)
        self.label_keys = tuple(label_keys)
        self.weight_keys = tuple(weight_keys)
        self.grid_size = int(grid_size)
        self.occupancy_layout = str(occupancy_layout)
        self.mode = mode
        self.bbox_min = (
            None if bbox_min is None else np.asarray(bbox_min, dtype=np.float32).reshape(3)
        )
        self.bbox_span = (
            None
            if bbox_span is None
            else np.asarray(bbox_span, dtype=np.float32).reshape(3)
        )
        self.base = DrivAerNetDataset(
            input_keys=("vertices",),
            label_keys=self.label_keys,
            weight_keys=self.weight_keys,
            subset_dir=subset_dir,
            ids_file=ids_file,
            root_dir=root_dir,
            csv_file=csv_file,
            num_points=num_points,
            transform=transform,
            pointcloud_exist=pointcloud_exist,
            train_fractions=train_fractions,
            mode=mode,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        input_dict, label_dict, weight_dict = self.base.__getitem__(
            idx, apply_augmentations=False
        )
        raster = (
            points_to_voxel_occupancy
            if self.occupancy_layout in {"voxels", "volume", "3d"}
            else points_to_multiview_occupancy
        )
        occupancy, geom_size = raster(
            input_dict["vertices"],
            self.grid_size,
            bbox_min=self.bbox_min,
            bbox_span=self.bbox_span,
        )
        labels = {
            key: np.asarray(value, dtype=np.float32).reshape(-1)
            for key, value in label_dict.items()
        }
        weights = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in weight_dict.items()
        }
        return (
            {self.input_keys[0]: occupancy, GEOM_SIZE_KEY: geom_size},
            labels,
            weights,
        )


class FNOCdNet(base.Arch):
    """2D FNO backbone with mean/max pooling for scalar Cd regression.

    The head predicts a residual around the training-set Cd mean so the
    network can start near the unconditional baseline without blocking
    gradients (no last-layer zero-init, no ``× std``).

    Args:
        input_keys: Occupancy tensor key, e.g. ``("occupancy",)``.
        output_keys: Drag coefficient key, e.g. ``("cd_value",)``.
        n_modes_height: Fourier modes along height.
        n_modes_width: Fourier modes along width.
        n_modes_depth: If set, use ``TFNO3dNet`` on a voxel occupancy.
        hidden_channels: FNO hidden width.
        in_channels: Occupancy channels (3 views or 1 volume).
        field_channels: FNO output channels pooled into the Cd head. Defaults to 32.
        lifting_channels: Hidden width of the lifting MLP. Defaults to 64.
        projection_channels: Hidden width of the projection MLP. Defaults to 64.
        n_layers: Number of Fourier layers. Defaults to 4.
        head_hidden: Hidden width of the Cd MLP head. Defaults to 128.
        cd_mean: Training-set Cd mean added to the residual head.
        cd_std: Unused in forward; kept so checkpoints stay compatible.
        use_size_features: Concatenate per-sample extents to the pooled field.
        norm: FNO block normalization, e.g. ``group_norm``.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        n_modes_height: int,
        n_modes_width: int,
        hidden_channels: int,
        n_modes_depth: Optional[int] = None,
        in_channels: int = 3,
        field_channels: int = 32,
        lifting_channels: int = 64,
        projection_channels: int = 64,
        n_layers: int = 4,
        head_hidden: int = 128,
        cd_mean: float = 0.0,
        cd_std: float = 1.0,
        use_size_features: bool = True,
        norm: Optional[str] = "group_norm",
        factorization: str = "dense",
        **kwargs,
    ):
        super().__init__()
        self.input_keys = tuple(input_keys)
        self.output_keys = tuple(output_keys)
        self.field_channels = int(field_channels)
        self.use_size_features = bool(use_size_features)
        self.spatial_dims = 3 if n_modes_depth is not None else 2
        backbone_kwargs = dict(
            input_keys=self.input_keys,
            output_keys=("field",),
            n_modes_height=int(n_modes_height),
            n_modes_width=int(n_modes_width),
            hidden_channels=int(hidden_channels),
            in_channels=int(in_channels),
            out_channels=self.field_channels,
            lifting_channels=int(lifting_channels),
            projection_channels=int(projection_channels),
            n_layers=int(n_layers),
            factorization=str(factorization),
            rank=float(kwargs.get("rank", 1.0)),
            skip="linear",
            mlp_skip="soft-gating",
            norm=norm,
        )
        if self.spatial_dims == 3:
            self.backbone = TFNO3dNet(
                n_modes_depth=int(n_modes_depth),
                **backbone_kwargs,
            )
        else:
            self.backbone = TFNO2dNet(**backbone_kwargs)
        head_in = self.field_channels * 2 + (3 if self.use_size_features else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, int(head_hidden)),
            nn.GELU(),
            nn.Linear(int(head_hidden), 1),
        )
        self.register_buffer(
            "cd_mean", paddle.to_tensor([float(cd_mean)], dtype="float32")
        )
        self.register_buffer(
            "cd_std", paddle.to_tensor([float(max(cd_std, 1e-8))], dtype="float32")
        )

    def forward(self, x: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        field = self.backbone({self.input_keys[0]: x[self.input_keys[0]]})["field"]
        flat = field.flatten(start_axis=-self.spatial_dims, stop_axis=-1)
        pooled = paddle.concat(
            [flat.mean(axis=-1), paddle.max(flat, axis=-1)], axis=-1
        )
        if self.use_size_features:
            size = x[GEOM_SIZE_KEY]
            if size.ndim != 2:
                size = size.reshape([pooled.shape[0], -1])
            pooled = paddle.concat([pooled, size], axis=-1)
        residual = self.head(pooled)
        cd_value = residual + self.cd_mean
        return {self.output_keys[0]: cd_value}


def build_fno_model(
    cfg,
    cd_mean: float = 0.0,
    cd_std: float = 1.0,
) -> FNOCdNet:
    """Construct ``FNOCdNet`` from a Hydra model config block."""
    arch = str(cfg.MODEL.get("arch", "FNO")).upper()
    n_modes_depth = cfg.MODEL.get("n_modes_depth", None)
    if n_modes_depth is None and arch in {"FNO3D", "TFNO3D", "TFNO3DNET"}:
        raise ValueError("3D FNO configs must set MODEL.n_modes_depth")
    default_in = 1 if n_modes_depth is not None else 3
    return FNOCdNet(
        input_keys=tuple(cfg.MODEL.input_keys),
        output_keys=tuple(cfg.MODEL.output_keys),
        n_modes_height=int(cfg.MODEL.n_modes_height),
        n_modes_width=int(cfg.MODEL.n_modes_width),
        n_modes_depth=None if n_modes_depth is None else int(n_modes_depth),
        hidden_channels=int(cfg.MODEL.hidden_channels),
        in_channels=int(cfg.MODEL.get("in_channels", default_in)),
        field_channels=int(cfg.MODEL.get("field_channels", 32)),
        lifting_channels=int(cfg.MODEL.get("lifting_channels", 64)),
        projection_channels=int(cfg.MODEL.get("projection_channels", 64)),
        n_layers=int(cfg.MODEL.get("n_layers", 4)),
        head_hidden=int(cfg.MODEL.get("head_hidden", 128)),
        cd_mean=cd_mean,
        cd_std=cd_std,
        use_size_features=bool(cfg.MODEL.get("use_size_features", True)),
        norm=cfg.MODEL.get("norm", "group_norm"),
        factorization=str(cfg.MODEL.get("factorization", "dense")),
        rank=float(cfg.MODEL.get("rank", 1.0)),
    )
