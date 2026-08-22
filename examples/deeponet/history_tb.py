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

"""Parse PaddleScience train.log and export curves via TensorBoardX."""

from __future__ import annotations

import os
import re
from os import path as osp
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_TRAIN_HEAD_RE = re.compile(r"\[Train\]\[Epoch\s+(\d+)/(\d+)\]\[Iter\s+(\d+)/(\d+)\]")
_EVAL_AVG_RE = re.compile(r"\[Eval\]\[Epoch\s+(\d+)\]\[Avg\]\s+(.+)$")
_EVAL_BEST_RE = re.compile(r"\[Eval\]\[Epoch\s+(\d+)\]\[best metric:\s+([0-9.eE+-]+)\]")
_KV_RE = re.compile(r"([A-Za-z0-9_./]+):\s*([^,]+)")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _parse_kv_fields(text: str) -> Dict[str, float]:
    fields: Dict[str, float] = {}
    for key, raw in _KV_RE.findall(text):
        value = raw.strip().rstrip("s")
        try:
            fields[key] = float(value)
        except ValueError:
            continue
    return fields


def parse_train_log(log_path: str) -> Dict[str, List[Tuple[int, float]]]:
    """Parse a ppsci train.log into {metric_name: [(step, value), ...]}."""
    series: Dict[str, List[Tuple[int, float]]] = {}
    with open(log_path, "r", encoding="utf-8", errors="replace") as fp:
        for raw in fp:
            line = _strip_ansi(raw).strip()
            train_m = _TRAIN_HEAD_RE.search(line)
            if train_m:
                epoch = int(train_m.group(1))
                tail = line[train_m.end() :]
                for key, value in _parse_kv_fields(tail).items():
                    series.setdefault(f"train/{key}", []).append((epoch, value))
                continue

            eval_m = _EVAL_AVG_RE.search(line)
            if eval_m:
                epoch = int(eval_m.group(1))
                for key, value in _parse_kv_fields(eval_m.group(2)).items():
                    series.setdefault(f"eval/{key}", []).append((epoch, value))
                continue

            best_m = _EVAL_BEST_RE.search(line)
            if best_m:
                epoch = int(best_m.group(1))
                series.setdefault("eval/best_metric", []).append(
                    (epoch, float(best_m.group(2)))
                )
    return series


def _dedupe_last(points: Sequence[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Keep the last value when the same step is logged more than once."""
    by_step: Dict[int, float] = {}
    for step, value in points:
        by_step[step] = value
    return sorted(by_step.items(), key=lambda item: item[0])


def write_tensorboard(
    series: Dict[str, List[Tuple[int, float]]],
    tbd_dir: str,
    run_name: str,
) -> str:
    """Write scalar series and summary figures to a TensorBoardX logdir."""
    logdir = osp.join(tbd_dir, run_name)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir=logdir)
    cleaned = {name: _dedupe_last(points) for name, points in series.items() if points}

    for name, points in cleaned.items():
        for step, value in points:
            writer.add_scalar(name, value, global_step=step)

    figures = _build_curve_figures(cleaned)
    for tag, fig in figures.items():
        writer.add_figure(tag, fig, global_step=0)
        plt.close(fig)

    writer.flush()
    writer.close()
    return logdir


def _build_curve_figures(
    series: Dict[str, List[Tuple[int, float]]],
) -> Dict[str, plt.Figure]:
    figures: Dict[str, plt.Figure] = {}

    def _xy(name: str):
        points = series.get(name) or []
        if not points:
            return None, None
        xs, ys = zip(*points)
        return list(xs), list(ys)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x, y = _xy("train/loss")
    if x:
        ax.semilogy(x, y, label="train/loss")
    x, y = _xy("eval/G_eval/loss")
    if x:
        ax.semilogy(x, y, marker="o", label="eval/G_eval/loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Train / Eval Loss")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    figures["curves/loss"] = fig

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x, y = _xy("eval/G_eval/L2Rel.G")
    if x:
        ax.plot(x, y, marker="o", label="eval L2Rel")
    x, y = _xy("eval/best_metric")
    if x:
        ax.plot(x, y, marker="s", label="best metric")
    ax.set_xlabel("epoch")
    ax.set_ylabel("L2Rel")
    ax.set_title("Evaluation L2 Relative Error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    figures["curves/l2rel"] = fig

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    x, y = _xy("train/lr")
    if x:
        axes[0].plot(x, y, label="lr")
        axes[0].set_ylabel("lr")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    x, y = _xy("train/ips")
    if x:
        axes[1].plot(x, y, color="tab:orange", label="ips")
        axes[1].set_ylabel("ips")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    axes[1].set_xlabel("epoch")
    fig.suptitle("Learning Rate / Throughput")
    figures["curves/lr_ips"] = fig

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x, y = _xy("train/batch_cost")
    if x:
        ax.plot(x, y, label="batch_cost (s)")
    x, y = _xy("train/reader_cost")
    if x:
        ax.plot(x, y, label="reader_cost (s)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("seconds")
    ax.set_title("Time Cost")
    ax.grid(True, alpha=0.3)
    ax.legend()
    figures["curves/time"] = fig

    return figures


def save_curve_pngs(
    series: Dict[str, List[Tuple[int, float]]],
    png_dir: str,
) -> List[str]:
    os.makedirs(png_dir, exist_ok=True)
    paths: List[str] = []
    cleaned = {name: _dedupe_last(points) for name, points in series.items() if points}
    for tag, fig in _build_curve_figures(cleaned).items():
        filename = tag.replace("/", "_") + ".png"
        path = osp.join(png_dir, filename)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def find_train_logs(log_path: str) -> List[str]:
    if osp.isfile(log_path):
        return [log_path]
    if not osp.isdir(log_path):
        return []
    found: List[str] = []
    for root, _dirs, files in os.walk(log_path):
        if "train.log" in files:
            found.append(osp.join(root, "train.log"))
    return sorted(found)


def _run_name_from_log(log_file: str, log_root: str) -> str:
    run_dir = osp.dirname(osp.abspath(log_file))
    try:
        rel = osp.relpath(run_dir, osp.abspath(log_root))
    except ValueError:
        rel = osp.basename(run_dir)
    name = rel.replace(os.sep, "_")
    return name if name not in (".", "") else osp.basename(run_dir)


def export_history(
    log_path: str,
    tbd_dir: str,
) -> List[str]:
    """Convert one or more historical train.log files into TensorBoardX runs."""
    logs = find_train_logs(log_path)
    if not logs:
        raise FileNotFoundError(f"No train.log found under: {log_path}")

    log_root = log_path if osp.isdir(log_path) else osp.dirname(osp.abspath(log_path))
    written: List[str] = []
    for log_file in logs:
        series = parse_train_log(log_file)
        if not series:
            continue
        run_name = _run_name_from_log(log_file, log_root)
        logdir = write_tensorboard(series, tbd_dir, run_name)
        png_dir = osp.join(osp.dirname(log_file), "visual_history")
        save_curve_pngs(series, png_dir)
        written.append(logdir)
    if not written:
        raise ValueError(f"Parsed no metrics from logs under: {log_path}")
    return written
