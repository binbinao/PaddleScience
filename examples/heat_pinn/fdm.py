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

from __future__ import annotations

import argparse
import itertools
import os
from os import path as osp
from typing import Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEMP_SCALE = 75.0
BOUNDARY_TEMPERATURE = {
    "x=-l": 75.0,
    "x=+l": 0.0,
    "y=-l": 50.0,
    "y=+l": 0.0,
}


def solve(n: int, l: float) -> np.ndarray:
    """
    Solves the heat equation using the finite difference method.
    Reference: https://github.com/314arhaam/heat-pinn/blob/main/codes/heatman.ipynb

    Args:
        n (int): The number of grid points in each direction.
        l (float): The half length of the square domain, i.e. [-l, l] x [-l, l].

    Returns:
        np.ndarray: A 2D array containing the temperature values at each grid point.
    """
    if n < 2:
        raise ValueError(f"n should be greater than 1, but got {n}")

    b = np.zeros([n, n], dtype="float64")
    matrix = np.zeros([n**2, n**2], dtype="float64")
    for k, (i, j) in enumerate(itertools.product(range(n), range(n))):
        stencil = np.zeros([n, n], dtype="float64")
        stencil[i, j] = -4.0
        if i != 0:
            stencil[i - 1, j] = 1.0
        else:
            b[i, j] += -BOUNDARY_TEMPERATURE["y=-l"]
        if i != n - 1:
            stencil[i + 1, j] = 1.0
        else:
            b[i, j] += -BOUNDARY_TEMPERATURE["y=+l"]
        if j != 0:
            stencil[i, j - 1] = 1.0
        else:
            b[i, j] += -BOUNDARY_TEMPERATURE["x=-l"]
        if j != n - 1:
            stencil[i, j + 1] = 1.0
        else:
            b[i, j] += -BOUNDARY_TEMPERATURE["x=+l"]
        matrix[k, :] = stencil.reshape(1, n**2)

    temperature = np.linalg.solve(matrix, b.reshape(n**2, 1))
    return temperature.reshape([n, n])


def build_dataset(
    n: int, l: float = 1.0, normalize: bool = True
) -> Tuple[dict, dict, np.ndarray]:
    """Build coordinate-label arrays from FDM results for PINNs training/evaluation."""
    coord = np.linspace(-l, l, n, dtype="float32")
    x_grid, y_grid = np.meshgrid(coord, coord, indexing="ij")
    temperature = solve(n, l).T.astype("float32")
    label = temperature / TEMP_SCALE if normalize else temperature
    input_data = {
        "x": x_grid.reshape([-1, 1]).astype("float32"),
        "y": y_grid.reshape([-1, 1]).astype("float32"),
    }
    label_data = {"u": label.reshape([-1, 1]).astype("float32")}
    return input_data, label_data, temperature


def save_results(
    output_dir: str, input_data: dict, label_data: dict, temperature: np.ndarray
):
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        osp.join(output_dir, "fdm_solution.npz"),
        x=input_data["x"],
        y=input_data["y"],
        u=label_data["u"],
        temperature=temperature,
    )


def plot_temperature(input_data: dict, n: int, temperature: np.ndarray, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    x = input_data["x"].reshape(n, n)
    y = input_data["y"].reshape(n, n)
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(x, y, temperature, cmap="magma")
    plt.colorbar(label="T")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("FDM temperature field")
    plt.axis("square")
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, "fdm_temperature.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Solve 2D steady heat equation by FDM.")
    parser.add_argument("--n", type=int, default=100, help="Grid size in each direction.")
    parser.add_argument("--length", type=float, default=1.0, help="Half length of domain.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_heat_pinn/fdm_results",
        help="Directory for FDM npz and figure outputs.",
    )
    args = parser.parse_args()

    input_data, label_data, temperature = build_dataset(args.n, args.length)
    save_results(args.output_dir, input_data, label_data, temperature)
    plot_temperature(input_data, args.n, temperature, args.output_dir)
    print(f"FDM results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
