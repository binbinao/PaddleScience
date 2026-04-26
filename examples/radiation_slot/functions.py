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
Custom functions for PINN-based radiation slot simulation.

Includes PML (Perfectly Matched Layer) implementation, input/output
transformations, and loss functions adapted from hPINNs architecture.
"""

from typing import Dict, List, Tuple

import numpy as np
import paddle
import paddle.nn.functional as F

# =============================================================================
# Constants
# =============================================================================

# Default domain for radiation slot simulation (normalized coordinates)
# Physical domain is normalized so that wavelength ~ O(1)
BOX = np.array([[-2.0, -2.0], [2.0, 3.0]])  # Default domain (matches hPINNs scale)

# PML layer thickness (normalized units)
DPML = 1.0

# Normalized angular frequency (matches hPINNs: OMEGA = 2π)
# Physical frequency enters only through coordinate normalization
OMEGA = 2 * np.pi

# Maximum conductivity for PML
SIGMA0 = -np.log(1e-20) / (4 * DPML**3 / 3)

# Extended domain including PML
l_BOX = BOX + np.array([[-DPML, -DPML], [DPML, DPML]])

# Augmented Lagrangian parameters
beta = 2.0
mu = 2

# Global variables for training state
lambda_re: np.ndarray = None
lambda_im: np.ndarray = None
loss_weight: List[float] = None
train_mode: str = "soft"  # 'soft', 'penalty', or 'aug_lag'

# Loss logging
loss_log: List[float] = []
loss_obj: float = 0.0
lambda_log: List[Tuple[np.ndarray, np.ndarray]] = []


# =============================================================================
# Input/Output Transformations
# =============================================================================

def transform_in(input_dict: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
    """Apply input transformation (Fourier features) for PINN.
    
    Uses periodic features for the waveguide domain.
    
    Args:
        input_dict: Dictionary with 'x' and 'y' tensors
        
    Returns:
        Transformed input dictionary with Fourier features
    """
    x = input_dict["x"]
    y = input_dict["y"]
    
    # Periodic features for x (waveguide direction)
    p = BOX[1][0] - BOX[0][0] + 2 * DPML
    w = 2 * np.pi / p
    
    input_transformed = {}
    
    # Add Fourier features for x
    for t in range(1, 7):
        input_transformed[f"x_cos_{t}"] = paddle.cos(t * w * x)
        input_transformed[f"x_sin_{t}"] = paddle.sin(t * w * x)
    
    # Keep y as is (transverse direction)
    input_transformed["y"] = y
    input_transformed["y_cos_1"] = paddle.cos(OMEGA * y)
    input_transformed["y_sin_1"] = paddle.sin(OMEGA * y)
    
    return input_transformed


def transform_out_all(input_dict: Dict[str, paddle.Tensor], var: paddle.Tensor) -> paddle.Tensor:
    """Apply output transformation to enforce Dirichlet boundary conditions.
    
    Uses sigmoid-based transformation to ensure zero at boundaries.
    
    Args:
        input_dict: Input dictionary
        var: Raw network output
        
    Returns:
        Transformed output with boundary enforcement
    """
    y = input_dict["y"]
    
    # Get boundary y values
    y_min = BOX[0][1] - DPML
    y_max = BOX[1][1] + DPML
    
    # Sigmoid-based boundary enforcement
    # t = (1 - exp(a - y)) * (1 - exp(y - b)) ensures t = 0 at y = a, b
    t = (1 - paddle.exp(y_min - y)) * (1 - paddle.exp(y - y_max))
    
    return t * var


def transform_out_real_part(input_dict: Dict[str, paddle.Tensor], out_dict: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
    """Transform real part of electric field output.
    
    Args:
        input_dict: Input dictionary
        out_dict: Output dictionary with 'e_re' key
        
    Returns:
        Transformed output dictionary
    """
    e_re = out_dict["e_re"]
    trans_out = transform_out_all(input_dict, e_re)
    return {"e_real": trans_out}


def transform_out_imaginary_part(input_dict: Dict[str, paddle.Tensor], out_dict: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
    """Transform imaginary part of electric field output.
    
    Args:
        input_dict: Input dictionary
        out_dict: Output dictionary with 'e_im' key
        
    Returns:
        Transformed output dictionary
    """
    e_im = out_dict["e_im"]
    trans_out = transform_out_all(input_dict, e_im)
    return {"e_imaginary": trans_out}


def transform_out_epsilon(input_dict: Dict[str, paddle.Tensor], out_dict: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
    """Transform dielectric constant output to be in [1, 12] range.
    
    Args:
        input_dict: Input dictionary
        out_dict: Output dictionary with 'eps' key
        
    Returns:
        Transformed output dictionary
    """
    eps = out_dict["eps"]
    # Map sigmoid output [0,1] to [1, 12]
    eps = F.sigmoid(eps) * 11 + 1
    return {"epsilon": eps}


# =============================================================================
# PML (Perfectly Matched Layer) Implementation
# =============================================================================

def _sigma_1(d: np.ndarray) -> np.ndarray:
    """First-order sigma function for PML."""
    return SIGMA0 * d**2 * np.heaviside(d, 0)


def _sigma_2(d: np.ndarray) -> np.ndarray:
    """Second-order sigma function for PML."""
    return 2 * SIGMA0 * d * np.heaviside(d, 0)


def sigma(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Calculate sigma profile for PML.
    
    sigma(x) = 0 if a < x < b, otherwise grows cubically from zero.
    
    Args:
        x: Position array
        a, b: PML boundaries
        
    Returns:
        Sigma values
    """
    return _sigma_1(a - x) + _sigma_1(x - b)


def dsigma(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Calculate derivative of sigma for PML."""
    return -_sigma_2(a - x) + _sigma_2(x - b)


def perfectly_matched_layers(
    x: paddle.Tensor,
    y: paddle.Tensor,
) -> Tuple[paddle.Tensor, ...]:
    """Apply PML technique for Helmholtz equation.
    
    Based on arXiv:2108.05348. Returns the A and B coefficients
    for the PML-transformed Helmholtz equation.
    
    Args:
        x: x-coordinates tensor
        y: y-coordinates tensor
        
    Returns:
        Tuple of (A1, B1, A2, B2, A3, B3, A4, B4) for PML transformation
    """
    x_np = x.numpy()
    y_np = y.numpy()
    
    # x-direction PML parameters
    sigma_x = sigma(x_np, BOX[0][0], BOX[1][0])
    AB1 = 1 / (1 + 1j / OMEGA * sigma_x)**2
    A1, B1 = AB1.real, AB1.imag
    
    dsigma_x = dsigma(x_np, BOX[0][0], BOX[1][0])
    AB2 = -1j / OMEGA * dsigma_x * AB1 / (1 + 1j / OMEGA * sigma_x)
    A2, B2 = AB2.real, AB2.imag
    
    # y-direction PML parameters
    sigma_y = sigma(y_np, BOX[0][1], BOX[1][1])
    AB3 = 1 / (1 + 1j / OMEGA * sigma_y)**2
    A3, B3 = AB3.real, AB3.imag
    
    dsigma_y = dsigma(y_np, BOX[0][1], BOX[1][1])
    AB4 = -1j / OMEGA * dsigma_y * AB3 / (1 + 1j / OMEGA * sigma_y)
    A4, B4 = AB4.real, AB4.imag
    
    return (
        paddle.to_tensor(A1, dtype=x.dtype),
        paddle.to_tensor(B1, dtype=x.dtype),
        paddle.to_tensor(A2, dtype=x.dtype),
        paddle.to_tensor(B2, dtype=x.dtype),
        paddle.to_tensor(A3, dtype=x.dtype),
        paddle.to_tensor(B3, dtype=x.dtype),
        paddle.to_tensor(A4, dtype=x.dtype),
        paddle.to_tensor(B4, dtype=x.dtype),
    )


# =============================================================================
# Loss Functions
# =============================================================================

def init_lambda(
    output_dict: Dict[str, paddle.Tensor],
    bound: int,
) -> None:
    """Initialize Lagrange multipliers for augmented Lagrangian method.
    
    Args:
        output_dict: Output dictionary containing x, y tensors
        bound: Number of boundary points
    """
    global lambda_re, lambda_im, loss_weight
    
    x = output_dict["x"]
    y = output_dict["y"]
    
    # Initialize lambdas to zero
    lambda_re = np.zeros((len(x[bound:]), 1), dtype=paddle.get_default_dtype())
    lambda_im = np.zeros((len(y[bound:]), 1), dtype=paddle.get_default_dtype())
    
    # Set loss weights
    if train_mode == "aug_lag":
        loss_weight = [0.5 * mu] * 2 + [1.0, 1.0] + [1.0]
    else:
        loss_weight = [0.5 * mu] * 2 + [0.0, 0.0] + [1.0]


def update_lambda(
    output_dict: Dict[str, paddle.Tensor],
    bound: int,
) -> None:
    """Update Lagrange multipliers.
    
    Args:
        output_dict: Output dictionary
        bound: Number of boundary points
    """
    global lambda_re, lambda_im, lambda_log
    
    loss_re, loss_im = compute_real_and_imaginary_loss(output_dict)
    loss_re = loss_re[bound:]
    loss_im = loss_im[bound:]
    
    lambda_re = lambda_re + mu * loss_re.numpy()
    lambda_im = lambda_im + mu * loss_im.numpy()
    
    lambda_log.append((lambda_re.copy().squeeze(), lambda_im.copy().squeeze()))


def update_mu() -> None:
    """Update mu parameter for augmented Lagrangian."""
    global mu, loss_weight
    mu *= beta
    loss_weight[:2] = [0.5 * mu] * 2


def compute_real_and_imaginary_loss(
    output_dict: Dict[str, paddle.Tensor],
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Compute real and imaginary parts of Helmholtz loss with PML.
    
    Args:
        output_dict: Output dictionary with field components and derivatives
        
    Returns:
        Tuple of (loss_re, loss_im) tensors
    """
    x = output_dict["x"]
    y = output_dict["y"]
    
    e_re = output_dict["e_real"]
    e_im = output_dict["e_imaginary"]
    eps = output_dict["epsilon"]
    
    # Get derivatives
    de_re_x = output_dict["de_re_x"]
    de_re_y = output_dict["de_re_y"]
    de_re_xx = output_dict["de_re_xx"]
    de_re_yy = output_dict["de_re_yy"]
    
    de_im_x = output_dict["de_im_x"]
    de_im_y = output_dict["de_im_y"]
    de_im_xx = output_dict["de_im_xx"]
    de_im_yy = output_dict["de_im_yy"]
    
    # Apply PML transformation
    a1, b1, a2, b2, a3, b3, a4, b4 = perfectly_matched_layers(x, y)
    
    # Real part of Helmholtz equation with PML
    loss_re = (
        (a1 * de_re_xx + a2 * de_re_x + a3 * de_re_yy + a4 * de_re_y) / OMEGA
        - (b1 * de_im_xx + b2 * de_im_x + b3 * de_im_yy + b4 * de_im_y) / OMEGA
        + eps * OMEGA * e_re
    )
    
    # Imaginary part of Helmholtz equation with PML
    loss_im = (
        (a1 * de_im_xx + a2 * de_im_x + a3 * de_im_yy + a4 * de_im_y) / OMEGA
        + (b1 * de_re_xx + b2 * de_re_x + b3 * de_re_yy + b4 * de_re_y) / OMEGA
        + eps * OMEGA * e_im
    )
    
    return loss_re, loss_im


def pde_loss_fun(
    output_dict: Dict[str, paddle.Tensor],
    *args,
) -> Dict[str, paddle.Tensor]:
    """Compute PDE loss for Helmholtz equation.
    
    Supports augmented Lagrangian method for constraint enforcement.
    
    Args:
        output_dict: Output dictionary
        
    Returns:
        Dictionary with 'pde' loss
    """
    global loss_log
    
    bound = int(output_dict["bound"].flatten()[0])
    loss_re, loss_im = compute_real_and_imaginary_loss(output_dict)
    
    # Only consider interior points (after boundary)
    loss_re = loss_re[bound:]
    loss_im = loss_im[bound:]
    
    # MSE losses
    loss_eqs1 = paddle.mean(loss_re**2)
    loss_eqs2 = paddle.mean(loss_im**2)
    
    # Augmented Lagrangian terms
    if lambda_im is None:
        init_lambda(output_dict, bound)
    
    loss_lag1 = paddle.mean(loss_re * lambda_re)
    loss_lag2 = paddle.mean(loss_im * lambda_im)
    
    # Combine losses
    losses = (
        loss_weight[0] * loss_eqs1
        + loss_weight[1] * loss_eqs2
        + loss_weight[2] * loss_lag1
        + loss_weight[3] * loss_lag2
    )
    
    # Log losses for monitoring
    loss_log.append(float(loss_eqs1 + loss_eqs2))
    loss_log.append(float(loss_lag1 + loss_lag2))
    
    return {"pde": losses}


def eval_loss_fun(
    output_dict: Dict[str, paddle.Tensor],
    *args,
) -> Dict[str, paddle.Tensor]:
    """Compute evaluation loss.
    
    Args:
        output_dict: Output dictionary
        
    Returns:
        Evaluation loss dictionary
    """
    loss_re, loss_im = compute_real_and_imaginary_loss(output_dict)
    loss = paddle.mean(loss_re**2 + loss_im**2)
    return {"eval": loss}


def eval_metric_fun(
    output_dict: Dict[str, paddle.Tensor],
    *args,
) -> Dict[str, paddle.Tensor]:
    """Compute evaluation metric (MSE).
    
    Args:
        output_dict: Output dictionary
        
    Returns:
        Metric dictionary
    """
    loss_re, loss_im = compute_real_and_imaginary_loss(output_dict)
    metric = paddle.mean((loss_re**2 + loss_im**2))
    return {"eval_metric": metric}


# =============================================================================
# Utility Functions
# =============================================================================

def set_params(
    train_mode_val: str,
    output_dir: str,
    dataset_path: str,
    dataset_path_valid: str,
) -> None:
    """Set global parameters for plotting and training.
    
    Args:
        train_mode_val: Training mode ('soft', 'penalty', 'aug_lag')
        output_dir: Output directory
        dataset_path: Training dataset path
        dataset_path_valid: Validation dataset path
    """
    global train_mode, loss_log, loss_obj, lambda_log
    
    train_mode = train_mode_val
    loss_log = []
    loss_obj = 0.0
    lambda_log = []


def reset_state() -> None:
    """Reset all training state variables."""
    global lambda_re, lambda_im, loss_weight, loss_log, loss_obj, lambda_log
    
    lambda_re = None
    lambda_im = None
    loss_weight = None
    loss_log = []
    loss_obj = 0.0
    lambda_log = []
