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
PINN (Physics-Informed Neural Network) Solver for radar radiation slot.

Implements hPINNs architecture for solving the 2D Helmholtz equation
with PML boundaries and TE10 excitation, adapted from the hPINNs example.
"""

import os
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

import paddle
import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from omegaconf import DictConfig

from .geometry import RadiationSlotGeometry, generate_random_samples, generate_boundary_samples
from .functions import (
    transform_in,
    transform_out_real_part,
    transform_out_imaginary_part,
    transform_out_epsilon,
    pde_loss_fun,
    eval_loss_fun,
    eval_metric_fun,
    set_params,
    reset_state,
)


@dataclass
class PINNSolverConfig:
    """Configuration for PINN solver.
    
    Attributes:
        num_interior: Number of interior points for training
        num_boundary: Number of boundary points per edge
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        train_mode: 'soft', 'penalty', or 'aug_lag'
        hidden_size: Number of hidden units per layer
        num_layers: Number of hidden layers
    """
    num_interior: int = 10000
    num_boundary: int = 200
    num_epochs: int = 1000
    learning_rate: float = 0.001
    train_mode: str = "soft"
    hidden_size: int = 64
    num_layers: int = 4
    output_dir: str = "outputs/pinn"


class PINNSolver:
    """Physics-Informed Neural Network solver for radiation slot simulation.
    
    Uses three parallel MLP networks to predict:
    - Real part of electric field (e_real)
    - Imaginary part of electric field (e_imaginary)
    - Relative permittivity (epsilon)
    
    The Helmholtz equation is enforced through physics-based loss functions.
    """
    
    def __init__(
        self,
        geometry: RadiationSlotGeometry,
        frequency: float = 15.0,
        config: PINNSolverConfig = None,
    ):
        """Initialize PINN solver.
        
        Args:
            geometry: Radiation slot geometry parameters
            frequency: Operating frequency in GHz
            config: Solver configuration
        """
        self.geometry = geometry
        self.frequency = frequency
        self.config = config or PINNSolverConfig()
        
        # Physical constants
        self.c = 3e10  # speed of light (cm/s)
        self.omega = 2 * np.pi * frequency * 1e9
        
        # Update PML parameters based on geometry
        self._update_pml_params()
        
        # Reset training state
        reset_state()
        
        # Store model and solver references (set during training)
        self.model = None
        self.solver = None
        self.output_dir = None
    
    def _update_pml_params(self) -> None:
        """Update PML parameters based on geometry.
        
        Uses normalized coordinates where OMEGA = 2π (matching hPINNs).
        The physical domain is mapped to a normalized domain of similar
        scale to the original hPINNs example (BOX ~ [-2,-2] to [2,3]).
        """
        import examples.radiation_slot.functions as func_mod
        
        # Keep the default hPINNs-scale BOX and DPML (already set in functions.py)
        # BOX = [[-2,-2],[2,3]], DPML = 1, OMEGA = 2π
        # These are the same values used in the hPINNs holography example
        # and are known to produce well-conditioned loss functions.
        
        # Store normalization factors for converting between physical and
        # normalized coordinates during predict/post-processing
        x_min, y_min = self.geometry.domain_min
        x_max, y_max = self.geometry.domain_max
        
        # Physical domain size
        self.phys_x_range = (x_min, x_max)
        self.phys_y_range = (y_min, y_max)
        
        # Normalized domain from functions.py
        self.norm_x_range = (func_mod.BOX[0][0], func_mod.BOX[1][0])
        self.norm_y_range = (func_mod.BOX[0][1], func_mod.BOX[1][1])
        
        # Scale factors: norm = (phys - phys_min) / (phys_max - phys_min) * (norm_max - norm_min) + norm_min
        self.x_scale = (self.norm_x_range[1] - self.norm_x_range[0]) / (x_max - x_min)
        self.y_scale = (self.norm_y_range[1] - self.norm_y_range[0]) / (y_max - y_min)
        self.x_offset = self.norm_x_range[0] - x_min * self.x_scale
        self.y_offset = self.norm_y_range[0] - y_min * self.y_scale
    
    def _build_model(self) -> ppsci.arch.ModelList:
        """Build the PINN model with three parallel MLPs.
        
        Returns:
            ModelList containing real, imaginary, and epsilon networks
        """
        # Input keys match the output of transform_in (Fourier features)
        fourier_input_keys = (
            "x_cos_1", "x_sin_1", "x_cos_2", "x_sin_2",
            "x_cos_3", "x_sin_3", "x_cos_4", "x_sin_4",
            "x_cos_5", "x_sin_5", "x_cos_6", "x_sin_6",
            "y", "y_cos_1", "y_sin_1",
        )
        
        # Real part network
        model_re = ppsci.arch.MLP(
            input_keys=fourier_input_keys,
            output_keys=("e_re",),
            num_layers=self.config.num_layers,
            hidden_size=self.config.hidden_size,
            activation="tanh",
        )
        
        # Imaginary part network
        model_im = ppsci.arch.MLP(
            input_keys=fourier_input_keys,
            output_keys=("e_im",),
            num_layers=self.config.num_layers,
            hidden_size=self.config.hidden_size,
            activation="tanh",
        )
        
        # Permittivity network
        model_eps = ppsci.arch.MLP(
            input_keys=fourier_input_keys,
            output_keys=("eps",),
            num_layers=self.config.num_layers,
            hidden_size=self.config.hidden_size,
            activation="tanh",
        )
        
        # Register input transformations (Fourier features)
        model_re.register_input_transform(transform_in)
        model_im.register_input_transform(transform_in)
        model_eps.register_input_transform(transform_in)
        
        # Register output transformations (boundary enforcement)
        model_re.register_output_transform(transform_out_real_part)
        model_im.register_output_transform(transform_out_imaginary_part)
        model_eps.register_output_transform(transform_out_epsilon)
        
        return ppsci.arch.ModelList((model_re, model_im, model_eps))
    
    def _build_output_expr(self) -> Dict[str, Any]:
        """Build output expressions including field and derivatives.
        
        Returns:
            Dictionary mapping output names to expressions
        """
        # Field outputs
        output_expr = {
            "x": lambda out: out["x"],
            "y": lambda out: out["y"],
            "e_real": lambda out: out["e_real"],
            "e_imaginary": lambda out: out["e_imaginary"],
            "epsilon": lambda out: out["epsilon"],
        }
        
        # First derivatives
        output_expr["de_re_x"] = lambda out: jacobian(out["e_real"], out["x"])
        output_expr["de_re_y"] = lambda out: jacobian(out["e_real"], out["y"])
        output_expr["de_im_x"] = lambda out: jacobian(out["e_imaginary"], out["x"])
        output_expr["de_im_y"] = lambda out: jacobian(out["e_imaginary"], out["y"])
        
        # Second derivatives
        output_expr["de_re_xx"] = lambda out: hessian(out["e_real"], out["x"])
        output_expr["de_re_yy"] = lambda out: hessian(out["e_real"], out["y"])
        output_expr["de_im_xx"] = lambda out: hessian(out["e_imaginary"], out["x"])
        output_expr["de_im_yy"] = lambda out: hessian(out["e_imaginary"], out["y"])
        
        return output_expr
    
    def _build_interior_dataset(self) -> Dict:
        """Build interior point dataset in normalized coordinates.
        
        Samples uniformly in the extended (BOX + PML) normalized domain.
        """
        import examples.radiation_slot.functions as func_mod
        
        x_min = func_mod.l_BOX[0][0]
        x_max = func_mod.l_BOX[1][0]
        y_min = func_mod.l_BOX[0][1]
        y_max = func_mod.l_BOX[1][1]
        
        n = self.config.num_interior
        x = np.random.uniform(x_min, x_max, (n, 1)).astype(np.float32)
        y = np.random.uniform(y_min, y_max, (n, 1)).astype(np.float32)
        
        return {"x": x, "y": y}
    
    def _build_boundary_dataset(self) -> Dict:
        """Build boundary point dataset in normalized coordinates.
        
        PEC boundary: E = 0 on the edges of the extended domain.
        """
        import examples.radiation_slot.functions as func_mod
        
        x_min = func_mod.l_BOX[0][0]
        x_max = func_mod.l_BOX[1][0]
        y_min = func_mod.l_BOX[0][1]
        y_max = func_mod.l_BOX[1][1]
        
        nb = self.config.num_boundary
        
        # Left / right boundaries
        left_x = np.full((nb, 1), x_min, dtype=np.float32)
        left_y = np.linspace(y_min, y_max, nb).reshape(-1, 1).astype(np.float32)
        right_x = np.full((nb, 1), x_max, dtype=np.float32)
        right_y = left_y.copy()
        
        # Bottom / top boundaries
        bot_x = np.linspace(x_min, x_max, nb).reshape(-1, 1).astype(np.float32)
        bot_y = np.full((nb, 1), y_min, dtype=np.float32)
        top_x = bot_x.copy()
        top_y = np.full((nb, 1), y_max, dtype=np.float32)
        
        return {
            "x": np.concatenate([left_x, right_x, bot_x, top_x], axis=0),
            "y": np.concatenate([left_y, right_y, bot_y, top_y], axis=0),
        }
    
    def train(self, output_dir: str = None, fdfd_result: Dict = None) -> Dict:
        """Train the PINN model.
        
        If fdfd_result is provided, uses FDFD solution as supervision labels
        (data-driven mode). Otherwise falls back to pure physics mode.
        
        Args:
            output_dir: Output directory for checkpoints and logs
            fdfd_result: Optional FDFD result dict with keys
                'field_real', 'field_imag', 'mesh' (with 'xx', 'yy')
        """
        import time
        start_time = time.time()
        
        paddle.framework.core.set_prim_eager_enabled(True)
        
        self.output_dir = output_dir or self.config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        set_params(self.config.train_mode, self.output_dir, None, None)
        
        model = self._build_model()
        self.model = model
        
        constraint = {}
        
        if fdfd_result is not None:
            # ── Data-driven mode: FDFD solution as labels ──
            mesh = fdfd_result['mesh']
            xx = mesh['xx'].flatten().astype(np.float32)
            yy = mesh['yy'].flatten().astype(np.float32)
            e_re = fdfd_result['field_real'].flatten().astype(np.float32)
            e_im = fdfd_result['field_imag'].flatten().astype(np.float32)
            
            # Normalize field values so network outputs are O(1)
            self.field_scale = float(max(np.max(np.abs(e_re)), np.max(np.abs(e_im)), 1.0))
            e_re_norm = e_re / self.field_scale
            e_im_norm = e_im / self.field_scale
            
            # Map physical coords → normalized coords for the network
            x_norm = (xx * self.x_scale + self.x_offset).astype(np.float32)
            y_norm = (yy * self.y_scale + self.y_offset).astype(np.float32)
            
            # Subsample for training efficiency
            n_total = len(xx)
            n_use = min(n_total, self.config.num_interior)
            idx = np.random.choice(n_total, n_use, replace=False)
            
            data_input = {
                "x": x_norm[idx, np.newaxis],
                "y": y_norm[idx, np.newaxis],
            }
            data_label = {
                "e_real": e_re_norm[idx, np.newaxis],
                "e_imaginary": e_im_norm[idx, np.newaxis],
            }
            
            data_expr = {
                "e_real": lambda out: out["e_real"],
                "e_imaginary": lambda out: out["e_imaginary"],
            }
            
            data_constraint = ppsci.constraint.SupervisedConstraint(
                {"dataset": {"name": "IterableNamedArrayDataset",
                             "input": data_input, "label": data_label}},
                ppsci.loss.MSELoss(),
                data_expr,
                name="fdfd_data",
            )
            constraint[data_constraint.name] = data_constraint
        else:
            # ── Pure physics mode: PDE residual only ──
            output_expr = self._build_output_expr()
            output_expr["bound"] = lambda out: out["bound"]
            
            interior_data = self._build_interior_dataset()
            boundary_data = self._build_boundary_dataset()
            num_boundary = len(boundary_data["x"])
            
            train_x = np.concatenate([boundary_data["x"], interior_data["x"]])
            train_y = np.concatenate([boundary_data["y"], interior_data["y"]])
            train_bound = np.full_like(train_x, num_boundary, dtype=np.float32)
            
            label_keys = ("x", "y", "bound", "e_real", "e_imaginary", "epsilon",
                          "de_re_x", "de_re_y", "de_re_xx", "de_re_yy",
                          "de_im_x", "de_im_y", "de_im_xx", "de_im_yy")
            input_dict = {"x": train_x, "y": train_y, "bound": train_bound}
            label_dict = {k: train_x for k in label_keys}
            
            pde_constraint = ppsci.constraint.SupervisedConstraint(
                {"dataset": {"name": "IterableNamedArrayDataset",
                             "input": input_dict, "label": label_dict}},
                ppsci.loss.FunctionalLoss(pde_loss_fun),
                output_expr,
                name="helmholtz_pde",
            )
            constraint[pde_constraint.name] = pde_constraint
        
        optimizer = ppsci.optimizer.Adam(self.config.learning_rate)(model)
        
        self.solver = ppsci.solver.Solver(
            model, constraint, self.output_dir, optimizer, None,
            self.config.num_epochs, 1,
            save_freq=max(1, self.config.num_epochs // 5),
        )
        
        self.solver.train()
        
        return {'train_time': time.time() - start_time,
                'epochs': self.config.num_epochs,
                'output_dir': self.output_dir}
    
    def predict(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 1024,
    ) -> Dict[str, np.ndarray]:
        """Predict electric field at given physical coordinates.
        
        Automatically maps physical coordinates to normalized space,
        runs inference, and returns results.
        
        Args:
            x: x-coordinates in physical space (cm)
            y: y-coordinates in physical space (cm)
            batch_size: Batch size for inference
            
        Returns:
            Dictionary with predicted fields
        """
        if self.solver is None:
            raise RuntimeError("Solver not trained. Call train() first.")
        
        original_shape = np.asarray(x).shape
        
        # Convert physical coordinates to normalized coordinates
        x_flat = np.asarray(x).flatten()
        y_flat = np.asarray(y).flatten()
        x_norm = (x_flat * self.x_scale + self.x_offset).astype(np.float32)
        y_norm = (y_flat * self.y_scale + self.y_offset).astype(np.float32)
        
        input_dict = {
            "x": x_norm[:, np.newaxis],
            "y": y_norm[:, np.newaxis],
        }
        
        expr_dict = {
            "e_real": lambda out: out["e_real"],
            "e_imaginary": lambda out: out["e_imaginary"],
            "epsilon": lambda out: out["epsilon"],
        }
        
        output_dict = self.solver.predict(
            input_dict, expr_dict,
            batch_size=batch_size, return_numpy=True,
        )
        
        e_real = output_dict["e_real"].reshape(original_shape)
        e_imaginary = output_dict["e_imaginary"].reshape(original_shape)
        epsilon = output_dict["epsilon"].reshape(original_shape)
        
        # Rescale back to physical units if trained with normalized FDFD data
        scale = getattr(self, 'field_scale', 1.0)
        e_real = e_real * scale
        e_imaginary = e_imaginary * scale
        
        return {
            'e_real': e_real,
            'e_imaginary': e_imaginary,
            'epsilon': epsilon,
            'magnitude': np.sqrt(e_real**2 + e_imaginary**2),
        }
    
    def load(self, model_path: str) -> None:
        """Load a pretrained model.
        
        Args:
            model_path: Path to model checkpoint
        """
        model = self._build_model()
        
        self.solver = ppsci.solver.Solver(
            model,
            pretrained_model_path=model_path,
        )


def create_helmholtz_equation(model: ppsci.arch.ModelList, omega: float) -> Dict:
    """Create Helmholtz equation object for SPINN-style solving.
    
    Args:
        model: Neural network model
        omega: Angular frequency
        
    Returns:
        Dictionary with Helmholtz equation
    """
    from ppsci.equation.pde import Helmholtz
    
    # For complex Helmholtz, we need custom implementation
    # Here we return a simplified version
    return {
        "Helmholtz": Helmholtz(2, omega, model)
    }
