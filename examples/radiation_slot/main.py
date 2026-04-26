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
Main entry point for radar radiation slot simulation.

This module provides a complete workflow for simulating electromagnetic
field distributions in a rectangular waveguide with a radiation slot,
supporting both FDFD (Finite Difference Frequency Domain) and PINN
(Physics-informed Neural Network) methods.

Usage:
    # Training mode
    python main.py --mode train
    
    # Evaluation mode
    python main.py --mode eval
    
    # Inference mode
    python main.py --mode infer
    
    # Full simulation with frequency sweep
    python main.py --mode sweep --frequency-start 12 --frequency-stop 18
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import logger
from ppsci.utils import misc

# Import local modules
from .geometry import RadiationSlotGeometry, generate_mesh, generate_random_samples, generate_boundary_samples
from .fdfd_solver import FDFDSolver
from .postprocess import FieldVisualizer, RadiationPatternCalculator, SParameterCalculator, DataExporter
from .comparator import SolutionComparator, ComparisonVisualizer, ComparisonReport
from .functions import (
    transform_in,
    transform_out_real_part,
    transform_out_imaginary_part,
    transform_out_epsilon,
    perfectly_matched_layers,
    pde_loss_fun,
    eval_loss_fun,
    eval_metric_fun,
    set_params,
    reset_state,
)


# =============================================================================
# Configuration
# =============================================================================

# Default geometry parameters
DEFAULT_GEOMETRY = {
    'waveguide_width': 1.02,      # cm - WR-90 standard
    'waveguide_height': 0.51,     # cm
    'slot_length': 2.0,           # cm
    'slot_position': 0.5,         # cm
    'medium_epsilon': 1.0,
    'medium_mu': 1.0,
    'pml_thickness': 1.0,         # cm
    'mesh_size': 0.02,            # cm (lambda/10 at 15GHz)
}

# Default simulation parameters
DEFAULT_SIMULATION = {
    'frequency': 15.0,            # GHz - center of Ku band
    'frequency_start': 12.0,      # GHz
    'frequency_stop': 18.0,       # GHz
    'num_frequency_points': 13,   # 13 points = 0.5GHz spacing
}

# Default PINN training parameters
DEFAULT_TRAINING = {
    'epochs': 20000,
    'iters_per_epoch': 1,
    'learning_rate': 0.001,
    'batch_size': 1024,
    'num_interior_points': 10000,
    'num_boundary_points': 500,
    'train_mode': 'soft',         # 'soft', 'penalty', or 'aug_lag'
}

# Default model architecture
DEFAULT_MODEL = {
    'num_layers': 4,
    'hidden_size': 64,
    'activation': 'tanh',
}


# =============================================================================
# PINN Solver
# =============================================================================

class PINNSolver:
    """PINN solver for radiation slot simulation.
    
    Uses physics-informed neural networks to solve the Helmholtz equation
    with PML boundary conditions.
    """
    
    def __init__(
        self,
        geometry: RadiationSlotGeometry,
        frequency: float = 15.0,
        model_config: Dict[str, Any] = None,
        train_config: Dict[str, Any] = None,
    ):
        """Initialize PINN solver.
        
        Args:
            geometry: Radiation slot geometry
            frequency: Operating frequency in GHz
            model_config: Neural network architecture configuration
            train_config: Training configuration
        """
        self.geometry = geometry
        self.frequency = frequency
        self.model_config = model_config or DEFAULT_MODEL.copy()
        self.train_config = train_config or DEFAULT_TRAINING.copy()
        
        # Angular frequency
        self.omega = 2 * np.pi * frequency * 1e9
        
        # Setup transforms
        self._setup_transforms()
        
        # Setup models
        self._setup_models()
        
        # Setup constraints
        self._setup_constraints()
        
        # Setup validator
        self._setup_validator()
        
        # Training state
        self.is_trained = False
        self.training_history = []
    
    def _setup_transforms(self):
        """Setup input/output transforms for PINN."""
        # Update global functions with current geometry
        from .functions import BOX, OMEGA
        
        # Update BOX based on geometry
        margin = self.geometry.pml_thickness
        x_min = -self.geometry.slot_length / 2 - margin
        x_max = self.geometry.slot_length / 2 + margin
        y_min = -self.geometry.waveguide_width / 2 - margin
        y_max = self.geometry.waveguide_width / 2 + margin
        
        BOX[0] = [x_min, y_min]
        BOX[1] = [x_max, y_max]
        OMEGA = self.omega
    
    def _setup_models(self):
        """Setup neural network models for real/imaginary parts and epsilon."""
        # Real part model
        self.model_re = ppsci.arch.MLP(
            input_keys=["x", "y"],
            output_keys=["e_re"],
            num_layers=self.model_config['num_layers'],
            hidden_size=self.model_config['hidden_size'],
            activation=self.model_config['activation'],
        )
        
        # Imaginary part model
        self.model_im = ppsci.arch.MLP(
            input_keys=["x", "y"],
            output_keys=["e_im"],
            num_layers=self.model_config['num_layers'],
            hidden_size=self.model_config['hidden_size'],
            activation=self.model_config['activation'],
        )
        
        # Epsilon model
        self.model_eps = ppsci.arch.MLP(
            input_keys=["x", "y"],
            output_keys=["eps"],
            num_layers=self.model_config['num_layers'],
            hidden_size=self.model_config['hidden_size'],
            activation=self.model_config['activation'],
        )
        
        # Register transforms
        self.model_re.register_input_transform(transform_in)
        self.model_im.register_input_transform(transform_in)
        self.model_eps.register_input_transform(transform_in)
        
        self.model_re.register_output_transform(transform_out_real_part)
        self.model_im.register_output_transform(transform_out_imaginary_part)
        self.model_eps.register_output_transform(transform_out_epsilon)
        
        # Model list
        self.model_list = ppsci.arch.ModelList((self.model_re, self.model_im, self.model_eps))
    
    def _setup_constraints(self):
        """Setup PINN constraints."""
        from .functions import train_mode
        
        # Set training mode
        train_mode = self.train_config.get('train_mode', 'soft')
        set_params(train_mode, "./outputs", None, None)
        
        # Label keys
        label_keys = ("x", "y", "bound", "e_real", "e_imaginary", "epsilon")
        label_keys_derivative = (
            "de_re_x", "de_re_y", "de_re_xx", "de_re_yy",
            "de_im_x", "de_im_y", "de_im_xx", "de_im_yy",
        )
        
        # Output expressions
        self.output_expr = {
            "x": lambda out: out["x"],
            "y": lambda out: out["y"],
            "bound": lambda out: out["bound"],
            "e_real": lambda out: out["e_real"],
            "e_imaginary": lambda out: out["e_imaginary"],
            "epsilon": lambda out: out["epsilon"],
            "de_re_x": lambda out: jacobian(out["e_real"], out["x"]),
            "de_re_y": lambda out: jacobian(out["e_real"], out["y"]),
            "de_re_xx": lambda out: hessian(out["e_real"], out["x"]),
            "de_re_yy": lambda out: hessian(out["e_real"], out["y"]),
            "de_im_x": lambda out: jacobian(out["e_imaginary"], out["x"]),
            "de_im_y": lambda out: jacobian(out["e_imaginary"], out["y"]),
            "de_im_xx": lambda out: hessian(out["e_imaginary"], out["x"]),
            "de_im_yy": lambda out: hessian(out["e_imaginary"], out["y"]),
        }
        
        # PDE constraint
        self.pde_constraint = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "IterableNamedArrayDataset",
                    "input_keys": ("x", "y", "bound"),
                    "label_keys": label_keys + label_keys_derivative,
                    "alias_dict": {
                        "e_real": "x",
                        "e_imaginary": "x",
                        "epsilon": "x",
                        **{k: "x" for k in label_keys_derivative},
                    },
                },
            },
            ppsci.loss.FunctionalLoss(pde_loss_fun),
            self.output_expr,
            name="pde_constraint",
        )
    
    def _setup_validator(self):
        """Setup validation callback."""
        self.validator = None
    
    def train(
        self,
        output_dir: str,
        pretrained_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train the PINN model.
        
        Args:
            output_dir: Output directory for checkpoints
            pretrained_path: Path to pretrained model (optional)
            
        Returns:
            Training history dictionary
        """
        from .functions import reset_state
        
        # Reset state
        reset_state()
        
        # Setup optimizer
        optimizer = ppsci.optimizer.Adam(self.train_config['learning_rate'])
        
        # Setup solver
        solver = ppsci.solver.Solver(
            self.model_list,
            {self.pde_constraint.name: self.pde_constraint},
            output_dir,
            optimizer,
            epochs=self.train_config['epochs'],
            iters_per_epoch=self.train_config['iters_per_epoch'],
            save_freq=1000,
            seed=42,
        )
        
        # Train
        solver.train()
        
        self.is_trained = True
        self.training_history = solver.history
        
        return solver.history
    
    def predict(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Predict electric field at given coordinates.
        
        Args:
            x: x-coordinates array
            y: y-coordinates array
            
        Returns:
            Dictionary with field components
        """
        if not self.is_trained:
            logger.warning("Model not trained. Loading pretrained model if available.")
        
        input_data = {
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
        }
        
        output = self.model_list.predict(input_data, return_numpy=True)
        
        return {
            'field_real': output['e_real'],
            'field_imaginary': output['e_imaginary'],
            'epsilon': output['epsilon'],
            'field': output['e_real'] + 1j * output['e_imaginary'],
        }
    
    def solve(
        self,
        mesh: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Solve for electric field on mesh.
        
        Args:
            mesh: Mesh dictionary with 'xx' and 'yy'
            
        Returns:
            Solution dictionary with field data
        """
        xx = mesh['xx']
        yy = mesh['yy']
        
        # Flatten and predict
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        
        result = self.predict(x_flat[:, np.newaxis], y_flat[:, np.newaxis])
        
        # Reshape results
        field_shape = xx.shape
        
        return {
            'field': result['field'].reshape(field_shape),
            'field_real': result['field_real'].reshape(field_shape),
            'field_imag': result['field_imaginary'].reshape(field_shape),
            'field_magnitude': np.abs(result['field']).reshape(field_shape),
            'field_phase': np.angle(result['field']).reshape(field_shape),
            'epsilon': result['epsilon'].reshape(field_shape),
            'mesh': mesh,
            'frequency': self.frequency,
            'omega': self.omega,
        }


# =============================================================================
# Main Simulation Functions
# =============================================================================

def run_fdfd_simulation(
    geometry_config: Dict[str, float] = None,
    frequency: float = 15.0,
    frequency_sweep: bool = False,
    frequency_start: float = 12.0,
    frequency_stop: float = 18.0,
    output_dir: str = './outputs_fdfd',
) -> Dict[str, Any]:
    """Run FDFD simulation for radiation slot.
    
    Args:
        geometry_config: Geometry parameters
        frequency: Single frequency for simulation
        frequency_sweep: Whether to perform frequency sweep
        frequency_start: Start frequency for sweep
        frequency_stop: Stop frequency for sweep
        output_dir: Output directory
        
    Returns:
        Simulation results dictionary
    """
    geometry_config = geometry_config or DEFAULT_GEOMETRY.copy()
    geometry = RadiationSlotGeometry(**geometry_config)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if frequency_sweep:
        solver = FDFDSolver(geometry, frequency=frequency_start)
        results = solver.solve_frequency_sweep(frequency_start, frequency_stop)
    else:
        solver = FDFDSolver(geometry, frequency=frequency)
        results = solver.solve()
    
    # Add metadata
    results['geometry'] = geometry_config
    results['solver_type'] = 'fdfd'
    results['output_dir'] = output_dir
    
    logger.message(f"FDFD simulation completed. Output saved to {output_dir}")
    
    return results


def run_pinn_simulation(
    geometry_config: Dict[str, float] = None,
    frequency: float = 15.0,
    model_config: Dict[str, Any] = None,
    train_config: Dict[str, Any] = None,
    output_dir: str = './outputs_pinn',
    pretrained_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run PINN simulation for radiation slot.
    
    Args:
        geometry_config: Geometry parameters
        frequency: Operating frequency
        model_config: Neural network configuration
        train_config: Training configuration
        output_dir: Output directory
        pretrained_path: Path to pretrained model
        
    Returns:
        Simulation results dictionary
    """
    geometry_config = geometry_config or DEFAULT_GEOMETRY.copy()
    model_config = model_config or DEFAULT_MODEL.copy()
    train_config = train_config or DEFAULT_TRAINING.copy()
    
    geometry = RadiationSlotGeometry(**geometry_config)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize PINN solver
    pinn_solver = PINNSolver(
        geometry=geometry,
        frequency=frequency,
        model_config=model_config,
        train_config=train_config,
    )
    
    # Train model
    history = pinn_solver.train(output_dir, pretrained_path)
    
    # Generate mesh and solve
    mesh = generate_mesh(geometry, frequency)
    results = pinn_solver.solve(mesh)
    results['history'] = history
    results['geometry'] = geometry_config
    results['solver_type'] = 'pinn'
    results['output_dir'] = output_dir
    
    logger.message(f"PINN simulation completed. Output saved to {output_dir}")
    
    return results


def run_comparison(
    fdfd_results: Dict[str, Any],
    pinn_results: Dict[str, Any],
    output_dir: str = './outputs/comparison',
) -> Dict[str, Any]:
    """Run comparison between FDFD and PINN results.
    
    Args:
        fdfd_results: FDFD simulation results
        pinn_results: PINN simulation results
        output_dir: Output directory
        
    Returns:
        Comparison report dictionary
    """
    from .comparator import ComparisonReport
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate report
    reporter = ComparisonReport(output_dir)
    report = reporter.generate_report(
        fdfd_results,
        pinn_results,
        fdfd_results.get('mesh', {}),
        fdfd_results.get('frequency', 15.0),
        fdfd_results.get('geometry', {}),
    )
    
    return report


def run_full_simulation(
    geometry_config: Dict[str, float] = None,
    frequency: float = 15.0,
    frequency_sweep: bool = False,
    frequency_start: float = 12.0,
    frequency_stop: float = 18.0,
    model_config: Dict[str, Any] = None,
    train_config: Dict[str, Any] = None,
    base_output_dir: str = './outputs',
) -> Dict[str, Any]:
    """Run complete simulation workflow with both FDFD and PINN.
    
    Args:
        geometry_config: Geometry parameters
        frequency: Operating frequency
        frequency_sweep: Whether to sweep frequencies
        frequency_start: Start frequency
        frequency_stop: Stop frequency
        model_config: PINN model configuration
        train_config: PINN training configuration
        base_output_dir: Base output directory
        
    Returns:
        Complete simulation results
    """
    geometry_config = geometry_config or DEFAULT_GEOMETRY.copy()
    model_config = model_config or DEFAULT_MODEL.copy()
    train_config = train_config or DEFAULT_TRAINING.copy()
    
    # Run FDFD simulation
    fdfd_results = run_fdfd_simulation(
        geometry_config=geometry_config,
        frequency=frequency,
        frequency_sweep=frequency_sweep,
        frequency_start=frequency_start,
        frequency_stop=frequency_stop,
        output_dir=os.path.join(base_output_dir, 'fdfd'),
    )
    
    # Run PINN simulation
    pinn_results = run_pinn_simulation(
        geometry_config=geometry_config,
        frequency=frequency,
        model_config=model_config,
        train_config=train_config,
        output_dir=os.path.join(base_output_dir, 'pinn'),
    )
    
    # Run comparison
    comparison = run_comparison(
        fdfd_results,
        pinn_results,
        output_dir=os.path.join(base_output_dir, 'comparison'),
    )
    
    # Post-process results
    postprocess_results = run_postprocessing(
        fdfd_results,
        pinn_results,
        output_dir=os.path.join(base_output_dir, 'postprocess'),
    )
    
    return {
        'fdfd': fdfd_results,
        'pinn': pinn_results,
        'comparison': comparison,
        'postprocess': postprocess_results,
    }


def run_postprocessing(
    fdfd_results: Dict[str, Any],
    pinn_results: Dict[str, Any],
    output_dir: str = './outputs/postprocess',
) -> Dict[str, Any]:
    """Run post-processing on simulation results.
    
    Args:
        fdfd_results: FDFD results
        pinn_results: PINN results
        output_dir: Output directory
        
    Returns:
        Post-processing results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get mesh from results
    mesh = fdfd_results.get('mesh', pinn_results.get('mesh', {}))
    geometry = fdfd_results.get('geometry', {})
    
    # Field visualization
    field_vis = FieldVisualizer(output_dir=os.path.join(output_dir, 'fields'))
    field_vis.visualize_field(
        fdfd_results,
        mesh,
        title="FDFD Electric Field",
        save_name="fdfd_field",
    )
    field_vis.visualize_field(
        pinn_results,
        mesh,
        title="PINN Electric Field",
        save_name="pinn_field",
    )
    field_vis.visualize_comparison(
        fdfd_results,
        pinn_results,
        mesh,
    )
    
    # Radiation pattern
    radiation_vis = RadiationPatternCalculator(geometry, frequency=15.0)
    
    # Use FDFD field for pattern calculation (more accurate)
    field = fdfd_results.get('field', np.zeros_like(mesh['xx']))
    angles, pattern = radiation_vis.calculate_far_field(field, mesh)
    radiation_vis.visualize_pattern(angles, pattern)
    
    # S-parameters
    s_param_calc = SParameterCalculator(geometry, frequency=15.0)
    s11 = s_param_calc.calculate_s11_from_field(fdfd_results, mesh)
    s21 = s_param_calc.calculate_s21_from_field(fdfd_results, mesh)
    
    logger.message(f"S11 (reflection): {s11}")
    logger.message(f"S21 (transmission): {s21}")
    
    # Data export
    exporter = DataExporter(output_dir=os.path.join(output_dir, 'data'))
    exporter.export_field_data(fdfd_results, mesh, 'fdfd_field')
    exporter.export_field_data(pinn_results, mesh, 'pinn_field')
    
    return {
        'radiation_pattern': {'angles': angles, 'pattern': pattern},
        's_parameters': {'s11': s11, 's21': s21},
    }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main function for CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Radar Radiation Slot Simulation'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['fdfd', 'pinn', 'compare', 'full', 'postprocess'],
        help='Simulation mode'
    )
    parser.add_argument(
        '--frequency',
        type=float,
        default=15.0,
        help='Operating frequency in GHz'
    )
    parser.add_argument(
        '--frequency-start',
        type=float,
        default=12.0,
        help='Start frequency for sweep'
    )
    parser.add_argument(
        '--frequency-stop',
        type=float,
        default=18.0,
        help='Stop frequency for sweep'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--waveguide-width',
        type=float,
        default=1.02,
        help='Waveguide width in cm'
    )
    parser.add_argument(
        '--slot-length',
        type=float,
        default=2.0,
        help='Slot length in cm'
    )
    
    args = parser.parse_args()
    
    # Setup geometry
    geometry_config = {
        'waveguide_width': args.waveguide_width,
        'slot_length': args.slot_length,
    }
    
    # Run simulation based on mode
    if args.mode == 'fdfd':
        results = run_fdfd_simulation(
            geometry_config=geometry_config,
            frequency=args.frequency,
            frequency_sweep=(args.frequency_stop > args.frequency_start),
            frequency_start=args.frequency_start,
            frequency_stop=args.frequency_stop,
            output_dir=os.path.join(args.output_dir, 'fdfd'),
        )
    
    elif args.mode == 'pinn':
        results = run_pinn_simulation(
            geometry_config=geometry_config,
            frequency=args.frequency,
            output_dir=os.path.join(args.output_dir, 'pinn'),
        )
    
    elif args.mode == 'compare':
        # Load existing results for comparison
        logger.message("Loading FDFD results...")
        fdfd_results = np.load(os.path.join(args.output_dir, 'fdfd', 'results.npz'))
        logger.message("Loading PINN results...")
        pinn_results = np.load(os.path.join(args.output_dir, 'pinn', 'results.npz'))
        
        comparison = run_comparison(
            fdfd_results,
            pinn_results,
            output_dir=os.path.join(args.output_dir, 'comparison'),
        )
        results = {'comparison': comparison}
    
    elif args.mode == 'postprocess':
        logger.message("Loading FDFD results...")
        fdfd_results = np.load(os.path.join(args.output_dir, 'fdfd', 'results.npz'))
        logger.message("Loading PINN results...")
        pinn_results = np.load(os.path.join(args.output_dir, 'pinn', 'results.npz'))
        
        postprocess = run_postprocessing(
            fdfd_results,
            pinn_results,
            output_dir=os.path.join(args.output_dir, 'postprocess'),
        )
        results = {'postprocess': postprocess}
    
    else:  # full
        results = run_full_simulation(
            geometry_config=geometry_config,
            frequency=args.frequency,
            frequency_sweep=(args.frequency_stop > args.frequency_start),
            frequency_start=args.frequency_start,
            frequency_stop=args.frequency_stop,
            base_output_dir=args.output_dir,
        )
    
    # Print summary
    logger.message("\n" + "=" * 60)
    logger.message("SIMULATION COMPLETE")
    logger.message("=" * 60)
    
    if 'comparison' in results:
        metrics = results['comparison']['comparison_metrics']
        logger.message(f"L2 Error: {metrics['l2_relative_error']:.4e}")
        logger.message(f"Correlation: {metrics['pearson_correlation']:.4f}")
    
    logger.message(f"Results saved to: {args.output_dir}")
    logger.message("=" * 60 + "\n")
    
    return results


if __name__ == '__main__':
    main()
