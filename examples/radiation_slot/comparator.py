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
Comparison module for validating PINN solutions against FDFD benchmarks.

Provides multi-metric comparison, error analysis, and convergence studies
for the radar radiation slot simulation.
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import interpolate
from scipy import integrate

from ppsci.utils import logger


# =============================================================================
# Comparison Metrics
# =============================================================================

class ComparisonMetrics:
    """Compute various comparison metrics between two solutions.
    
    Implements L2 error, maximum error, correlation, and other metrics
    commonly used for validating PINN solutions against traditional
    numerical methods.
    """
    
    @staticmethod
    def l2_relative_error(
        reference: np.ndarray,
        prediction: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Compute L2 relative error.
        
        ||u_ref - u_pred||_2 / ||u_ref||_2
        
        Args:
            reference: Reference solution
            prediction: Predicted solution
            weights: Optional weighting array (e.g., for spatial integration)
            
        Returns:
            L2 relative error
        """
        ref_flat = reference.flatten()
        pred_flat = prediction.flatten()
        
        if weights is not None:
            weights_flat = weights.flatten()
            numerator = np.sqrt(np.sum((ref_flat - pred_flat)**2 * weights_flat))
            denominator = np.sqrt(np.sum(ref_flat**2 * weights_flat))
        else:
            numerator = np.linalg.norm(ref_flat - pred_flat)
            denominator = np.linalg.norm(ref_flat)
        
        return numerator / (denominator + 1e-15)
    
    @staticmethod
    def max_absolute_error(
        reference: np.ndarray,
        prediction: np.ndarray,
    ) -> float:
        """Compute maximum absolute error.
        
        max(|u_ref - u_pred|)
        
        Args:
            reference: Reference solution
            prediction: Predicted solution
            
        Returns:
            Maximum absolute error
        """
        return np.max(np.abs(reference - prediction))
    
    @staticmethod
    def mean_absolute_error(
        reference: np.ndarray,
        prediction: np.ndarray,
    ) -> float:
        """Compute mean absolute error.
        
        Args:
            reference: Reference solution
            prediction: Predicted solution
            
        Returns:
            Mean absolute error
        """
        return np.mean(np.abs(reference - prediction))
    
    @staticmethod
    def root_mean_square_error(
        reference: np.ndarray,
        prediction: np.ndarray,
    ) -> float:
        """Compute root mean square error.
        
        Args:
            reference: Reference solution
            prediction: Predicted solution
            
        Returns:
            RMSE
        """
        return np.sqrt(np.mean((reference - prediction)**2))
    
    @staticmethod
    def relative_root_mean_square_error(
        reference: np.ndarray,
        prediction: np.ndarray,
    ) -> float:
        """Compute relative root mean square error.
        
        RMSE / ||u_ref||_2
        
        Args:
            reference: Reference solution
            prediction: Predicted solution
            
        Returns:
            Relative RMSE
        """
        rmse = ComparisonMetrics.root_mean_square_error(reference, prediction)
        ref_norm = np.sqrt(np.mean(reference**2))
        return rmse / (ref_norm + 1e-15)
    
    @staticmethod
    def pearson_correlation(
        reference: np.ndarray,
        prediction: np.ndarray,
    ) -> float:
        """Compute Pearson correlation coefficient.
        
        Args:
            reference: Reference solution
            prediction: Predicted solution
            
        Returns:
            Correlation coefficient in [-1, 1]
        """
        ref_flat = reference.flatten()
        pred_flat = prediction.flatten()
        
        return np.corrcoef(ref_flat, pred_flat)[0, 1]
    
    @staticmethod
    def spearman_correlation(
        reference: np.ndarray,
        prediction: np.ndarray,
    ) -> float:
        """Compute Spearman rank correlation coefficient.
        
        More robust to outliers than Pearson correlation.
        
        Args:
            reference: Reference solution
            prediction: Predicted solution
            
        Returns:
            Spearman correlation coefficient
        """
        from scipy.stats import spearmanr
        ref_flat = reference.flatten()
        pred_flat = prediction.flatten()
        
        corr, _ = spearmanr(ref_flat, pred_flat)
        return corr
    
    @staticmethod
    def compute_all_metrics(
        reference: np.ndarray,
        prediction: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all comparison metrics.
        
        Args:
            reference: Reference solution
            prediction: Predicted solution
            
        Returns:
            Dictionary of all computed metrics
        """
        return {
            'l2_relative_error': ComparisonMetrics.l2_relative_error(reference, prediction),
            'max_absolute_error': ComparisonMetrics.max_absolute_error(reference, prediction),
            'mean_absolute_error': ComparisonMetrics.mean_absolute_error(reference, prediction),
            'root_mean_square_error': ComparisonMetrics.root_mean_square_error(reference, prediction),
            'relative_rmse': ComparisonMetrics.relative_root_mean_square_error(reference, prediction),
            'pearson_correlation': ComparisonMetrics.pearson_correlation(reference, prediction),
            'spearman_correlation': ComparisonMetrics.spearman_correlation(reference, prediction),
        }


# =============================================================================
# Solution Comparator
# =============================================================================

class SolutionComparator:
    """Compare FDFD and PINN solutions for the radiation slot problem.
    
    Provides comprehensive comparison including:
    - Point-wise error analysis
    - Field distribution comparison
    - Frequency sweep comparison
    - Convergence analysis
    """
    
    def __init__(
        self,
        output_dir: str = './outputs/comparison',
        metrics: List[str] = None,
    ):
        """Initialize solution comparator.
        
        Args:
            output_dir: Directory for comparison results
            metrics: List of metrics to compute (default: all)
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.metrics = metrics or [
            'l2_relative_error',
            'max_absolute_error',
            'relative_rmse',
            'pearson_correlation',
        ]
    
    def compare_fields(
        self,
        fdfd_result: Dict[str, np.ndarray],
        pinn_result: Dict[str, np.ndarray],
        mesh: Dict[str, np.ndarray],
        region: str = 'full',
    ) -> Dict[str, float]:
        """Compare electric field solutions from FDFD and PINN.
        
        Args:
            fdfd_result: FDFD solution result
            pinn_result: PINN solution result
            mesh: Mesh dictionary
            region: Comparison region ('full', 'waveguide', 'slot', 'pml')
            
        Returns:
            Dictionary of comparison metrics
        """
        # Extract field data
        fdfd_field = fdfd_result.get('field', np.zeros_like(mesh['xx']))
        pinn_field = pinn_result.get('field', np.zeros_like(mesh['xx']))
        
        # Apply region mask if specified
        mask = np.ones_like(fdfd_field, dtype=bool)
        xx = mesh['xx']
        
        if region == 'waveguide':
            # Only compare within waveguide
            a = WAVEGUIDE_WIDTH_CM / 2
            mask = np.abs(mesh['yy']) <= a
        elif region == 'slot':
            # Only compare in radiation slot region
            slot_pos = 0.5
            slot_len = 2.0
            mask = (np.abs(xx - slot_pos) < slot_len / 2) | (np.abs(xx + slot_pos) < slot_len / 2)
        elif region == 'pml':
            # Only compare in PML region
            mask = ~((np.abs(xx) < 2) & (np.abs(mesh['yy']) < 1))
        
        fdfd_field_region = fdfd_field[mask]
        pinn_field_region = pinn_field[mask]
        
        # Compute metrics
        metrics = ComparisonMetrics.compute_all_metrics(fdfd_field_region, pinn_field_region)
        
        # Log results
        logger.message("=" * 50)
        logger.message("Field Comparison Results:")
        logger.message(f"  L2 Relative Error: {metrics['l2_relative_error']:.4e}")
        logger.message(f"  Max Absolute Error: {metrics['max_absolute_error']:.4e}")
        logger.message(f"  Relative RMSE: {metrics['relative_rmse']:.4e}")
        logger.message(f"  Pearson Correlation: {metrics['pearson_correlation']:.4f}")
        logger.message("=" * 50)
        
        return metrics
    
    def compare_frequency_sweep(
        self,
        fdfd_sweep: Dict[str, Any],
        pinn_sweep: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Compare frequency sweep results.
        
        Compares S-parameters and field magnitudes across frequency range.
        
        Args:
            fdfd_sweep: FDFD frequency sweep results
            pinn_sweep: PINN frequency sweep results
            
        Returns:
            Dictionary with comparison arrays
        """
        fdfd_freqs = fdfd_sweep.get('frequencies', [])
        pinn_freqs = pinn_sweep.get('frequencies', [])
        
        # Interpolate to common frequency grid
        common_freqs = np.union1d(fdfd_freqs, pinn_freqs)
        
        # Interpolate magnitudes
        fdfd_magnitudes = fdfd_sweep.get('field_magnitudes', np.zeros(len(fdfd_freqs)))
        pinn_magnitudes = pinn_sweep.get('field_magnitudes', np.zeros(len(pinn_freqs)))
        
        if len(fdfd_magnitudes) > 0:
            fdfd_interp = interpolate.interp1d(fdfd_freqs, fdfd_magnitudes, fill_value='extrapolate')
            fdfd_common = fdfd_interp(common_freqs)
        else:
            fdfd_common = np.zeros(len(common_freqs))
        
        if len(pinn_magnitudes) > 0:
            pinn_interp = interpolate.interp1d(pinn_freqs, pinn_magnitudes, fill_value='extrapolate')
            pinn_common = pinn_interp(common_freqs)
        else:
            pinn_common = np.zeros(len(common_freqs))
        
        # Compute point-wise errors
        errors = np.abs(fdfd_common - pinn_common) / (np.abs(fdfd_common) + 1e-10)
        
        return {
            'frequencies': common_freqs,
            'fdfd_magnitudes': fdfd_common,
            'pinn_magnitudes': pinn_common,
            'relative_errors': errors,
        }
    
    def convergence_study(
        self,
        pinn_results: List[Dict[str, np.ndarray]],
        fdfd_reference: Dict[str, np.ndarray],
        mesh_info: List[Dict[str, float]],
    ) -> Dict[str, List[float]]:
        """Perform convergence study with varying mesh densities.
        
        Args:
            pinn_results: List of PINN results with different mesh densities
            fdfd_reference: FDFD result as reference solution
            mesh_info: List of mesh information for each result
            
        Returns:
            Dictionary with convergence data
        """
        l2_errors = []
        correlations = []
        mesh_sizes = []
        
        for i, pinn_result in enumerate(pinn_results):
            metrics = self.compare_fields(fdfd_reference, pinn_result, mesh_info[i])
            l2_errors.append(metrics['l2_relative_error'])
            correlations.append(metrics['pearson_correlation'])
            mesh_sizes.append(mesh_info[i].get('mesh_size', 0))
        
        return {
            'mesh_sizes': mesh_sizes,
            'l2_errors': l2_errors,
            'correlations': correlations,
        }


# =============================================================================
# Visualization
# =============================================================================

class ComparisonVisualizer:
    """Visualize comparison results between FDFD and PINN solutions."""
    
    def __init__(self, output_dir: str = './outputs/comparison'):
        """Initialize comparison visualizer.
        
        Args:
            output_dir: Directory for saving figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 13,
        })
    
    def visualize_field_comparison(
        self,
        fdfd_field: np.ndarray,
        pinn_field: np.ndarray,
        mesh: Dict[str, np.ndarray],
        save_name: str = 'field_comparison',
    ) -> plt.Figure:
        """Create 4-panel comparison figure.
        
        Args:
            fdfd_field: FDFD field solution
            pinn_field: PINN field solution
            mesh: Mesh dictionary
            save_name: Filename for saved figure
            
        Returns:
            Matplotlib Figure with subplots
        """
        xx = mesh['xx']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        fields = [
            ('FDFD Solution', np.abs(fdfd_field)),
            ('PINN Solution', np.abs(pinn_field)),
            ('Absolute Error', np.abs(fdfd_field - pinn_field)),
            ('Relative Error', np.abs(fdfd_field - pinn_field) / (np.abs(fdfd_field) + 1e-10)),
        ]
        
        for ax, (title, data) in zip(axes.flatten(), fields):
            # Set colormap
            if 'Error' in title:
                cmap = 'hot'
                vmin = 0
                vmax = np.percentile(data.flatten(), 99)
            else:
                cmap = 'viridis'
                vmin = np.percentile(data.flatten(), 1)
                vmax = np.percentile(data.flatten(), 99)
            
            im = ax.pcolormesh(xx, mesh['yy'], data, cmap=cmap, shading='gouraud',
                              vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, format='%.2e')
            ax.set_xlabel('x (cm)')
            ax.set_ylabel('y (cm)')
            ax.set_title(title)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(
            os.path.join(self.output_dir, f"{save_name}.png"),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        
        logger.message(f"Saved field comparison to {self.output_dir}")
        
        return fig
    
    def visualize_error_distribution(
        self,
        fdfd_field: np.ndarray,
        pinn_field: np.ndarray,
        save_name: str = 'error_distribution',
    ) -> plt.Figure:
        """Visualize error distribution as histogram.
        
        Args:
            fdfd_field: FDFD field solution
            pinn_field: PINN field solution
            save_name: Filename for saved figure
            
        Returns:
            Matplotlib Figure
        """
        errors = (fdfd_field - pinn_field).flatten()
        relative_errors = np.abs(errors) / (np.abs(fdfd_field.flatten()) + 1e-10)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Absolute error histogram
        ax1 = axes[0]
        ax1.hist(errors.real, bins=50, alpha=0.7, label='Real', color='blue')
        ax1.hist(errors.imag, bins=50, alpha=0.7, label='Imaginary', color='green')
        ax1.set_xlabel('Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Absolute Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Relative error histogram
        ax2 = axes[1]
        ax2.hist(relative_errors, bins=50, color='red', alpha=0.7)
        ax2.set_xlabel('Relative Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Relative Error Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        stats = ComparisonMetrics.compute_all_metrics(fdfd_field, pinn_field)
        ax2.axvline(stats['relative_rmse'], color='black', linestyle='--',
                   label=f"RMSE: {stats['relative_rmse']:.2e}")
        ax2.legend()
        
        plt.tight_layout()
        
        plt.savefig(
            os.path.join(self.output_dir, f"{save_name}.png"),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        
        return fig
    
    def visualize_convergence(
        self,
        convergence_data: Dict[str, List[float]],
        save_name: str = 'convergence',
    ) -> plt.Figure:
        """Visualize convergence with mesh refinement.
        
        Args:
            convergence_data: Convergence data dictionary
            save_name: Filename for saved figure
            
        Returns:
            Matplotlib Figure
        """
        mesh_sizes = convergence_data['mesh_sizes']
        l2_errors = convergence_data['l2_errors']
        correlations = convergence_data['correlations']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # L2 error vs mesh size
        ax1 = axes[0]
        ax1.semilogy(mesh_sizes, l2_errors, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Mesh Size (cm)')
        ax1.set_ylabel('L2 Relative Error')
        ax1.set_title('Error Convergence with Mesh Refinement')
        ax1.grid(True, alpha=0.3)
        
        # Correlation vs mesh size
        ax2 = axes[1]
        ax2.plot(mesh_sizes, correlations, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Mesh Size (cm)')
        ax2.set_ylabel('Pearson Correlation')
        ax2.set_title('Correlation with Mesh Refinement')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plt.savefig(
            os.path.join(self.output_dir, f"{save_name}.png"),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        
        return fig
    
    def visualize_metrics_summary(
        self,
        metrics: Dict[str, float],
        save_name: str = 'metrics_summary',
    ) -> plt.Figure:
        """Create bar chart summary of comparison metrics.
        
        Args:
            metrics: Dictionary of computed metrics
            save_name: Filename for saved figure
            
        Returns:
            Matplotlib Figure
        """
        # Select metrics to display
        display_metrics = {
            'L2 Relative Error': metrics.get('l2_relative_error', 0),
            'Max Absolute Error': metrics.get('max_absolute_error', 0),
            'Relative RMSE': metrics.get('relative_rmse', 0),
            'Pearson Correlation': metrics.get('pearson_correlation', 0),
            'Spearman Correlation': metrics.get('spearman_correlation', 0),
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(display_metrics))
        values = list(display_metrics.values())
        labels = list(display_metrics.keys())
        
        # Use logarithmic scale for error metrics
        is_error = ['Error' in label or 'RMSE' in label for label in labels]
        
        colors = ['red' if err else 'blue' for err in is_error]
        bars = ax.bar(x, values, color=colors, alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel('Value')
        ax.set_title('FDFD vs PINN Comparison Metrics')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val < 1:
                label = f'{val:.2e}'
            else:
                label = f'{val:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   label, ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        plt.savefig(
            os.path.join(self.output_dir, f"{save_name}.png"),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        
        return fig


# =============================================================================
# Report Generation
# =============================================================================

class ComparisonReport:
    """Generate comprehensive comparison report."""
    
    def __init__(self, output_dir: str = './outputs/comparison'):
        """Initialize report generator.
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = output_dir
        self.comparator = SolutionComparator(output_dir)
        self.visualizer = ComparisonVisualizer(output_dir)
    
    def generate_report(
        self,
        fdfd_result: Dict[str, np.ndarray],
        pinn_result: Dict[str, np.ndarray],
        mesh: Dict[str, np.ndarray],
        frequency: float,
        geometry: Dict[str, float],
    ) -> Dict[str, Any]:
        """Generate complete comparison report.
        
        Args:
            fdfd_result: FDFD solution result
            pinn_result: PINN solution result
            mesh: Mesh dictionary
            frequency: Operating frequency
            geometry: Geometry parameters
            
        Returns:
            Dictionary containing all report data
        """
        # Compute metrics
        metrics = self.comparator.compare_fields(fdfd_result, pinn_result, mesh)
        
        # Generate visualizations
        self.visualizer.visualize_field_comparison(
            fdfd_result.get('field', 0),
            pinn_result.get('field', 0),
            mesh,
        )
        
        self.visualizer.visualize_error_distribution(
            fdfd_result.get('field', 0),
            pinn_result.get('field', 0),
        )
        
        self.visualizer.visualize_metrics_summary(metrics)
        
        # Create report summary
        report = {
            'simulation_info': {
                'frequency': frequency,
                'geometry': geometry,
                'mesh_size': mesh.get('mesh_size', 0),
                'num_points': mesh.get('num_points', 0),
            },
            'comparison_metrics': metrics,
            'pass_criteria': self._check_pass_criteria(metrics),
            'figures': {
                'field_comparison': 'field_comparison.png',
                'error_distribution': 'error_distribution.png',
                'metrics_summary': 'metrics_summary.png',
            },
        }
        
        # Log report
        logger.message("\n" + "=" * 60)
        logger.message("COMPARISON REPORT SUMMARY")
        logger.message("=" * 60)
        logger.message(f"Frequency: {frequency} GHz")
        logger.message(f"Mesh Size: {mesh.get('mesh_size', 0)} cm")
        logger.message("-" * 40)
        logger.message(f"L2 Relative Error: {metrics['l2_relative_error']:.4e}")
        logger.message(f"Max Error: {metrics['max_absolute_error']:.4e}")
        logger.message(f"Correlation: {metrics['pearson_correlation']:.4f}")
        logger.message("-" * 40)
        logger.message(f"Pass Criteria: {report['pass_criteria']}")
        logger.message("=" * 60 + "\n")
        
        return report
    
    def _check_pass_criteria(self, metrics: Dict[str, float]) -> bool:
        """Check if comparison passes quality criteria.
        
        Args:
            metrics: Comparison metrics
            
        Returns:
            True if all criteria are passed
        """
        # Acceptable error thresholds
        l2_threshold = 0.1  # 10% L2 error
        correlation_threshold = 0.9  # 90% correlation
        
        passed = (
            metrics['l2_relative_error'] < l2_threshold and
            metrics['pearson_correlation'] > correlation_threshold
        )
        
        return passed


# =============================================================================
# Utility Functions
# =============================================================================

def quick_compare(
    fdfd_field: np.ndarray,
    pinn_field: np.ndarray,
    output_dir: str = './outputs/comparison',
) -> Dict[str, float]:
    """Quick comparison function for simple use cases.
    
    Args:
        fdfd_field: FDFD field solution
        pinn_field: PINN field solution
        output_dir: Output directory
        
    Returns:
        Dictionary of comparison metrics
    """
    comparator = SolutionComparator(output_dir)
    metrics = comparator.compare_fields(
        {'field': fdfd_field},
        {'field': pinn_field},
        {'xx': np.array([[0]]), 'yy': np.array([[0]])},  # Dummy mesh
    )
    
    return metrics


def print_comparison_table(metrics: Dict[str, float]) -> None:
    """Print formatted comparison metrics table.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "=" * 60)
    print("FDFD vs PINN Comparison Results")
    print("=" * 60)
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    
    for key, value in metrics.items():
        if abs(value) < 1:
            formatted = f"{value:.4e}"
        else:
            formatted = f"{value:.4f}"
        print(f"{key:<25} {formatted:<15}")
    
    print("=" * 60 + "\n")
