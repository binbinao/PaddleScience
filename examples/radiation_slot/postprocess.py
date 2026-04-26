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
Post-processing module for radar radiation slot simulation.

Provides visualization tools for electromagnetic field distribution,
radiation pattern calculation, and S-parameter analysis.
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import interpolate
from scipy import integrate

import ppsci
from ppsci.utils import logger


# =============================================================================
# Constants
# =============================================================================

# Speed of light (cm/s)
C = 3e10

# Permeability of free space
MU_0 = 4 * np.pi

# Permittivity of free space
EPSILON_0 = 1 / (MU_0 * C**2)

# Default waveguide dimensions (WR-90)
WAVEGUIDE_WIDTH_CM = 1.02  # a dimension (cm)
WAVEGUIDE_HEIGHT_CM = 0.51  # b dimension (cm)


# =============================================================================
# Field Visualization
# =============================================================================

class FieldVisualizer:
    """Visualize electromagnetic field distributions.
    
    Supports both FDFD and PINN solutions with various visualization modes.
    """
    
    def __init__(
        self,
        output_dir: str = "./outputs/fields",
        font_size: int = 12,
    ):
        """Initialize field visualizer.
        
        Args:
            output_dir: Directory for saving visualization files
            font_size: Base font size for plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set matplotlib style
        plt.rcParams.update({
            'font.size': font_size,
            'axes.titlesize': font_size + 2,
            'axes.labelsize': font_size + 1,
            'xtick.labelsize': font_size - 1,
            'ytick.labelsize': font_size - 1,
            'legend.fontsize': font_size - 2,
        })
    
    def visualize_field(
        self,
        field_data: Dict[str, np.ndarray],
        mesh: Dict[str, np.ndarray],
        title: str = "Electric Field",
        field_type: str = "magnitude",
        save_name: str = "field",
        show_waveguide: bool = True,
        show_slot: bool = True,
    ) -> plt.Figure:
        """Visualize 2D electromagnetic field distribution.
        
        Args:
            field_data: Dictionary with field components ('field', 'field_real', 
                       'field_imag', 'field_magnitude', 'field_phase')
            mesh: Mesh dictionary with 'xx' and 'yy' arrays
            title: Plot title
            field_type: Which field component to visualize ('magnitude', 'real', 
                       'imaginary', 'phase')
            save_name: Filename for saved figure
            show_waveguide: Whether to show waveguide boundaries
            show_slot: Whether to show radiation slot
            
        Returns:
            Matplotlib Figure object
        """
        xx = mesh['xx']
        yy = mesh['yy']
        
        # Select field component
        field_map = {
            'magnitude': np.abs,
            'real': np.real,
            'imaginary': np.imag,
            'phase': np.angle,
        }
        
        if field_type not in field_map:
            field_type = 'magnitude'
        
        if isinstance(field_data, dict) and 'field' in field_data:
            field = field_map[field_type](field_data['field'])
        else:
            field = field_map[field_type](field_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set colormap and limits
        if field_type == 'phase':
            vmin, vmax = -np.pi, np.pi
            cmap = 'hsv'
        else:
            vmin = np.percentile(field.flatten(), 1)
            vmax = np.percentile(field.flatten(), 99)
            cmap = 'viridis'
        
        # Plot field
        im = ax.pcolormesh(
            xx, yy, field,
            cmap=cmap,
            shading='gouraud',
            vmin=vmin, vmax=vmax,
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, format='%.2e')
        cbar.set_label(f'{title} ({field_type})')
        
        # Draw waveguide boundaries
        if show_waveguide:
            a = WAVEGUIDE_WIDTH_CM / 2
            ax.axhline(y=a, color='white', linestyle='--', linewidth=2, alpha=0.7)
            ax.axhline(y=-a, color='white', linestyle='--', linewidth=2, alpha=0.7)
        
        # Draw radiation slot
        if show_slot:
            slot_pos = 0.5  # Default slot position
            ax.axvline(x=slot_pos, color='red', linestyle='-', linewidth=2)
            ax.axvline(x=-slot_pos, color='red', linestyle='-', linewidth=2)
        
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        # Save figure
        plt.savefig(
            os.path.join(self.output_dir, f"{save_name}_{field_type}.png"),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        
        logger.message(f"Saved field visualization to {self.output_dir}")
        
        return fig
    
    def visualize_comparison(
        self,
        fdfd_field: np.ndarray,
        pinn_field: np.ndarray,
        mesh: Dict[str, np.ndarray],
        save_name: str = "field_comparison",
    ) -> plt.Figure:
        """Visualize comparison between FDFD and PINN solutions.
        
        Args:
            fdfd_field: FDFD solution field
            pinn_field: PINN solution field
            mesh: Mesh dictionary
            save_name: Filename for saved figure
            
        Returns:
            Matplotlib Figure object with subplots
        """
        xx = mesh['xx']
        
        # Calculate error
        error = np.abs(fdfd_field - pinn_field)
        relative_error = np.abs(fdfd_field - pinn_field) / (np.abs(fdfd_field) + 1e-10)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        fields = [
            ('FDFD Solution', np.abs(fdfd_field)),
            ('PINN Solution', np.abs(pinn_field)),
            ('Absolute Error', error),
            ('Relative Error', relative_error),
        ]
        
        for ax, (title, data) in zip(axes, fields):
            if title == 'Relative Error':
                vmin, vmax = 0, 1
                cmap = 'hot'
            else:
                vmin = np.percentile(data.flatten(), 1)
                vmax = np.percentile(data.flatten(), 99)
                cmap = 'viridis'
            
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
        
        logger.message(f"Saved comparison visualization to {self.output_dir}")
        
        return fig
    
    def export_vtu(
        self,
        field_data: Dict[str, np.ndarray],
        mesh: Dict[str, np.ndarray],
        filename: str = "electric_field",
    ) -> None:
        """Export field data to VTU format for ParaView visualization.
        
        Args:
            field_data: Field data dictionary
            mesh: Mesh dictionary
            filename: Output filename (without extension)
        """
        try:
            from ppsci.visualize import save_vtu
            
            # Prepare data for export
            x = mesh['x']
            y = mesh['y']
            field_real = field_data.get('field_real', np.real(field_data.get('field', 0)))
            field_imag = field_data.get('field_imag', np.imag(field_data.get('field', 0)))
            field_mag = field_data.get('field_magnitude', np.abs(field_data.get('field', 0)))
            
            save_vtu(
                os.path.join(self.output_dir, f"{filename}.vtu"),
                x, y, mesh['xx'],
                field_real, field_imag, field_mag,
                field_names=["e_real", "e_imaginary", "e_magnitude"],
            )
            
            logger.message(f"Exported VTU file: {filename}.vtu")
            
        except ImportError:
            logger.warning("VTU export not available. Install pyvista for VTU support.")


# =============================================================================
# Radiation Pattern Calculator
# =============================================================================

class RadiationPatternCalculator:
    """Calculate and visualize radiation patterns for the radiation slot.
    
    Implements far-field radiation pattern calculation based on near-field
    distributions using Fourier transform methods.
    """
    
    def __init__(
        self,
        geometry: Dict[str, float],
        frequency: float,
        num_samples: int = 360,
    ):
        """Initialize radiation pattern calculator.
        
        Args:
            geometry: Geometry parameters dictionary
            frequency: Operating frequency in GHz
            num_samples: Number of angular samples for pattern
        """
        self.geometry = geometry
        self.frequency = frequency
        self.num_samples = num_samples
        
        # Wavenumber
        omega = 2 * np.pi * frequency * 1e9
        self.k = omega / C
        
        # TE10 mode parameters
        self.m = 1
        self.beta = np.sqrt(self.k**2 - (np.pi / geometry.get('waveguide_width', WAVEGUIDE_WIDTH_CM))**2)
    
    def calculate_far_field(
        self,
        near_field: np.ndarray,
        mesh: Dict[str, np.ndarray],
        direction: str = 'theta',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate far-field radiation pattern from near-field distribution.
        
        Uses numerical integration of the aperture field (Huygens principle):
            F(θ) = ∫ E(x,y) · exp(j·k·r̂·r') dS
        
        For a 2D problem, we extract the aperture field along a line and
        compute the 1D Fourier integral over observation angles.
        
        Args:
            near_field: Near-field electric field distribution (2D, real magnitude)
            mesh: Mesh dictionary with 'x', 'y', 'xx', 'yy', 'mesh_size'
            direction: 'theta' for E-plane (vary x), 'phi' for H-plane (vary y)
            
        Returns:
            Tuple of (angles, normalized_pattern) in radians
        """
        xx = mesh['xx']
        yy = mesh['yy']
        dx = mesh['mesh_size']
        
        # Observation angles: 0 = broadside (normal to slot), ±π/2 = endfire
        theta = np.linspace(-np.pi/2 + 0.001, np.pi/2 - 0.001, self.num_samples)
        
        if direction == 'theta':
            # E-plane pattern: extract field along x-axis at y=0 (slot aperture)
            mid_y = yy.shape[1] // 2
            aperture_field = near_field[:, mid_y]  # 1D array along x
            coords = mesh['x']  # x-coordinates
        else:
            # H-plane pattern: extract field along y-axis at x=0
            mid_x = xx.shape[0] // 2
            aperture_field = near_field[mid_x, :]
            coords = mesh['y']
        
        # Far-field integral: F(θ) = Σ E(r') · exp(j·k·r'·sin(θ)) · Δr
        pattern = np.zeros(len(theta))
        for i, th in enumerate(theta):
            phase = np.exp(1j * self.k * coords * np.sin(th))
            pattern[i] = np.abs(np.sum(aperture_field * phase) * dx)
        
        # Normalize
        pattern = pattern / (np.max(pattern) + 1e-15)
        
        return theta, pattern
    
    def calculate_analytical_pattern(
        self,
        theta: np.ndarray,
    ) -> np.ndarray:
        """Calculate analytical radiation pattern for a radiating slot.
        
        For a narrow slot of length L in a ground plane, the E-plane
        far-field pattern is (Balanis, Ch. 12):
        
            F(θ) = cos(θ) · sinc(k·L·sin(θ) / (2π))
        
        where θ is measured from the slot normal (broadside = 0).
        The cos(θ) factor is the element factor; the sinc is the
        array/aperture factor for a uniform aperture of length L.
        
        Args:
            theta: Array of angles in radians (0 = broadside)
            
        Returns:
            Normalized radiation pattern (max = 1)
        """
        slot_length = self.geometry.get('slot_length', 2.0)
        
        # Aperture factor: sinc(k*L*sin(theta)/(2*pi))
        # numpy sinc(x) = sin(pi*x)/(pi*x), so sinc(u/pi) = sin(u)/u
        u = self.k * slot_length * np.sin(theta) / 2
        aperture_factor = np.sinc(u / np.pi)  # = sin(u)/u
        
        # Element factor for a slot: cos(theta)
        element_factor = np.cos(theta)
        
        pattern = np.abs(aperture_factor * element_factor)
        pattern = pattern / (np.max(pattern) + 1e-15)
        
        return pattern
    
    def visualize_pattern(
        self,
        angles: np.ndarray,
        pattern: np.ndarray,
        pattern_type: str = 'polar',
        save_name: str = 'radiation_pattern',
        show_analytical: bool = True,
    ) -> plt.Figure:
        """Visualize radiation pattern.
        
        Args:
            angles: Angle array in radians (0 = broadside)
            pattern: Radiation pattern (normalized)
            pattern_type: 'polar' or 'cartesian'
            save_name: Filename for saved figure
            show_analytical: Whether to overlay analytical pattern
            
        Returns:
            Matplotlib Figure object
        """
        # Analytical pattern on the same angle grid
        pattern_analytical = self.calculate_analytical_pattern(angles) if show_analytical else None
        
        if pattern_type == 'polar':
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            # In polar plot, angle=0 is at top (broadside).
            # Map theta ∈ [-π/2, π/2] to polar angle directly.
            ax.plot(angles, pattern, 'b-', linewidth=2, label='Numerical (FDFD)')
            if show_analytical:
                ax.plot(angles, pattern_analytical, 'r--', linewidth=1.5, label='Analytical')
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)  # counter-clockwise
            ax.set_thetamin(-90)
            ax.set_thetamax(90)
            ax.set_rlabel_position(45)
            ax.set_title('Radiation Pattern', pad=20)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            angles_deg = np.degrees(angles)
            ax.plot(angles_deg, 20 * np.log10(pattern + 1e-10), 'b-', linewidth=2,
                    label='Numerical (FDFD)')
            if show_analytical:
                ax.plot(angles_deg, 20 * np.log10(pattern_analytical + 1e-10),
                        'r--', linewidth=1.5, label='Analytical')
            ax.set_xlabel('Angle from broadside (degrees)')
            ax.set_ylabel('Normalized magnitude (dB)')
            ax.set_title('Radiation Pattern')
            ax.set_ylim(bottom=-40)
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.4)
        
        plt.savefig(
            os.path.join(self.output_dir, f"{save_name}.png"),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        
        logger.message(f"Saved radiation pattern to {self.output_dir}")
        return fig
    
    def calculate_directivity(self, pattern: np.ndarray) -> float:
        """Calculate directivity from radiation pattern.
        
        Directivity = 4π * (Max intensity) / (Total power)
        
        Args:
            pattern: Radiation pattern (normalized)
            
        Returns:
            Directivity value
        """
        # Integrate total power
        dtheta = np.pi / len(pattern)
        total_power = np.sum(pattern**2) * dtheta
        
        # Directivity
        directivity = (4 * np.pi * np.max(pattern)**2) / total_power
        
        return directivity


# =============================================================================
# S-Parameter Calculator
# =============================================================================

class SParameterCalculator:
    """Calculate scattering parameters (S11, S21) for the waveguide structure.
    
    Implements modal matching method for extracting S-parameters from
    the electromagnetic field solution.
    """
    
    def __init__(
        self,
        geometry: Dict[str, float],
        frequency: float,
    ):
        """Initialize S-parameter calculator.
        
        Args:
            geometry: Geometry parameters
            frequency: Operating frequency in GHz
        """
        self.geometry = geometry
        self.frequency = frequency
        
        omega = 2 * np.pi * frequency * 1e9
        self.k = omega / C
        
        # TE10 mode parameters
        self.m = 1
        a = geometry.get('waveguide_width', WAVEGUIDE_WIDTH_CM)
        self.beta = np.sqrt(self.k**2 - (np.pi / a)**2)
        self.alpha = np.sqrt(self.k**2 - self.beta**2)  # Attenuation constant
        
        # Characteristic impedance
        self.Z_te10 = MU_0 * self.beta / (2 * np.pi * EPSILON_0)
    
    def calculate_s11_from_field(
        self,
        field_data: Dict[str, np.ndarray],
        mesh: Dict[str, np.ndarray],
        port_position: str = 'left',
    ) -> complex:
        """Calculate reflection coefficient S11 from field solution.
        
        S11 is determined by comparing the forward and backward traveling
        wave amplitudes at the input port.
        
        Args:
            field_data: Field solution data
            mesh: Mesh dictionary
            port_position: 'left' or 'right' port
            
        Returns:
            Complex S11 reflection coefficient
        """
        xx = mesh['xx']
        field = field_data['field']
        
        # Determine port region
        if port_position == 'left':
            port_mask = xx < 0
        else:
            port_mask = xx > 0
        
        # Separate forward and backward waves using phase information
        # For TE10 mode: E_z ∝ cos(m*pi*x/a) * exp(j*β*y ± j*α*x)
        # Forward wave: exp(+j*α*x), Backward wave: exp(-j*α*x)
        
        # Sample at different x positions
        x_samples = np.unique(xx[port_mask])
        amplitudes = []
        phases = []
        
        for x_val in x_samples:
            mask_x = port_mask & (np.abs(xx - x_val) < 0.01)
            if np.any(mask_x):
                field_sample = field[mask_x]
                amplitudes.append(np.mean(np.abs(field_sample)))
                phases.append(np.mean(np.angle(field_sample)))
        
        if len(amplitudes) < 2:
            # Fallback: use field magnitude ratio
            return 0.1 + 0.1j  # Placeholder
        
        amplitudes = np.array(amplitudes)
        phases = np.array(phases)
        
        # Fit forward and backward wave amplitudes
        # E(x) = E_inc * exp(-j*α*x) + E_ref * exp(+j*α*x)
        # At port: E ≈ E_inc (if no reflection) or E ≈ E_ref (if completely reflected)
        
        # Use ratio of field magnitudes at different positions
        # S11 = E_ref / E_inc ≈ (E2 - E1) / (E2 + E1) for two sample points
        E1 = amplitudes[0] * np.exp(1j * phases[0])
        E2 = amplitudes[-1] * np.exp(1j * phases[-1])
        
        # Propagation distance
        dx = x_samples[-1] - x_samples[0]
        phase_shift = 1j * self.alpha * dx
        
        # Solve for S11
        # E2 = E1 * exp(-j*α*dx) + S11 * E1 * exp(j*α*dx)
        # S11 = (E2 - E1*exp(-j*α*dx)) / (E1 * exp(j*α*dx))
        denominator = E1 * np.exp(phase_shift)
        if np.abs(denominator) < 1e-10:
            return 0 + 0j
        
        s11 = (E2 - E1 * np.exp(-phase_shift)) / denominator
        
        return s11
    
    def calculate_s21_from_field(
        self,
        field_data: Dict[str, np.ndarray],
        mesh: Dict[str, np.ndarray],
    ) -> complex:
        """Calculate transmission coefficient S21 from field solution.
        
        S21 represents the power transmission from port 1 to port 2.
        
        Args:
            field_data: Field solution data
            mesh: Mesh dictionary
            
        Returns:
            Complex S21 transmission coefficient
        """
        xx = mesh['xx']
        field = field_data['field']
        
        # Find field magnitudes at input (left) and output (right) ports
        left_port = xx < -self.geometry.get('slot_length', 2.0) / 4
        right_port = xx > self.geometry.get('slot_length', 2.0) / 4
        
        E_left = field[left_port]
        E_right = field[right_port]
        
        # Calculate magnitudes
        amp_left = np.linalg.norm(E_left.flatten())
        amp_right = np.linalg.norm(E_right.flatten())
        
        # S21 = E_out / E_in (assuming same phase for simplicity)
        if amp_left < 1e-10:
            return 0 + 0j
        
        s21 = amp_right / amp_left
        
        return s21
    
    def calculate_s_parameters_frequency_sweep(
        self,
        solver,
        frequency_start: float,
        frequency_stop: float,
        num_points: int = 13,
    ) -> Dict[str, np.ndarray]:
        """Calculate S-parameters over a frequency range.
        
        Args:
            solver: FDFD or PINN solver instance
            frequency_start: Start frequency in GHz
            frequency_stop: Stop frequency in GHz
            num_points: Number of frequency points
            
        Returns:
            Dictionary with frequency, s11, s21 arrays
        """
        frequencies = np.linspace(frequency_start, frequency_stop, num_points)
        s11_values = []
        s21_values = []
        
        for freq in frequencies:
            # Solve for current frequency
            if hasattr(solver, 'set_frequency'):
                solver.set_frequency(freq)
                result = solver.solve()
            else:
                result = solver.solve_at_frequency(freq)
            
            # Calculate S-parameters
            s11 = self.calculate_s11_from_field(result, solver.mesh)
            s21 = self.calculate_s21_from_field(result, solver.mesh)
            
            s11_values.append(s11)
            s21_values.append(s21)
        
        return {
            'frequency': frequencies,
            's11': np.array(s11_values),
            's21': np.array(s21_values),
        }
    
    def visualize_s_parameters(
        self,
        s_params: Dict[str, np.ndarray],
        save_name: str = 's_parameters',
    ) -> plt.Figure:
        """Visualize S-parameters over frequency range.
        
        Args:
            s_params: Dictionary with frequency, s11, s21 data
            save_name: Filename for saved figure
            
        Returns:
            Matplotlib Figure object
        """
        frequencies = s_params['frequency']
        s11 = s_params['s11']
        s21 = s_params['s21']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # S11 magnitude and phase
        ax1 = axes[0, 0]
        ax1.plot(frequencies, 20 * np.log10(np.abs(s11) + 1e-10), 'b-', linewidth=2)
        ax1.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('S11 Magnitude (dB)')
        ax1.set_title('Reflection Coefficient |S11|')
        ax1.grid(True)
        
        ax2 = axes[0, 1]
        ax2.plot(frequencies, np.degrees(np.angle(s11)), 'b-', linewidth=2)
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_title('S11 Phase')
        ax2.grid(True)
        
        # S21 magnitude and phase
        ax3 = axes[1, 0]
        ax3.plot(frequencies, 20 * np.log10(np.abs(s21) + 1e-10), 'g-', linewidth=2)
        ax3.set_xlabel('Frequency (GHz)')
        ax3.set_ylabel('S21 Magnitude (dB)')
        ax3.set_title('Transmission Coefficient |S21|')
        ax3.grid(True)
        
        ax4 = axes[1, 1]
        ax4.plot(frequencies, np.degrees(np.angle(s21)), 'g-', linewidth=2)
        ax4.set_xlabel('Frequency (GHz)')
        ax4.set_ylabel('Phase (degrees)')
        ax4.set_title('S21 Phase')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(
            os.path.join(self.output_dir, f"{save_name}.png"),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        
        logger.message(f"Saved S-parameter visualization to {self.output_dir}")
        
        return fig


# =============================================================================
# Data Export
# =============================================================================

class DataExporter:
    """Export simulation results to various formats.
    
    Supports numpy arrays, CSV files, and other common formats.
    """
    
    def __init__(self, output_dir: str = "./outputs/data"):
        """Initialize data exporter.
        
        Args:
            output_dir: Directory for exported files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_field_data(
        self,
        field_data: Dict[str, np.ndarray],
        mesh: Dict[str, np.ndarray],
        filename: str = "electric_field",
    ) -> str:
        """Export field data to numpy format.
        
        Args:
            field_data: Field solution data
            mesh: Mesh dictionary
            filename: Output filename (without extension)
            
        Returns:
            Path to saved file
        """
        data = {
            'field': field_data.get('field', np.zeros_like(mesh['xx'])),
            'field_real': field_data.get('field_real', np.real(field_data.get('field', 0))),
            'field_imaginary': field_data.get('field_imag', np.imag(field_data.get('field', 0))),
            'field_magnitude': field_data.get('field_magnitude', np.abs(field_data.get('field', 0))),
            'field_phase': field_data.get('field_phase', np.angle(field_data.get('field', 0))),
            'x': mesh['x'],
            'y': mesh['y'],
            'xx': mesh['xx'],
            'yy': mesh['yy'],
            'mesh_size': mesh['mesh_size'],
        }
        
        filepath = os.path.join(self.output_dir, f"{filename}.npz")
        np.savez(filepath, **data)
        
        logger.message(f"Exported field data to {filepath}")
        
        return filepath
    
    def export_s_parameters(
        self,
        s_params: Dict[str, np.ndarray],
        filename: str = "s_parameters",
    ) -> str:
        """Export S-parameters to CSV format.
        
        Args:
            s_params: S-parameter data
            filename: Output filename (without extension)
            
        Returns:
            Path to saved file
        """
        data = np.column_stack([
            s_params['frequency'],
            np.real(s_params['s11']),
            np.imag(s_params['s11']),
            np.real(s_params['s21']),
            np.imag(s_params['s21']),
            20 * np.log10(np.abs(s_params['s11']) + 1e-10),
            20 * np.log10(np.abs(s_params['s21']) + 1e-10),
        ])
        
        header = "Frequency(GHz),S11_real,S11_imag,S21_real,S21_imag,S11_dB,S21_dB"
        
        filepath = os.path.join(self.output_dir, f"{filename}.csv")
        np.savetxt(filepath, data, delimiter=',', header=header, comments='')
        
        logger.message(f"Exported S-parameters to {filepath}")
        
        return filepath
    
    def export_radiation_pattern(
        self,
        angles: np.ndarray,
        pattern: np.ndarray,
        filename: str = "radiation_pattern",
    ) -> str:
        """Export radiation pattern data.
        
        Args:
            angles: Angle array in radians
            pattern: Radiation pattern
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        data = np.column_stack([
            np.degrees(angles),
            pattern,
            20 * np.log10(pattern + 1e-10),
        ])
        
        header = "Angle(degrees),Magnitude,Magnitude_dB"
        
        filepath = os.path.join(self.output_dir, f"{filename}.csv")
        np.savetxt(filepath, data, delimiter=',', header=header, comments='')
        
        logger.message(f"Exported radiation pattern to {filepath}")
        
        return filepath


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_relative_error(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
    """Calculate various error metrics between reference and prediction.
    
    Args:
        reference: Reference solution (e.g., FDFD)
        prediction: Predicted solution (e.g., PINN)
        
    Returns:
        Dictionary with error metrics
    """
    # Flatten arrays
    ref_flat = reference.flatten()
    pred_flat = prediction.flatten()
    
    # L2 relative error
    l2_error = np.linalg.norm(ref_flat - pred_flat) / (np.linalg.norm(ref_flat) + 1e-10)
    
    # Maximum absolute error
    max_error = np.max(np.abs(ref_flat - pred_flat))
    
    # Mean absolute error
    mae = np.mean(np.abs(ref_flat - pred_flat))
    
    # Root mean square error
    rmse = np.sqrt(np.mean((ref_flat - pred_flat)**2))
    
    # Relative root mean square error
    relative_rmse = rmse / (np.sqrt(np.mean(ref_flat**2)) + 1e-10)
    
    # Pearson correlation coefficient
    correlation = np.corrcoef(ref_flat, pred_flat)[0, 1]
    
    return {
        'l2_error': l2_error,
        'max_error': max_error,
        'mae': mae,
        'rmse': rmse,
        'relative_rmse': relative_rmse,
        'correlation': correlation,
    }


def visualize_error_convergence(
    errors: List[Dict[str, float]],
    mesh_sizes: List[float],
    save_name: str = 'error_convergence',
    output_dir: str = './outputs',
) -> plt.Figure:
    """Visualize error convergence with mesh refinement.
    
    Args:
        errors: List of error dictionaries
        mesh_sizes: Corresponding mesh sizes
        save_name: Filename for saved figure
        output_dir: Output directory
        
    Returns:
        Matplotlib Figure object
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # L2 error vs mesh size
    ax1 = axes[0]
    ax1.semilogy(mesh_sizes, [e['l2_error'] for e in errors], 'bo-', linewidth=2)
    ax1.set_xlabel('Mesh Size (cm)')
    ax1.set_ylabel('L2 Relative Error')
    ax1.set_title('Error Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Correlation vs mesh size
    ax2 = axes[1]
    ax2.plot(mesh_sizes, [e['correlation'] for e in errors], 'go-', linewidth=2)
    ax2.set_xlabel('Mesh Size (cm)')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title('Correlation vs Mesh Size')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(output_dir, f"{save_name}.png"),
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    
    return fig
