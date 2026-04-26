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
Geometry module for radar radiation slot simulation.

Defines the rectangular waveguide radiation slot structure and provides
mesh generation functions for both FDFD and PINN solvers.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class RadiationSlotGeometry:
    """Rectangular waveguide radiation slot geometry parameters.
    
    For TE10 mode simulation in Ku-band (12-18 GHz).
    
    Attributes:
        waveguide_width: Width of the waveguide (cm), typically 1.02 cm for TE10
        waveguide_height: Height of the waveguide (cm), typically 0.51 cm
        slot_length: Length of the radiation slot (cm)
        slot_position: Position of the slot along waveguide axis (cm)
        medium_epsilon: Dielectric permittivity of the medium (default: air = 1.0)
        medium_mu: Magnetic permeability of the medium (default: air = 1.0)
        pml_thickness: Perfectly Matched Layer thickness (cm)
        mesh_size: Target mesh size for FDFD (cm)
    """
    
    # Waveguide dimensions (cm) - standard WR-90 waveguide
    waveguide_width: float = 1.02  # a (wide dimension)
    waveguide_height: float = 0.51  # b (narrow dimension)
    
    # Slot parameters (cm)
    slot_length: float = 2.0
    slot_position: float = 0.5  # center position along waveguide
    
    # Medium properties
    medium_epsilon: float = 1.0
    medium_mu: float = 1.0
    
    # Simulation parameters
    pml_thickness: float = 1.0  # PML layer thickness
    mesh_size: float = 0.02  # lambda/10 for typical Ku-band
    
    # Domain bounds (auto-computed or user-specified)
    domain_min: Tuple[float, float] = field(default=None)
    domain_max: Tuple[float, float] = field(default=None)
    
    def __post_init__(self):
        """Compute derived geometry parameters."""
        if self.domain_min is None or self.domain_max is None:
            # Compute domain bounds based on geometry
            # Add PML thickness on all sides
            margin = self.pml_thickness
            
            # X direction: slot length + margins
            x_min = -self.slot_length / 2 - margin
            x_max = self.slot_length / 2 + margin
            
            # Y direction: waveguide width + margins (centered at origin)
            y_min = -self.waveguide_width / 2 - margin
            y_max = self.waveguide_width / 2 + margin
            
            self.domain_min = (x_min, y_min)
            self.domain_max = (x_max, y_max)
    
    @property
    def cutoff_frequency(self) -> float:
        """Calculate TE10 cutoff frequency in GHz."""
        c = 3e10  # speed of light (cm/s)
        f_c = c / (2 * self.waveguide_width * np.sqrt(self.medium_epsilon)) / 1e9
        return f_c
    
    @property
    def wavenumber(self) -> float:
        """Calculate wavenumber k = omega * sqrt(mu * epsilon) for given frequency."""
        return 2 * np.pi * np.sqrt(self.medium_mu * self.medium_epsilon)
    
    def get_mesh_bounds(self, frequency: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get mesh bounds as numpy arrays.
        
        Args:
            frequency: Frequency in GHz (used to adjust mesh size)
            
        Returns:
            Tuple of (lower_bounds, upper_bounds) as numpy arrays
        """
        return (np.array([self.domain_min[0], self.domain_min[1]]),
                np.array([self.domain_max[0], self.domain_max[1]]))


def generate_mesh(
    geometry: RadiationSlotGeometry,
    frequency: float = 15.0,
) -> Dict[str, np.ndarray]:
    """Generate uniform grid mesh for FDFD solver.
    
    Args:
        geometry: Radiation slot geometry parameters
        frequency: Frequency in GHz for mesh density calculation
        
    Returns:
        Dictionary containing:
            - 'x': x-coordinates (1D array)
            - 'y': y-coordinates (1D array)
            - 'xx': 2D meshgrid of x
            - 'yy': 2D meshgrid of y
            - 'epsilon': Relative permittivity at each point
            - 'mesh_size': Actual mesh size used
            - 'num_points': Total number of mesh points
    """
    c = 3e10  # speed of light (cm/s)
    omega = 2 * np.pi * frequency * 1e9
    
    # Calculate mesh size: lambda/10
    wavelength = c / frequency / 1e9 / np.sqrt(geometry.medium_epsilon)
    mesh_size = geometry.mesh_size if geometry.mesh_size is not None else wavelength / 10
    
    # Get domain bounds
    x_min, y_min = geometry.domain_min
    x_max, y_max = geometry.domain_max
    
    # Generate uniform grid
    x = np.arange(x_min, x_max + mesh_size, mesh_size)
    y = np.arange(y_min, y_max + mesh_size, mesh_size)
    
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # Set permittivity (1 for air, could vary for different media)
    epsilon = np.ones_like(xx) * geometry.medium_epsilon
    
    return {
        'x': x,
        'y': y,
        'xx': xx,
        'yy': yy,
        'epsilon': epsilon,
        'mesh_size': mesh_size,
        'num_points': xx.size,
        'omega': omega,
        'wavelength': wavelength,
    }


def generate_pml_mesh(
    geometry: RadiationSlotGeometry,
    frequency: float = 15.0,
) -> Dict[str, np.ndarray]:
    """Generate mesh with PML layers for PINN solver.
    
    The mesh includes PML layers on all boundaries for absorbing outgoing waves.
    
    Args:
        geometry: Radiation slot geometry parameters
        frequency: Frequency in GHz
        
    Returns:
        Dictionary containing:
            - 'x': x-coordinates including PML region
            - 'y': y-coordinates including PML region
            - 'xx': 2D meshgrid
            - 'yy': 2D meshgrid
            - 'pml_mask': Boolean mask for PML region
            - 'core_mask': Boolean mask for core (non-PML) region
            - 'mesh_size': Mesh size
    """
    c = 3e10
    omega = 2 * np.pi * frequency * 1e9
    
    # Calculate mesh size
    wavelength = c / frequency / 1e9 / np.sqrt(geometry.medium_epsilon)
    mesh_size = geometry.mesh_size if geometry.mesh_size is not None else wavelength / 10
    
    # Extended domain with PML
    x_min = geometry.domain_min[0] - geometry.pml_thickness
    x_max = geometry.domain_max[0] + geometry.pml_thickness
    y_min = geometry.domain_min[1] - geometry.pml_thickness
    y_max = geometry.domain_max[1] + geometry.pml_thickness
    
    # Generate grid
    x = np.arange(x_min, x_max + mesh_size, mesh_size)
    y = np.arange(y_min, y_max + mesh_size, mesh_size)
    
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # Create PML mask (points outside core domain)
    core_x_min, core_y_min = geometry.domain_min
    core_x_max, core_y_max = geometry.domain_max
    
    pml_mask = (xx < core_x_min) | (xx > core_x_max) | \
               (yy < core_y_min) | (yy > core_y_max)
    core_mask = ~pml_mask
    
    return {
        'x': x,
        'y': y,
        'xx': xx,
        'yy': yy,
        'pml_mask': pml_mask,
        'core_mask': core_mask,
        'mesh_size': mesh_size,
        'num_points': xx.size,
        'omega': omega,
    }


def generate_random_samples(
    geometry: RadiationSlotGeometry,
    num_samples: int = 10000,
    include_pml: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate random sample points for PINN training.
    
    Args:
        geometry: Radiation slot geometry parameters
        num_samples: Number of random samples
        include_pml: Whether to include PML region
        
    Returns:
        Dictionary containing:
            - 'x': Random x-coordinates (1D array)
            - 'y': Random y-coordinates (1D array)
            - 'samples': 2D array of [x, y] pairs
    """
    if include_pml:
        x_min = geometry.domain_min[0] - geometry.pml_thickness
        x_max = geometry.domain_max[0] + geometry.pml_thickness
        y_min = geometry.domain_min[1] - geometry.pml_thickness
        y_max = geometry.domain_max[1] + geometry.pml_thickness
    else:
        x_min, y_min = geometry.domain_min
        x_max, y_max = geometry.domain_max
    
    x = np.random.uniform(x_min, x_max, num_samples)
    y = np.random.uniform(y_min, y_max, num_samples)
    
    return {
        'x': x[:, np.newaxis].astype(np.float32),
        'y': y[:, np.newaxis].astype(np.float32),
        'samples': np.stack([x, y], axis=-1).astype(np.float32),
    }


def generate_boundary_samples(
    geometry: RadiationSlotGeometry,
    num_boundary: int = 200,
) -> Dict[str, np.ndarray]:
    """Generate boundary sample points for PINN boundary conditions.
    
    Args:
        geometry: Radiation slot geometry parameters
        num_boundary: Number of boundary samples per edge
        
    Returns:
        Dictionary containing boundary samples for each edge:
            - 'left': Left boundary samples
            - 'right': Right boundary samples
            - 'bottom': Bottom boundary samples
            - 'top': Top boundary samples
    """
    x_min, y_min = geometry.domain_min
    x_max, y_max = geometry.domain_max
    
    # Generate samples for each boundary edge
    t = np.linspace(0, 1, num_boundary)
    
    # Left boundary: x = x_min, y varies
    left_x = np.full(num_boundary, x_min)
    left_y = np.linspace(y_min, y_max, num_boundary)
    
    # Right boundary: x = x_max, y varies
    right_x = np.full(num_boundary, x_max)
    right_y = np.linspace(y_min, y_max, num_boundary)
    
    # Bottom boundary: y = y_min, x varies
    bottom_x = np.linspace(x_min, x_max, num_boundary)
    bottom_y = np.full(num_boundary, y_min)
    
    # Top boundary: y = y_max, x varies
    top_x = np.linspace(x_min, x_max, num_boundary)
    top_y = np.full(num_boundary, y_max)
    
    return {
        'left': {
            'x': left_x[:, np.newaxis].astype(np.float32),
            'y': left_y[:, np.newaxis].astype(np.float32),
        },
        'right': {
            'x': right_x[:, np.newaxis].astype(np.float32),
            'y': right_y[:, np.newaxis].astype(np.float32),
        },
        'bottom': {
            'x': bottom_x[:, np.newaxis].astype(np.float32),
            'y': bottom_y[:, np.newaxis].astype(np.float32),
        },
        'top': {
            'x': top_x[:, np.newaxis].astype(np.float32),
            'y': top_y[:, np.newaxis].astype(np.float32),
        },
    }


def get_excitation_region(
    geometry: RadiationSlotGeometry,
    mesh: Dict[str, np.ndarray],
    buffer: float = 0.1,
) -> np.ndarray:
    """Get mask for the excitation region (waveguide input).
    
    For TE10 mode excitation at the left end of the waveguide.
    
    Args:
        geometry: Radiation slot geometry
        mesh: Mesh dictionary with 'xx' and 'yy'
        buffer: Buffer region around the excitation area
        
    Returns:
        Boolean mask for excitation region
    """
    xx = mesh['xx']
    yy = mesh['yy']
    
    # Excitation region: left end of waveguide
    x_max_excite = geometry.slot_position - buffer
    
    # Waveguide cross-section
    y_min = -geometry.waveguide_width / 2
    y_max = geometry.waveguide_width / 2
    
    mask = (xx < x_max_excite) & (yy >= y_min) & (yy <= y_max)
    return mask


def get_slot_region(
    geometry: RadiationSlotGeometry,
    mesh: Dict[str, np.ndarray],
    buffer: float = 0.1,
) -> np.ndarray:
    """Get mask for the radiation slot region.
    
    Args:
        geometry: Radiation slot geometry
        mesh: Mesh dictionary with 'xx' and 'yy'
        buffer: Buffer region around the slot
        
    Returns:
        Boolean mask for slot region
    """
    xx = mesh['xx']
    yy = mesh['yy']
    
    # Slot region
    x_min_slot = geometry.slot_position - geometry.slot_length / 2 - buffer
    x_max_slot = geometry.slot_position + geometry.slot_length / 2 + buffer
    
    # Full width (open to air)
    mask = (xx >= x_min_slot) & (xx <= x_max_slot)
    return mask
