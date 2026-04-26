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
FDFD (Finite Difference Frequency Domain) Solver for radar radiation slot.

Implements 2D Helmholtz equation solver with PEC boundary conditions and
TE10 mode excitation for electromagnetic field simulation.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any
from scipy.sparse import diags, csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, gmres
import warnings

warnings.filterwarnings('ignore')

from .geometry import RadiationSlotGeometry, generate_mesh, get_excitation_region


class FDFDSolver:
    """Finite Difference Frequency Domain solver for electromagnetic field simulation.
    
    Solves the 2D Helmholtz equation for TE mode electric field:
        (d²/dx² + d²/dy² + k²ε) E_z = J (source term)
    
    For TE10 mode in rectangular waveguide, the field is:
        E_z(x,y) = E_0 * cos(m*pi*x/a) * exp(j*β*y)
    where m=1 for TE10 mode.
    """
    
    def __init__(
        self,
        geometry: RadiationSlotGeometry,
        frequency: float = 15.0,
        solver_type: str = 'direct',
    ):
        """Initialize FDFD solver.
        
        Args:
            geometry: Radiation slot geometry parameters
            frequency: Operating frequency in GHz
            solver_type: 'direct' (spsolve) or 'iterative' (gmres)
        """
        self.geometry = geometry
        self.frequency = frequency
        self.solver_type = solver_type
        
        # Physical constants
        self.c = 3e10  # speed of light (cm/s)
        self.omega = 2 * np.pi * frequency * 1e9  # angular frequency
        
        # Generate mesh
        self.mesh = generate_mesh(geometry, frequency)
        self.dx = self.mesh['mesh_size']
        self.dy = self.mesh['mesh_size']
        
        # Wavenumber: k = omega / c * sqrt(mu * epsilon)
        self.k = self.omega / self.c * np.sqrt(
            self.geometry.medium_mu * self.geometry.medium_epsilon
        )
        self.beta = np.sqrt(self.k**2 - (np.pi / self.geometry.waveguide_width)**2)
        
        # Mesh dimensions
        self.nx = len(self.mesh['x'])
        self.ny = len(self.mesh['y'])
        
        # Pre-compute masks
        self.excitation_mask = get_excitation_region(geometry, self.mesh)
        
        # Solver statistics
        self.solve_time = 0.0
        self.iterations = 0
    
    def _build_laplacian_matrix(self) -> csc_matrix:
        """Build 5-point stencil Laplacian matrix for 2D grid.
        
        Returns:
            Sparse Laplacian matrix in CSC format
        """
        # Create diagonals for 5-point stencil
        # Center: 4/(dx*dy)
        # X neighbors: -1/(dx*dx)
        # Y neighbors: -1/(dy*dy)
        
        center = 4.0 / (self.dx * self.dy)
        diag_x = -1.0 / (self.dx * self.dx)
        diag_y = -1.0 / (self.dy * self.dy)
        
        # Build diagonal matrices
        main_diag = np.full(self.nx * self.ny, center)
        x_diag = np.full(self.nx * self.ny, diag_x)
        y_diag = np.full(self.nx * self.ny, diag_y)
        
        # For edges, adjust stencil
        x_diag[:self.ny] = 0  # Left boundary
        x_diag[-self.ny:] = 0  # Right boundary
        
        y_diag[::self.ny] = 0  # Bottom boundary
        y_diag[self.ny-1::self.ny] = 0  # Top boundary
        
        # Create sparse matrix using diags
        offsets = [0, 1, -1, self.ny, -self.ny]
        laplacian = diags(
            [main_diag, x_diag, x_diag, y_diag, y_diag],
            offsets,
            shape=(self.nx * self.ny, self.nx * self.ny),
            format='csc'
        )
        
        return laplacian
    
    def _build_permittivity_matrix(self) -> csc_matrix:
        """Build diagonal matrix with permittivity values.
        
        Returns:
            Diagonal matrix with ε values
        """
        epsilon = self.mesh['epsilon'].flatten()
        return diags(epsilon, shape=(self.nx * self.ny, self.nx * self.ny), format='csc')
    
    def _build_source_vector(self) -> np.ndarray:
        """Build source vector for TE10 excitation.
        
        The source term represents the impressed current J = -jωE.
        For TE10 mode, we excite at the waveguide input with a cosine distribution.
        
        Returns:
            Source vector (complex)
        """
        source = np.zeros(self.nx * self.ny, dtype=complex)
        
        # Get mesh coordinates
        xx = self.mesh['xx']
        yy = self.mesh['yy']
        
        # TE10 mode has E_z ∝ cos(m*pi*x/a)
        m = 1  # TE10 mode
        a = self.geometry.waveguide_width
        
        # Only excite in waveguide region
        waveguide_mask = (np.abs(yy) <= a / 2) & (xx < 0)  # Left half
        
        # Source distribution: cosine in x, uniform in y
        source_magnitude = np.cos(m * np.pi * xx / a)
        source_magnitude = source_magnitude * waveguide_mask.astype(float)
        
        # Normalize and set source
        source_magnitude = source_magnitude.flatten()
        source_magnitude = source_magnitude / (np.max(np.abs(source_magnitude)) + 1e-10)
        
        # For FDFD, we solve the real part
        # Source is typically a delta function or Gaussian at excitation point
        source = source_magnitude.astype(complex) * 1e6  # Amplify source
        
        return source
    
    def _apply_pec_boundary(self, matrix: csc_matrix, vector: np.ndarray) -> Tuple[csc_matrix, np.ndarray]:
        """Apply Perfect Electric Conductor (PEC) boundary conditions.
        
        PEC boundaries require E = 0. We modify the matrix to set boundary 
        values to zero by setting diagonal to 1 and off-diagonals to 0.
        
        Args:
            matrix: System matrix
            vector: Right-hand side vector
            
        Returns:
            Modified matrix and vector
        """
        # Get boundary indices
        boundary_indices = self._get_boundary_indices()
        
        # Convert to CSR for efficient row access
        matrix_csr = matrix.tocsr()
        
        for idx in boundary_indices:
            # Set all entries in this row to 0 except diagonal
            row = matrix_csr.getrow(idx)
            row.data[:] = 0
            row.indices[:] = idx
            row.data[:] = 1.0
            matrix_csr[idx, :] = row
            
            # Set boundary value to 0
            vector[idx] = 0.0
        
        return matrix_csr.tocsc(), vector
    
    def _get_boundary_indices(self) -> np.ndarray:
        """Get indices of boundary nodes.
        
        Returns:
            Array of boundary node indices
        """
        xx = self.mesh['xx']
        yy = self.mesh['yy']
        
        x_min, y_min = self.geometry.domain_min
        x_max, y_max = self.geometry.domain_max
        
        # Boundary mask
        boundary_mask = (np.abs(xx - x_min) < self.dx / 2) | \
                        (np.abs(xx - x_max) < self.dx / 2) | \
                        (np.abs(yy - y_min) < self.dy / 2) | \
                        (np.abs(yy - y_max) < self.dy / 2)
        
        boundary_flat = boundary_mask.flatten()
        return np.where(boundary_flat)[0]
    
    def solve(self) -> Dict[str, Any]:
        """Solve the Helmholtz equation for the electric field.
        
        Returns:
            Dictionary containing:
                - 'field': Electric field solution (2D complex array)
                - 'field_real': Real part of field
                - 'field_imag': Imaginary part of field
                - 'field_magnitude': Magnitude of field
                - 'solve_time': Time taken for solution
                - 'iterations': Number of iterations (for iterative solver)
        """
        import time
        start_time = time.time()
        
        # Build matrices
        laplacian = self._build_laplacian_matrix()
        permittivity = self._build_permittivity_matrix()
        source = self._build_source_vector()
        
        # Helmholtz equation: (Laplacian + k^2 * ε) * E = J
        # Discretized: (L + k^2 * diag(ε)) * E = J
        
        # System matrix: Laplacian + k^2 * permittivity
        system_matrix = laplacian + self.k**2 * permittivity
        
        # Apply PEC boundary conditions
        system_matrix, source = self._apply_pec_boundary(system_matrix, source)
        
        # Solve system
        if self.solver_type == 'direct':
            solution = spsolve(system_matrix, source)
            self.iterations = 0
        else:
            solution, info = gmres(system_matrix, source, rtol=1e-6, maxiter=1000)
            self.iterations = info
        
        self.solve_time = time.time() - start_time
        
        # Reshape to 2D
        field_2d = solution.reshape(self.nx, self.ny)
        
        return {
            'field': field_2d,
            'field_real': np.real(field_2d),
            'field_imag': np.imag(field_2d),
            'field_magnitude': np.abs(field_2d),
            'field_phase': np.angle(field_2d),
            'solve_time': self.solve_time,
            'iterations': self.iterations,
            'mesh': self.mesh,
            'frequency': self.frequency,
            'omega': self.omega,
        }
    
    def solve_frequency_sweep(
        self,
        frequency_start: float,
        frequency_stop: float,
        num_points: int = 13,
    ) -> Dict[str, Any]:
        """Solve Helmholtz equation over a frequency range.
        
        Args:
            frequency_start: Start frequency in GHz
            frequency_stop: Stop frequency in GHz
            num_points: Number of frequency points
            
        Returns:
            Dictionary containing solutions at all frequencies
        """
        frequencies = np.linspace(frequency_start, frequency_stop, num_points)
        results = {}
        
        for freq in frequencies:
            self.frequency = freq
            self.omega = 2 * np.pi * freq * 1e9
            self.k = self.omega / self.c * np.sqrt(
                self.geometry.medium_mu * self.geometry.medium_epsilon
            )
            
            result = self.solve()
            results[freq] = result
        
        results['frequencies'] = frequencies
        return results


class TE10Excitation:
    """TE10 mode excitation generator for waveguide ports."""
    
    def __init__(
        self,
        waveguide_width: float,
        amplitude: float = 1.0,
        phase: float = 0.0,
    ):
        """Initialize TE10 excitation.
        
        Args:
            waveguide_width: Width of waveguide (a dimension)
            amplitude: Field amplitude
            phase: Phase offset in radians
        """
        self.a = waveguide_width
        self.amplitude = amplitude
        self.phase = phase
    
    def electric_field(
        self,
        x: np.ndarray,
        y: np.ndarray,
        propagation_distance: float = 0.0,
    ) -> np.ndarray:
        """Calculate TE10 electric field distribution.
        
        E_z(x, y, z) = E_0 * cos(m*pi*x/a) * exp(j*β*z)
        
        Args:
            x: x-coordinates
            y: y-coordinates
            propagation_distance: Propagation distance z (cm)
            
        Returns:
            Electric field E_z (complex)
        """
        m = 1  # TE10 mode
        
        # Transverse field: cos(m*pi*x/a)
        transverse = np.cos(m * np.pi * x / self.a)
        
        # Longitudinal phase: exp(j*β*z)
        longitudinal = np.exp(1j * propagation_distance)
        
        # Apply cutoff outside waveguide
        waveguide_mask = np.abs(y) <= self.a / 2
        transverse = transverse * waveguide_mask.astype(float)
        
        return self.amplitude * transverse * longitudinal * np.exp(1j * self.phase)
    
    def magnetic_field(
        self,
        x: np.ndarray,
        y: np.ndarray,
        propagation_distance: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate TE10 magnetic field components.
        
        For TE10 mode:
            H_y = (j * k * E_0 / ω * μ) * sin(m*pi*x/a) * exp(j*β*z)
            H_x = 0
        
        Args:
            x: x-coordinates
            y: y-coordinates
            propagation_distance: Propagation distance z (cm)
            
        Returns:
            Tuple of (H_x, H_y) magnetic field components
        """
        m = 1
        
        # H_y component
        h_y = np.sin(m * np.pi * x / self.a)
        h_y = h_y * np.exp(1j * propagation_distance)
        
        # Apply cutoff
        waveguide_mask = np.abs(y) <= self.a / 2
        h_y = h_y * waveguide_mask.astype(float)
        
        return np.zeros_like(h_y), self.amplitude * h_y


def compute_reflection_coefficient(
    incident_field: np.ndarray,
    reflected_field: np.ndarray,
) -> complex:
    """Compute reflection coefficient from field measurements.
    
    Args:
        incident_field: Incident field (TE10 mode)
        reflected_field: Reflected field at port
        
    Returns:
        Reflection coefficient S11
    """
    # S11 = reflected / incident
    if np.linalg.norm(incident_field) < 1e-10:
        return 0.0 + 0.0j
    
    ratio = np.linalg.norm(reflected_field) / np.linalg.norm(incident_field)
    
    # Assume same phase for simplicity
    return ratio


def compute_transmission_coefficient(
    incident_field: np.ndarray,
    transmitted_field: np.ndarray,
) -> complex:
    """Compute transmission coefficient from field measurements.
    
    Args:
        incident_field: Incident field
        transmitted_field: Transmitted field at output port
        
    Returns:
        Transmission coefficient S21
    """
    if np.linalg.norm(incident_field) < 1e-10:
        return 0.0 + 0.0j
    
    ratio = np.linalg.norm(transmitted_field) / np.linalg.norm(incident_field)
    return ratio
