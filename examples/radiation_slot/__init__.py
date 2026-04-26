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
Radar Radiation Slot Simulation Package

This package provides a complete simulation framework for radar rectangular 
waveguide radiation slots using both traditional FDFD method and PINN 
(Physics-Informed Neural Network) method.

Main modules:
- geometry: Geometry definition and mesh generation for radiation slot
- fdfd_solver: Finite Difference Frequency Domain solver
- postprocess: Post-processing for field visualization, radiation pattern, and S-parameters
- comparator: Comparison and validation between FDFD and PINN results

Example usage:
    >>> from radiation_slot import RadiationSlotGeometry, FDFDSolver
    >>> 
    >>> # Create geometry
    >>> geometry = RadiationSlotGeometry(waveguide_width=1.02, slot_length=2.0)
    >>> 
    >>> # Run FDFD simulation
    >>> solver = FDFDSolver(geometry, frequency=15.0)
    >>> result = solver.solve()
    >>> 
    >>> # Visualize results
    >>> from radiation_slot.postprocess import FieldVisualizer
    >>> vis = FieldVisualizer('./outputs')
    >>> vis.visualize_field(result, mesh)
"""

from .geometry import (
    RadiationSlotGeometry,
    generate_mesh,
    generate_pml_mesh,
    generate_random_samples,
    generate_boundary_samples,
)
from .fdfd_solver import FDFDSolver, TE10Excitation
from .postprocess import (
    FieldVisualizer,
    RadiationPatternCalculator,
    SParameterCalculator,
    DataExporter,
)
from .comparator import (
    SolutionComparator,
    ComparisonVisualizer,
    ComparisonReport,
    ComparisonMetrics,
    quick_compare,
)

__all__ = [
    # Geometry
    "RadiationSlotGeometry",
    "generate_mesh",
    "generate_pml_mesh",
    "generate_random_samples",
    "generate_boundary_samples",
    # FDFD Solver
    "FDFDSolver",
    "TE10Excitation",
    # Post-processing
    "FieldVisualizer",
    "RadiationPatternCalculator",
    "SParameterCalculator",
    "DataExporter",
    # Comparator
    "SolutionComparator",
    "ComparisonVisualizer",
    "ComparisonReport",
    "ComparisonMetrics",
    "quick_compare",
]

__version__ = "1.0.0"
