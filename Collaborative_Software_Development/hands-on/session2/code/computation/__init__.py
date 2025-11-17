# computation/__init__.py
"""
Small 2D advection–diffusion toy model for the collaboration workshop.
"""

from .solver import (
    max_stable_dt,
    apply_dirichlet_boundary,
    step_advection_diffusion,
)
