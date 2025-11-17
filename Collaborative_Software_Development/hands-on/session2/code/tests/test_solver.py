# tests/test_solver.py
import numpy as np

from computation.solver import (
    apply_dirichlet_boundary,
    max_stable_dt,
    step_advection_diffusion,
)


def test_nonnegative_field_under_stable_diffusion():
    ny, nx = 32, 32
    u = np.zeros((ny, nx))
    u[ny // 2, nx // 2] = 1.0

    D = 0.1
    dx = dy = 1.0
    dt = 0.5 * max_stable_dt(D, dx, dy)  # safe

    u = apply_dirichlet_boundary(u, value=0.0)
    for _ in range(50):
        u = step_advection_diffusion(u, D=D, vx=0.0, vy=0.0, dt=dt, dx=dx, dy=dy)
        u = apply_dirichlet_boundary(u, value=0.0)

    assert np.all(u >= -1e-12)


def test_impulse_spreads_but_total_mass_decreases_with_dirichlet():
    ny, nx = 32, 32
    u = np.zeros((ny, nx))
    u[ny // 2, nx // 2] = 1.0

    D = 0.1
    dx = dy = 1.0
    dt = 0.5 * max_stable_dt(D, dx, dy)

    u = apply_dirichlet_boundary(u, value=0.0)
    initial_mass = u.sum()

    for _ in range(50):
        u = step_advection_diffusion(u, D=D, vx=0.0, vy=0.0, dt=dt, dx=dx, dy=dy)
        u = apply_dirichlet_boundary(u, value=0.0)

    assert u.sum() < initial_mass
    assert u[ny // 2, nx // 2] < 1.0
