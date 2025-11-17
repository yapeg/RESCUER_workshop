# computation/solver.py
import numpy as np


def max_stable_dt(D: float, dx: float, dy: float) -> float:
    """
    Compute a conservative stability bound for 2D explicit diffusion.

    Parameters
    ----------
    D : float
        Diffusion coefficient.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    float
        Maximum stable time step for the explicit 2D scheme.
    """
    return 0.5 * (dx * dx * dy * dy) / (D * (dx * dx + dy * dy))


def apply_dirichlet_boundary(u: np.ndarray, value: float = 0.0) -> np.ndarray:
    """
    Apply homogeneous Dirichlet boundary conditions to the field.

    Parameters
    ----------
    u : ndarray, shape (ny, nx)
        Concentration field to modify in place.
    value : float, default 0.0
        Boundary value.

    Returns
    -------
    ndarray
        The modified concentration field (same object as input).
    """
    u[0, :] = value
    u[-1, :] = value
    u[:, 0] = value
    u[:, -1] = value
    return u


def step_advection_diffusion(
    u: np.ndarray,
    D: float,
    vx: float,
    vy: float,
    dt: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Advance the 2D advection–diffusion equation by one explicit time step.

    The PDE is:
        du/dt = D (d2u/dx2 + d2u/dy2) - vx du/dx - vy du/dy

    Parameters
    ----------
    u : ndarray of shape (ny, nx)
        Current concentration field.
    D : float
        Diffusion coefficient.
    vx : float
        Advection velocity in x-direction.
    vy : float
        Advection velocity in y-direction.
    dt : float
        Time step size.
    dx : float
        Grid spacing in x-direction.
    dy : float
        Grid spacing in y-direction.

    Returns
    -------
    ndarray
        Updated concentration field after one explicit time step
        (a new array is returned; the input is not modified).

    Notes
    -----
    * Diffusion: 5-point Laplacian stencil.
    * Advection: simple upwind scheme based on the sign of vx, vy.
    * Boundary conditions are not applied here; call `apply_dirichlet_boundary`
      outside this function if needed.
    """
    ny, nx = u.shape
    unew = u.copy()

    # Second derivatives (diffusion)
    d2udx2 = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / (dx * dx)
    d2udy2 = (u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :]) / (dy * dy)

    # Central region indices
    i_slice = slice(1, ny - 1)
    j_slice = slice(1, nx - 1)

    # Start with diffusion contribution
    unew[i_slice, j_slice] += dt * (
        D * (d2udx2[1:-1, :] + d2udy2[:, 1:-1])
    )

    # Advection (upwind)
    # du/dx
    dudx = np.zeros_like(u)
    if vx >= 0.0:
        dudx[:, 1:] = (u[:, 1:] - u[:, :-1]) / dx
    else:
        dudx[:, :-1] = (u[:, 1:] - u[:, :-1]) / dx

    # du/dy
    dudy = np.zeros_like(u)
    if vy >= 0.0:
        dudy[1:, :] = (u[1:, :] - u[:-1, :]) / dy
    else:
        dudy[:-1, :] = (u[1:, :] - u[:-1, :]) / dy

    unew[i_slice, j_slice] += -dt * (
        vx * dudx[i_slice, j_slice] + vy * dudy[i_slice, j_slice]
    )

    return unew
