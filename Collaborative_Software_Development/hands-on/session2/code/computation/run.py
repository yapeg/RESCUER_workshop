# computation/run.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .io_utils import (
    SimulationConfig,
    load_simple_config,
    read_initial_condition,
    write_field_csv,
)
from .plotting import plot_field
from .solver import (
    apply_dirichlet_boundary,
    max_stable_dt,
    step_advection_diffusion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D advection–diffusion toy model for workshop."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.csv",
        help="Path to simulation config CSV.",
    )
    parser.add_argument(
        "--ic",
        type=str,
        default="configs/base_ic.csv",
        help="Path to initial condition CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for CSV and plots.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting of the final field.",
    )
    return parser.parse_args()


def run_simulation(
    cfg: SimulationConfig,
    u0: np.ndarray,
    output_dir: str | Path,
    make_plot: bool = True,
) -> None:
    """
    Run the advection–diffusion simulation.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation parameters.
    u0 : ndarray, shape (ny, nx)
        Initial condition.
    output_dir : str or Path
        Output directory.
    make_plot : bool
        Whether to generate a plot of the final field.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ny, nx = u0.shape
    if nx != cfg.nx or ny != cfg.ny:
        raise ValueError(
            f"Initial condition shape {(ny, nx)} does not match "
            f"config (ny={cfg.ny}, nx={cfg.nx})."
        )

    # Stability check for diffusion part
    if cfg.D > 0.0:
        dt_max = max_stable_dt(cfg.D, dx=1.0, dy=1.0)
        if cfg.dt > dt_max:
            raise ValueError(
                f"Unstable time step: dt={cfg.dt} > dt_max={dt_max:.3g}. "
                "Reduce dt or D."
            )

    u = u0.copy()
    u = apply_dirichlet_boundary(u, value=0.0)
    num_outs=0
    csv_path = output_dir / f"field_{num_outs}.csv"
    write_field_csv(csv_path, u)
    num_outs += 1

    for j in range(cfg.nsteps):
        u = step_advection_diffusion(
            u,
            D=cfg.D,
            vx=cfg.vx,
            vy=cfg.vy,
            dt=cfg.dt,
            dx=1.0,
            dy=1.0,
        )
        u = apply_dirichlet_boundary(u, value=0.0)

        if j % 200 == 0:
            csv_path = output_dir / f"field_{j}.csv"
            write_field_csv(csv_path, u)

            if make_plot:
                plot_path = output_dir / f"field_{j}.png"
            title = (
                f"Advection–diffusion (D={cfg.D}, vx={cfg.vx}, vy={cfg.vy}, "
                    f"dt={cfg.dt}, step={j})"
            )
            plot_field(u, out_path=plot_path, title=title)

def main() -> None:
    args = parse_args()
    cfg = load_simple_config(args.config)
    u0 = read_initial_condition(args.ic)
    run_simulation(cfg, u0, args.output_dir, make_plot=not args.no_plot)


if __name__ == "__main__":
    main()
