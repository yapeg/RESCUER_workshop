# computation/io_utils.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class SimulationConfig:
    nx: int
    ny: int
    D: float
    vx: float
    vy: float
    dt: float
    nsteps: int


def read_initial_condition(path: str | Path) -> np.ndarray:
    """
    Read a 2D initial condition from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to a CSV file containing a rectangular array of floats.

    Returns
    -------
    ndarray
        2D array of shape (ny, nx) with the initial field.
    """
    path = Path(path)
    data = np.loadtxt(path, delimiter=",")
    return data


def write_field_csv(path: str | Path, u: np.ndarray) -> None:
    """
    Write a 2D field to a CSV file.

    Parameters
    ----------
    path : str or Path
        Output CSV path.
    u : ndarray
        2D array to write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, u, delimiter=",")


def load_simple_config(path: str | Path) -> SimulationConfig:
    """
    Load a minimal configuration from a CSV-like file.

    Expects a file with header 'key,value' and rows like:
        nx,64
        ny,64
        D,0.1
        vx,0.5
        vy,0.0
        dt,0.01
        nsteps,200

    Parameters
    ----------
    path : str or Path
        Path to the config CSV.

    Returns
    -------
    SimulationConfig
        Parsed configuration.
    """
    path = Path(path)
    params: dict[str, float] = {}
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            key, value = row[0].strip(), row[1].strip()
            params[key] = float(value)

    return SimulationConfig(
        nx=int(params["nx"]),
        ny=int(params["ny"]),
        D=float(params["D"]),
        vx=float(params["vx"]),
        vy=float(params["vy"]),
        dt=float(params["dt"]),
        nsteps=int(params["nsteps"]),
    )
