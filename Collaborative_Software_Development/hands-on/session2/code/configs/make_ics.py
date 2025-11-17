# configs/make_ics.py
import numpy as np
from pathlib import Path


def make_impulse_ic(path: str, nx: int = 64, ny: int = 64) -> None:
    u0 = np.zeros((ny, nx))
    u0[ny // 2, nx // 2] = 1.0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, u0, delimiter=",")


def make_gaussian_ic(path: str, nx: int = 64, ny: int = 64, sigma: float = 0.2) -> None:
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    u0 = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, u0, delimiter=",")


if __name__ == "__main__":
    make_impulse_ic("configs/base_ic.csv")
