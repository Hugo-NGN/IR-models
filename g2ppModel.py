"""
Core G2++ model components shared by simulation and calibration modules.
"""

#TODO : Add a functionnality to calibrate phi(t) 
#       1. Fitting initial term-structure
#       2. Joint calibration (constant, linear or spline...) with the 2 factors parameters
#       3. (Optional) time varying intercept (using Kalman filter)   [could be interesting to link to functionnal data]


from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

__all__ = ["G2ppModel", "SimulationResult"]

def _default_phi_factory(r0: float) -> Callable[[float], float]:
    """Return a constant shift matching the initial short rate."""

    def _phi(_: float) -> float:
        return r0

    return _phi


@dataclass
class SimulationResult:
    x_paths: np.ndarray
    y_paths: np.ndarray
    short_rate_paths: np.ndarray
    phi_grid: np.ndarray
    time_grid: np.ndarray


def _g2pp_em_kernel(
    x_paths: np.ndarray,
    y_paths: np.ndarray,
    dW: np.ndarray,
    a: float,
    b: float,
    sigma: float,
    eta: float,
    dt: float,
    sqrt_dt: float,
) -> None:
    n_simulations, n_steps, _ = dW.shape
    for j in range(n_simulations):
        for i in range(n_steps):
            x_curr = x_paths[j, i]
            y_curr = y_paths[j, i]
            x_paths[j, i + 1] = x_curr + (-a * x_curr) * dt + sigma * sqrt_dt * dW[j, i, 0]
            y_paths[j, i + 1] = y_curr + (-b * y_curr) * dt + eta * sqrt_dt * dW[j, i, 1]


class G2ppModel:
    """
    G2++ model implementation with Monte Carlo simulation using Euler-Maruyama.
    """

    def __init__(
        self,
        a: float,
        b: float,
        sigma: float,
        eta: float,
        rho: float,
        x0: float = 0.0,
        y0: float = 0.0,
        phi: Optional[Callable[[float], float]] = None,
        r0: float = 0.02,
    ) -> None:
        """
        Parameters
        ----------
        a, b : float
            Mean-reversion speeds (>0) for factors x and y.
        sigma, eta : float
            Volatilities (>0) of factors x and y.
        rho : float
            Correlation between Brownian motions (-1 < rho < 1).
        x0, y0 : float
            Initial factor levels.
        phi : callable, optional
            Deterministic shift ensuring fit to initial curve. If None,
            defaults to a constant shift that matches r0.
        r0 : float
            Initial short rate used when phi is None.
        """
        if a <= 0 or b <= 0:
            raise ValueError("Mean-reversion speeds a and b must be positive.")
        if sigma <= 0 or eta <= 0:
            raise ValueError("Volatilities sigma and eta must be positive.")
        if not (-1 < rho < 1):
            raise ValueError("Correlation rho must lie in (-1, 1).")

        self.a = a
        self.b = b
        self.sigma = sigma
        self.eta = eta
        self.rho = rho
        self.x0 = x0
        self.y0 = y0
        self.r0 = r0
        self.phi = phi if phi is not None else _default_phi_factory(r0 - x0 - y0)

    def _phi_vector(self, time_grid: np.ndarray) -> np.ndarray:
        return np.array([self.phi(t) for t in time_grid])

    def simulate_paths(
        self,
        T: float,
        n_steps: int,
        n_simulations: int,
        random_seed: Optional[int] = None,
        return_factors: bool = True,
    ) -> SimulationResult:
        """
        Simulate factor and short-rate paths via Euler-Maruyama.

        Returns
        -------
        SimulationResult containing x, y, short-rate paths, and time grid.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        time_grid = np.linspace(0.0, T, n_steps + 1)
        phi_grid = self._phi_vector(time_grid)

        x_paths = np.zeros((n_simulations, n_steps + 1))
        y_paths = np.zeros_like(x_paths)

        x_paths[:, 0] = self.x0
        y_paths[:, 0] = self.y0

        corr_matrix = np.array([[1.0, self.rho], [self.rho, 1.0]])
        chol = np.linalg.cholesky(corr_matrix)

        z = np.random.normal(size=(n_simulations, n_steps, 2))
        dW = np.einsum("ijk,kl->ijl", z, chol.T)

        _g2pp_em_kernel(
            x_paths,
            y_paths,
            dW,
            self.a,
            self.b,
            self.sigma,
            self.eta,
            dt,
            sqrt_dt,
        )

        short_rate_paths = x_paths + y_paths + phi_grid[np.newaxis, :]

        if not return_factors:
            warnings.warn("return_factors flag will be removed in future versions.", FutureWarning)

        return SimulationResult(
            x_paths=x_paths,
            y_paths=y_paths,
            short_rate_paths=short_rate_paths,
            phi_grid=phi_grid,
            time_grid=time_grid,
        )

    def theoretical_mean(self, t: float) -> float:
        """
        The expected short rate equals phi(t) because x and y have zero mean.
        """
        return self.phi(t)

