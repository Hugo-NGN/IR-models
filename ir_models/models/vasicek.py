"""
Core Vasicek model implementation shared by simulation and parameter estimation modules.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["VasicekModel"]


class VasicekModel:
    """
    Classic Vasicek one-factor short-rate model.
    """

    def __init__(self, kappa: float, theta: float, sigma: float, r0: float) -> None:
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0

    def simulate_path(
        self,
        T: float,
        n_steps: int,
        n_simulations: int = 1,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate interest-rate paths using Euler-Maruyama discretisation.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = self.r0

        dW = np.random.normal(0.0, 1.0, size=(n_simulations, n_steps))

        for i in range(n_steps):
            r_current = paths[:, i]
            drift = self.kappa * (self.theta - r_current) * dt
            diffusion = self.sigma * sqrt_dt * dW[:, i]
            paths[:, i + 1] = r_current + drift + diffusion

        return paths

    def simulate_path_vectorized(
        self,
        T: float,
        n_steps: int,
        n_simulations: int = 1,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Vectorised simulation helper (same interface as simulate_path).
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = self.r0

        dW = np.random.normal(0.0, 1.0, size=(n_simulations, n_steps))

        for i in range(n_steps):
            r_current = paths[:, i]
            paths[:, i + 1] = (
                r_current
                + self.kappa * (self.theta - r_current) * dt
                + self.sigma * sqrt_dt * dW[:, i]
            )

        return paths

    def get_time_grid(self, T: float, n_steps: int) -> np.ndarray:
        return np.linspace(0.0, T, n_steps + 1)

    def theoretical_mean(self, t: float) -> float:
        return self.theta + (self.r0 - self.theta) * np.exp(-self.kappa * t)

    def theoretical_variance(self, t: float) -> float:
        return (self.sigma**2 / (2.0 * self.kappa)) * (1.0 - np.exp(-2.0 * self.kappa * t))

