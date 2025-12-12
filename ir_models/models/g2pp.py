"""
Core G2++ model components shared by simulation and parameter estimation modules.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

__all__ = ["G2ppModel", "SimulationResult", "G2ppZeroCouponPricing"]

def _default_phi_factory(r0: float) -> Callable[[float], float]:
    """Return a constant shift matching the initial short rate."""

    def _phi(_: float) -> float:
        return r0

    return _phi


class G2ppZeroCouponPricing:
    """
    Zero-coupon bond pricing and forward rate calculations for G2++ model.
    
    In the G2++ model, the zero-coupon bond price is given by:
        P(t, T) = A(t, T) * exp(-B1(t, T)*x1(t) - B2(t, T)*x2(t))
    
    where:
        B1(t, T) = (1 - exp(-a*(T-t))) / a
        B2(t, T) = (1 - exp(-b*(T-t))) / b
        A(t, T) = exp(integral_t^T phi(s) ds - V(t, T))
        
    and V(t, T) captures the variance contribution.
    """
    
    def __init__(
        self,
        a: float,
        b: float,
        sigma: float,
        eta: float,
        rho: float,
        phi: Callable[[float], float],
    ) -> None:
        """
        Parameters
        ----------
        a, b : float
            Mean-reversion speeds for factors x and y.
        sigma, eta : float
            Volatilities of factors x and y.
        rho : float
            Correlation between Brownian motions.
        phi : callable
            Deterministic shift function φ(t).
        """
        self.a = a
        self.b = b
        self.sigma = sigma
        self.eta = eta
        self.rho = rho
        self.phi = phi
    
    def B1(self, t: float, T: float) -> float:
        """Calculate B1(t, T) = (1 - exp(-a*(T-t))) / a."""
        tau = T - t
        if tau <= 0:
            return 0.0
        if self.a < 1e-8:
            return tau
        return (1.0 - np.exp(-self.a * tau)) / self.a
    
    def B2(self, t: float, T: float) -> float:
        """Calculate B2(t, T) = (1 - exp(-b*(T-t))) / b."""
        tau = T - t
        if tau <= 0:
            return 0.0
        if self.b < 1e-8:
            return tau
        return (1.0 - np.exp(-self.b * tau)) / self.b
    
    def V(self, t: float, T: float) -> float:
        """
        Calculate the variance contribution V(t, T) for the A(t, T) term.
        
        V(t, T) = (sigma^2)/(2*a^2) * (T - t + 2/a * exp(-a*(T-t)) - 1/(2*a) * exp(-2*a*(T-t)) - 3/(2*a))
                + (eta^2)/(2*b^2) * (T - t + 2/b * exp(-b*(T-t)) - 1/(2*b) * exp(-2*b*(T-t)) - 3/(2*b))
                + rho * sigma * eta / (a*b) * (T - t + (exp(-a*(T-t)) - 1)/a + (exp(-b*(T-t)) - 1)/b 
                                                - (exp(-(a+b)*(T-t)) - 1)/(a+b))
        """
        tau = T - t
        if tau <= 0:
            return 0.0
        
        # First factor contribution
        if self.a < 1e-8:
            v1 = (self.sigma**2 / 6.0) * tau**3
        else:
            exp_a = np.exp(-self.a * tau)
            exp_2a = np.exp(-2.0 * self.a * tau)
            v1 = (self.sigma**2) / (2.0 * self.a**2) * (
                tau 
                + 2.0 / self.a * exp_a 
                - 1.0 / (2.0 * self.a) * exp_2a 
                - 3.0 / (2.0 * self.a)
            )
        
        # Second factor contribution
        if self.b < 1e-8:
            v2 = (self.eta**2 / 6.0) * tau**3
        else:
            exp_b = np.exp(-self.b * tau)
            exp_2b = np.exp(-2.0 * self.b * tau)
            v2 = (self.eta**2) / (2.0 * self.b**2) * (
                tau 
                + 2.0 / self.b * exp_b 
                - 1.0 / (2.0 * self.b) * exp_2b 
                - 3.0 / (2.0 * self.b)
            )
        
        # Cross-correlation contribution
        if self.a < 1e-8 and self.b < 1e-8:
            v_cross = self.rho * self.sigma * self.eta * tau**3 / 3.0
        elif self.a < 1e-8:
            exp_b = np.exp(-self.b * tau)
            v_cross = self.rho * self.sigma * self.eta / self.b * (
                tau + (exp_b - 1.0) / self.b - tau / 2.0
            )
        elif self.b < 1e-8:
            exp_a = np.exp(-self.a * tau)
            v_cross = self.rho * self.sigma * self.eta / self.a * (
                tau + (exp_a - 1.0) / self.a - tau / 2.0
            )
        else:
            exp_a = np.exp(-self.a * tau)
            exp_b = np.exp(-self.b * tau)
            exp_ab = np.exp(-(self.a + self.b) * tau)
            v_cross = self.rho * self.sigma * self.eta / (self.a * self.b) * (
                tau 
                + (exp_a - 1.0) / self.a 
                + (exp_b - 1.0) / self.b 
                - (exp_ab - 1.0) / (self.a + self.b)
            )
        
        return v1 + v2 + v_cross
    
    def phi_integral(self, t: float, T: float, n_points: int = 100) -> float:
        """
        Numerical integration of phi from t to T using trapezoidal rule.
        """
        tau = T - t
        if tau <= 0:
            return 0.0
        
        s_grid = np.linspace(t, T, n_points)
        phi_values = np.array([self.phi(s) for s in s_grid])
        return np.trapz(phi_values, s_grid)
    
    def A(self, t: float, T: float) -> float:
        """
        Calculate A(t, T) = exp(integral_t^T phi(s) ds - V(t, T)).
        """
        phi_int = self.phi_integral(t, T)
        v = self.V(t, T)
        return np.exp(phi_int - v)
    
    def zero_coupon_price(
        self, 
        t: float, 
        T: float, 
        x1: float, 
        x2: float
    ) -> float:
        """
        Calculate zero-coupon bond price P(t, T).
        
        Parameters
        ----------
        t : float
            Current time.
        T : float
            Maturity time.
        x1, x2 : float
            Current values of the two factors.
        
        Returns
        -------
        float
            Zero-coupon bond price P(t, T).
        """
        if T <= t:
            return 1.0
        
        b1 = self.B1(t, T)
        b2 = self.B2(t, T)
        a_val = self.A(t, T)
        
        return a_val * np.exp(-b1 * x1 - b2 * x2)
    
    def euribor_rate(
        self,
        t: float,
        T: float,
        delta: float,
        x1: float,
        x2: float,
    ) -> float:
        """
        Calculate EURIBOR rate L(t; T, T+δ) = (1/δ) * (P(t,T) / P(t,T+δ) - 1).
        
        Parameters
        ----------
        t : float
            Current time.
        T : float
            Start date of the forward period.
        delta : float
            Tenor in years (e.g., 0.25 for 3M).
        x1, x2 : float
            Current values of the two factors.
        
        Returns
        -------
        float
            Forward LIBOR/EURIBOR rate.
        """
        P_T = self.zero_coupon_price(t, T, x1, x2)
        P_T_delta = self.zero_coupon_price(t, T + delta, x1, x2)
        
        if P_T_delta <= 0:
            return 0.0
        
        return (1.0 / delta) * (P_T / P_T_delta - 1.0)


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

