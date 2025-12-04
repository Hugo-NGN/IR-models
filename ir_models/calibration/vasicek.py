"""
Calibrate the Vasicek model to historical short-rate data using MLE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

try:
    from scipy.optimize import minimize
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for Vasicek calibration. Install via `pip install scipy`."
    ) from exc

from ir_models.models.vasicek import VasicekModel


@dataclass
class VasicekCalibrationResult:
    params: Dict[str, float]
    log_likelihood: float
    optimized: bool
    message: str
    fitted_model: VasicekModel
    residuals: np.ndarray


class VasicekMLECalibrator:
    """
    Maximum-likelihood Vasicek calibration for equally spaced observations.
    """

    def __init__(self, rates: Sequence[float], dt: float) -> None:
        rates_array = np.asarray(rates, dtype=float)
        if rates_array.ndim != 1:
            raise ValueError("rates must be a 1D sequence.")
        if rates_array.size < 2:
            raise ValueError("Need at least two observations to calibrate the model.")
        if dt <= 0:
            raise ValueError("dt must be positive.")

        self.rates = rates_array
        self.dt = dt
        self._x = rates_array[:-1]
        self._y = rates_array[1:]

    # ------------------------------------------------------------------ #
    # Likelihood helpers
    # ------------------------------------------------------------------ #
    def _transition_mean(self, kappa: float, theta: float) -> np.ndarray:
        exp_term = np.exp(-kappa * self.dt)
        return theta + (self._x - theta) * exp_term

    def _transition_variance(self, kappa: float, sigma: float) -> float:
        return (sigma**2 / (2.0 * kappa)) * (1.0 - np.exp(-2.0 * kappa * self.dt))

    def _log_likelihood(self, params: Dict[str, float]) -> float:
        kappa = params["kappa"]
        theta = params["theta"]
        sigma = params["sigma"]

        if kappa <= 0 or sigma <= 0:
            return -np.inf

        mean = self._transition_mean(kappa, theta)
        var = self._transition_variance(kappa, sigma)
        if var <= 0 or not np.isfinite(var):
            return -np.inf

        residuals = self._y - mean
        log_like = -0.5 * np.sum(np.log(2.0 * np.pi * var) + residuals**2 / var)
        return log_like

    def _negative_log_likelihood(self, vector: np.ndarray) -> float:
        params = {"kappa": vector[0], "theta": vector[1], "sigma": vector[2]}
        log_like = self._log_likelihood(params)
        if not np.isfinite(log_like):
            return 1e6
        return -log_like

    def _residuals(self, params: Dict[str, float]) -> np.ndarray:
        mean = self._transition_mean(params["kappa"], params["theta"])
        return self._y - mean

    # ------------------------------------------------------------------ #
    # Initial guess using OLS on AR(1) form
    # ------------------------------------------------------------------ #
    def ols_initial_guess(self) -> Dict[str, float]:
        x = self._x
        y = self._y
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        beta = numerator / denominator if denominator > 0 else 0.99
        beta = np.clip(beta, 1e-6, 0.9999)
        alpha = y_mean - beta * x_mean

        kappa_guess = -np.log(beta) / self.dt
        theta_guess = alpha / (1.0 - beta)

        fitted = alpha + beta * x
        residuals = y - fitted
        variance = np.var(residuals, ddof=1)
        variance = max(variance, 1e-8)
        sigma_guess = np.sqrt(
            variance * 2.0 * kappa_guess / (1.0 - np.exp(-2.0 * kappa_guess * self.dt))
        )

        return {
            "kappa": float(kappa_guess),
            "theta": float(theta_guess),
            "sigma": float(max(sigma_guess, 1e-4)),
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(
        self,
        initial_guess: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, tuple[float, float]]] = None,
        optimizer_kwargs: Optional[Dict[str, float]] = None,
    ) -> VasicekCalibrationResult:
        if initial_guess is None:
            initial_guess = self.ols_initial_guess()

        guess_vec = np.array(
            [
                initial_guess.get("kappa", 0.5),
                initial_guess.get("theta", np.mean(self.rates)),
                initial_guess.get("sigma", 0.02),
            ],
            dtype=float,
        )

        if bounds is None:
            bounds = {
                "kappa": (1e-4, 5.0),
                "theta": (-0.1, 0.5),
                "sigma": (1e-4, 1.0),
            }

        bound_list = [bounds["kappa"], bounds["theta"], bounds["sigma"]]

        opt_kwargs = {"method": "L-BFGS-B"}
        if optimizer_kwargs:
            opt_kwargs.update(optimizer_kwargs)

        result = minimize(
            fun=self._negative_log_likelihood,
            x0=guess_vec,
            bounds=bound_list,
            **opt_kwargs,
        )

        params = {
            "kappa": result.x[0],
            "theta": result.x[1],
            "sigma": result.x[2],
        }
        log_like = self._log_likelihood(params)
        residuals = self._residuals(params)

        fitted_model = VasicekModel(
            kappa=params["kappa"],
            theta=params["theta"],
            sigma=params["sigma"],
            r0=self.rates[0],
        )

        return VasicekCalibrationResult(
            params=params,
            log_likelihood=log_like,
            optimized=result.success,
            message=result.message,
            fitted_model=fitted_model,
            residuals=residuals,
        )


# ---------------------------------------------------------------------- #
# Helpers for demos / testing
# ---------------------------------------------------------------------- #
def generate_synthetic_rates(
    model: VasicekModel,
    T: float,
    n_steps: int,
    measurement_std: float = 0.0,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    if random_seed is not None:
        np.random.seed(random_seed)
    paths = model.simulate_path(T=T, n_steps=n_steps, n_simulations=1)
    rates = paths[0]
    if measurement_std > 0:
        noise = np.random.normal(scale=measurement_std, size=rates.shape)
        rates = rates + noise
    return rates


def example_calibration() -> VasicekCalibrationResult:
    true_model = VasicekModel(kappa=0.4, theta=0.04, sigma=0.015, r0=0.03)
    dt = 1.0 / 252.0
    n_steps = 1000
    rates = generate_synthetic_rates(
        model=true_model,
        T=dt * n_steps,
        n_steps=n_steps,
        measurement_std=0.0005,
        random_seed=123,
    )

    calibrator = VasicekMLECalibrator(rates=rates, dt=dt)
    result = calibrator.fit()

    print("Calibration success:", result.optimized)
    print("Message:", result.message)
    print("Log-likelihood:", result.log_likelihood)
    for name, value in result.params.items():
        print(f"  {name}: {value:.6f}")
    return result


if __name__ == "__main__":
    example_calibration()

