"""
Parameter estimation for the G2++ model using a Kalman filter and
maximum-likelihood approach.

The implementation assumes we observe (possibly noisy) short rates
`r_obs(t_i)` sampled at regular intervals `dt`. The short rate is
modelled as
    r(t) = x(t) + y(t) + phi(t)
with (correlated) Ornstein-Uhlenbeck factors x, y governed by

    dx = -a x dt + sigma dW1
    dy = -b y dt + eta  dW2,   corr[dW1, dW2] = rho.
"""

#TODO : Complete the methods to caibrate phi(t) 


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import argparse
import warnings

try:
    from scipy.optimize import minimize  # type: ignore
    _SCIPY_AVAILABLE = True
except Exception:
    # Defer import error until optimisation is actually requested so users can
    # inspect CLI help or use non-optimisation parts of the module.
    minimize = None  # type: ignore
    _SCIPY_AVAILABLE = False

from g2ppModel import G2ppModel


def _default_phi(_: float) -> float:
    """Default shift function returning zero."""
    return 0.0


def _stationary_variance(kappa: float, vol: float) -> float:
    return vol**2 / (2.0 * kappa)


def _discrete_ou_coefficients(kappa: float, vol: float, dt: float) -> float:
    """
    Standard deviation of the OU increment over dt (zero-drift component).
    """
    return vol * math.sqrt((1.0 - math.exp(-2.0 * kappa * dt)) / (2.0 * kappa))


def _transition_matrices(
    a: float, b: float, sigma: float, eta: float, rho: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact discretisation for two correlated OU factors.

    Returns
    -------
    F : np.ndarray
        State transition matrix (2x2, diagonal).
    Q : np.ndarray
        Process noise covariance matrix.
    """
    f1 = math.exp(-a * dt)
    f2 = math.exp(-b * dt)
    F = np.array([[f1, 0.0], [0.0, f2]])

    std_x = _discrete_ou_coefficients(a, sigma, dt)
    std_y = _discrete_ou_coefficients(b, eta, dt)

    Q = np.array(
        [
            [std_x**2, rho * std_x * std_y],
            [rho * std_x * std_y, std_y**2],
        ]
    )
    return F, Q


@dataclass
class EstimationResult:
    params: Dict[str, float]
    log_likelihood: float
    optimized: bool
    message: str
    filtered_states: np.ndarray
    state_covariances: np.ndarray


class G2ppKalmanMLE:
    """
    Calibrate G2++ parameters from short-rate observations using MLE.
    """

    def __init__(
        self,
        observations: Sequence[float],
        dt: float,
        phi: Optional[Callable[[float], float]] = None,
        measurement_var: float = 1e-6,
    ) -> None:
        if dt <= 0:
            raise ValueError("Time step dt must be positive.")
        self.observations = np.asarray(observations, dtype=float)
        if self.observations.ndim != 1:
            raise ValueError("Observations must be a 1D sequence.")
        self.n_obs = self.observations.size
        self.dt = dt
        self.phi = phi if phi is not None else _default_phi
        self.measurement_var = measurement_var

    # --------------------------------------------------------------------- #
    # Kalman filter
    # --------------------------------------------------------------------- #
    def _initialize_state(self, a: float, b: float, sigma: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
        """Use stationary distribution as prior."""
        mean = np.zeros(2, dtype=float)
        var_x = _stationary_variance(a, sigma)
        var_y = _stationary_variance(b, eta)
        cov = np.diag([var_x, var_y])
        return mean, cov

    def _kalman_filter(
        self,
        params: Dict[str, float],
    ) -> tuple[float, np.ndarray, np.ndarray]:
        a = params["a"]
        b = params["b"]
        sigma = params["sigma"]
        eta = params["eta"]
        rho = params["rho"]
        meas_var = params["measurement_var"]

        if not (-0.999 < rho < 0.999):
            return -np.inf, np.empty((0, 2)), np.empty((0, 2, 2))

        F, Q = _transition_matrices(a, b, sigma, eta, rho, self.dt)
        state_mean, state_cov = self._initialize_state(a, b, sigma, eta)

        H = np.array([[1.0, 1.0]])  # observation matrix
        R = np.array([[meas_var]])
        log_likelihood = 0.0

        filtered_means = np.zeros((self.n_obs, 2))
        filtered_covs = np.zeros((self.n_obs, 2, 2))

        phi_values = np.array([self.phi(i * self.dt) for i in range(self.n_obs)])

        for idx, obs in enumerate(self.observations):
            pred_mean = state_mean
            pred_cov = state_cov

            obs_pred = (H @ pred_mean)[0] + phi_values[idx]
            innovation = obs - obs_pred

            S = H @ pred_cov @ H.T + R
            S_scalar = S[0, 0]
            if S_scalar <= 0:
                return -np.inf, filtered_means, filtered_covs

            log_likelihood -= 0.5 * (
                np.log(2.0 * np.pi)
                + np.log(S_scalar)
                + innovation**2 / S_scalar
            )

            K = (pred_cov @ H.T) / S_scalar  # 2x1
            state_mean = pred_mean + (K.flatten() * innovation)
            state_cov = pred_cov - K @ H @ pred_cov

            filtered_means[idx] = state_mean
            filtered_covs[idx] = state_cov

            # Propagate to next step if not last observation
            if idx < self.n_obs - 1:
                state_mean = F @ state_mean
                state_cov = F @ state_cov @ F.T + Q

        return log_likelihood, filtered_means, filtered_covs

    # ------------------------------------------------------------------ #
    # Optimization interface
    # ------------------------------------------------------------------ #
    @staticmethod
    def _param_dict_from_vector(vec: np.ndarray) -> Dict[str, float]:
        a, b, sigma, eta, rho, meas_var = vec
        return {
            "a": float(a),
            "b": float(b),
            "sigma": float(sigma),
            "eta": float(eta),
            "rho": float(rho),
            "measurement_var": float(meas_var),
        }

    def _negative_log_likelihood(self, vec: np.ndarray) -> float:
        params = self._param_dict_from_vector(vec)
        log_like, _, _ = self._kalman_filter(params)
        if not np.isfinite(log_like):
            return 1e6
        return -log_like

    def fit(
        self,
        initial_guess: Dict[str, float],
        bounds: Optional[Dict[str, tuple[float, float]]] = None,
        optimizer_kwargs: Optional[Dict[str, float]] = None,
    ) -> EstimationResult:
        """
        Run numerical optimisation to maximise the Kalman log-likelihood.
        """
        guess_vec = np.array(
            [
                initial_guess.get("a", 0.1),
                initial_guess.get("b", 0.3),
                initial_guess.get("sigma", 0.02),
                initial_guess.get("eta", 0.015),
                initial_guess.get("rho", 0.0),
                initial_guess.get("measurement_var", self.measurement_var),
            ],
            dtype=float,
        )

        if bounds is None:
            bounds = {
                "a": (1e-4, 5.0),
                "b": (1e-4, 5.0),
                "sigma": (1e-4, 0.2),
                "eta": (1e-4, 0.2),
                "rho": (-0.999, 0.999),
                "measurement_var": (1e-8, 0.01),
            }

        bound_list = [
            bounds["a"],
            bounds["b"],
            bounds["sigma"],
            bounds["eta"],
            bounds["rho"],
            bounds["measurement_var"],
        ]

        opt_kwargs = {"method": "L-BFGS-B"}
        if optimizer_kwargs:
            opt_kwargs.update(optimizer_kwargs)

        if not _SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for G2++ calibration. Install via `pip install scipy`."
            )

        result = minimize(
            fun=self._negative_log_likelihood,
            x0=guess_vec,
            bounds=bound_list,
            **opt_kwargs,
        )

        params = self._param_dict_from_vector(result.x)
        log_like, filtered_means, filtered_covs = self._kalman_filter(params)

        return EstimationResult(
            params=params,
            log_likelihood=log_like,
            optimized=result.success,
            message=result.message,
            filtered_states=filtered_means,
            state_covariances=filtered_covs,
        )


# ---------------------------------------------------------------------- #
# Utility to generate synthetic data & demonstrate calibration workflow
# ---------------------------------------------------------------------- #
def generate_synthetic_short_rates(
    true_params: Dict[str, float],
    dt: float,
    n_steps: int,
    measurement_std: float = 0.001,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Create synthetic short-rate data using the simulation module.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    model = G2ppModel(
        a=true_params["a"],
        b=true_params["b"],
        sigma=true_params["sigma"],
        eta=true_params["eta"],
        rho=true_params["rho"],
        x0=true_params.get("x0", 0.0),
        y0=true_params.get("y0", 0.0),
        r0=true_params.get("r0", 0.02),
    )

    sim = model.simulate_paths(
        T=dt * n_steps,
        n_steps=n_steps,
        n_simulations=1,
        random_seed=random_seed,
    )
    short_rates = sim.short_rate_paths[0]
    noise = np.random.normal(scale=measurement_std, size=short_rates.shape)
    return short_rates + noise


def example_estimation(phi_method: str = "fit_term_structure") -> EstimationResult:
    """
    Example workflow: generate data, run Kalman MLE, print results.
    """
    true_params = {
        "a": 0.2,
        "b": 0.5,
        "sigma": 0.03,
        "eta": 0.02,
        "rho": -0.6,
        "r0": 0.025,
    }
    dt = 1.0 / 12.0  # monthly
    n_steps = 600

    observations = generate_synthetic_short_rates(
        true_params=true_params,
        dt=dt,
        n_steps=n_steps,
        measurement_std=0.001,
        random_seed=7,
    )

    # Default behaviour: fit a deterministic shift phi(t) that captures the
    # initial term-structure. The user can choose another method via CLI.
    def _compute_phi_from_method(obs: np.ndarray, dt: float, method: str) -> Callable[[float], float]:
        """
        Return a callable phi(t) based on the requested method.

        Currently supported methods:
        - "fit_term_structure": a simple smoothed/interpolated version of the
          observed short-rate series (default for demonstrations).
        - "joint": not implemented; falls back to "fit_term_structure" with a warning.
        - "kalman_time_varying": not implemented; falls back to "fit_term_structure" with a warning.
        """
        n = obs.size
        times = np.arange(n) * dt

        if method == "fit_term_structure":
            # simple smoothing via moving average then linear interpolation
            window = max(3, int(0.02 * n))  # small window or at least 3
            kernel = np.ones(window) / window
            smooth = np.convolve(obs, kernel, mode="same")

            # Adjust the entire smooth curve so that it matches the last
            # observed short-rate (i.e. respect the last observed term
            # structure). This applies a constant shift equal to the
            # difference between the last observation and the smoothed
            # value at the last time.
            if n > 0:
                offset = float(obs[-1] - smooth[-1])
                smooth = smooth + offset

            def phi(t: float) -> float:
                return float(np.interp(t, times, smooth))

            return phi

        if method == "joint":
            warnings.warn(
                "Joint calibration for phi is not implemented; falling back to fit_term_structure"
            )
            return _compute_phi_from_method(obs, dt, "fit_term_structure")

        if method == "kalman_time_varying":
            warnings.warn(
                "Time-varying Kalman calibration for phi is not implemented; falling back to fit_term_structure"
            )
            return _compute_phi_from_method(obs, dt, "fit_term_structure")

        raise ValueError(f"Unknown phi calibration method: {method}")

    phi_func = _compute_phi_from_method(observations, dt, phi_method)

    estimator = G2ppKalmanMLE(
        observations=observations,
        dt=dt,
        phi=phi_func,
        measurement_var=1e-6,
    )

    initial_guess = {
        "a": 0.1,
        "b": 0.4,
        "sigma": 0.02,
        "eta": 0.015,
        "rho": -0.4,
        "measurement_var": 1e-4,
    }

    result = estimator.fit(initial_guess=initial_guess)
    print("Optimisation success:", result.optimized)
    print("Message:", result.message)
    print("Log-likelihood:", result.log_likelihood)
    for key, value in result.params.items():
        print(f"  {key}: {value:.6f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G2++ calibration example")
    parser.add_argument(
        "--phi-method",
        type=str,
        default="fit_term_structure",
        choices=["fit_term_structure", "joint", "kalman_time_varying"],
        help="Method to calibrate phi(t). Default: fit_term_structure",
    )
    args = parser.parse_args()

    # Run example with chosen phi calibration method
    example_estimation(phi_method=args.phi_method)

