"""
Parameter estimation for the G2++ model using a Kalman filter and
maximum-likelihood approach.

The implementation can work in two modes:

1. SHORT_RATE mode (legacy): Observes short rates r_obs(t_i) = x(t) + y(t) + phi(t)
   
2. EURIBOR mode: Observes EURIBOR rates (e.g., 3M) which are forward rates
   calculated from zero-coupon bond prices:
       L(t; T, T+δ) = (1/δ) * (P(t,T) / P(t,T+δ) - 1)
   
   This is the correct approach for calibrating to market EURIBOR data,
   as EURIBOR is not a short rate but a discrete forward rate.

The factors x, y are (correlated) Ornstein-Uhlenbeck processes:
    dx = -a x dt + sigma dW1
    dy = -b y dt + eta  dW2,   corr[dW1, dW2] = rho.
"""

#TODO : Complete the methods to calibrate phi(t) 


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

from ir_models.models.g2pp import G2ppModel, G2ppZeroCouponPricing


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
    Estimate G2++ parameters from observations using MLE via Kalman filter.
    
    Supports two observation modes:
    - 'short_rate': Direct observation of short rate r(t) = x(t) + y(t) + phi(t)
    - 'euribor': Observation of EURIBOR forward rates L(t; T, T+δ)
    """

    def __init__(
        self,
        observations: Sequence[float],
        dt: float,
        phi: Optional[Callable[[float], float]] = None,
        measurement_var: float = 1e-6,
        observation_mode: str = "euribor",
        euribor_tenor: float = 0.25,
        euribor_start_offset: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        observations : Sequence[float]
            Observed rates (either short rates or EURIBOR rates depending on mode).
        dt : float
            Time step between observations (in years).
        phi : callable, optional
            Deterministic shift function φ(t). Defaults to zero.
        measurement_var : float
            Initial guess for measurement noise variance.
        observation_mode : str
            Either 'short_rate' or 'euribor' (default: 'euribor').
        euribor_tenor : float
            Tenor δ for EURIBOR in years (e.g., 0.25 for 3M). Only used in euribor mode.
        euribor_start_offset : float
            Time offset T in L(t; T, T+δ). Usually 0 for spot-starting EURIBOR.
        """
        if dt <= 0:
            raise ValueError("Time step dt must be positive.")
        self.observations = np.asarray(observations, dtype=float)
        if self.observations.ndim != 1:
            raise ValueError("Observations must be a 1D sequence.")
        self.n_obs = self.observations.size
        self.dt = dt
        self.phi = phi if phi is not None else _default_phi
        self.measurement_var = measurement_var
        
        if observation_mode not in ["short_rate", "euribor"]:
            raise ValueError("observation_mode must be 'short_rate' or 'euribor'")
        self.observation_mode = observation_mode
        self.euribor_tenor = euribor_tenor
        self.euribor_start_offset = euribor_start_offset

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

    def _observation_function(
        self,
        state: np.ndarray,
        t: float,
        pricing: G2ppZeroCouponPricing,
    ) -> float:
        """
        Compute predicted observation given state [x1, x2].
        
        - In 'short_rate' mode: r(t) = x1 + x2 + phi(t)
        - In 'euribor' mode: L(t; T, T+δ) via zero-coupon prices
        """
        x1, x2 = state
        
        if self.observation_mode == "short_rate":
            return x1 + x2 + self.phi(t)
        elif self.observation_mode == "euribor":
            T_start = t + self.euribor_start_offset
            return pricing.euribor_rate(t, T_start, self.euribor_tenor, x1, x2)
        else:
            raise ValueError(f"Unknown observation mode: {self.observation_mode}")
    
    def _observation_jacobian(
        self,
        state: np.ndarray,
        t: float,
        pricing: G2ppZeroCouponPricing,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute Jacobian of observation function w.r.t. state via finite differences.
        Returns H = [∂h/∂x1, ∂h/∂x2] as a (1, 2) array.
        """
        x1, x2 = state
        h0 = self._observation_function(state, t, pricing)
        
        # Perturb x1
        state_dx1 = np.array([x1 + epsilon, x2])
        h_dx1 = self._observation_function(state_dx1, t, pricing)
        dh_dx1 = (h_dx1 - h0) / epsilon
        
        # Perturb x2
        state_dx2 = np.array([x1, x2 + epsilon])
        h_dx2 = self._observation_function(state_dx2, t, pricing)
        dh_dx2 = (h_dx2 - h0) / epsilon
        
        return np.array([[dh_dx1, dh_dx2]])

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

        R = np.array([[meas_var]])
        log_likelihood = 0.0

        filtered_means = np.zeros((self.n_obs, 2))
        filtered_covs = np.zeros((self.n_obs, 2, 2))

        # Create pricing object for EURIBOR calculations
        pricing = G2ppZeroCouponPricing(a, b, sigma, eta, rho, self.phi)

        for idx, obs in enumerate(self.observations):
            t = idx * self.dt
            pred_mean = state_mean
            pred_cov = state_cov

            # Compute predicted observation
            if self.observation_mode == "short_rate":
                # Linear observation: h(x) = x1 + x2 + phi(t)
                H = np.array([[1.0, 1.0]])
                obs_pred = self._observation_function(pred_mean, t, pricing)
            elif self.observation_mode == "euribor":
                # Nonlinear observation: use Extended Kalman Filter
                obs_pred = self._observation_function(pred_mean, t, pricing)
                H = self._observation_jacobian(pred_mean, t, pricing)
            else:
                raise ValueError(f"Unknown observation mode: {self.observation_mode}")

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
    
    def _param_dict_from_optimized_vector(self, vec: np.ndarray) -> Dict[str, float]:
        """
        Reconstruct full parameter dictionary from optimization vector,
        combining optimized parameters with fixed parameters.
        """
        params = {}
        # Fill in optimized parameters
        for i, name in enumerate(self.param_names_to_optimize):
            params[name] = float(vec[i])
        # Fill in fixed parameters
        for name, value in self.fixed_params.items():
            params[name] = float(value)
        return params

    def _negative_log_likelihood(self, vec: np.ndarray) -> float:
        params = self._param_dict_from_optimized_vector(vec)
        log_like, _, _ = self._kalman_filter(params)
        if not np.isfinite(log_like):
            return 1e6
        return -log_like

    def fit(
        self,
        initial_guess: Dict[str, float],
        bounds: Optional[Dict[str, tuple[float, float]]] = None,
        optimizer_kwargs: Optional[Dict[str, float]] = None,
        fixed_params: Optional[Dict[str, float]] = None,
    ) -> EstimationResult:
        """
        Run numerical optimisation to maximise the Kalman log-likelihood.
        
        Parameters
        ----------
        initial_guess : Dict[str, float]
            Initial values for parameters to be estimated.
        bounds : Optional[Dict[str, tuple[float, float]]]
            Bounds for each parameter.
        optimizer_kwargs : Optional[Dict[str, float]]
            Additional arguments for scipy.optimize.minimize.
        fixed_params : Optional[Dict[str, float]]
            Parameters to keep fixed during estimation (e.g., {"a": 0.01, "b": 0.5}).
            These parameters will not be optimized.
        
        Returns
        -------
        EstimationResult
            Results including estimated parameters and diagnostics.
        """
        # Store fixed parameters
        self.fixed_params = fixed_params if fixed_params is not None else {}
        
        # List of all parameter names in order
        all_param_names = ["a", "b", "sigma", "eta", "rho", "measurement_var"]
        
        # Determine which parameters to optimize
        self.param_names_to_optimize = [p for p in all_param_names if p not in self.fixed_params]
        
        # Build initial guess vector for parameters to optimize
        guess_vec = np.array(
            [initial_guess.get(p, 0.1) for p in self.param_names_to_optimize],
            dtype=float,
        )
        
        # Default bounds
        if bounds is None:
            bounds = {
                "a": (1e-4, 5.0),
                "b": (1e-4, 5.0),
                "sigma": (1e-4, 0.2),
                "eta": (1e-4, 0.2),
                "rho": (-0.999, 0.999),
                "measurement_var": (1e-8, 0.01),
            }
        
        # Build bounds list for parameters to optimize
        bound_list = [bounds[p] for p in self.param_names_to_optimize]

        opt_kwargs = {"method": "L-BFGS-B"}
        if optimizer_kwargs:
            opt_kwargs.update(optimizer_kwargs)

        if not _SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for G2++ parameter estimation. Install via `pip install scipy`."
            )

        result = minimize(
            fun=self._negative_log_likelihood,
            x0=guess_vec,
            bounds=bound_list,
            **opt_kwargs,
        )

        params = self._param_dict_from_optimized_vector(result.x)
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
# Utility to generate synthetic data & demonstrate parameter estimation workflow
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
                "Joint parameter estimation for phi is not implemented; falling back to fit_term_structure"
            )
            return _compute_phi_from_method(obs, dt, "fit_term_structure")

        if method == "kalman_time_varying":
            warnings.warn(
                "Time-varying Kalman parameter estimation for phi is not implemented; falling back to fit_term_structure"
            )
            return _compute_phi_from_method(obs, dt, "fit_term_structure")

        raise ValueError(f"Unknown phi estimation method: {method}")

    phi_func = _compute_phi_from_method(observations, dt, phi_method)

    estimator = G2ppKalmanMLE(
        observations=observations,
        dt=dt,
        phi=phi_func,
        measurement_var=1e-6,
    )

    initial_guess = {
        "a": 0.1,
        "b": 0.5,
        "sigma": 0.01,
        "eta": 0.01,
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
    parser = argparse.ArgumentParser(description="G2++ parameter estimation example")
    parser.add_argument(
        "--phi-method",
        type=str,
        default="fit_term_structure",
        choices=["fit_term_structure", "joint", "kalman_time_varying"],
        help="Method to estimate phi(t). Default: fit_term_structure",
    )
    args = parser.parse_args()

    # Run example with chosen phi estimation method
    example_estimation(phi_method=args.phi_method)

