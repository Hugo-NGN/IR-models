#!/usr/bin/env python3
"""
Example script for calibrating G2++ model to EURIBOR 3M data.

This script demonstrates the correct approach to calibrating the G2++ model
to market EURIBOR data by treating EURIBOR as a forward rate (not a short rate).

Key points:
- EURIBOR 3M is a forward rate: L(t; T, T+δ) = (1/δ) * (P(t,T)/P(t,T+δ) - 1)
- The calibration uses an Extended Kalman Filter (EKF) to estimate the latent
  short rate factors (x1, x2) from observed EURIBOR rates
- The observation equation is nonlinear, requiring Jacobian computation
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from ir_models.estimation.g2pp import G2ppKalmanMLE
from ir_models.models.g2pp import G2ppZeroCouponPricing


def load_euribor_data(file_path: str, start_date: str, end_date: str):
    """Load and filter EURIBOR data."""
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df_filtered = df.loc[mask].copy()
    df_filtered = df_filtered.sort_values('Date', ascending=True)
    
    rates = df_filtered['Dernier'].values / 100.0  # Convert to decimal
    dates = df_filtered['Date'].values
    
    print(f"Loaded {len(rates)} observations from {start_date} to {end_date}")
    return rates, dates


def phi_func(t: float) -> float:
    """
    Deterministic shift function.
    For this example, we use phi(t) = 0.
    
    In production, phi(t) should be calibrated to fit the initial term structure.
    """
    return 0.0


def calibrate_g2pp(
    rates: np.ndarray,
    dt: float = 1.0/12.0,
    fixed_params: dict = None,
    initial_guess: dict = None,
):
    """
    Calibrate G2++ model to EURIBOR 3M data.
    
    Parameters
    ----------
    rates : np.ndarray
        Observed EURIBOR 3M rates (as decimals, e.g., 0.03 for 3%).
    dt : float
        Time step in years (default: 1/12 for monthly data).
    fixed_params : dict, optional
        Parameters to keep fixed during estimation.
    initial_guess : dict, optional
        Initial parameter values for optimization.
    
    Returns
    -------
    result : EstimationResult
        Calibration results including estimated parameters and filtered states.
    """
    print("\n" + "="*70)
    print("G2++ CALIBRATION TO EURIBOR 3M")
    print("="*70)
    
    # Default fixed parameters
    if fixed_params is None:
        fixed_params = {
            "a": 0.01,  # 1% mean reversion for first factor
            "b": 0.50,  # 50% mean reversion for second factor
        }
    
    # Default initial guess
    if initial_guess is None:
        initial_guess = {
            "sigma": 0.09,
            "eta": 0.09,
            "rho": -0.1,
            "measurement_var": 1e-5,
        }
    
    print("\nInitializing estimator in EURIBOR mode...")
    print(f"  Observation mode: EURIBOR forward rates")
    print(f"  Tenor: 3M (δ = 0.25 years)")
    print(f"  Time step: {dt:.4f} years")
    
    estimator = G2ppKalmanMLE(
        observations=rates,
        dt=dt,
        phi=phi_func,
        measurement_var=1e-6,
        observation_mode="euribor",      # EURIBOR mode
        euribor_tenor=0.25,               # 3M = 0.25 years
        euribor_start_offset=0.0,         # Spot-starting EURIBOR
    )
    
    print("\nFixed parameters:")
    for key, value in fixed_params.items():
        print(f"  {key}: {value:.6f}")
    
    print("\nInitial guess for free parameters:")
    for key, value in initial_guess.items():
        print(f"  {key}: {value:.6f}")
    
    print("\nRunning optimization (this may take a minute)...")
    result = estimator.fit(
        initial_guess=initial_guess,
        fixed_params=fixed_params,
    )
    
    print("\n" + "="*70)
    print("ESTIMATION RESULTS")
    print("="*70)
    print(f"Optimization success: {result.optimized}")
    print(f"Log-likelihood: {result.log_likelihood:.4f}")
    
    print("\nEstimated parameters:")
    for key, value in result.params.items():
        status = "[FIXED]" if key in fixed_params else "[ESTIMATED]"
        print(f"  {key:20s}: {value:.6f}  {status}")
    
    return result


def compute_fitted_euribor(
    filtered_states: np.ndarray,
    params: dict,
    dt: float,
) -> np.ndarray:
    """
    Compute fitted EURIBOR rates from filtered factor states.
    """
    pricing = G2ppZeroCouponPricing(
        a=params["a"],
        b=params["b"],
        sigma=params["sigma"],
        eta=params["eta"],
        rho=params["rho"],
        phi=phi_func,
    )
    
    n_obs = filtered_states.shape[0]
    fitted_euribor = np.zeros(n_obs)
    
    for i in range(n_obs):
        t = i * dt
        x1, x2 = filtered_states[i]
        T_start = t + 0.0  # Spot-starting
        fitted_euribor[i] = pricing.euribor_rate(t, T_start, 0.25, x1, x2)
    
    return fitted_euribor


def plot_results(rates, dates, result, dt):
    """Plot calibration results."""
    fitted_euribor = compute_fitted_euribor(
        result.filtered_states,
        result.params,
        dt,
    )
    
    x_fitted = result.filtered_states[:, 0]
    y_fitted = result.filtered_states[:, 1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Observed vs Fitted EURIBOR
    ax = axes[0, 0]
    ax.plot(dates, rates, 'b-', label='Observed EURIBOR 3M', linewidth=1.5, alpha=0.7)
    ax.plot(dates, fitted_euribor, 'r--', label='Fitted EURIBOR 3M (EKF)', linewidth=1.5)
    ax.set_title('Observed vs Fitted EURIBOR 3M', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Residuals
    ax = axes[0, 1]
    residuals = rates - fitted_euribor
    ax.plot(dates, residuals * 10000, 'k-', linewidth=1.0)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Residuals (basis points)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual (bp)')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mae = np.mean(np.abs(residuals)) * 10000
    rmse = np.sqrt(np.mean(residuals**2)) * 10000
    ax.text(0.02, 0.98, f'MAE: {mae:.2f} bp\nRMSE: {rmse:.2f} bp',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Factor x(t)
    ax = axes[1, 0]
    ax.plot(dates, x_fitted, 'g-', label='Factor x₁(t)', linewidth=1.5)
    ax.set_title('Fitted Factor x₁(t)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('x₁(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Factor y(t)
    ax = axes[1, 1]
    ax.plot(dates, y_fitted, 'm-', label='Factor x₂(t)', linewidth=1.5)
    ax.set_title('Fitted Factor x₂(t)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('x₂(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('g2pp_euribor_calibration.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: g2pp_euribor_calibration.png")
    plt.show()


def main():
    """Main execution function."""
    # Configuration
    data_file = 'data/ERB3M_historic.xlsx'
    start_date = '2021-09-01'
    end_date = '2024-08-31'
    dt = 1.0 / 12.0  # Monthly data
    
    # Load data
    rates, dates = load_euribor_data(data_file, start_date, end_date)
    
    # Calibrate model
    result = calibrate_g2pp(rates, dt=dt)
    
    # Print fit statistics
    fitted_euribor = compute_fitted_euribor(result.filtered_states, result.params, dt)
    residuals = rates - fitted_euribor
    
    print("\n" + "="*70)
    print("FIT STATISTICS")
    print("="*70)
    print(f"Mean Absolute Error:     {np.mean(np.abs(residuals))*10000:.4f} bp")
    print(f"Root Mean Squared Error: {np.sqrt(np.mean(residuals**2))*10000:.4f} bp")
    print(f"Max Absolute Error:      {np.max(np.abs(residuals))*10000:.4f} bp")
    print(f"Correlation:             {np.corrcoef(rates, fitted_euribor)[0,1]:.6f}")
    
    # Plot results
    plot_results(rates, dates, result, dt)


if __name__ == "__main__":
    main()
