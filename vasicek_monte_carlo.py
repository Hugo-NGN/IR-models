"""
Vasicek Interest Rate Model Monte Carlo utilities.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from vasicek_model import VasicekModel

def plot_simulations(
    paths: np.ndarray,
    time_grid: np.ndarray,
    model: VasicekModel,
    n_paths_to_plot: int = 10,
    show_mean: bool = True,
    show_confidence_bands: bool = True,
    confidence_level: float = 0.95
):
    """
    Plot Monte Carlo simulation results.
    
    Parameters:
    -----------
    paths : np.ndarray
        Simulated paths (n_simulations, n_steps + 1)
    time_grid : np.ndarray
        Time points
    model : VasicekModel
        Vasicek model instance
    n_paths_to_plot : int
        Number of individual paths to display
    show_mean : bool
        Whether to show theoretical mean
    show_confidence_bands : bool
        Whether to show confidence bands
    confidence_level : float
        Confidence level for bands (default: 0.95)
    """
    n_simulations, n_steps = paths.shape
    n_steps = n_steps - 1
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual paths (sample)
    n_plot = min(n_paths_to_plot, n_simulations)
    for i in range(n_plot):
        plt.plot(time_grid, paths[i, :], alpha=0.3, linewidth=0.5, color='lightblue')
    
    # Plot empirical mean
    empirical_mean = np.mean(paths, axis=0)
    plt.plot(time_grid, empirical_mean, 'b-', linewidth=2, label='Empirical Mean')
    
    # Plot theoretical mean
    if show_mean:
        theoretical_mean = np.array([model.theoretical_mean(t) for t in time_grid])
        plt.plot(time_grid, theoretical_mean, 'r--', linewidth=2, label='Theoretical Mean')
    
    # Plot confidence bands
    if show_confidence_bands:
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(paths, lower_percentile, axis=0)
        upper_bound = np.percentile(paths, upper_percentile, axis=0)
        
        plt.fill_between(
            time_grid, lower_bound, upper_bound,
            alpha=0.2, color='gray', label=f'{confidence_level*100:.0f}% Confidence Band'
        )
    
    plt.xlabel('Time (years)', fontsize=12)
    plt.ylabel('Interest Rate', fontsize=12)
    plt.title(f'Vasicek Model: {n_simulations} Monte Carlo Simulations', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def example_usage():
    """
    Example usage of the Vasicek model simulation.
    """
    # Model parameters
    kappa = 0.5      # Mean reversion speed
    theta = 0.05     # Long-term mean (5%)
    sigma = 0.02     # Volatility (2%)
    r0 = 0.03        # Initial rate (3%)
    
    # Simulation parameters
    T = 5.0          # 5 years
    n_steps = 252    # Daily steps (assuming 252 trading days per year)
    n_simulations = 10000  # Number of Monte Carlo paths
    
    # Create model
    model = VasicekModel(kappa=kappa, theta=theta, sigma=sigma, r0=r0)
    
    # Run simulation
    print("Running Monte Carlo simulation...")
    paths = model.simulate_path(
        T=T,
        n_steps=n_steps,
        n_simulations=n_simulations,
        random_seed=42
    )
    
    # Get time grid
    time_grid = model.get_time_grid(T, n_steps)
    
    # Print statistics
    print(f"\nSimulation Statistics:")
    print(f"  Number of simulations: {n_simulations}")
    print(f"  Time steps: {n_steps}")
    print(f"  Time horizon: {T} years")
    print(f"\nFinal Rate Statistics (at t={T}):")
    final_rates = paths[:, -1]
    print(f"  Mean: {np.mean(final_rates):.4f}")
    print(f"  Std Dev: {np.std(final_rates):.4f}")
    print(f"  Min: {np.min(final_rates):.4f}")
    print(f"  Max: {np.max(final_rates):.4f}")
    print(f"\nTheoretical Mean (at t={T}): {model.theoretical_mean(T):.4f}")
    print(f"Theoretical Std Dev (at t={T}): {np.sqrt(model.theoretical_variance(T)):.4f}")
    
    # Plot results
    plot_simulations(
        paths=paths,
        time_grid=time_grid,
        model=model,
        n_paths_to_plot=50,
        show_mean=True,
        show_confidence_bands=True
    )
    
    return model, paths, time_grid


if __name__ == "__main__":
    example_usage()

