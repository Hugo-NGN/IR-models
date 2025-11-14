"""
G2++ (two-factor Gaussian) interest rate model with Monte Carlo simulation.
The model assumes the short rate is
    r(t) = x(t) + y(t) + phi(t)
where x and y follow correlated Ornstein-Uhlenbeck processes:
    dx = -a x dt + sigma dW1
    dy = -b y dt + eta dW2
with corr(dW1, dW2) = rho.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from g2ppModel import G2ppModel, SimulationResult


def plot_short_rate_paths(
    sim_result: SimulationResult,
    n_paths_to_plot: int = 20,
    show_mean: bool = True,
    confidence_level: float = 0.95,
) -> None:
    """
    Visualize simulated short-rate paths and summary statistics.
    """
    short_paths = sim_result.short_rate_paths
    time_grid = sim_result.time_grid
    n_sim = short_paths.shape[0]

    plt.figure(figsize=(12, 7))

    for idx in range(min(n_paths_to_plot, n_sim)):
        plt.plot(time_grid, short_paths[idx], 
        #color="steelblue", 
        alpha=0.25,
         linewidth=0.8)

    empirical_mean = short_paths.mean(axis=0)
    plt.plot(time_grid, empirical_mean, color="blue", linewidth=2, label="Empirical mean")

    if show_mean:
        plt.plot(
            time_grid,
            sim_result.phi_grid,
            "r--",
            linewidth=1.5,
            label="Theoretical mean",
        )

    alpha = 1 - confidence_level
    lower = np.percentile(short_paths, alpha / 2 * 100, axis=0)
    upper = np.percentile(short_paths, (1 - alpha / 2) * 100, axis=0)
    plt.fill_between(time_grid, lower, upper, color="gray", alpha=0.2, label=f"{confidence_level:.0%} band")

    plt.title("G2++ Short Rate Monte Carlo Simulation")
    plt.xlabel("Time (years)")
    plt.ylabel("Short rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def example_usage() -> SimulationResult:
    """
    Demonstrate simulating G2++ short-rate paths.
    """
    model = G2ppModel(
        a=0.1,
        b=0.3,
        sigma=0.02,
        eta=0.015,
        rho=-0.75,
        x0=0.0,
        y0=0.0,
        r0=0.025,
    )

    T = 5.0
    n_steps = 360
    n_simulations = 50000

    print("Running G2++ Monte Carlo simulation...")
    sim_result = model.simulate_paths(
        T=T,
        n_steps=n_steps,
        n_simulations=n_simulations,
        random_seed=123,
    )

    final_rates = sim_result.short_rate_paths[:, -1]
    print("Summary statistics at T = %.2f years" % T)
    print(f"  Mean: {final_rates.mean():.4f}")
    print(f"  Std:  {final_rates.std():.4f}")
    print(f"  Min:  {final_rates.min():.4f}")
    print(f"  Max:  {final_rates.max():.4f}")

    plot_short_rate_paths(sim_result, n_paths_to_plot=250)
    return sim_result


if __name__ == "__main__":
    example_usage()

