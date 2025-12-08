
import pandas as pd
import numpy as np
import sys
import os

# Add project root to sys.path to allow imports from ir_models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ir_models.estimation.g2pp import G2ppKalmanMLE

def main():
    # Load data
    file_path = os.path.join(os.path.dirname(__file__), '../data/ERB3M_historic.xlsx')
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    
    # Filter for 2021-2023 (3 years)
    # User asked for "between 2021 and 2024 (3 years)", so likely [2021, 2024) or [2021, 2023] inclusive.
    # 2021, 2022, 2023 is 3 years.
    start_date = '2021-01-01'
    end_date = '2023-12-31'
    
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df_filtered = df.loc[mask].copy()
    
    # Sort by date ascending (oldest first)
    df_filtered = df_filtered.sort_values('Date', ascending=True)
    
    print(f"Data filtered from {start_date} to {end_date}.")
    print(f"Number of observations: {len(df_filtered)}")
    
    if len(df_filtered) < 10:
        print("Error: Not enough data points.")
        return

    # Extract rates (Dernier column)
    # Assuming values are in percent (e.g. 2.06), convert to decimal (0.0206)
    rates = df_filtered['Dernier'].values / 100.0
    
    # Time step (Monthly data)
    dt = 1.0 / 12.0
    
    print("Estimating G2++ model parameters...")
    
    # Initialize estimator
    # We use the default phi method (fit_term_structure) which is now the default in the class/example
    # But we need to pass it explicitly if we use the class directly?
    # The class G2ppKalmanMLE takes a phi function.
    # In the example_estimation function in g2pp_estimation.py, it constructs a phi function.
    # I should replicate that logic or import the helper if possible.
    # The helper _compute_phi_from_method is inside example_estimation, so I can't import it.
    # I will reimplement the simple smoothing phi here.
    
    def create_phi(obs, dt):
        n = obs.size
        times = np.arange(n) * dt
        # Simple smoothing
        window = max(3, int(0.02 * n))
        kernel = np.ones(window) / window
        smooth = np.convolve(obs, kernel, mode="same")
        
        # Adjust to match last observation
        if n > 0:
            offset = float(obs[-1] - smooth[-1])
            smooth = smooth + offset
            
        def phi(t):
            return float(np.interp(t, times, smooth))
        return phi

    phi_func = create_phi(rates, dt)
    
    estimator = G2ppKalmanMLE(
        observations=rates,
        dt=dt,
        phi=phi_func,
        measurement_var=1e-6
    )
    
    # Initial guess
    initial_guess = {
        "a": 0.1,
        "b": 0.4,
        "sigma": 0.01,
        "eta": 0.01,
        "rho": -0.5,
        "measurement_var": 1e-5,
    }
    
    result = estimator.fit(initial_guess=initial_guess)
    
    print("\nParameter Estimation Results:")
    print(f"Optimisation success: {result.optimized}")
    print(f"Log-likelihood: {result.log_likelihood:.4f}")
    print("Estimated Parameters:")
    for key, value in result.params.items():
        print(f"  {key}: {value:.6f}")

if __name__ == "__main__":
    main()
