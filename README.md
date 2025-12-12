# IR-models

This repository contains reference implementations for two short-rate interest-rate models and associated
parameter estimation and Monte-Carlo simulation utilities:

- Vasicek model (single-factor)
- G2++ model (two-factor Gaussian)

The README below follows the simple plan in the project root: separate sections for MODEL, PARAMETER ESTIMATION and
MC SIMULATION for each model.

## 📚 Documentation

**Complete technical documentation** is available in LaTeX format:

📖 **[G2++ Methodology Documentation](docs/G2PP_Methodology.tex)**  - Complete mathematical and computational description including:
- Stochastic dynamics and factor models
- Zero-coupon bond pricing with analytical formulas
- EURIBOR calculation as forward rates
- Kalman Filter and Extended Kalman Filter for parameter estimation
- Monte Carlo simulation via Euler-Maruyama
- Full code-to-math correspondence


*Document will be added soon...*

## Requirements

- Python 3.10+
- numpy
- scipy

Install dependencies with:

```
pip install -r requirements.txt
```

## VASICEK

### Model

Files:

- `ir_models/models/vasicek.py` — implementation of the Vasicek (Ornstein–Uhlenbeck) short-rate model, simulation routines

What it does:

- Simulates short-rate paths under the Vasicek dynamics.
Mathematical formulation

- Short-rate SDE (Vasicek):

	$\displaystyle dr_t = \kappa (\theta - r_t)\,dt + \sigma\,dW_t$.

	Here $\kappa>0$ is the mean-reversion speed, $\theta$ the long-run mean, and $\sigma>0$ the volatility.

- Exact first two moments (used in the code's helpers):

	$\displaystyle \mathbb{E}[r_t] = \theta + (r_0-\theta)e^{-\kappa t}$,

	$\displaystyle \mathrm{Var}(r_t) = \frac{\sigma^2}{2\kappa}\bigl(1-e^{-2\kappa t}\bigr).$

- Discrete-time Euler-Maruyama used for simulation in code:

	$\displaystyle r_{t+\Delta t} = r_t + \kappa(\theta-r_t)\Delta t + \sigma\sqrt{\Delta t}\,Z,$

	where $Z\sim\mathcal{N}(0,1)$.

### Parameter Estimation

Files:

- `ir_models/estimation/vasicek.py` — parameter estimation utilities (MLE/Kalman or closed-form where applicable).

Usage:

- Run the parameter estimation script to fit Vasicek parameters to observed short-rate series.
  ```bash
  python3 -m ir_models.estimation.vasicek
  ```

Mathematical details used by the estimator

- For equally spaced observations with spacing $\Delta t$ the Vasicek dynamics imply an AR(1)-type transition
	for consecutive samples $r_{n}\to r_{n+1}$:

	$\displaystyle r_{n+1} = \theta + (r_n-\theta)e^{-\kappa\Delta t} + \varepsilon_{n+1},$

	with $\varepsilon_{n+1}\sim\mathcal{N}(0,\;v)$ and

	$\displaystyle v = \frac{\sigma^2}{2\kappa}\bigl(1-e^{-2\kappa\Delta t}\bigr).$

- The estimator constructs the Gaussian log-likelihood

	$\displaystyle \ell(\kappa,\theta,\sigma) = -\tfrac12\sum_n\Bigl(\log(2\pi v) + \frac{(r_{n+1}-\mu_n)^2}{v}\Bigr)$

	where $\mu_n=\theta+(r_n-\theta)e^{-\kappa\Delta t}$. The code maximises this (numerically via SciPy).

- The code also computes an OLS-based initial guess by mapping the AR(1) coefficients to $\kappa,\theta,\sigma$ via

	$\beta\approx e^{-\kappa\Delta t},\quad \alpha = \theta(1-\beta)$

	which gives $\kappa = -\log\beta/\Delta t$ and $\theta = \alpha/(1-\beta)$ as used in `ols_initial_guess()`.

### MC Simulation

Files:

- `ir_models/simulation/vasicek.py` — Monte Carlo drivers that use `ir_models/models/vasicek.py` to generate scenarios for pricing/risk.

Usage:

- The Monte Carlo file includes example runners to produce path ensembles and simple aggregation/plotting helpers.

## G2++

### Model

Files:

- `ir_models/models/g2pp.py` — core G2++ model implementation and path simulator for two correlated OU factors.

What it does:

- Simulates x and y factors and constructs short-rate r(t) = x(t) + y(t) + phi(t). The `phi` shift handles the
	deterministic part of the term structure.

Mathematical formulation

- Two-factor Gaussian (G2++) dynamics for the zero-mean factors $x,y$ used in the code: uytreza	

	$\displaystyle dx_t = -a\,x_t\,dt + \sigma\,dW_t^{(1)},$

	$\displaystyle dy_t = -b\,y_t\,dt + \eta\,dW_t^{(2)},$

	with $\mathrm{corr}(dW^{(1)},dW^{(2)})=\rho$. The short rate is assembled as

	$\displaystyle r_t = x_t + y_t + \phi(t)$,

	where $\phi(t)$ is a deterministic shift (used to fit the observed term structure).

- Simulation: Euler–Maruyama discretisation is used in `simulate_paths`:

	$\displaystyle x_{t+\Delta t} = x_t - a x_t\Delta t + \sigma\sqrt{\Delta t}\,Z^{(1)}$, 

	$\displaystyle y_{t+\Delta t} = y_t - b y_t\Delta t + \eta\sqrt{\Delta t}\,Z^{(2)}$, 

	where $(Z^{(1)},Z^{(2)})$ are correlated standard normals with correlation $\rho$ (constructed via Cholesky).

Theoretical moments used by the code

- Each OU factor has stationary variance

	$\displaystyle \mathrm{Var}(x_t)=\frac{\sigma^2}{2a},\qquad \mathrm{Var}(y_t)=\frac{\eta^2}{2b},$

	and the discrete-time increment standard deviation used when forming the process noise covariance is

	$\displaystyle s(\kappa,\nu,\Delta t)=\nu\sqrt{\frac{1-e^{-2\kappa\Delta t}}{2\kappa}},$

	matching the implementation in `_discrete_ou_coefficients()`.

### Zero-Coupon Bond Pricing and EURIBOR Calculation

The G2++ model provides analytical formulas for zero-coupon bond prices:

$$P(t, T) = A(t, T) \cdot \exp(-B_1(t,T) x_1(t) - B_2(t,T) x_2(t))$$

where:

$$B_i(t,T) = \frac{1 - e^{-a_i (T-t)}}{a_i}, \quad i = 1,2$$

$$A(t,T) = \exp\left( \int_t^T \phi(s) ds - V(t,T) \right)$$

and $V(t,T)$ captures the variance contribution from the stochastic factors.

From these bond prices, we can compute **EURIBOR rates** (or any forward LIBOR rate) as:

$$L(t; T, T+\delta) = \frac{1}{\delta} \left( \frac{P(t,T)}{P(t,T+\delta)} - 1 \right)$$

where $\delta$ is the tenor (e.g., 0.25 for 3 months).

**Important:** EURIBOR is **not** a short rate—it is a discrete forward rate derived from zero-coupon bond prices. This distinction is crucial for proper calibration.

Implementation: See `G2ppZeroCouponPricing` class in `ir_models/models/g2pp.py`.

### Parameter Estimation

Files:

- `ir_models/estimation/g2pp.py` — Extended Kalman filter / MLE estimator for the G2++ factors. Supports two observation modes:
	- **Short rate mode** (legacy): Direct observation of $r(t) = x(t) + y(t) + \phi(t)$
	- **EURIBOR mode** (recommended): Observation of forward rates $L(t; T, T+\delta)$ computed from zero-coupon bond prices

phi estimation details:

- The estimator supports a deterministic shift function $\phi(t)$ that can be provided externally or set to zero.
- For production calibration, $\phi(t)$ should be fitted to the initial term structure of interest rates.

Extended Kalman Filter (EKF) details for EURIBOR mode

- State vector: $\mathbf{x}_n = [x_n,\;y_n]^\top$ evolves linearly via the exact discrete transition

	$\displaystyle \mathbf{x}_{n+1} = F\mathbf{x}_n + \mathbf{u}_n,$

	where $F=\mathrm{diag}(e^{-a\Delta t},\;e^{-b\Delta t})$ and $\mathbf{u}_n\sim\mathcal{N}(0,Q)$ with $Q$ the
	process noise covariance assembled from the OU increment standard deviations and the correlation $\rho$ (see
	`_transition_matrices` in the code).

- Observation model (EURIBOR with measurement noise):

	$\displaystyle z_n = h(\mathbf{x}_n, t_n) + \varepsilon_n,$

	where $h(\mathbf{x}_n, t_n) = L(t_n; T, T+\delta)$ is the **nonlinear** EURIBOR function computed from ZC bond prices,
	and $\varepsilon_n\sim\mathcal{N}(0,\,R)$ captures measurement variance. 
	
- The Extended Kalman Filter linearizes the observation equation at each time step:

	$\displaystyle H_n = \nabla h|_{\mathbf{x}_n} \approx \begin{bmatrix} \frac{\partial L}{\partial x_1} & \frac{\partial L}{\partial x_2} \end{bmatrix}$

	computed via finite differences. The EKF then proceeds with standard Kalman update equations using this time-varying Jacobian.

- Legacy observation model (short-rate observations with measurement noise):

	$\displaystyle z_n = H\mathbf{x}_n + \phi(t_n) + \varepsilon_n,$

	where $H=[1\;1]$. The standard Kalman filter is used in this linear case.
	candidate parameters $(a,b,\sigma,\eta,\rho)$.

- The implementation computes the log-likelihood incrementally inside the Kalman loop and the outer optimiser
	(SciPy) maximises this likelihood to obtain parameter estimates.

Usage examples:

Show help for the G2++ parameter estimation example (no SciPy required to show help):

```bash
python3 -m ir_models.estimation.g2pp --help
```

Run the example (SciPy required):

```bash
pip install -r requirements.txt
python3 -m ir_models.estimation.g2pp
```

Choose a phi method (currently `joint` and `kalman_time_varying` will warn and fall back):

```bash
python3 -m ir_models.estimation.g2pp --phi-method fit_term_structure
```

### MC Simulation

Files:

- `ir_models/simulation/g2pp.py` — Monte Carlo driver using `ir_models/models/g2pp.py` for scenario generation.

Usage:

- The Monte Carlo script demonstrates path generation and basic statistics; adjust seeds and parameters inside the
	script or refactor into CLI flags as needed.

## Examples & Notes

- Each parameter estimation file includes an `example_estimation()` helper demonstrating synthetic data generation and a
	full estimation run.
- Parameter estimation uses SciPy's `minimize` (L-BFGS-B by default).