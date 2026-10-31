"""Release–recapture simulation and temperature fitting.

Both simulation and fitting take depth U/h in MHz, waist in μm, and wavelength
in nm. Simulation takes temperature in K and times in s; fitting takes times
in μs and returns temperature in μK.
The model uses harmonic thermal initial conditions in a Gaussian tweezer.
"""

from dataclasses import dataclass
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import h, hbar, k
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t

def gaussian_tweezer_potential(positions, *, waist, depth, rayleigh):
    """3D potential in joules, with zero energy at infinity; z is the beam axis."""
    x, y, z = positions.T
    axial_factor = 1 + (z / rayleigh)**2
    return -depth / axial_factor * np.exp(
        -2 * (x**2 + y**2) / (waist**2 * axial_factor)
    )

def release_recapture(temperature, times, *, waist_um, depth_MHz, wavelength_nm,
                      mass=1.44316e-25,
                      samples=50_000, seed=42, zero_point=True):
    """Simulate release–recapture with temperature in K and release times in s.

    Trap inputs are the Gaussian 1/e² intensity waist ``waist_um`` in μm,
    positive depth U/h ``depth_MHz`` in MHz (not angular frequency), and laser
    ``wavelength_nm`` in nm. Internally, convert lengths to metres and depth
    to joules, then derive the Rayleigh range as pi * waist**2 / wavelength.
    ``mass`` is in kg and defaults to Rb-87 (1.44316e-25 kg).
    ``zero_point=True`` includes zero-point widths; False uses classical widths.

    Return probability, Monte Carlo standard error, and the initially bound
    fraction. Accurate initial harmonic approximation requires k*T << depth_J;
    the classical capture test is approximate near T=0.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError("times must be a finite, nonnegative 1D array in seconds.")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative, in kelvin.")
    trap_values = np.array([mass, waist_um, depth_MHz, wavelength_nm], dtype=float)
    if not np.all(np.isfinite(trap_values)) or np.any(trap_values <= 0):
        raise ValueError("mass (kg), waist_um, depth_MHz, and wavelength_nm must be finite and positive.")
    if not isinstance(samples, (int, np.integer)) or samples < 1:
        raise ValueError("samples must be a positive integer.")
    waist = waist_um * 1e-6
    wavelength = wavelength_nm * 1e-9
    depth = h * depth_MHz * 1e6
    rayleigh = np.pi * waist**2 / wavelength
    frequencies = np.array([np.sqrt(4 * depth / (mass * waist**2))] * 2
                           + [np.sqrt(2 * depth / (mass * rayleigh**2))])
    if zero_point:
        # hbar*omega*(nbar + 1/2): exact thermal Wigner second moments.
        mode_energy = (hbar * frequencies / 2 if temperature == 0 else
                       hbar * frequencies / (2 * np.tanh(hbar * frequencies / (2 * k * temperature))))
    else:
        mode_energy = np.full(3, k * temperature)
    sigma_position = np.sqrt(mode_energy / (mass * frequencies**2))
    sigma_velocity = np.sqrt(mode_energy / mass)

    rng = np.random.default_rng(seed)
    positions = rng.normal(size=(samples, 3)) * sigma_position
    velocities = rng.normal(size=(samples, 3)) * sigma_velocity
    kinetic_energy = 0.5 * mass * np.sum(velocities**2, axis=1)
    potential_args = dict(waist=waist, depth=depth, rayleigh=rayleigh)
    initially_bound = (kinetic_energy + gaussian_tweezer_potential(positions, **potential_args)) < 0
    retained = int(initially_bound.sum())
    if retained == 0:
        raise ValueError("No initially trapped samples; check temperature and trap depth.")
    positions = positions[initially_bound]
    velocities = velocities[initially_bound]
    kinetic_energy = kinetic_energy[initially_bound]

    probability = np.empty(times.size)
    # Reuse trajectories across times; pointwise MC errors are correlated in time.
    for j, release_time in enumerate(times):
        final_positions = positions + velocities * release_time
        final_energy = kinetic_energy + gaussian_tweezer_potential(final_positions, **potential_args)
        probability[j] = np.mean(final_energy < 0)
    standard_error = np.sqrt(probability * (1 - probability) / retained)
    return probability, standard_error, retained / samples


@dataclass
class ReleaseRecaptureFit:
    """Fit outputs; covariance/CI rows are [temperature in μK, survival].

    ``curve_ci95`` has shape (2, n_curve_times), with lower/upper rows.
    ``curve_mc_stderr`` is separate simulation sampling uncertainty, scaled
    by fitted survival; it is not included in ``curve_ci95``.
    """

    temperature_uK: float
    survival: float
    parameter_covariance: np.ndarray
    parameter_ci95: np.ndarray
    time_us: np.ndarray
    data: np.ndarray
    curve_time_us: np.ndarray
    curve_probability: np.ndarray
    curve_ci95: np.ndarray
    curve_mc_stderr: np.ndarray
    residual_sse: float
    near_bound: bool

    @property
    def parameter_stderr(self):
        """Approximate 1σ errors for [temperature in μK, survival]."""
        return np.sqrt(np.diag(self.parameter_covariance))

    def summary(self):
        """Return temperature, survival, and uncertainty assumptions for printing."""
        errors = self.parameter_stderr
        ci = self.parameter_ci95
        return (
            f'Temperature: {self.temperature_uK:.2f} ± {errors[0]:.2f} μK (1σ); '
            f'95% CI: [{ci[0, 0]:.2f}, {ci[0, 1]:.2f}] μK\n'
            f'Fitted initial survival: {self.survival:.4f} ± {errors[1]:.4f} (1σ); '
            f'95% CI: [{ci[1, 0]:.4f}, {ci[1, 1]:.4f}]\n'
            'Approximate pointwise 95% confidence band from fitted-parameter covariance.\n'
            'Assumes independent, equal-variance residuals; excludes Monte Carlo '
            'and trap-calibration uncertainty.'
            + ('\nFit is near a parameter bound; symmetric confidence intervals may be unreliable.'
               if self.near_bound else '')
        )


def fit_release_recapture(time_us, recapture, *, waist_um, depth_MHz, wavelength_nm,
                          mass=1.44316e-25,
                          samples=50_000, seed=42, zero_point=True,
                          temperature_bounds_uK=(0.1, 200.0)):
    """Fit temperature and survival to ``survival * release_recapture(T, t)``.

    Parameters
    ----------
    time_us, recapture : array-like
        Release times in μs and average recapture probabilities. Accepts a
        1D scan or an (n, 1) column; paired nonfinite values are omitted and
        times are sorted. At least three valid points spanning times are needed.
    waist_um : float
        Gaussian 1/e² intensity waist in micrometres (μm).
    wavelength_nm : float
        Laser wavelength in nanometres (nm). Both lengths are converted to
        metres internally, then rayleigh = pi * waist**2 / wavelength.
    depth_MHz : float
        Positive trap depth U/h in MHz (ordinary frequency, not angular).
        Converted internally to joules as h * depth_MHz * 1e6.
    mass : float, optional
        Atom mass in kg; defaults to Rb-87 (1.44316e-25 kg).
        All trap parameters are held fixed during the fit.
    samples, seed, zero_point
        Simulation settings. An integer seed reuses the same trajectories
        across evaluations for a deterministic objective. zero_point defaults
        to True; set False to use classical thermal widths.
    temperature_bounds_uK : pair of float
        Positive, increasing search limits in μK.

    Returns
    -------
    ReleaseRecaptureFit
        Fitted parameters, covariance, 95% intervals, sorted data, and a dense
        fitted curve with a pointwise 95% confidence band. No plotting or printing.

    Notes
    -----
    Uses unweighted least squares, a coarse temperature search, and bounded
    derivative-free refinement. Uncertainty is the local linear approximation
    from fit residuals, including temperature–survival covariance. It assumes
    independent, equal-variance residuals and is not a prediction interval.
    Monte Carlo and trap-calibration uncertainty are excluded. Confidence
    intervals can be unreliable near parameter bounds or zero-point saturation.
    """
    time_us = np.asarray(time_us, dtype=float)
    if time_us.ndim == 2 and time_us.shape[1] == 1:
        time_us = time_us[:, 0]
    data = np.asarray(recapture, dtype=float)
    if time_us.ndim != 1 or data.ndim != 1 or time_us.shape != data.shape:
        raise ValueError('Provide one time and one recapture probability per point.')
    valid = np.isfinite(time_us) & np.isfinite(data)
    order = np.argsort(time_us[valid])
    time_us, data = time_us[valid][order], data[valid][order]
    if len(time_us) < 3 or np.ptp(time_us) == 0:
        raise ValueError('Need at least three valid points spanning release times.')
    if np.any(time_us < 0) or np.any((data < 0) | (data > 1)):
        raise ValueError('Times must be nonnegative and recapture probabilities must be in [0, 1].')
    bounds = np.asarray(temperature_bounds_uK, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or not 0 < bounds[0] < bounds[1]:
        raise ValueError('temperature_bounds_uK must contain two positive, increasing limits.')
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Use a nonnegative integer seed for deterministic fitting.')

    # The simulator validates trap inputs and converts to SI units internally.
    simulation_args = dict(mass=mass, waist_um=waist_um, depth_MHz=depth_MHz,
                           wavelength_nm=wavelength_nm,
                           samples=samples, seed=seed, zero_point=zero_point)

    def probability_at(temperature_uK, times_us):
        return release_recapture(temperature_uK * 1e-6, times_us * 1e-6, **simulation_args)

    def fit_at_temperature(temperature_uK):
        probability, _, _ = probability_at(temperature_uK, time_us)
        norm = np.dot(probability, probability)
        if norm == 0:
            return np.inf, 0.0
        survival = np.clip(np.dot(data, probability) / norm, 0, 1)
        return np.sum((data - survival * probability)**2), survival

    # Avoid numerical derivatives of discrete Monte Carlo capture decisions.
    grid = np.geomspace(*bounds, 41)
    errors = np.array([fit_at_temperature(T)[0] for T in grid])
    best = int(np.argmin(errors))
    if not np.isfinite(errors[best]):
        raise ValueError('No model recapture over this time/temperature range.')
    fit = minimize_scalar(
        lambda T: fit_at_temperature(T)[0],
        bounds=(grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]),
        method='bounded', options={'xatol': 0.01},
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    temperature_uK = min([fit.x, grid[best]], key=lambda T: fit_at_temperature(T)[0])
    residual_sse, survival = fit_at_temperature(temperature_uK)

    curve_time_us = np.linspace(time_us.min(), time_us.max(), 300)
    probability, mc_stderr, _ = probability_at(temperature_uK, curve_time_us)
    # A finite step smooths the discrete capture test; common random numbers
    # reduce simulation noise in the derivative.
    derivative_step = max(0.5, 0.02 * temperature_uK)
    lower_T = max(0.0, temperature_uK - derivative_step)
    upper_T = temperature_uK + derivative_step
    all_times = np.concatenate((time_us, curve_time_us))
    low, _, _ = probability_at(lower_T, all_times)
    high, _, _ = probability_at(upper_T, all_times)
    derivative = (high - low) / (upper_T - lower_T)
    probability_data, _, _ = probability_at(temperature_uK, time_us)
    jacobian = np.column_stack((survival * derivative[:len(data)], probability_data))
    if np.linalg.matrix_rank(jacobian) < 2:
        raise ValueError('Temperature and survival are not identifiable from these data.')
    dof = len(data) - 2
    covariance = (residual_sse / dof) * np.linalg.inv(jacobian.T @ jacobian)
    stderr = np.sqrt(np.diag(covariance))
    critical = student_t.ppf(0.975, dof)
    parameter_ci95 = (np.array([temperature_uK, survival])[:, None]
                      + critical * stderr[:, None] * np.array([-1, 1]))
    curve_jacobian = np.column_stack((survival * derivative[len(data):], probability))
    curve_stderr = np.sqrt(np.maximum(0, np.einsum(
        'ij,jk,ik->i', curve_jacobian, covariance, curve_jacobian,
    )))
    curve_probability = survival * probability
    curve_ci95 = np.clip(curve_probability
                         + critical * curve_stderr * np.array([[-1], [1]]), 0, 1)
    near_bound = best in (0, len(grid) - 1) or survival in (0, 1)
    if near_bound:
        warnings.warn('Fit is near a parameter bound; symmetric confidence intervals may be unreliable.',
                      RuntimeWarning, stacklevel=2)
    return ReleaseRecaptureFit(
        temperature_uK=float(temperature_uK), survival=float(survival),
        parameter_covariance=covariance, parameter_ci95=parameter_ci95,
        time_us=time_us, data=data, curve_time_us=curve_time_us,
        curve_probability=curve_probability, curve_ci95=curve_ci95,
        curve_mc_stderr=survival * mc_stderr, residual_sse=float(residual_sse),
        near_bound=near_bound,
    )


def plot_release_recapture_fit(result : ReleaseRecaptureFit, ax=None):
    """Plot data, fit, and the fitted-parameter 95% confidence band; return axes.

    Pass an existing Matplotlib axes to control figure size/style. This does
    not call ``show()`` or save a file.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result.time_us, result.data, 'o', markersize=4, label='Data', zorder=3)
    line, = ax.plot(result.curve_time_us, result.curve_probability, '-',
                    label=f'Release–recapture fit: T = {result.temperature_uK:.2f} '
                          f'± {result.parameter_stderr[0]:.2f} μK (1σ)')
    ax.fill_between(result.curve_time_us, result.curve_ci95[0], result.curve_ci95[1],
                    color=line.get_color(), alpha=0.3, linewidth=0,
                    label='95% confidence band (fitted parameters)')
    ax.set(xlabel='Release time (μs)', ylabel='Average recapture probability',
           title='Release–recapture temperature fit', ylim=(-0.02, 1.02))
    ax.legend(fontsize=11)
    return ax

