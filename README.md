# Local analysis library

`data_analysis_helper` is a Python package inside `DailyAnalysis`. Import it
directly from notebooks or scripts run from this folder; no installation is
needed. NumPy and SciPy are required for release–recapture fitting, and Matplotlib
is required for plotting. The existing `py314` environment has these dependencies.

## Release–recapture temperature fit

```python
import matplotlib.pyplot as plt
from data_analysis_helper.release_recapture import (
    fit_release_recapture,
    plot_release_recapture_fit,
)

trap = dict(
    waist_um=0.95,               # 1/e² intensity radius, μm
    depth_MHz=15.627464342495682,  # U/h in MHz; equivalent to U/k = 0.75 mK
    wavelength_nm=850,           # laser wavelength, nm
)
result = fit_release_recapture(
    exp.param_sets_array[:, 0],       # release time, μs
    exp.compute_recapture_average(),  # probability, 0–1
    **trap,
    samples=50_000,
    seed=42,
    temperature_bounds_uK=(0.1, 200.0),
)
print(result.summary())
ax = plot_release_recapture_fit(result)
ax.figure.tight_layout()
plt.show()
```

The fit accepts arrays independently of `ExperimentData`. It fits temperature
and initial survival using unweighted least squares. Missing/nonfinite pairs are
omitted and times are sorted. The fixed seed makes repeated fits reproducible.
Rb-87 mass (`mass=1.44316e-25` kg) and `zero_point=True` are defaults and can
be overridden. Use `waist_um` in μm and `wavelength_nm` in nm. Trap depth is
`U/h` in MHz; the function converts lengths to metres and depth to joules,
then derives the Rayleigh range internally.
For existing calls, replace `depth` in joules with `depth_MHz = depth / (h * 1e6)`
replace `waist` in metres with `waist_um = waist * 1e6`, and replace
`rayleigh` with the laser `wavelength_nm` in nm.

Useful result attributes:

| Attribute | Meaning |
| --- | --- |
| `temperature_uK`, `survival` | Best-fit temperature in μK and survival factor |
| `parameter_stderr` | Approximate 1σ errors in order `[temperature, survival]` |
| `parameter_covariance` | Full 2×2 covariance, including parameter correlation |
| `parameter_ci95` | 95% intervals, rows `[temperature, survival]`, columns `[lower, upper]` |
| `time_us`, `data` | Filtered, sorted experimental arrays |
| `curve_time_us`, `curve_probability` | Dense fitted curve |
| `curve_ci95` | Pointwise 95% confidence band, rows `[lower, upper]` |
| `curve_mc_stderr` | Separate Monte Carlo standard error of the scaled curve |
| `residual_sse` | Sum of squared residuals |
| `near_bound` | Whether the fit is near a search limit or survival bound |

The shaded confidence band propagates both fitted parameters and their
correlation. It uses a local linear approximation and assumes independent,
equal-variance residuals. It is a confidence band for the mean curve, not a
prediction interval for new measurements. Trap-calibration and Monte Carlo
uncertainty are excluded; `curve_mc_stderr` is provided separately. Intervals
near parameter bounds or zero-point saturation require care.

## Simulation alone

```python
import numpy as np
from data_analysis_helper.release_recapture import release_recapture

# Same trap units and defaults as fit_release_recapture().
simulation_trap = dict(
    waist_um=0.95,
    depth_MHz=15.627464342495682,  # U/h in MHz; equivalent to U/k = 0.75 mK
    wavelength_nm=850,
)
probability, mc_stderr, initially_bound_fraction = release_recapture(
    50e-6,                  # temperature in K
    np.linspace(0, 50e-6, 101),  # release times in seconds
    **simulation_trap,
    samples=50_000,
    seed=42,
)
```

The simulator converts trap inputs to SI internally. Its temperature and time
arguments remain in **kelvin and seconds**; the fitter takes times in μs and
returns temperature in μK. Both default to Rb-87 mass and `zero_point=True`.

The simulation uses harmonic thermal initial conditions, ballistic release,
and an energy-based recapture test in a Gaussian tweezer. Its harmonic initial
approximation requires `k*T << h*depth_MHz*1e6`. `zero_point=True` includes zero-point
widths, with an approximate classical capture test near zero temperature.

Add future analysis modules beside `release_recapture.py`, keeping calculations
independent of notebook globals and documenting input/output units.

Run the focused checks from `DailyAnalysis`:

```sh
python -m unittest discover -s data_analysis_helper/tests -p 'test_release_recapture.py'
```
