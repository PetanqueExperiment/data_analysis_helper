# Local analysis library

`data_analysis_helper` is a Python package inside `DailyAnalysis`. Import it
directly from notebooks or scripts run from this folder; no installation is
needed. NumPy and SciPy are required for release–recapture fitting, and Matplotlib
is required for plotting. The existing `py314` environment has these dependencies.

## Release–recapture temperature fit

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k
from data_analysis_helper.release_recapture import (
    fit_release_recapture,
    plot_release_recapture_fit,
)

waist = 0.95e-6  # 1/e² intensity radius, m
trap = dict(
    mass=1.44316e-25,         # Rb-87, kg
    waist=waist,
    depth=k * 0.75e-3,       # positive trap depth, J
    rayleigh=np.pi * waist**2 / 850e-9,  # m
)
result = fit_release_recapture(
    exp.param_sets_array[:, 0],       # release time, μs
    exp.compute_recapture_average(),  # probability, 0–1
    **trap,
    samples=50_000,
    seed=42,
    zero_point=True,
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
from data_analysis_helper.release_recapture import release_recapture

probability, mc_stderr, initially_bound_fraction = release_recapture(
    50e-6,                  # temperature in K
    np.linspace(0, 50e-6, 101),  # release times in seconds
    **trap,
    samples=50_000,
    seed=42,
    zero_point=True,
)
```

The simulation uses harmonic thermal initial conditions, ballistic release,
and an energy-based recapture test in a Gaussian tweezer. Its harmonic initial
approximation requires `k*T << depth`. `zero_point=True` includes zero-point
widths, with an approximate classical capture test near zero temperature.

Add future analysis modules beside `release_recapture.py`, keeping calculations
independent of notebook globals and documenting input/output units.

Run the focused checks from `DailyAnalysis`:

```sh
python -m unittest discover -s tests -p 'test_release_recapture.py'
```
