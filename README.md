# data_analysis_helper

Analysis helpers for PyCam tweezer experiments: load an experiment folder,
compute loading / recapture probabilities, plot them, and fit models
(Rabi, Ramsey, release–recapture temperature).

## Setup

No installation: add the folder that *contains* `data_analysis_helper` to the
path, then import.

```python
import sys
sys.path.append(r"C:\Users\PETANQUE-PC\Desktop\Tom")

from data_analysis_helper import *
```

Requires NumPy, SciPy and Matplotlib (the Anaconda base environment has them).

## Package layout

| Module | Contents |
| --- | --- |
| `experiment` | `ExperimentData`: load an experiment folder, compute loading / recapture |
| `data_plotter` | `plot_*` functions taking an `ExperimentData` |
| `models` | `rabi_osc`, `ramsey_spectro` fit models for `scipy.optimize.curve_fit` |
| `release_recapture` | Release–recapture simulation and temperature fit |

Everything listed below is re-exported by the package, so
`from data_analysis_helper import ExperimentData, plot_recapture` works.

## Experiment folders

```
my_scan/
  manifest.jsonl   # one JSON object per parameter set: {"tag": ..., "data_file": ...}
  rois.json        # {"centers": [...]}, one per tweezer site
  data/            # one .npy per manifest entry, shape (shots, images, sites)
```

A manifest `tag` lists the scanned parameters as `device+parameter+value`,
separated by `;` for multi-axis scans, e.g. `AOM_Tweezers+amplitude+0.43` or
`AOM+amplitude+0.2;Wait+time+10`. Every entry must scan the same axes in the
same order. Tags without `+` are runs without a scan.

## Loading and computing

```python
exp = ExperimentData("2026/11/09/my_scan", threshold=217)  # prints a summary
# or ExperimentData("my_scan", base_dir=r"D:\Experimental_Data\2026\11\09")
```

A relative path is relative to the current working directory (the notebook's
folder). `print_summary=False` silences the summary.

| Attribute | Meaning |
| --- | --- |
| `data_array` | Counts, shape `(param_sets, shots, images, sites)`; NaN for missing files and for shots padded up to the longest file |
| `param_sets_array` | Scan values (float), shape `(param_sets, scan_axes)`, in manifest order |
| `parameters_labels` | `(device, parameter)` for each scan axis |
| `atom_number`, `scan_nb_axes`, `total_nb_of_sets_of_params` | Number of sites, scan axes, parameter sets |
| `experiment_folder_path` | Absolute path of the folder |
| `threshold` | Default count threshold for "atom present" |

| Method | Returns |
| --- | --- |
| `compute_loading_array()` | Fraction of (valid shot, site) pairs loaded in image 1, per parameter set |
| `compute_recapture_per_atom()` | Per site: shots bright in image 2 / shots loaded in image 1; shape `(param_sets, sites)` |
| `compute_recapture_average()` | Mean of the above over sites that were loaded at least once |
| `compute_recapture_grid(x_axis=0)` | `(x_vals, y_vals, grid)` of the average recapture for a 2-axis scan |
| `get_scan_values(axis)` | Sorted unique values of one scan axis |
| `valid_shots()` | Boolean mask `(param_sets, shots)`, `False` for padded shots |

All `compute_*` methods take an optional `threshold` (default `exp.threshold`)
and return float arrays with **NaN where there is no value** (nothing loaded,
missing file), so filter with `np.isfinite` before fitting:

```python
y = exp.compute_recapture_average()
ok = np.isfinite(y)
popt, pcov = curve_fit(model, exp.param_sets_array[ok, 0], y[ok])
```

### Slicing a scan

Derive smaller experiments from a loaded one, without re-reading the disk.
The original is left unchanged.

| Method | Returns |
| --- | --- |
| `split_by_axis(axis)` | `{value: ExperimentData}`, one experiment per value of `axis`, with that axis removed |
| `fix_axis(axis, value)` | The slice where `axis == value`, with that axis removed; chainable |
| `subset(selection)` | Only the parameter sets in a boolean mask, index list or slice; all axes kept |

```python
# 3-axis scan, 5 values on axis 2 -> five 2-axis experiments
for value, sub in exp.split_by_axis(2).items():
    plot_recapture_2d(sub)

line = exp.fix_axis(2, 10.0).fix_axis(0, 0.3)       # down to a 1D scan
early = exp.subset(exp.param_sets_array[:, 0] < 5)  # arbitrary selection
```

Removed axes are recorded in `fixed_parameters`, and `exp.label` (used in
plot legends and titles) appends them to the folder path, e.g.
`D:\...\my_scan [Wait time = 10]`.

## Plotting

| Function | Plot |
| --- | --- |
| `plot_recapture(exp)` | Average recapture vs the first scan axis |
| `plot_loading(exp)` | Loading probability vs the first scan axis |
| `plot_recapture_per_atom(exp, style="map" \| "curve")` | Recapture of each site (1-axis scan) |
| `plot_recapture_2d(exp, style="map" \| "curve", x_axis=0)` | Heatmap, or one curve per value of the other axis (2-axis scan) |
| `plot_fluorescence(exp, image_index)` | Mean counts above / below threshold in one image |
| `plot_fluo_recapture(exp)` | Image-2 counts of loaded sites vs empty sites |

- Single-axes plots take `ax=` and return the axes. With no `ax`, curves draw
  on the current axes (so a loop overlays several experiments) and 2D maps
  open a new figure.
- The two fluorescence plots take `fig=` and return the figure.
- 1D plots are drawn in increasing x order and skip points with no value.
- Legends and titles show the absolute experiment path; pass `label=` to
  override it.
- To get the plotted numbers, use the `compute_*` methods (or
  `ax.lines[-1].get_xdata()` / `get_ydata()`).

### Styling is up to you

The helpers set no style: no markers, line styles, colors, grid, limits,
figure size or legend position. Choose them with

- rcParams or a style sheet: `plt.style.use("lab.mplstyle")` or
  `plt.rcParams.update({...})`;
- keyword arguments, forwarded to `Axes.plot` (curves), `Axes.pcolormesh`
  (maps) or `Axes.errorbar` (fluorescence panels);
- the returned axes or figure: `ax.set_ylim(-0.03, 1.03)`, `ax.legend(loc=(1.02, 0))`.

```python
plt.rcParams.update({
    "lines.marker": "o", "errorbar.capsize": 3,
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.6,
})

for exp in experiments:
    plot_recapture(exp)

plot_recapture_2d(exp, style="map", vmin=0, vmax=1, cmap="viridis")
plot_fluo_recapture(exp, fig=plt.figure(figsize=(14, 4)), fmt="o-")
```

## Fit models

`rabi_osc(t, rabi_f, delta, phi, amplitude, offset, decay_time)` and
`ramsey_spectro(f, offset, amplitude, f_rabi, f_resonance, T, phi=0)` have the
`f(x, *params)` signature of `curve_fit`; see their docstrings for the
physics and units (keep them consistent, e.g. MHz and μs). To fix a
parameter, wrap the model:

```python
popt, _ = curve_fit(lambda f, o, a, fr, f0: ramsey_spectro(f, o, a, fr, f0, T=5.0), x, y, p0=...)
```

## Release–recapture temperature fit

```python
trap = dict(
    waist_um=0.95,                 # 1/e² intensity radius, μm
    depth_MHz=15.627464342495682,  # U/h in MHz; equivalent to U/k = 0.75 mK
    wavelength_nm=850,             # laser wavelength, nm
)
result = fit_release_recapture(
    exp.param_sets_array[:, 0],       # release time, μs
    exp.compute_recapture_average(),  # probability, 0–1
    **trap,
    samples=50_000,
    seed=42,
    temperature_bounds_uK=(0.1, 200.0),
)  # prints result.summary() and the trap parameters
ax = plot_release_recapture_fit(result)
```

The fit accepts plain arrays, independently of `ExperimentData`. It fits
temperature and initial survival using unweighted least squares. Nonfinite
pairs are omitted and times are sorted. The fixed seed makes repeated fits
reproducible. Rb-87 mass (`mass=1.44316e-25` kg) and `zero_point=True` are
defaults and can be overridden. Use `waist_um` in μm and `wavelength_nm` in
nm. Trap depth is `U/h` in MHz; the function converts lengths to metres and
depth to joules, then derives the Rayleigh range internally.

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

### Simulation alone

```python
probability, mc_stderr, initially_bound_fraction = release_recapture(
    50e-6,                       # temperature in K
    np.linspace(0, 50e-6, 101),  # release times in seconds
    **trap,
    samples=50_000,
    seed=42,
)
```

The simulator converts trap inputs to SI internally. Its temperature and time
arguments are in **kelvin and seconds**; the fitter takes times in μs and
returns temperature in μK. Both default to Rb-87 mass and `zero_point=True`.

The simulation uses harmonic thermal initial conditions, ballistic release,
and an energy-based recapture test in a Gaussian tweezer. Its harmonic initial
approximation requires `k*T << h*depth_MHz*1e6`. `zero_point=True` includes
zero-point widths, with an approximate classical capture test near zero
temperature.

## Arena light-shift analysis (`arena/`)

A separate subpackage, not re-exported by `from data_analysis_helper import *`:

```python
from data_analysis_helper import arena as als

result = als.analyze_arena_ramsey(exp, condition={"Arena AOM amplitude": 0.05},
                                  grating=(41.38, 51.18))  # fits, beam, step, one figure
```

From a Ramsey scan of the arena sites (the last `n_arena` ROIs) and the
reservoir (all other ROIs): Ramsey fit of each site, light shift
`f0(site) - f0(reservoir)` in kHz, Gaussian fit of the arena beam center on the
SLM trap-file positions (μm), and the SLM grating step that brings the arena
onto the beam. The files follow the analysis: `ramsey.py`, `beam.py`,
`analysis.py`, then `plots.py`; see `arena/__init__.py`.

`analyze_arena_ramsey` takes a frequency-only scan. For scans with other axes,
`fit_arena_scans` returns one `ArenaFits` per condition; fit the beam of each
with `fit_beam_center`.

## Development

Keep calculations independent of notebook globals, document input/output
units, and leave plot styling to the caller. Run the tests from the folder
that contains `data_analysis_helper`:

```sh
cd C:\Users\PETANQUE-PC\Desktop\Tom
python -m unittest discover -s data_analysis_helper/tests
```

On a Windows console, set `PYTHONIOENCODING=utf-8` first, otherwise the "μ" in
`fit_release_recapture`'s printout raises `UnicodeEncodeError`.
