# data_analysis_helper — clarity and ease of review first

This is the analysis code I care most about. I review every change line by
line, so code must be easy to read and check against the physics, even at
the cost of brevity, speed or generality.

## Priorities (in order)
1. Correct physics, and visibly so: a reader can check each formula against
   the paper/notes without running it.
2. Easy to review: small diffs, one purpose per change, no drive-by edits.
3. Readable: plain, explicit code over clever or compact code.
4. Performance and generality only when I ask for them.

## Style (match the existing modules)
- Plain functions and NumPy arrays. Add a class or dataclass only when it
  clearly simplifies use (like `ExperimentData`, `fit result`); no
  inheritance hierarchies, decorators, metaprogramming or hidden state.
- Units in names for physical quantities: `waist_um`, `depth_MHz`,
  `time_us`, `temperature_uK`. Convert to SI explicitly, in one visible
  place, near the top of the function.
- Physics parameters as keyword-only arguments (`*, waist_um, depth_MHz`).
- Google-style docstrings stating the units of every input and output and
  the model/assumptions (approximations, validity range).
- Comments explain *why* (which formula, which convention, why a factor of
  2 or pi), not what the code does. Reference the formula/source when useful.
- Intermediate results get named variables rather than one long expression,
  when that makes the formula easier to match to the physics.
- Missing values are NaN; never silently drop or fill data. Validate inputs
  and raise `ValueError` with a message that names the expected unit.
- Calculations never depend on notebook globals. Plot helpers set no style
  (no colors, markers, grid, limits); they take `ax=`/`fig=` and `**kwargs`.
- Dependencies: NumPy, SciPy, Matplotlib only. Ask before adding anything.

## How to work here
- Before changing a formula, model or fit: say what changes physically and
  why, and wait for my agreement if it changes results.
- Keep the public API stable (the README tables). If it must change, update
  `README.md` and `__init__.py` re-exports in the same change.
- Add or update a test in `tests/` for every behaviour change; prefer tests
  that check against an analytic limit or a known number.
- Run tests from `C:\Users\PETANQUE-PC\Desktop\Tom`:
  `python -m unittest discover -s data_analysis_helper/tests`
  (set `PYTHONIOENCODING=utf-8` first on a Windows console).
- After a change, summarise it in a few lines: what changed, what physics it
  affects, how it was checked.
