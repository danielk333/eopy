# eopy

Earth orbits in Python: a compact, fast propagation designed around
NumPy arrays and function calls so computationally heavy pieces can later move
to low level implementations without changing the public shape of the code.

## What is implemented

- DOP853/Dormand-Prince 8(5,3) propagation through SciPy `solve_ivp`.
- Batch Cartesian propagation for states shaped `(N, 6)`.
- Central Earth gravity plus analytic J2.
- A low-degree spherical-harmonic API placeholder for Holmes-Featherstone style
  gravity fields.
- DTM2000-like density hook and MSIS density through `pymsis`.
- Drag acceleration with co-rotating atmosphere and ballistic coefficient input.
- Isotropic solar radiation pressure with cylindrical Earth shadowing.
- Point-mass third-body perturbations with fixed or callable position providers,
  including a lazy Astropy-backed ephemeris helper.
- Basic propagation events through SciPy-compatible callables, with helpers for
  altitude crossing and cylindrical eclipse boundaries.
- Frame/time helper wrappers that centralize `spacecoords` and `astropy` usage.

The DTM2000 and high-degree Holmes-Featherstone paths are intentionally
marked with TODOs where Orekit uses large empirical datasets, frame handling and
optimized recursions. The current code favors a stable, vectorized interface and
testable propagation plumbing over pretending those models are complete.

## Example

```python
import numpy as np

from eopy.constants import EARTH_MU
from eopy.gravity import GravityField
from eopy.integrators import Dop853Options
from eopy.propagator import propagate

r = 7000e3
v = np.sqrt(EARTH_MU / r)
states = np.array([
    [r, 0.0, 0.0, 0.0, v, 0.0],
    [r + 1000.0, 0.0, 0.0, 0.0, np.sqrt(EARTH_MU / (r + 1000.0)), 0.0],
])

t, propagated, stats = propagate(
    states,
    0.0,
    3600.0,
    gravity=GravityField(degree=2),
    options=Dop853Options(rtol=1e-10, atol=1e-3, max_step=60.0),
)
```

More runnable examples live in `examples/` and are covered by pytest so they
stay synchronized with the public API.
