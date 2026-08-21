# eopy

Earth orbits in Python: a compact, fast propagation designed around NumPy arrays and function calls
so computationally heavy pieces can later move to low level implementations without changing the
public shape of the code.

## What is implemented

- DOP853/Dormand-Prince 8(5,3) propagation through SciPy `solve_ivp`.
- Batch Cartesian propagation for states shaped `(N, 6)`.
- Central Earth gravity plus analytic J2.
- A low-degree spherical-harmonic and Holmes-Featherstone style gravity fields (TODO).
- DTM2000 density (TODO) and MSIS density through `pymsis`.
- Drag acceleration with co-rotating atmosphere and ballistic coefficient input.
- Isotropic solar radiation pressure with cylindrical Earth shadowing.
- Point-mass third-body perturbations with fixed or callable position providers, including a lazy
  Astropy-backed ephemeris helper.
- Basic propagation events through SciPy-compatible callables.
- On-demand data download and parsing helpers for external gravity, EOP and space-weather inputs.
- A C++ library for the most used functions loaded with `ctypes`, intended for the computationally
  heavy gravity paths.

Large operational data files are not bundled with the package. Use `EOPY_DATA_PATH` to choose a
cache directory, or let eopy use the platform cache directory. The `eopy.data` module can download
CelesTrak space-weather/EOP CSVs and arbitrary ICGEM-compatible gravity model files, and can parse
`.gfc` gravity coefficients into contiguous NumPy arrays.

Native C++ library is built as a shared library with a plain C ABI and loaded from Python with
`ctypes`. It falls back to Python when the library has not been built. Developers can point the
loader at an out-of-tree build with `EOPY_NATIVE_LIBRARY=/path/to/libeopykernels.so`.

For source-tree development, build the native library into the package directory
with:

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
```

Warning, the C++ code is in development and mostly placeholders.

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

For native gravity and external data workflows:

```bash
python examples/download_data_files.py /tmp/eopy-data \
  --no-space-weather \
  --no-eop

python examples/native_gravity_model.py \
  --gravity-model /tmp/eopy-data/EIGEN-6S4_v2.gfc \
  --native-library /path/to/libeopykernels.so \
  --degree 4 \
  --order 4
```

Both examples default to the ICGEM EIGEN-6S4 v2 `.gfc` URL when no gravity URL
or local gravity model path is supplied.
