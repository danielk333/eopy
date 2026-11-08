"""Run from outside the checkout after installing a release wheel."""

from importlib.metadata import version
from importlib.resources import files

import numpy as np

import eopy
from eopy import native

assert native.available(), native.library_diagnostics()
assert version("eopy") == eopy.__version__, "Package metadata and Python versions differ"
assert native.version() == eopy.__version__, "Native library is stale; rebuild or reinstall eopy"
assert files("eopy").joinpath("assets/dtm_2000.txt").is_file()
assert files("eopy").joinpath("assets/Orekit-NOTICE.txt").is_file()
r = 7e6
initial = np.array([r, 0, 0, 0, np.sqrt(eopy.EARTH_MU / r), 0])
period = 2 * np.pi * np.sqrt(r**3 / eopy.EARTH_MU)
_, final, _ = eopy.propagate(
    initial,
    0.0,
    period,
    gravity=eopy.GravityField(degree=0),
    options=eopy.Dop853Options(rtol=1e-12, atol=1e-7, max_step=45),
)
np.testing.assert_allclose(final[:3], initial[:3], atol=0.003)
print(f"eopy {eopy.__version__}: installed native propagation verified")
