"""Float64 arrays; dimensions are documented at each public interface."""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

NDArray_N: TypeAlias = NDArray[np.float64]
NDArray_NxM: TypeAlias = NDArray[np.float64]
