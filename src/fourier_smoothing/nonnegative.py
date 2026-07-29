"""Tolerance-aware validation of numerically nonnegative arrays."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_NEGATIVITY_RTOL = 1.0e-12
DEFAULT_NEGATIVITY_ATOL = 1.0e-15


def clip_roundoff_nonnegative(
    values: ArrayLike,
    name: str = "values",
    *,
    rtol: float = DEFAULT_NEGATIVITY_RTOL,
    atol: float = DEFAULT_NEGATIVITY_ATOL,
) -> NDArray[np.float64]:
    """Validate nonnegativity and clip only roundoff-scale undershoots.

    Values below ``-(atol + rtol * max(1, max(abs(values))))`` are treated as
    materially negative and raise ``ValueError``. Values inside that tolerance
    are clipped to zero. This prevents invalid transition or density inputs from
    being silently repaired while retaining protection against FFT roundoff.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite")
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0.0 or atol < 0.0:
        raise ValueError("nonnegativity tolerances must be finite and nonnegative")

    scale = max(1.0, float(np.max(np.abs(array))))
    tolerance = float(atol + rtol * scale)
    minimum = float(np.min(array))
    if minimum < -tolerance:
        raise ValueError(
            f"{name} contains materially negative values: minimum={minimum:.6e}, "
            f"allowed_roundoff={tolerance:.6e}"
        )
    return np.maximum(array, 0.0)
