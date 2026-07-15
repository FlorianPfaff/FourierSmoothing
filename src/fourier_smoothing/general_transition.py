"""General Fourier transition operators for torus smoothing."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .smoother import reverse_frequencies


def fourier_general_backward_predict(
    next_message_coefficients: ArrayLike,
    transition_coefficients: ArrayLike,
) -> NDArray[np.complex128]:
    """Apply the adjoint of a general torus transition in Fourier coefficients.

    The transition density is represented as

    ``p(y | x) = sum_{k,l} T[k,l] exp(i k*y) exp(i l*x)``.

    Its coefficient tensor must have shape
    ``next_message_coefficients.shape + current_coefficient_shape``: the first
    axes belong to the next state ``y`` and the remaining axes to the current
    state ``x``. The backward information coefficients are

    ``B[l] = (2*pi)^d sum_k T[k,l] U[-k]``,

    where ``d`` is the next-state torus dimension. Coefficient arrays use NumPy
    FFT order, with the zero frequency at index zero on every axis.

    This operation costs ``O(N_next * N_current)`` for a dense transition tensor.
    Additive identity dynamics should use the diagonal/Hadamard specialization
    in :func:`torus_identity_backward_predict_fourier` instead.
    """

    message = np.asarray(next_message_coefficients, dtype=np.complex128)
    transition = np.asarray(transition_coefficients, dtype=np.complex128)
    if message.ndim < 1:
        raise ValueError("next_message_coefficients must have at least one dimension")
    if transition.ndim <= message.ndim:
        raise ValueError(
            "transition_coefficients must contain next-state axes followed by at least one current-state axis"
        )
    if transition.shape[: message.ndim] != message.shape:
        raise ValueError(
            "the leading transition coefficient axes must match next_message_coefficients: "
            f"got {transition.shape[:message.ndim]} and {message.shape}"
        )

    next_axes = tuple(range(message.ndim))
    reversed_message = reverse_frequencies(message)
    result = np.tensordot(reversed_message, transition, axes=(next_axes, next_axes))
    return (2.0 * np.pi) ** message.ndim * np.asarray(result, dtype=np.complex128)


def fourier_general_forward_predict(
    current_density_coefficients: ArrayLike,
    transition_coefficients: ArrayLike,
    *,
    next_dimension: int,
) -> NDArray[np.complex128]:
    """Apply a general torus transition to current density coefficients.

    ``transition_coefficients`` uses the same layout as in
    :func:`fourier_general_backward_predict`: the first ``next_dimension`` axes
    belong to the next state and the remaining axes to the current state. The
    forward coefficients are

    ``P[k] = (2*pi)^d sum_l T[k,l] F[-l]``,

    where ``d`` is the current-state torus dimension.
    """

    current = np.asarray(current_density_coefficients, dtype=np.complex128)
    transition = np.asarray(transition_coefficients, dtype=np.complex128)
    if current.ndim < 1:
        raise ValueError("current_density_coefficients must have at least one dimension")
    if next_dimension < 1:
        raise ValueError("next_dimension must be positive")
    if transition.ndim != next_dimension + current.ndim:
        raise ValueError(
            "transition coefficient dimensionality must equal next_dimension plus the current-state dimension"
        )
    if transition.shape[next_dimension:] != current.shape:
        raise ValueError(
            "the trailing transition coefficient axes must match current_density_coefficients: "
            f"got {transition.shape[next_dimension:]} and {current.shape}"
        )

    current_axes = tuple(range(next_dimension, transition.ndim))
    reversed_current = reverse_frequencies(current)
    result = np.tensordot(transition, reversed_current, axes=(current_axes, tuple(range(current.ndim))))
    return (2.0 * np.pi) ** current.ndim * np.asarray(result, dtype=np.complex128)


def transition_grid_to_fourier(
    transition_values: ArrayLike,
    *,
    next_dimension: int,
) -> NDArray[np.complex128]:
    """Convert an equidistant transition grid to joint Fourier coefficients.

    ``transition_values`` must use next-state grid axes first and current-state
    grid axes second. ``next_dimension`` is checked explicitly to make the axis
    convention visible at call sites.
    """

    values = np.asarray(transition_values)
    if next_dimension < 1 or next_dimension >= values.ndim:
        raise ValueError("next_dimension must split transition_values into nonempty next/current axis groups")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("transition_values must be non-empty and finite")
    return np.fft.fftn(values) / values.size
