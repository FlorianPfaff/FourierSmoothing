"""General Fourier transition operators for torus smoothing."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .smoother import (
    FourierSmoothingResult,
    fourier_to_grid,
    multiply_fourier_truncated,
    multiply_fourier_via_grid,
    normalize_fourier_density,
    reverse_frequencies,
)


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


def fourier_general_transition_smoother(
    filtered_coefficients: ArrayLike,
    likelihood_coefficients: ArrayLike,
    transition_coefficients: ArrayLike | None,
    *,
    normalize_backward: bool = True,
    clip_negative_products: bool = False,
    multiplication: str = "truncated_convolution",
) -> FourierSmoothingResult:
    """Fixed-interval Fourier smoother for dense general transition tensors.

    ``filtered_coefficients`` and ``likelihood_coefficients`` have shape
    ``(T, *coefficient_shape)``. A time-invariant transition tensor has shape
    ``coefficient_shape + coefficient_shape`` with next-state axes first. A
    time-varying sequence has shape
    ``(T-1, *coefficient_shape, *coefficient_shape)``.

    The recursion uses the adjoint contraction in
    :func:`fourier_general_backward_predict`. Coefficient products can use
    aliasing-free linear convolution followed by truncation or same-grid
    multiplication for direct comparison with a grid smoother.
    """

    filtered = np.asarray(filtered_coefficients, dtype=np.complex128)
    likelihoods = np.asarray(likelihood_coefficients, dtype=np.complex128)
    if filtered.ndim < 2 or filtered.shape[0] < 1:
        raise ValueError("filtered_coefficients must have shape (T, *coefficient_shape)")
    if likelihoods.shape != filtered.shape:
        raise ValueError("likelihood_coefficients must have the same shape as filtered_coefficients")

    time_steps = filtered.shape[0]
    coefficient_shape = filtered.shape[1:]
    transition_sequence, time_varying = _validate_transition_sequence(
        transition_coefficients,
        time_steps,
        coefficient_shape,
    )

    backward = np.zeros_like(filtered, dtype=np.complex128)
    smoothed = np.zeros_like(filtered, dtype=np.complex128)
    normalizers = np.empty(time_steps, dtype=np.complex128)

    backward[-1][(0,) * len(coefficient_shape)] = 1.0
    smoothed[-1], normalizers[-1] = normalize_fourier_density(
        _multiply_coefficients(
            filtered[-1],
            backward[-1],
            multiplication=multiplication,
            clip_negative=clip_negative_products,
        )
    )

    for t in range(time_steps - 2, -1, -1):
        future = _multiply_coefficients(
            likelihoods[t + 1],
            backward[t + 1],
            multiplication=multiplication,
            clip_negative=clip_negative_products,
        )
        transition_t = transition_sequence[t] if time_varying else transition_sequence
        backward_t = fourier_general_backward_predict(future, transition_t)
        if backward_t.shape != coefficient_shape:
            raise ValueError(
                f"transition at time {t} returned coefficient shape {backward_t.shape}, expected {coefficient_shape}"
            )
        if normalize_backward:
            scale = float(np.max(np.abs(fourier_to_grid(backward_t))))
            if np.isfinite(scale) and scale > 0.0:
                backward_t = backward_t / scale
        backward[t] = backward_t
        smoothed[t], normalizers[t] = normalize_fourier_density(
            _multiply_coefficients(
                filtered[t],
                backward_t,
                multiplication=multiplication,
                clip_negative=clip_negative_products,
            )
        )

    return FourierSmoothingResult(
        smoothed_coefficients=smoothed,
        backward_messages=backward,
        normalizers=normalizers,
    )


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


def _validate_transition_sequence(
    transition_coefficients: ArrayLike | None,
    time_steps: int,
    coefficient_shape: tuple[int, ...],
) -> tuple[NDArray[np.complex128], bool]:
    transition_shape = coefficient_shape + coefficient_shape
    if time_steps == 1:
        if transition_coefficients is None:
            return np.empty((0, *transition_shape), dtype=np.complex128), True
        transition = np.asarray(transition_coefficients, dtype=np.complex128)
        if transition.shape not in (transition_shape, (0, *transition_shape)):
            raise ValueError(
                f"transition_coefficients must have shape {transition_shape} for a single-state sequence"
            )
        return transition, transition.ndim == len(transition_shape) + 1

    if transition_coefficients is None:
        raise ValueError("transition_coefficients are required for sequences with more than one time step")
    transition = np.asarray(transition_coefficients, dtype=np.complex128)
    if transition.shape == transition_shape:
        return transition, False
    time_varying_shape = (time_steps - 1, *transition_shape)
    if transition.shape == time_varying_shape:
        return transition, True
    raise ValueError(
        f"transition_coefficients must have shape {transition_shape} or {time_varying_shape}, got {transition.shape}"
    )


def _multiply_coefficients(
    left: NDArray[np.complex128],
    right: NDArray[np.complex128],
    *,
    multiplication: str,
    clip_negative: bool,
) -> NDArray[np.complex128]:
    if clip_negative:
        return multiply_fourier_via_grid(left, right, clip_negative=True)
    if multiplication == "grid":
        return multiply_fourier_via_grid(left, right)
    if multiplication in ("truncated_convolution", "linear_convolution", "coefficient"):
        return multiply_fourier_truncated(left, right, output_shape=left.shape)
    raise ValueError(
        "multiplication must be 'truncated_convolution' or 'grid', "
        f"got {multiplication!r}"
    )
