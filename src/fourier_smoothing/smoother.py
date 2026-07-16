"""Fourier and grid fixed-interval smoothing on hypertori."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class GridSmoothingResult:
    """Grid smoother output."""

    smoothed: NDArray[np.float64]
    backward_messages: NDArray[np.float64]
    normalizers: NDArray[np.float64]


@dataclass(frozen=True)
class FourierSmoothingResult:
    """Fourier identity smoother output."""

    smoothed_coefficients: NDArray[np.complex128]
    backward_messages: NDArray[np.complex128]
    normalizers: NDArray[np.complex128]


@dataclass(frozen=True)
class TorusAdditiveGridTransition:
    """Backward predictor for x[t+1] = x[t] + w[t] on an equidistant torus grid.

    ``noise_density`` is interpreted as the density of ``w`` evaluated at the
    grid offsets. It may have shape ``grid_shape`` or ``(T-1, *grid_shape)``.
    """

    noise_density: NDArray[np.float64]
    cell_volume: float
    normalize_noise: bool = True

    @classmethod
    def for_grid_shape(
        cls,
        noise_density: ArrayLike,
        grid_shape: Iterable[int],
        *,
        normalize_noise: bool = True,
    ) -> "TorusAdditiveGridTransition":
        shape = _shape_tuple(grid_shape)
        return cls(
            noise_density=_as_real(noise_density, "noise_density"),
            cell_volume=cell_volume_for_grid(shape),
            normalize_noise=normalize_noise,
        )

    def __call__(self, next_message: ArrayLike, t: int) -> NDArray[np.float64]:
        message = _as_real(next_message, "next_message")
        noise = self._noise_at(t, message.shape)
        if self.normalize_noise:
            noise = normalize_grid_density(noise, self.cell_volume)

        # beta[j] = int p_w(y - x_j) * message(y) dy.
        # On the periodic grid this is a circular cross-correlation.
        beta = np.fft.ifftn(np.conj(np.fft.fftn(noise)) * np.fft.fftn(message)).real
        beta *= self.cell_volume
        return np.maximum(beta, 0.0)

    def _noise_at(self, t: int, grid_shape: tuple[int, ...]) -> NDArray[np.float64]:
        noise = self.noise_density
        if noise.shape == grid_shape:
            return noise
        if noise.ndim == len(grid_shape) + 1 and noise.shape[1:] == grid_shape:
            return noise[t]
        raise ValueError(f"noise shape {noise.shape} is incompatible with grid shape {grid_shape}")


@dataclass(frozen=True)
class DenseGridTransition:
    """Dense transition-density matrix for grid or histogram smoothers.

    ``transition_density[i, j]`` is interpreted as
    ``p(x_{t+1}=x_i | x_t=x_j)`` evaluated or averaged on the grid. The
    backward prediction computes ``sum_i omega_i K[i, j] message[i]``.
    """

    transition_density: NDArray[np.float64]
    grid_shape: tuple[int, ...]
    cell_volume: float
    normalize_columns: bool = True

    @classmethod
    def for_grid_shape(
        cls,
        transition_density: ArrayLike,
        grid_shape: Iterable[int],
        *,
        cell_volume: float | None = None,
        normalize_columns: bool = True,
    ) -> "DenseGridTransition":
        shape = _shape_tuple(grid_shape)
        matrix = _as_real(transition_density, "transition_density")
        n_grid = int(np.prod(shape))
        if matrix.shape not in ((n_grid, n_grid),) and not (
            matrix.ndim == 3 and matrix.shape[1:] == (n_grid, n_grid)
        ):
            raise ValueError(
                f"transition density shape {matrix.shape} is incompatible with flattened grid size {n_grid}"
            )
        if cell_volume is None:
            cell_volume = cell_volume_for_grid(shape)
        return cls(matrix, shape, float(cell_volume), normalize_columns=normalize_columns)

    def __call__(self, next_message: ArrayLike, t: int) -> NDArray[np.float64]:
        message = _as_real(next_message, "next_message")
        if message.shape != self.grid_shape:
            raise ValueError(f"next_message shape {message.shape} does not match grid shape {self.grid_shape}")
        matrix = self._matrix_at(t)
        values = self.cell_volume * matrix.T @ message.reshape(-1)
        return np.maximum(values.reshape(self.grid_shape), 0.0)

    def forward_predict(self, current_density: ArrayLike, t: int) -> NDArray[np.float64]:
        """Apply the forward transition to grid density values."""

        density = _as_real(current_density, "current_density")
        if density.shape != self.grid_shape:
            raise ValueError(f"current_density shape {density.shape} does not match grid shape {self.grid_shape}")
        matrix = self._matrix_at(t)
        values = self.cell_volume * matrix @ density.reshape(-1)
        return np.maximum(values.reshape(self.grid_shape), 0.0)

    def _matrix_at(self, t: int) -> NDArray[np.float64]:
        matrix = self.transition_density
        if matrix.ndim == 3:
            matrix = matrix[t]
        if self.normalize_columns:
            column_integrals = np.sum(matrix, axis=0, keepdims=True) * self.cell_volume
            if not np.all(np.isfinite(column_integrals)) or np.any(column_integrals <= 0.0):
                raise ValueError("transition-density columns must have positive finite integrals")
            matrix = matrix / column_integrals
        return matrix


def torus_grid(grid_shape: Iterable[int]) -> tuple[NDArray[np.float64], ...]:
    """Return a broadcastable equidistant grid on [0, 2*pi)^d."""

    shape = _shape_tuple(grid_shape)
    axes = [np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) for n in shape]
    return tuple(np.meshgrid(*axes, indexing="ij"))


def cell_volume_for_grid(grid_shape: Iterable[int]) -> float:
    """Uniform quadrature weight for an equidistant grid on T^d."""

    shape = _shape_tuple(grid_shape)
    return float((2.0 * np.pi) ** len(shape) / np.prod(shape))


def normalize_grid_density(values: ArrayLike, cell_volume: float) -> NDArray[np.float64]:
    """Normalize grid values so that sum(values) * cell_volume = 1."""

    arr = np.maximum(_as_real(values, "values"), 0.0)
    integral = float(np.sum(arr) * cell_volume)
    if not np.isfinite(integral) or integral <= 0.0:
        raise ValueError(f"density integral must be positive and finite, got {integral}")
    return arr / integral


def grid_backward_information_smoother(
    filtered: ArrayLike,
    likelihoods: ArrayLike,
    backward_predict: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    *,
    cell_volume: float | None = None,
    normalize_backward: bool = True,
) -> GridSmoothingResult:
    """Fixed-interval smoother using a backward information recursion.

    ``filtered[t]`` is p(x_t | z_1, ..., z_t). ``likelihoods[t]`` is p(z_t|x_t).
    The returned density is proportional to ``filtered[t] * beta[t]``.
    """

    f = np.maximum(_as_real(filtered, "filtered"), 0.0)
    ell = np.maximum(_as_real(likelihoods, "likelihoods"), 0.0)
    _check_time_grid_pair(f, ell)
    if cell_volume is None:
        cell_volume = cell_volume_for_grid(f.shape[1:])

    steps = f.shape[0]
    beta = np.empty_like(f, dtype=np.float64)
    smoothed = np.empty_like(f, dtype=np.float64)
    normalizers = np.empty(steps, dtype=np.float64)

    beta[-1] = 1.0
    smoothed[-1], normalizers[-1] = _normalize_product(f[-1], beta[-1], cell_volume)

    for t in range(steps - 2, -1, -1):
        beta_t = np.maximum(_as_real(backward_predict(ell[t + 1] * beta[t + 1], t), "beta_t"), 0.0)
        if beta_t.shape != f.shape[1:]:
            raise ValueError(f"backward_predict returned {beta_t.shape}, expected {f.shape[1:]}")
        if normalize_backward:
            scale = float(np.max(beta_t))
            if np.isfinite(scale) and scale > 0.0:
                beta_t = beta_t / scale
        beta[t] = beta_t
        smoothed[t], normalizers[t] = _normalize_product(f[t], beta_t, cell_volume)

    return GridSmoothingResult(smoothed, beta, normalizers)


def grid_to_fourier(values: ArrayLike) -> NDArray[np.complex128]:
    """Convert equidistant grid values to Fourier coefficients in NumPy FFT order."""

    arr = np.asarray(values)
    if arr.ndim == 0:
        raise ValueError("values must have at least one dimension")
    return np.fft.fftn(arr) / arr.size


def fourier_to_grid(
    coefficients: ArrayLike,
    *,
    grid_shape: Iterable[int] | None = None,
    real: bool = True,
):
    """Evaluate Fourier coefficients on an equidistant grid.

    If ``grid_shape`` is omitted, the coefficient shape is used. If
    ``grid_shape`` is larger, the coefficient tensor is zero-padded in the
    centered frequency ordering before evaluation. If it is smaller, the
    coefficient tensor is center-truncated.
    """

    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if grid_shape is not None:
        coeffs = resize_fourier_coefficients(coeffs, grid_shape)
    values = np.fft.ifftn(coeffs * coeffs.size)
    return values.real if real else values


def resize_fourier_coefficients(coefficients: ArrayLike, output_shape: Iterable[int]) -> NDArray[np.complex128]:
    """Center-pad or center-crop FFT-order Fourier coefficients to ``output_shape``."""

    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if coeffs.ndim == 0:
        raise ValueError("coefficients must have at least one dimension")
    output_shape = _shape_tuple(output_shape)
    if len(output_shape) != coeffs.ndim:
        raise ValueError(f"output_shape must have length {coeffs.ndim}, got {len(output_shape)}")
    return np.fft.ifftshift(_center_pad_or_crop(np.fft.fftshift(coeffs), output_shape))


def truncate_fourier_coefficients(coefficients: ArrayLike, output_shape: Iterable[int]) -> NDArray[np.complex128]:
    """Alias for :func:`resize_fourier_coefficients` for explicit truncation sites."""

    return resize_fourier_coefficients(coefficients, output_shape)


def reverse_frequencies(coefficients: ArrayLike) -> NDArray[np.complex128]:
    """Return coeffs indexed by the negated frequency multi-index: out[k]=c[-k]."""

    out = np.asarray(coefficients, dtype=np.complex128)
    for axis, n in enumerate(out.shape):
        out = np.take(out, (-np.arange(n)) % n, axis=axis)
    return out


def normalize_fourier_density(coefficients: ArrayLike) -> tuple[NDArray[np.complex128], np.complex128]:
    """Normalize density coefficients using integral = (2*pi)^d * c[0,...,0]."""

    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if coeffs.ndim == 0:
        raise ValueError("coefficients must have at least one dimension")
    integral = (2.0 * np.pi) ** coeffs.ndim * coeffs[(0,) * coeffs.ndim]
    if not np.isfinite(integral.real) or abs(integral) <= 0.0:
        raise ValueError(f"density integral must be nonzero and finite, got {integral}")
    return coeffs / integral, integral


def multiply_fourier_via_grid(
    left: ArrayLike,
    right: ArrayLike,
    *,
    clip_negative: bool = False,
) -> NDArray[np.complex128]:
    """Multiply two Fourier representations by transforming to the grid.

    This computes the product on the implicit same-size grid and therefore
    realizes a circular/aliased coefficient convolution. It is useful for exact
    equivalence with the grid smoother, but not for aliasing-free coefficient-only
    multiplication claims.
    """

    a = np.asarray(left, dtype=np.complex128)
    b = np.asarray(right, dtype=np.complex128)
    if a.shape != b.shape:
        raise ValueError(f"coefficient shapes must match, got {a.shape} and {b.shape}")
    values = fourier_to_grid(a) * fourier_to_grid(b)
    if clip_negative:
        values = np.maximum(values, 0.0)
    return grid_to_fourier(values)


def multiply_fourier_truncated(
    left: ArrayLike,
    right: ArrayLike,
    *,
    output_shape: Iterable[int] | None = None,
) -> NDArray[np.complex128]:
    """Multiply two Fourier representations by linear convolution and truncation.

    The coefficient arrays are assumed to be in NumPy FFT order, i.e. the zero
    frequency is stored at index ``0``. The function computes the aliasing-free
    linear convolution of the represented coefficient tensors and center-truncates
    or center-pads the result to ``output_shape``. If ``output_shape`` is omitted,
    the shape of ``left`` is used.
    """

    a = np.asarray(left, dtype=np.complex128)
    b = np.asarray(right, dtype=np.complex128)
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("coefficient arrays must have at least one dimension")
    if a.ndim != b.ndim:
        raise ValueError(f"coefficient dimensions must match, got {a.ndim} and {b.ndim}")
    if output_shape is None:
        output_shape = a.shape
    output_shape = _shape_tuple(output_shape)
    if len(output_shape) != a.ndim:
        raise ValueError(f"output_shape must have length {a.ndim}, got {len(output_shape)}")

    full_shape = tuple(sa + sb - 1 for sa, sb in zip(a.shape, b.shape))
    axes = tuple(range(a.ndim))
    a_centered = np.fft.fftshift(a)
    b_centered = np.fft.fftshift(b)
    full_centered = np.fft.ifftn(
        np.fft.fftn(a_centered, full_shape, axes=axes)
        * np.fft.fftn(b_centered, full_shape, axes=axes),
        axes=axes,
    )
    truncated_centered = _center_pad_or_crop(full_centered, output_shape)
    return np.fft.ifftshift(truncated_centered)


def torus_identity_backward_predict_fourier(
    next_message_coefficients: ArrayLike,
    noise_coefficients: ArrayLike,
) -> NDArray[np.complex128]:
    """Backward prediction for additive torus dynamics in Fourier coefficients.

    For p_w(y-x) and future message u(y), beta_k = (2*pi)^d p_w[-k] u_k.
    """

    u = np.asarray(next_message_coefficients, dtype=np.complex128)
    w = np.asarray(noise_coefficients, dtype=np.complex128)
    if u.shape != w.shape:
        raise ValueError(f"coefficient shapes must match, got {u.shape} and {w.shape}")
    return (2.0 * np.pi) ** u.ndim * reverse_frequencies(w) * u


def fourier_identity_smoother(
    filtered_coefficients: ArrayLike,
    likelihood_coefficients: ArrayLike,
    noise_coefficients: ArrayLike,
    *,
    normalize_backward: bool = True,
    clip_negative_products: bool = False,
    multiplication: str = "truncated_convolution",
) -> FourierSmoothingResult:
    """Fixed-interval smoother for Fourier identity coefficients on T^d.

    Parameters
    ----------
    multiplication
        ``"truncated_convolution"`` performs aliasing-free coefficient
        convolution followed by truncation. ``"grid"`` reproduces the older
        grid-transform multiplication and is useful for equivalence checks with
        a grid implementation.
    """

    f = np.asarray(filtered_coefficients, dtype=np.complex128)
    ell = np.asarray(likelihood_coefficients, dtype=np.complex128)
    _check_time_grid_pair(f, ell)
    steps, coeff_shape = f.shape[0], f.shape[1:]

    beta = np.zeros_like(f, dtype=np.complex128)
    smoothed = np.zeros_like(f, dtype=np.complex128)
    normalizers = np.empty(steps, dtype=np.complex128)

    beta[-1][(0,) * len(coeff_shape)] = 1.0
    smoothed[-1], normalizers[-1] = normalize_fourier_density(
        _multiply_fourier_coefficients(
            f[-1],
            beta[-1],
            multiplication=multiplication,
            clip_negative=clip_negative_products,
        )
    )

    for t in range(steps - 2, -1, -1):
        future = _multiply_fourier_coefficients(
            ell[t + 1],
            beta[t + 1],
            multiplication=multiplication,
            clip_negative=clip_negative_products,
        )
        beta_t = torus_identity_backward_predict_fourier(future, _noise_coeffs_at(noise_coefficients, t, coeff_shape))
        if normalize_backward:
            scale = float(np.max(np.abs(fourier_to_grid(beta_t))))
            if np.isfinite(scale) and scale > 0.0:
                beta_t = beta_t / scale
        beta[t] = beta_t
        smoothed[t], normalizers[t] = normalize_fourier_density(
            _multiply_fourier_coefficients(
                f[t],
                beta_t,
                multiplication=multiplication,
                clip_negative=clip_negative_products,
            )
        )

    return FourierSmoothingResult(smoothed, beta, normalizers)


def _multiply_fourier_coefficients(
    left: ArrayLike,
    right: ArrayLike,
    *,
    multiplication: str,
    clip_negative: bool,
) -> NDArray[np.complex128]:
    if clip_negative:
        return multiply_fourier_via_grid(left, right, clip_negative=True)
    if multiplication == "grid":
        return multiply_fourier_via_grid(left, right)
    if multiplication in ("truncated_convolution", "linear_convolution", "coefficient"):
        return multiply_fourier_truncated(left, right, output_shape=np.asarray(left).shape)
    raise ValueError(
        "multiplication must be 'truncated_convolution' or 'grid', "
        f"got {multiplication!r}"
    )


def _noise_coeffs_at(noise_coefficients: ArrayLike, t: int, coeff_shape: tuple[int, ...]) -> NDArray[np.complex128]:
    noise = np.asarray(noise_coefficients, dtype=np.complex128)
    if noise.shape == coeff_shape:
        return noise
    if noise.ndim == len(coeff_shape) + 1 and noise.shape[1:] == coeff_shape:
        return noise[t]
    raise ValueError(f"noise coefficient shape {noise.shape} is incompatible with {coeff_shape}")


def _normalize_product(a: NDArray[np.float64], b: NDArray[np.float64], cell_volume: float):
    product = np.maximum(a * b, 0.0)
    normalizer = float(np.sum(product) * cell_volume)
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError(f"smoothing normalizer must be positive and finite, got {normalizer}")
    return product / normalizer, normalizer


def _as_real(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be non-empty and finite")
    return arr


def _shape_tuple(grid_shape: Iterable[int]) -> tuple[int, ...]:
    shape = tuple(int(n) for n in grid_shape)
    if not shape or any(n <= 0 for n in shape):
        raise ValueError("grid_shape must contain positive dimensions")
    return shape


def _center_pad_or_crop(values: NDArray[np.complex128], target_shape: tuple[int, ...]) -> NDArray[np.complex128]:
    if values.ndim != len(target_shape):
        raise ValueError("target_shape must have the same dimensionality as values")
    result = np.zeros(target_shape, dtype=values.dtype)
    source_slices = []
    target_slices = []
    for source_len, target_len in zip(values.shape, target_shape):
        overlap = min(source_len, target_len)
        source_start = (source_len - overlap) // 2
        target_start = (target_len - overlap) // 2
        source_slices.append(slice(source_start, source_start + overlap))
        target_slices.append(slice(target_start, target_start + overlap))
    result[tuple(target_slices)] = values[tuple(source_slices)]
    return result


def _check_time_grid_pair(a: NDArray, b: NDArray) -> None:
    if a.ndim < 2 or a.shape[0] < 1:
        raise ValueError("arrays must have shape (T, *grid_shape)")
    if a.shape != b.shape:
        raise ValueError(f"input shapes must match, got {a.shape} and {b.shape}")
