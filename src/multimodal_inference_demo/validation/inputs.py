"""Small shared validators without data access, random state or silent repair."""
from collections.abc import Sequence
from numbers import Integral, Real
import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


def identifier(value: str, *, name: str = 'identifier') -> str:
    """Require a nonempty string with no surrounding whitespace."""
    if not isinstance(value, str):
        raise TypeError(f'{name} must be a string')
    if not value or value != value.strip():
        raise ValueError(f'{name} must be nonempty and have no surrounding whitespace')
    return value


def names(values: Sequence[str], *, name: str = 'modalities', allow_empty: bool = False) -> tuple[str, ...]:
    """Snapshot unique names in supplied order, rejecting a bare string."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f'{name} must be a sequence of names')
    result = tuple(identifier(v, name=name) for v in values)
    if not result and not allow_empty:
        raise ValueError(f'{name} must not be empty')
    if len(set(result)) != len(result):
        raise ValueError(f'{name} contains duplicate names')
    return result


def integer(value: int, *, name: str, minimum: int = 1) -> int:
    """Require a true integer at or above the declared minimum."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer')
    if value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return int(value)


def nonnegative_number(value: float, *, name: str) -> float:
    """Require a finite nonnegative real value, rejecting booleans and strings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def finite_vector(value: ArrayLike, *, name: str = 'vector', dimension: int | None = None) -> NDArray[np.float64]:
    """Return an independent finite float64 1D vector with optional exact dimension."""
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu':
        raise TypeError(f'{name} must contain real numeric values')
    result = np.array(raw, dtype=np.float64, copy=True)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a nonempty finite 1D vector')
    if dimension is not None and len(result) != dimension:
        raise ValueError(f'{name} dimension differs from the declared schema')
    return result


def probability_vector(value: ArrayLike) -> tuple[float, ...]:
    """Validate [K], K>=2; renormalize roundoff only within absolute sum tolerance 1e-6."""
    p = finite_vector(value, name='probabilities')
    total = float(p.sum())
    if len(p) < 2 or (p < 0).any() or (p > 1).any() or abs(total-1.) > 1e-6:
        raise ValueError('Probabilities require >=2 classes, values in [0,1] and sum 1 within 1e-6')
    # A sum of K float64 probabilities has rounding error on the order of K*eps.
    # Leave distributions within that bound unchanged so repeated validation is
    # idempotent and cannot make confidence drift away from max(stored p).
    if abs(total-1.) > len(p) * np.finfo(np.float64).eps:
        p = p / total
    return tuple(float(v) for v in p)
