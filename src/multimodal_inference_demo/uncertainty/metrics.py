"""Confidence and natural-log entropy; these do not fit or imply calibration."""
import numpy as np
from numpy.typing import ArrayLike
from ..validation.inputs import probability_vector


def entropy(probabilities: ArrayLike, *, normalized: bool = False) -> float:
    """Return categorical entropy in nats, or divide by log(K) when normalized."""
    p = np.asarray(probability_vector(probabilities))
    logs = np.zeros_like(p)
    np.log(p, out=logs, where=p > 0)
    result = float(-np.sum(p * logs))
    return result / float(np.log(len(p))) if normalized else result


def confidence(probabilities: ArrayLike) -> float:
    """Return max(p) for a validated class distribution."""
    return max(probability_vector(probabilities))
