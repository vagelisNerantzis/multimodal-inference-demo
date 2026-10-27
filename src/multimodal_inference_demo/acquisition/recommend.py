"""Tiny recommendation adapter using supplied confidence gains, without future-feature access."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
import numpy as np
from ..inference.pipeline import InferenceResult
from ..validation.inputs import names, identifier, nonnegative_number


@dataclass(frozen=True)
class Recommendation:
    """An optional next measurement, its declared utility and readable decision reason."""
    recommended_modality: str | None
    estimated_utility: float
    reason: str

    def __post_init__(self) -> None:
        if self.recommended_modality is not None:
            identifier(self.recommended_modality, name='recommended modality')
        if isinstance(self.estimated_utility, (bool, np.bool_)) or not isinstance(self.estimated_utility, Real):
            raise TypeError('estimated_utility must be real')
        if not math.isfinite(self.estimated_utility):
            raise ValueError('estimated_utility must be finite')
        identifier(self.reason, name='recommendation reason')


def recommend_next(current: InferenceResult, available_modalities: Sequence[str],
                   estimated_gains: Mapping[str, float], costs: Mapping[str, float], *,
                   cost_weight: float = .02, confidence_threshold: float = .90) -> Recommendation:
    """Choose max(supplied confidence gain - cost_weight*cost), with lexical ties.

    Gains are external bounded forecasts, not measured future improvements.
    This adapter neither reads unacquired observations nor performs acquisition.
    """
    if not isinstance(current, InferenceResult):
        raise TypeError('current must be an InferenceResult')
    available = names(available_modalities, allow_empty=True)
    if not set(current.modalities_used).issubset(available):
        raise ValueError('Available modalities must include the currently used evidence')
    weight = nonnegative_number(cost_weight, name='cost_weight')
    threshold = nonnegative_number(confidence_threshold, name='confidence_threshold')
    if not 0 < threshold <= 1:
        raise ValueError('confidence_threshold must be in (0,1]')
    if not isinstance(estimated_gains, Mapping) or not isinstance(costs, Mapping):
        raise TypeError('estimated_gains and costs must be mappings')
    if not set(estimated_gains).issubset(available) or not set(costs).issubset(available):
        raise ValueError('Forecast or cost supplied for an unknown modality')
    if current.confidence >= threshold:
        return Recommendation(None, 0., 'Confidence threshold reached; no further modality recommended.')
    remaining = sorted(set(available) - set(current.modalities_used))
    if not remaining:
        return Recommendation(None, 0., 'No remaining modalities are available.')
    utilities = {}
    for name in remaining:
        if name not in estimated_gains or name not in costs:
            raise ValueError(f'Missing forecast or cost for candidate {name}')
        gain = estimated_gains[name]
        if isinstance(gain, (bool, np.bool_)) or not isinstance(gain, Real):
            raise TypeError('Estimated confidence gains must be real')
        gain = float(gain)
        if not math.isfinite(gain) or not -current.confidence <= gain <= 1-current.confidence:
            raise ValueError('Estimated confidence gain is nonfinite or exceeds feasible probability bounds')
        cost = nonnegative_number(costs[name], name='acquisition cost')
        utility = gain - weight*cost
        if not math.isfinite(utility):
            raise ValueError('Estimated acquisition utility is nonfinite')
        utilities[name] = utility
    best = min(utilities, key=lambda name: (-utilities[name], name))
    score = utilities[best]
    if score <= 0:
        return Recommendation(None, score, 'No positive estimated utility remains after the acquisition cost penalty.')
    return Recommendation(best, score, f'{best} recommended because its supplied expected confidence gain exceeds the acquisition cost penalty.')
