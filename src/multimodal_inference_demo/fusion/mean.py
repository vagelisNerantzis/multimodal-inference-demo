"""Small equal-weight fusion adapter; it does not learn alignment or modality weights."""
from collections.abc import Mapping
import numpy as np
from numpy.typing import ArrayLike, NDArray
from ..validation.inputs import identifier, finite_vector


def mean_fusion(embeddings: Mapping[str, ArrayLike]) -> NDArray[np.float64]:
    """Average shared-space [D] vectors in sorted-name order, independent of input ordering."""
    if not isinstance(embeddings, Mapping):
        raise TypeError('embeddings must be a modality-to-vector mapping')
    if not embeddings:
        raise ValueError('Cannot fuse an empty modality set')
    for name in embeddings:
        identifier(name, name='modality')
    values = [finite_vector(embeddings[name], name='embedding') for name in sorted(embeddings)]
    if len({len(value) for value in values}) != 1:
        raise ValueError('Modality embedding dimensions differ')
    fused = np.stack([value/len(values) for value in values]).sum(axis=0)
    return finite_vector(fused, name='fused embedding')
