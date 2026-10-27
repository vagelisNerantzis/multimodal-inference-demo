"""Seeded synthetic measurement geometry and its known fixed decoder."""
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from ..validation.inputs import identifier, integer, nonnegative_number, finite_vector

if TYPE_CHECKING:
    from ..data.synthetic import ModalitySpec


def _stable_seed(seed: int, *parts: str) -> int:
    """Derive a local RNG seed without Python's process-dependent hash()."""
    payload = json.dumps([seed, *parts], ensure_ascii=True, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest()[:8], 'big')


def _modality_basis(modality: str, feature_dim: int, seed: int) -> NDArray[np.float64]:
    """Return a deterministic orthogonal synthetic instrument basis."""
    rng = np.random.default_rng(_stable_seed(seed, 'basis-v1', modality, str(feature_dim)))
    matrix = rng.normal(size=(feature_dim, feature_dim))
    basis, triangular = np.linalg.qr(matrix)
    # Fix QR's arbitrary column signs using a positive diagonal convention.
    signs = np.where(np.diag(triangular) < 0, -1., 1.)
    return basis * signs[np.newaxis, :]


@dataclass(frozen=True)
class LinearEncoder:
    """Decode a known synthetic modality into a common embedding coordinate frame.

    Input is one finite [feature_dim] vector. Preprocessing divides by the
    declared instrument input_scale; the fixed projection reverses the seeded
    orthogonal measurement transform. Missing embedding coordinates are padded
    with zeros. This is known synthetic geometry, not a learned representation.
    """
    modality: str
    feature_dim: int
    embedding_dim: int
    seed: int = 42
    input_scale: float = 1.0
    _basis: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'modality', identifier(self.modality, name='modality'))
        object.__setattr__(self, 'feature_dim', integer(self.feature_dim, name='feature_dim'))
        object.__setattr__(self, 'embedding_dim', integer(self.embedding_dim, name='embedding_dim'))
        object.__setattr__(self, 'seed', integer(self.seed, name='seed', minimum=0))
        scale = nonnegative_number(self.input_scale, name='input_scale')
        if scale == 0:
            raise ValueError('input_scale must be positive')
        object.__setattr__(self, 'input_scale', scale)
        basis = _modality_basis(self.modality, self.feature_dim, self.seed)
        basis.setflags(write=False)
        object.__setattr__(self, '_basis', basis)

    @property
    def identifier(self) -> str:
        """Portable model revision/configuration identifier without local paths."""
        return (f'fixed-linear-v1:{self.modality}:f{self.feature_dim}:d{self.embedding_dim}'
                f':seed{self.seed}:scale{self.input_scale:.17g}')

    def encode(self, features: tuple[float, ...]) -> NDArray[np.float64]:
        """Return a finite [embedding_dim] vector without consuming any RNG state."""
        values = finite_vector(features, name='encoder features', dimension=self.feature_dim)
        with np.errstate(over='ignore', invalid='ignore'):
            decoded = (values / self.input_scale) @ self._basis
        decoded = finite_vector(decoded, name='decoded features', dimension=self.feature_dim)
        result = np.zeros(self.embedding_dim, dtype=np.float64)
        common = min(self.feature_dim, self.embedding_dim)
        result[:common] = decoded[:common]
        return result


def build_encoders(modalities: Mapping[str, 'ModalitySpec'], embedding_dim: int,
                   seed: int = 42) -> dict[str, LinearEncoder]:
    """Build separate known decoders for a declared modality specification mapping."""
    from ..data.synthetic import ModalitySpec
    if not isinstance(modalities, Mapping):
        raise TypeError('modalities must be a mapping of ModalitySpec values')
    if not modalities:
        raise ValueError('Cannot build encoders for an empty modality mapping')
    integer(embedding_dim, name='embedding_dim')
    integer(seed, name='seed', minimum=0)
    for name, spec in modalities.items():
        identifier(name, name='modality')
        if not isinstance(spec, ModalitySpec):
            raise TypeError('Every modality must have a ModalitySpec')
    return {name: LinearEncoder(name, spec.feature_dim, embedding_dim, seed, spec.feature_scale)
            for name, spec in sorted(modalities.items())}
