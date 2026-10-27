"""Deterministic, path-free provenance for selected observations and fixed model configuration."""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .. import __version__
from ..validation.inputs import identifier, integer, names, nonnegative_number


@dataclass(frozen=True)
class ProvenanceRecord:
    """Immutable report metadata with a content-derived run identifier; no wall clock."""
    sample_id: str
    run_id: str
    package_version: str
    modalities_used: tuple[str, ...]
    encoder_identifiers: tuple[tuple[str, str], ...]
    model_identifier: str
    seed: int
    prediction: str
    confidence: float
    entropy: float
    input_digest: str
    numpy_version: str

    def __post_init__(self) -> None:
        for key in ('sample_id','run_id','package_version','model_identifier','prediction','input_digest','numpy_version'):
            identifier(getattr(self, key), name=key)
        used = tuple(sorted(names(self.modalities_used)))
        encoders = tuple(sorted((identifier(m, name='modality'), identifier(v, name='encoder identifier'))
                                for m,v in self.encoder_identifiers))
        if tuple(m for m,_ in encoders) != used:
            raise ValueError('Encoder provenance must match modalities_used exactly')
        c = nonnegative_number(self.confidence, name='confidence')
        if c > 1:
            raise ValueError('Confidence must be <=1')
        object.__setattr__(self, 'modalities_used', used)
        object.__setattr__(self, 'encoder_identifiers', encoders)
        object.__setattr__(self, 'seed', integer(self.seed, name='seed', minimum=0))
        object.__setattr__(self, 'confidence', c)
        object.__setattr__(self, 'entropy', nonnegative_number(self.entropy, name='entropy'))

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-safe mapping without filesystem or environment paths."""
        return {'sample_id': self.sample_id, 'run_id': self.run_id, 'package_version': self.package_version,
                'modalities_used': list(self.modalities_used),
                'encoder_identifiers': {m:v for m,v in self.encoder_identifiers},
                'model_identifier': self.model_identifier, 'seed': self.seed,
                'prediction': self.prediction, 'confidence': self.confidence, 'entropy': self.entropy,
                'input_digest': self.input_digest, 'numpy_version': self.numpy_version}

    def to_json(self) -> str:
        """Serialize deterministically with sorted keys and no NaN/Inf values."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)


def create_record(*, sample_id: str, modalities_used: tuple[str, ...],
                  encoder_identifiers: tuple[tuple[str, str], ...], model_identifier: str,
                  seed: int, prediction: str, confidence: float, entropy: float,
                  observation_inputs: Mapping[str, tuple[str, tuple[float, ...]]]) -> ProvenanceRecord:
    """Hash only selected observation IDs/features and the declared inference configuration.

    Ground-truth labels and unacquired observations are intentionally excluded.
    A run identifier is a reproducibility fingerprint, not an authentication proof.
    """
    inputs = {m:{'observation_id': observation_inputs[m][0], 'features': list(observation_inputs[m][1])}
              for m in sorted(modalities_used)}
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    identity = {'sample_id':sample_id, 'input_digest':digest, 'modalities':sorted(modalities_used),
                'encoders':sorted(encoder_identifiers), 'model':model_identifier,
                'seed':seed, 'package_version':__version__, 'numpy_version':np.__version__}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return ProvenanceRecord(sample_id, run_id, __version__, modalities_used, encoder_identifiers,
                            model_identifier, seed, prediction, confidence, entropy, digest, np.__version__)
