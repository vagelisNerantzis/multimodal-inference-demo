"""Sample-aware inference over selected observations only; no ground-truth access."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
import math
import numpy as np
from ..data.synthetic import SyntheticSample
from ..models.encoders import LinearEncoder, build_encoders
from ..models.classifier import PrototypeClassifier
from ..fusion.mean import mean_fusion
from ..uncertainty.metrics import entropy as categorical_entropy
from ..provenance.record import ProvenanceRecord, create_record
from ..validation.inputs import identifier, names, finite_vector, probability_vector


@dataclass(frozen=True)
class InferenceResult:
    """One immutable physical-sample prediction, its selected views and provenance."""
    sample_id: str
    modalities_used: tuple[str, ...]
    per_modality_embeddings: Mapping[str, tuple[float, ...]]
    fused_embedding: tuple[float, ...]
    class_probabilities: tuple[float, ...]
    predicted_class: str
    confidence: float
    entropy: float
    provenance: ProvenanceRecord

    def __post_init__(self) -> None:
        identifier(self.sample_id, name='sample_id')
        used = tuple(sorted(names(self.modalities_used)))
        if set(self.per_modality_embeddings) != set(used):
            raise ValueError('Embedding modalities differ from modalities_used')
        fused = tuple(float(v) for v in finite_vector(self.fused_embedding, name='fused embedding'))
        embedded = {m:tuple(float(v) for v in finite_vector(self.per_modality_embeddings[m],
                    name='modality embedding', dimension=len(fused))) for m in used}
        probabilities = probability_vector(self.class_probabilities)
        if self.predicted_class != f'class_{int(np.argmax(probabilities))}':
            raise ValueError('Prediction differs from class probability argmax')
        if not math.isfinite(self.confidence) or not math.isclose(self.confidence, max(probabilities), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Confidence differs from maximum class probability')
        if not math.isfinite(self.entropy) or not math.isclose(self.entropy, categorical_entropy(probabilities), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Entropy differs from class distribution')
        if self.provenance.sample_id != self.sample_id or self.provenance.modalities_used != used:
            raise ValueError('Result identity differs from provenance')
        if self.provenance.prediction != self.predicted_class or self.provenance.confidence != self.confidence or self.provenance.entropy != self.entropy:
            raise ValueError('Result prediction differs from provenance')
        object.__setattr__(self, 'modalities_used', used)
        object.__setattr__(self, 'per_modality_embeddings', MappingProxyType(embedded))
        object.__setattr__(self, 'fused_embedding', fused)
        object.__setattr__(self, 'class_probabilities', probabilities)


def infer_sample(sample: SyntheticSample, modalities: Sequence[str], *,
                 encoders: Mapping[str, LinearEncoder] | None = None,
                 classifier: PrototypeClassifier | None = None) -> InferenceResult:
    """Encode, fuse and classify one physical sample's explicitly selected modalities.

    Defaults use known synthetic measurement geometry and fixed prototypes.
    true_class and unselected observation values are never read by inference.
    """
    if not isinstance(sample, SyntheticSample):
        raise TypeError('sample must be a SyntheticSample')
    used = tuple(sorted(names(modalities)))
    if not set(used).issubset(sample.available_modalities):
        raise ValueError('Requested modality is unavailable for this physical sample')
    encoders = build_encoders(sample.modality_specs, sample.embedding_dim, sample.seed) if encoders is None else encoders
    classifier = PrototypeClassifier(sample.n_classes, sample.embedding_dim) if classifier is None else classifier
    if classifier.n_classes != sample.n_classes:
        raise ValueError('Classifier class schema differs from the sample declaration')
    embedded = {}
    selected_inputs = {}
    identities = []
    for name in used:
        if name not in encoders:
            raise ValueError(f'No encoder supplied for modality {name}')
        encoder = encoders[name]
        observation = sample.observations[name]
        if observation.sample_id != sample.sample_id or observation.modality != name:
            raise ValueError('Observation identity differs from the physical sample/modality')
        if encoder.modality != name or encoder.embedding_dim != classifier.embedding_dim:
            raise ValueError('Encoder modality or embedding dimension differs from classifier schema')
        value = encoder.encode(observation.features)
        embedded[name] = tuple(float(v) for v in finite_vector(value, name='encoded observation', dimension=classifier.embedding_dim))
        selected_inputs[name] = (observation.observation_id, observation.features)
        identities.append((name, encoder.identifier))
    fused = tuple(float(v) for v in mean_fusion(embedded))
    prediction = classifier.predict(fused)
    uncertainty = categorical_entropy(prediction.probabilities)
    provenance = create_record(sample_id=sample.sample_id, modalities_used=used,
                encoder_identifiers=tuple(identities), model_identifier=classifier.identifier,
                seed=sample.seed, prediction=prediction.predicted_class, confidence=prediction.confidence,
                entropy=uncertainty, observation_inputs=selected_inputs)
    return InferenceResult(sample.sample_id, used, embedded, fused, prediction.probabilities,
                           prediction.predicted_class, prediction.confidence, uncertainty, provenance)
