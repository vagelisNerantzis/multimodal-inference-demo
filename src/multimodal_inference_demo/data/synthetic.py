"""Deterministic artificial observations sharing a declared class-axis geometry."""
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from ..models.encoders import _modality_basis, _stable_seed
from ..validation.inputs import identifier, integer, nonnegative_number, finite_vector, names


@dataclass(frozen=True)
class ModalitySpec:
    """Synthetic feature dimension, noise, cost, class strength and raw unit scale."""
    feature_dim: int
    noise: float
    cost: float
    signal_strength: float = 1.0
    feature_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, 'feature_dim', integer(self.feature_dim, name='feature_dim'))
        for name in ('noise', 'cost', 'signal_strength', 'feature_scale'):
            value = nonnegative_number(getattr(self, name), name=name)
            if name in ('signal_strength', 'feature_scale') and value == 0:
                raise ValueError(f'{name} must be positive')
            object.__setattr__(self, name, value)


def default_modalities() -> dict[str, ModalitySpec]:
    """Return illustrative generic views with deliberately different signal quality."""
    return {'macro': ModalitySpec(6, .30, 1., 1.),
            'spectral': ModalitySpec(8, .15, 2., 3.),
            'microscopy_low': ModalitySpec(10, .10, 3., 4.),
            'microscopy_high': ModalitySpec(12, .05, 6., 6.)}


@dataclass(frozen=True)
class Observation:
    """One immutable numeric view with explicit physical and observation identities."""
    sample_id: str
    modality: str
    features: tuple[float, ...]
    observation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'sample_id', identifier(self.sample_id, name='sample_id'))
        object.__setattr__(self, 'modality', identifier(self.modality, name='modality'))
        values = finite_vector(self.features, name='observation features')
        object.__setattr__(self, 'features', tuple(float(v) for v in values))
        observation_id = (f'{self.sample_id}:{self.modality}:0'
                          if self.observation_id is None else self.observation_id)
        object.__setattr__(self, 'observation_id', identifier(observation_id, name='observation_id'))


@dataclass(frozen=True)
class SyntheticSample:
    """One physical sample and its available numeric modalities.

    true_class is generation/evaluation metadata only. It is not an inference
    input. Declared modality specifications may include unavailable modalities;
    available_modalities must match the observations actually present.
    """
    sample_id: str
    true_class: int
    observations: Mapping[str, Observation]
    available_modalities: tuple[str, ...]
    modality_specs: Mapping[str, ModalitySpec]
    n_classes: int
    embedding_dim: int
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, 'sample_id', identifier(self.sample_id, name='sample_id'))
        classes = integer(self.n_classes, name='n_classes', minimum=2)
        dimension = integer(self.embedding_dim, name='embedding_dim')
        label = integer(self.true_class, name='true_class', minimum=0)
        seed = integer(self.seed, name='seed', minimum=0)
        if label >= classes or classes > dimension:
            raise ValueError('Invalid true_class or class/embedding dimension')
        if not isinstance(self.modality_specs, Mapping) or not self.modality_specs:
            raise ValueError('modality_specs must be a nonempty specification mapping')
        specs = {}
        for name, spec in self.modality_specs.items():
            identifier(name, name='modality')
            if not isinstance(spec, ModalitySpec):
                raise TypeError('Every modality must have a ModalitySpec')
            if spec.feature_dim < classes:
                raise ValueError('Every modality feature dimension must cover the class schema')
            specs[name] = spec
        available = names(self.available_modalities, name='available_modalities', allow_empty=True)
        if not isinstance(self.observations, Mapping):
            raise TypeError('observations must be a mapping')
        if set(available) != set(self.observations) or not set(available).issubset(specs):
            raise ValueError('Available modalities must match observations and declared specifications')
        observations = {}
        for name, observation in self.observations.items():
            if not isinstance(observation, Observation):
                raise TypeError('Every observation must be an Observation')
            if observation.sample_id != self.sample_id or observation.modality != name:
                raise ValueError('Observation identity/modality differs from its physical sample')
            if len(observation.features) != specs[name].feature_dim:
                raise ValueError('Observation feature dimension differs from its modality specification')
            observations[name] = observation
        ids = [observation.observation_id for observation in observations.values()]
        if len(set(ids)) != len(ids):
            raise ValueError('Observation IDs must be unique within a physical sample')
        object.__setattr__(self, 'true_class', label)
        object.__setattr__(self, 'n_classes', classes)
        object.__setattr__(self, 'embedding_dim', dimension)
        object.__setattr__(self, 'seed', seed)
        object.__setattr__(self, 'available_modalities', tuple(sorted(available)))
        object.__setattr__(self, 'observations', MappingProxyType(dict(sorted(observations.items()))))
        object.__setattr__(self, 'modality_specs', MappingProxyType(dict(sorted(specs.items()))))


def generate_samples(n_samples: int = 6, n_classes: int = 3,
                     modalities: Mapping[str, ModalitySpec] | None = None,
                     embedding_dim: int = 8, seed: int = 42) -> tuple[SyntheticSample, ...]:
    """Generate independent keyed noise for each physical sample/modality.

    Labels cycle over declared classes. A known class-axis signal is rotated
    into modality-specific raw coordinates and perturbed by Gaussian noise.
    The paired fixed encoders know that rotation; no representation is trained.
    Mapping insertion order and generation batch size do not change an existing
    sample's features. No global NumPy RNG is seeded or consumed.
    """
    count = integer(n_samples, name='n_samples')
    classes = integer(n_classes, name='n_classes', minimum=2)
    dimension = integer(embedding_dim, name='embedding_dim')
    seed = integer(seed, name='seed', minimum=0)
    specs = default_modalities() if modalities is None else modalities
    if not isinstance(specs, Mapping) or not specs:
        raise ValueError('modalities must be a nonempty ModalitySpec mapping')
    if classes > dimension:
        raise ValueError('n_classes cannot exceed embedding_dim')
    for name, spec in specs.items():
        identifier(name, name='modality')
        if not isinstance(spec, ModalitySpec):
            raise TypeError('Every modality must have a ModalitySpec')
        if spec.feature_dim < classes:
            raise ValueError('Every modality feature dimension must cover the class schema')
    samples = []
    for index in range(count):
        sample_id = f'synthetic_{index:03d}'
        label = index % classes
        observations = {}
        for name, spec in sorted(specs.items()):
            rng = np.random.default_rng(_stable_seed(seed, 'observation-v1', sample_id, name))
            latent = rng.normal(0, spec.noise, size=spec.feature_dim)
            latent[label] += spec.signal_strength
            with np.errstate(over='ignore', invalid='ignore'):
                features = spec.feature_scale * (_modality_basis(name, spec.feature_dim, seed) @ latent)
            observations[name] = Observation(sample_id, name, tuple(features))
        samples.append(SyntheticSample(sample_id, label, observations, tuple(sorted(specs)),
                                       specs, classes, dimension, seed))
    return tuple(samples)
