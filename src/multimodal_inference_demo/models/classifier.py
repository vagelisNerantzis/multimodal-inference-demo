"""Fixed class-axis prototypes and stable, synthetic softmax probabilities."""
from dataclasses import dataclass
import math
import numpy as np
from ..validation.inputs import integer, nonnegative_number, finite_vector, probability_vector


@dataclass(frozen=True)
class Classification:
    """Validated ordered probabilities, their argmax class and maximum probability."""
    probabilities: tuple[float, ...]
    predicted_class: str
    confidence: float

    def __post_init__(self) -> None:
        probabilities = probability_vector(self.probabilities)
        expected = f'class_{int(np.argmax(probabilities))}'
        confidence = nonnegative_number(self.confidence, name='confidence')
        if self.predicted_class != expected:
            raise ValueError('Predicted class disagrees with probability argmax')
        if not math.isclose(confidence, max(probabilities), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Confidence differs from maximum class probability')
        object.__setattr__(self, 'probabilities', probabilities)
        object.__setattr__(self, 'confidence', confidence)


@dataclass(frozen=True)
class PrototypeClassifier:
    """Classify [D] embeddings using fixed first-K coordinate prototypes.

    Prototypes are the first K standard basis vectors; no labels are fitted.
    Scores are prototype dot products divided by a fixed temperature. Resulting
    confidence is synthetic model certainty and is not a calibrated probability
    of correctness on any scientific population.
    """
    n_classes: int
    embedding_dim: int
    temperature: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, 'n_classes', integer(self.n_classes, name='n_classes', minimum=2))
        object.__setattr__(self, 'embedding_dim', integer(self.embedding_dim, name='embedding_dim'))
        if self.n_classes > self.embedding_dim:
            raise ValueError('n_classes cannot exceed the prototype embedding dimension')
        temperature = nonnegative_number(self.temperature, name='temperature')
        if temperature == 0:
            raise ValueError('temperature must be positive')
        object.__setattr__(self, 'temperature', temperature)

    @property
    def identifier(self) -> str:
        """Portable fixed-prototype revision, class schema and temperature."""
        return f'fixed-prototypes-v1:k{self.n_classes}:d{self.embedding_dim}:t{self.temperature:.17g}'

    def predict(self, embedding: tuple[float, ...]) -> Classification:
        """Return stable softmax probabilities over one finite sample embedding."""
        values = finite_vector(embedding, name='sample embedding', dimension=self.embedding_dim)
        with np.errstate(over='ignore', invalid='ignore'):
            logits = values[:self.n_classes] / self.temperature
        logits = finite_vector(logits, name='prototype logits', dimension=self.n_classes)
        # Large finite logit differences may become -inf, whose exp is safely 0.
        with np.errstate(over='ignore', under='ignore'):
            exponential = np.exp(logits - logits.max())
        probabilities = probability_vector(exponential / exponential.sum())
        prediction = f'class_{int(np.argmax(probabilities))}'
        return Classification(probabilities, prediction, max(probabilities))
