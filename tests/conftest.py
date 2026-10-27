"""Controlled synthetic fixtures use no random class ambiguity or external data."""
import pytest
from multimodal_inference_demo.data.synthetic import ModalitySpec, generate_samples
from multimodal_inference_demo.inference.pipeline import infer_sample


@pytest.fixture
def controlled_samples():
    modalities = {
        "first": ModalitySpec(6, 0., 1., signal_strength=1.),
        "second": ModalitySpec(8, 0., 2., signal_strength=3.),
        "third": ModalitySpec(10, 0., 6., signal_strength=6.),
    }
    return generate_samples(n_samples=3, n_classes=3, modalities=modalities,
                            embedding_dim=8, seed=42)


@pytest.fixture
def initial_result(controlled_samples):
    return infer_sample(controlled_samples[0], ["first"])
