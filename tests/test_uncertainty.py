"""Entropy and confidence are probability metrics, not correctness guarantees."""
import numpy as np
import pytest
from multimodal_inference_demo.uncertainty.metrics import entropy, confidence


def test_deterministic_distribution_entropy_and_confidence():
    assert entropy([1., 0., 0.]) == pytest.approx(0.)
    assert confidence([1., 0., 0.]) == 1.


def test_uniform_distribution_maximum_entropy():
    assert entropy([.25] * 4) == pytest.approx(np.log(4))
    assert entropy([.25] * 4, normalized=True) == pytest.approx(1.)


@pytest.mark.parametrize("probabilities", [[], [.3, .3], [1.1, -.1], [float("nan"), 1.], [float("inf"), 0.]])
def test_invalid_probabilities_are_rejected(probabilities):
    for metric in (entropy, confidence):
        with pytest.raises((ValueError, TypeError)):
            metric(probabilities)


def test_zero_probability_log_handling_is_safe():
    assert np.isfinite(entropy([.5, .5, 0.]))
    assert entropy([.5, .5, 0.]) == pytest.approx(np.log(2.))


def test_probability_validation_is_idempotent_at_machine_precision():
    from multimodal_inference_demo.validation.inputs import probability_vector
    # A controlled softmax-like vector whose sum differs from one by roundoff.
    p = (0.576116884765829, 0.21194155761708544, 0.2119415576170854)
    canonical = probability_vector(p)
    for _ in range(10):
        assert probability_vector(canonical) == canonical
    assert abs(sum(canonical)-1.) < 1e-14
