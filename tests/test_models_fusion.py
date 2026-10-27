"""Fixed model mechanics and deterministic shared-space fusion."""
import numpy as np
import pytest
from multimodal_inference_demo.models.encoders import LinearEncoder, build_encoders
from multimodal_inference_demo.models.classifier import PrototypeClassifier
from multimodal_inference_demo.fusion.mean import mean_fusion


def test_encoder_known_geometry_and_determinism(controlled_samples):
    sample = controlled_samples[0]
    encoders = build_encoders(sample.modality_specs, sample.embedding_dim, sample.seed)
    encoder = encoders["second"]
    features = sample.observations["second"].features
    expected = np.zeros(8)
    expected[sample.true_class] = 3.
    np.testing.assert_allclose(encoder.encode(features), expected, atol=1e-12)
    np.testing.assert_array_equal(encoder.encode(features), encoder.encode(features))


def test_modality_specific_encoder_geometry():
    features = np.array([1., 2., 3., 4., 5., 6.])
    first = LinearEncoder("first", 6, 8, seed=42)
    second = LinearEncoder("second", 6, 8, seed=42)
    assert not np.allclose(first.encode(features), second.encode(features))
    assert first.identifier != second.identifier


def test_encoder_output_shape_includes_padding():
    assert LinearEncoder("x", 4, 8).encode(np.ones(4)).shape == (8,)


@pytest.mark.parametrize("features", [[], [1., 2.], [[1., 2., 3., 4.]], [1., 2., 3., float("nan")]])
def test_encoder_rejects_invalid_shapes_and_values(features):
    with pytest.raises((ValueError, TypeError)):
        LinearEncoder("x", 4, 8).encode(features)


def test_classifier_has_valid_probabilities_and_known_class():
    classifier = PrototypeClassifier(3, 8)
    embedding = np.zeros(8)
    embedding[2] = 3.
    first, second = classifier.predict(embedding), classifier.predict(embedding)
    assert first.predicted_class == "class_2"
    assert np.isfinite(first.probabilities).all()
    assert min(first.probabilities) >= 0
    assert sum(first.probabilities) == pytest.approx(1.)
    assert first.confidence == max(first.probabilities)
    assert first.probabilities == second.probabilities


def test_classifier_softmax_handles_large_finite_logits():
    result = PrototypeClassifier(3, 8).predict([10000., 9999., 0., 0., 0., 0., 0., 0.])
    assert np.isfinite(result.probabilities).all()
    assert sum(result.probabilities) == pytest.approx(1.)
    assert result.predicted_class == "class_0"


@pytest.mark.parametrize("embedding", [[], np.ones((1, 8)), np.ones(7)])
def test_classifier_rejects_invalid_input_shape(embedding):
    with pytest.raises((ValueError, TypeError)):
        PrototypeClassifier(3, 8).predict(embedding)


def test_mean_fusion_correct_numerical_result():
    np.testing.assert_allclose(mean_fusion({"a": [1., 3.], "b": [3., 5.]}), [2., 4.])


def test_subset_fusion_is_permitted():
    np.testing.assert_array_equal(mean_fusion({"only": [1., 2.]}), [1., 2.])


def test_empty_fusion_is_rejected():
    with pytest.raises(ValueError):
        mean_fusion({})


def test_fusion_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError):
        mean_fusion({"a": [1., 2.], "b": [1., 2., 3.]})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_fusion_nonfinite_values_are_rejected(bad):
    with pytest.raises(ValueError):
        mean_fusion({"a": [1., bad]})
