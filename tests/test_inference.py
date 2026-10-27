"""Inference uses physical identity and selected evidence, never truth metadata."""
from dataclasses import replace
import numpy as np
import pytest
from multimodal_inference_demo.inference.pipeline import infer_sample
from multimodal_inference_demo.uncertainty.metrics import entropy


def test_result_propagates_sample_and_selected_modalities(controlled_samples):
    sample = controlled_samples[1]
    result = infer_sample(sample, ["second", "first"])
    assert result.sample_id == sample.sample_id
    assert result.modalities_used == ("first", "second")
    assert set(result.per_modality_embeddings) == {"first", "second"}
    assert np.shape(result.fused_embedding) == (8,)
    assert result.predicted_class == "class_1"


def test_repeated_inference_is_deterministic(controlled_samples):
    a = infer_sample(controlled_samples[0], ["first", "second"])
    b = infer_sample(controlled_samples[0], ["first", "second"])
    np.testing.assert_array_equal(a.fused_embedding, b.fused_embedding)
    assert a.class_probabilities == b.class_probabilities
    assert a.provenance.to_json() == b.provenance.to_json()


def test_modality_dictionary_order_does_not_change_fusion_prediction_or_provenance(controlled_samples):
    sample = controlled_samples[0]
    reverse_sample = replace(sample, observations=dict(reversed(list(sample.observations.items()))),
                             modality_specs=dict(reversed(list(sample.modality_specs.items()))))
    a = infer_sample(sample, ["first", "second", "third"])
    b = infer_sample(reverse_sample, ["third", "second", "first"])
    np.testing.assert_array_equal(a.fused_embedding, b.fused_embedding)
    assert a.class_probabilities == b.class_probabilities
    assert a.predicted_class == b.predicted_class
    assert a.provenance.run_id == b.provenance.run_id


def test_ground_truth_metadata_is_not_used_for_prediction(controlled_samples):
    sample = controlled_samples[0]
    changed_label = replace(sample, true_class=1)
    a, b = infer_sample(sample, ["first"]), infer_sample(changed_label, ["first"])
    assert a.class_probabilities == b.class_probabilities
    assert a.predicted_class == b.predicted_class
    assert a.provenance.run_id == b.provenance.run_id


def test_unacquired_feature_values_do_not_affect_inference_or_run_id(controlled_samples):
    sample = controlled_samples[0]
    observations = dict(sample.observations)
    third = observations["third"]
    observations["third"] = replace(third, features=tuple(v + 100 for v in third.features))
    changed = replace(sample, observations=observations)
    a, b = infer_sample(sample, ["first"]), infer_sample(changed, ["first"])
    assert a.class_probabilities == b.class_probabilities
    assert a.provenance.to_json() == b.provenance.to_json()


@pytest.mark.parametrize("selected", [[], ["unknown"], ["first", "first"]])
def test_invalid_selected_modalities_fail(controlled_samples, selected):
    with pytest.raises((ValueError, TypeError)):
        infer_sample(controlled_samples[0], selected)


def test_result_uncertainty_matches_probability_vector(initial_result):
    assert initial_result.confidence == max(initial_result.class_probabilities)
    assert initial_result.entropy == pytest.approx(entropy(initial_result.class_probabilities))
