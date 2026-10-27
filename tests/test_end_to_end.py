"""Controlled integration and the committed reproducible diagram."""
from pathlib import Path
import runpy
import numpy as np
from multimodal_inference_demo.inference.pipeline import infer_sample
from multimodal_inference_demo.acquisition.recommend import recommend_next


def test_controlled_fusion_improves_confidence_and_reduces_entropy(controlled_samples):
    sample = controlled_samples[0]
    first = infer_sample(sample, ["first"])
    second = infer_sample(sample, ["first", "second"])
    third = infer_sample(sample, ["first", "second", "third"])
    assert first.confidence < second.confidence < third.confidence
    assert first.entropy > second.entropy > third.entropy
    assert first.predicted_class == second.predicted_class == third.predicted_class == "class_0"


def test_recommendation_then_acquisition_updates_same_physical_sample(controlled_samples):
    sample = controlled_samples[0]
    initial = infer_sample(sample, ["first"])
    chosen = recommend_next(initial, sample.available_modalities,
                            {"second": .30, "third": .25}, {"second": 1., "third": 6.})
    assert chosen.recommended_modality == "second"
    updated = infer_sample(sample, [*initial.modalities_used, chosen.recommended_modality])
    repeated = infer_sample(sample, ["first", "second"])
    assert updated.sample_id == initial.sample_id == sample.sample_id
    assert updated.confidence > initial.confidence
    assert updated.entropy < initial.entropy
    np.testing.assert_array_equal(updated.fused_embedding, repeated.fused_embedding)
    assert updated.provenance.to_json() == repeated.provenance.to_json()


def test_workflow_svg_is_deterministic_and_matches_generator():
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(str(root / "assets/generate_workflow.py"))
    first = namespace["render_svg"]()
    second = namespace["render_svg"]()
    assert first == second
    assert first.encode("utf-8") == (root / "assets/workflow.svg").read_bytes()
