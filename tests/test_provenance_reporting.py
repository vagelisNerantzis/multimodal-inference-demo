"""Deterministic scientific records and readable reports avoid machine metadata."""
from dataclasses import replace
import importlib.metadata
import json
import re
import multimodal_inference_demo
from multimodal_inference_demo.inference.pipeline import infer_sample
from multimodal_inference_demo.acquisition.recommend import recommend_next
from multimodal_inference_demo.reporting.report import format_report


def test_provenance_captures_version_model_modalities_seed_and_prediction(initial_result):
    record = initial_result.provenance
    assert record.package_version == "0.1.0"
    assert record.sample_id == initial_result.sample_id
    assert record.modalities_used == initial_result.modalities_used
    assert record.seed == 42
    assert record.prediction == initial_result.predicted_class
    assert record.confidence == initial_result.confidence and record.entropy == initial_result.entropy
    assert record.encoder_identifiers and record.model_identifier and record.numpy_version


def test_provenance_serialization_is_stable_and_contains_no_private_paths(initial_result):
    record = initial_result.provenance
    text = record.to_json()
    assert text == record.to_json()
    assert json.loads(text) == record.to_dict()
    assert json.loads(text)["sample_id"] == initial_result.sample_id
    patterns = [r"(?i)(?<![a-z0-9])[a-z]:[\\/]", "file:" + "//", "/" + "home" + "/", "/" + "users" + "/"]
    assert not re.search("|".join(patterns), text)
    assert "timestamp" not in json.loads(text)


def test_selected_observation_changes_provenance_digest(controlled_samples):
    sample = controlled_samples[0]
    observations = dict(sample.observations)
    first = observations["first"]
    observations["first"] = replace(first, features=tuple(v + .01 for v in first.features))
    a = infer_sample(sample, ["first"])
    b = infer_sample(replace(sample, observations=observations), ["first"])
    assert a.provenance.input_digest != b.provenance.input_digest
    assert a.provenance.run_id != b.provenance.run_id


def test_report_includes_identity_prediction_uncertainty_and_recommendation(initial_result):
    chosen = recommend_next(initial_result, ["first", "second"], {"second": .2}, {"second": 1.})
    text = format_report(initial_result, chosen)
    assert initial_result.sample_id in text
    assert initial_result.predicted_class in text
    assert "first" in text and "second" in text
    assert "confidence" in text.lower() and "entropy" in text.lower()
    assert "reason" in text.lower() and initial_result.provenance.run_id in text


def test_report_without_recommendation_is_readable(initial_result):
    text = format_report(initial_result)
    assert initial_result.sample_id in text and initial_result.predicted_class in text
    assert "modalit" in text.lower()


def test_package_import_and_distribution_version():
    assert multimodal_inference_demo.__version__ == "0.1.0"
    assert importlib.metadata.version("multimodal-inference-demo") == "0.1.0"
