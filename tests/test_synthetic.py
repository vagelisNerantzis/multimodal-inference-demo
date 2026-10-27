"""Generation has explicit physical identities and modality-local seeded noise."""
from dataclasses import replace, FrozenInstanceError
import numpy as np
import pytest
from multimodal_inference_demo.data.synthetic import ModalitySpec, Observation, generate_samples


def test_generation_is_deterministic():
    first, second = generate_samples(), generate_samples()
    for a, b in zip(first, second):
        assert a.sample_id == b.sample_id and a.true_class == b.true_class
        for name in a.available_modalities:
            np.testing.assert_array_equal(a.observations[name].features, b.observations[name].features)


def test_unique_ids_and_shared_physical_class():
    samples = generate_samples(n_samples=9, n_classes=3)
    assert len({s.sample_id for s in samples}) == 9
    assert {s.true_class for s in samples} == {0, 1, 2}
    for sample in samples:
        assert all(observation.sample_id == sample.sample_id for observation in sample.observations.values())
        assert all(observation.modality == name for name, observation in sample.observations.items())


def test_custom_modalities_availability_and_feature_shapes():
    specs = {"sensor_z": ModalitySpec(4, .1, 1), "sensor_a": ModalitySpec(7, .2, 2)}
    samples = generate_samples(n_samples=2, n_classes=2, modalities=specs, embedding_dim=5)
    for sample in samples:
        assert set(sample.available_modalities) == set(specs) == set(sample.observations)
        assert sample.n_classes == 2 and sample.embedding_dim == 5
        assert all(np.shape(sample.observations[name].features) == (spec.feature_dim,) for name, spec in specs.items())


def test_seed_changes_noisy_features():
    a = generate_samples(seed=3)[0]
    b = generate_samples(seed=4)[0]
    assert any(not np.array_equal(a.observations[name].features, b.observations[name].features)
               for name in a.available_modalities)


def test_generator_does_not_depend_on_modality_mapping_order():
    specs = {"a": ModalitySpec(5, .2, 1), "z": ModalitySpec(8, .1, 2)}
    a = generate_samples(modalities=specs)[0]
    b = generate_samples(modalities=dict(reversed(list(specs.items()))))[0]
    for name in specs:
        np.testing.assert_array_equal(a.observations[name].features, b.observations[name].features)


@pytest.mark.parametrize("kwargs", [
    {"n_samples": 0}, {"n_classes": 0}, {"embedding_dim": 0}, {"seed": -1},
    {"n_classes": 4, "embedding_dim": 3},
    {"n_classes": 3, "modalities": {"too_small": ModalitySpec(2, 0, 1)}},
])
def test_invalid_generation_configuration(kwargs):
    with pytest.raises((ValueError, TypeError)):
        generate_samples(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"feature_dim": 0}, {"noise": -1.}, {"cost": -1.},
    {"signal_strength": 0.}, {"feature_scale": 0.},
])
def test_invalid_modality_spec(kwargs):
    values = dict(feature_dim=6, noise=.1, cost=1.)
    values.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        ModalitySpec(**values)


def test_sample_snapshots_are_immutable(controlled_samples):
    sample = controlled_samples[0]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        sample.sample_id = "changed"
    with pytest.raises(TypeError):
        sample.observations["extra"] = sample.observations["first"]


def test_sample_rejects_observation_from_different_physical_id(controlled_samples):
    sample = controlled_samples[0]
    observations = dict(sample.observations)
    observations["first"] = replace(observations["first"], sample_id="different_sample")
    with pytest.raises(ValueError):
        replace(sample, observations=observations)


def test_generation_and_encoding_do_not_mutate_global_numpy_rng():
    from multimodal_inference_demo.models.encoders import build_encoders
    before = np.random.get_state()
    sample = generate_samples(n_samples=1, seed=17)[0]
    encoders = build_encoders(sample.modality_specs, sample.embedding_dim, sample.seed)
    for name in sample.available_modalities:
        encoders[name].encode(sample.observations[name].features)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
