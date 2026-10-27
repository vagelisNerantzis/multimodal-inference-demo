"""Synthetic integration flow; no real data, learned models or acquisition claims."""
from multimodal_inference_demo.data.synthetic import generate_samples
from multimodal_inference_demo.inference.pipeline import infer_sample
from multimodal_inference_demo.acquisition.recommend import recommend_next
from multimodal_inference_demo.reporting.report import format_report


def main() -> None:
    """Infer, recommend, reveal a generated view and report a deterministic sample."""
    sample = generate_samples(seed=42)[0]
    used = ['macro']
    costs = {name: spec.cost for name,spec in sample.modality_specs.items()}
    # Declared forecast fractions of remaining confidence headroom. No candidate
    # feature values or ground-truth labels inform this illustrative adapter.
    fractions = {'spectral': .90, 'microscopy_low': .40, 'microscopy_high': 1., 'macro': .20}
    print('All inputs are generated; fixed geometry and prototypes are not trained.')
    initial = infer_sample(sample, used)
    result = initial
    while True:
        gains = {name: (1-result.confidence)*fractions[name]
                 for name in sample.available_modalities if name not in result.modalities_used}
        recommendation = recommend_next(result, sample.available_modalities, gains, costs,
                                         cost_weight=.02, confidence_threshold=.90)
        print('\n'+format_report(result, recommendation))
        if recommendation.recommended_modality is None:
            break
        used.append(recommendation.recommended_modality)
        result = infer_sample(sample, used)
    print(f'\nConfidence change: {initial.confidence:.6f} -> {result.confidence:.6f}')
    print(f'Entropy change: {initial.entropy:.6f} -> {result.entropy:.6f} nats')
    print('Final deterministic provenance:')
    print(result.provenance.to_json())


if __name__ == '__main__':
    main()
