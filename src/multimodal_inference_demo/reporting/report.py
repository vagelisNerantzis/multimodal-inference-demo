"""Human-readable reports consume results; they do not change inference or provenance."""
from ..inference.pipeline import InferenceResult
from ..acquisition.recommend import Recommendation


def format_report(result: InferenceResult, recommendation: Recommendation | None = None) -> str:
    """Render one concise physical-sample report from already computed values."""
    lines = [f'Sample: {result.sample_id}', f'Modalities used: {", ".join(result.modalities_used)}',
             f'Prediction: {result.predicted_class}', f'Confidence: {result.confidence:.6f}',
             f'Entropy: {result.entropy:.6f} nats']
    if recommendation is not None:
        lines += [f'Recommended next modality: {recommendation.recommended_modality or "None"}',
                  f'Estimated utility: {recommendation.estimated_utility:.6f}', f'Reason: {recommendation.reason}']
    lines.append(f'Run ID: {result.provenance.run_id}')
    return '\n'.join(lines)
