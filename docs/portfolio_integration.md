# Portfolio integration boundaries

This repository demonstrates compatible scientific concepts without importing,
copying large implementations from, or requiring installation of another
portfolio project. Every dependency is a normal package dependency; there are
no neighboring-checkout or machine-path dependencies.

| Repository | Scientific responsibility | Relation to this demo |
|---|---|---|
| scientific-ml-benchmarking | Physical-sample splits, leakage-safe evaluation, metrics, calibration, statistics and provenance | Inference preserves sample identity; it does not implement or claim benchmark validation or calibration. |
| multimodal-representation-learning | Modality encoders, common/aligned embeddings, fusion and representation evaluation | Fixed synthetic encoders and a small mean illustrate the interface; they do not reproduce trained shared spaces or learned material tokens. |
| adaptive-modality-acquisition | Candidate utility, costs, stopping rules and sequential acquisition histories | A small supplied-gain adapter recommends one next modality; it does not reproduce the full episode simulator, budget constraints or a learned router. |
| multimodal-inference-demo | End-to-end inference interaction, reporting and reproducibility | Generates selected observations, encodes, fuses, predicts, describes uncertainty and reports a traceable constructed run. |

## A compatible integration path

1. Establish immutable physical-sample membership and observation lineage in the
   dataset/evaluation layer. Preserve each specimen across modality joins.
2. Supply audited modality encoders with an explicit shared dimension and class
   schema. Common dimensions alone do not establish semantic alignment.
3. Aggregate repeated observations within each modality when that data model is
   needed. This demo currently provides one numeric vector per modality/sample.
4. Replace synthetic prototypes with an externally validated classifier while
   retaining sample-level output and explicit model/configuration identifiers.
5. Evaluate and calibrate confidence on the appropriate development partitions.
   A demo's maximum softmax probability is not a calibrated correctness estimate.
6. Supply acquisition utility forecasts without exposing actual unacquired
   measurements or test outcomes to the recommendation policy.
7. Record exactly the selected inputs and software/model configuration, then
   generate a human-readable report from the immutable inference output.

These are conceptual extension points. They introduce no runtime dependency or
claim of interoperability tests against the other repositories.

## Scientific safeguards

Synthetic labels are generation/evaluation metadata, never inference inputs.
Order-invariant modality processing is tested, and provenance excludes unused
features. Recommendations use declared forecasts; the example is not an oracle
for real measurement value. No real textile data or historical results are
bundled, and favorable constructed confidence changes are not evidence of
real-world generalization or optimal acquisition.
