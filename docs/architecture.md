# Architecture and data contracts

## Physical samples and modality observations

The physical sample is the unit of inference. A SyntheticSample carries an
explicit sample_id, ground-truth class metadata, declared class/embedding
schema, modality specifications and available observations. An Observation
has its own identity, sample ID, modality name and finite numeric feature vector.
The sample-level relationship must agree across these fields. This version has
one observation per modality; observation-level representation is not itself a
sample prediction until the selected representations are fused.

The generator is intentionally constructed: labels are cyclic class indices,
class signal lies in known coordinates, and per-modality seeded noise perturbs
it. A deterministic orthogonal transform and raw feature scale produce each
modality view. Feature dimensions and embedding dimension must accommodate the
class count. All observations are artificial; no files or datasets are read.

## Preprocessing and encoders

Each selected modality has a separate fixed LinearEncoder. It reverses the
known input scale and orthogonal synthetic transform, then truncates or pads
into the declared common embedding dimension. The common coordinates are aligned
by construction, not learned from data. Shapes, finite values and identity are
validated explicitly. Encoders preserve sample identity and never use the class
label to compute a representation.

## Fusion and sample classification

Mean fusion receives a nonempty mapping of equal-length embedding vectors.
Sorting modality names fixes the summation order, so dictionary insertion order
cannot change the fused vector. Subsets of explicitly available modalities are
valid; missing, unknown or repeated requested modalities fail rather than being
silently imputed. The result is one representation for one physical sample.

The fixed prototype classifier uses one known coordinate prototype per class.
Softmax is computed stably from prototype logits. It needs no external training
artifact. Probabilities, prediction and confidence describe the constructed
model, not a real scientific classifier or calibrated measurement.

## Uncertainty and recommendation

Entropy uses natural logarithms, with 0 log 0 defined as zero; normalized entropy
uses log(class_count) as its denominator. Confidence is max class probability.
Probability inputs have at least two classes, finite values in [0,1], and a
sum within absolute tolerance 1e-6 of one. Accepted sum drift larger than
K * float64 machine epsilon is normalized by the sum, where K is the class
count; drift within that summation-roundoff bound is left unchanged. Repeated
validation therefore preserves an already normalized distribution and keeps
confidence consistent with max(stored probabilities). Values outside the
validity tolerance fail; no coordinates are clipped or invalid distributions
silently repaired.

The recommendation adapter receives only current inference, available names,
externally supplied estimated confidence gains and costs. It does not inspect
unacquired observations or use true labels. It ranks eligible candidates by
estimated_gain - cost_weight * cost and resolves exact ties lexically. It can
return no recommendation when confidence is sufficient, no modality remains,
or the best utility is non-positive. Expected gains must respect the current
confidence's possible range. These supplied forecasts are not learned or
Bayesian expected information gain.

Acquiring a recommended synthetic observation is explicit in the example;
rerunning inference is a new run with a new selected-input record. The adapter
is not a full cost-budget episode implementation.

## Provenance and reporting

An immutable provenance record connects sample identity, selected modalities,
encoder/model identifiers, seed, package and NumPy versions, outputs, and hashes
of the relevant input/configuration. Canonical ordering and serialization allow
repeated inference to yield identical records without timestamps. True labels,
unselected feature values and machine paths do not participate in inference
provenance. This is traceability, not a signed authenticity proof.

Inference computation and report rendering are separate. format_report turns
an inference result and optional recommendation into concise human-readable text;
it does not recompute scientific outputs or make a new decision. The raw
structured record remains available for programmatic inspection.

## Boundaries and limitations

The pipeline is independently installable and uses NumPy only. It has no
cross-repository imports, private artifacts, GPU, notebook, web service or model
download. It supplies no train/validation/test assignment, model fitting,
calibration, uncertainty interval or real-world performance claim.

Real integration would require externally audited sample lineage, preprocessing,
class schema, independently validated encoders, calibration and forecast quality.
Separate physical samples before any crops, embeddings or modalities are made;
a demonstration using all synthetic observations is not a held-out benchmark.
