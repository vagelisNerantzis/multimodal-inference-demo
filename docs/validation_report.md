# Validation report

Validation date: 2026-10-27. Package version: **0.1.0**.

## Environment and installation

- Python **3.11.15**, NumPy **2.4.6**, pytest **9.1.1**.
- Created an isolated repository-local environment with `python -m venv .venv`.
  It does not inherit system site packages or depend on another portfolio repo.
- Executed `python -m pip install -e ".[dev]" --disable-pip-version-check
  --retries 0 --timeout 15`: passed, exit 0. Runtime dependency is NumPy only;
  pytest is a development extra. No PyTorch, pretrained model or dataset was
  installed. Build/cache/temp products were confined to ignored local storage.
- No local-path dependencies on other repositories are present. Public package
  dependencies are ordinary package-index requirements.

## Executed application validation

- `python -m compileall src tests examples assets`: passed, exit 0.
- `pytest -q`: **74 passed, 0 failed, 0 skipped**, no warnings, exit 0.
- `python examples/run_demo.py`: passed, exit 0. It was executed twice in
  separate processes and produced identical complete standard output.
- The README's full minimal Python example executed successfully and recommended
  spectral from the initial macro-only evidence.
- Package import/version assertion passed. Project metadata, installed
  distribution and `multimodal_inference_demo.__version__` all equal **0.1.0**.
- Application code, tests and demo require only CPU synthetic data, without
  network calls, external datasets, model downloads or GPU access. Dependency
  installation can access a package index; runtime does not require it.

## Coverage and consistency regression

The suite covers generated sample identity, modality availability, cyclic class
metadata, configurable noise and seeds, input shapes, fixed preprocessing and
encoders, prototype probabilities, entropy, selected-view inference, bounded
recommendation gains, cost penalties, ties/stopping, deterministic provenance,
report rendering and end-to-end confidence behavior.

Explicit invariants prove that reversing modality dictionaries leaves the fused
embedding, probabilities, prediction and run ID unchanged; changing ground-truth
metadata does not influence inference; changing unacquired feature values does
not affect the current result/provenance; and generation/encoding do not mutate
global NumPy RNG state.

The initial 73-case run found one exact-consistency failure: repeated probability
normalization could shift a value by one floating-point unit while retaining an
older confidence. The validator now leaves sums within K*float64_epsilon of one
unchanged and normalizes larger accepted drift only. This makes validation
idempotent while retaining the absolute validity tolerance of 1e-6. The existing
exact confidence=max(stored probabilities) assertion was retained, and one new
regression was added. The final suite has 74 passing cases.

## Observed synthetic end-to-end flow

All values are constructed integration checks, not learned performance,
calibration evidence or held-out scientific results.

- Sample: synthetic_000; seed 42; three declared classes; common dimension 8.
- Initial selected modality: macro. Prediction: class_0. Confidence:
  **0.670472**; entropy: **0.861256 nats** (rounded printed values).
- First recommendation: spectral, using supplied confidence-gain forecasts and
  a 0.02 cost weight. After adding it, confidence: **0.812947**; entropy:
  **0.610318 nats**.
- Second recommendation: microscopy_high. Final selected modalities in canonical
  order: macro, microscopy_high, spectral. Final prediction: class_0.
- Final confidence: **0.9413053162233799**; entropy:
  **0.26392659338014124 nats**. The 0.90 confidence threshold stops recommendation.
- Final report contains identity, selected modalities, prediction, confidence,
  entropy, optional recommendation/reason and run ID. Structured provenance
  includes input/configuration digests, package/NumPy version, encoder/model IDs,
  seed and outputs, with no timestamps or machine directories.
- Labels and unacquired feature values are not inputs to prediction or the
  recommendation adapter. Estimated gains are supplied illustrative forecasts,
  not measurements from unavailable observations.

## SVG reproducibility and CI

- `python assets/generate_workflow.py`: passed, exit 0.
- `python assets/generate_workflow.py --check`: passed, exit 0.
- After staging the SVG, the exact CI command `git diff --exit-code -- assets/workflow.svg`
  passed after regeneration (exit 0).
- The generated SVG is pinned to LF in `.gitattributes`. A Git checkout-index
  round trip into ignored local storage preserved its exact bytes on Windows.
- SVG bytes before/after regeneration matched exactly; repeated rendering is
  also covered by pytest. XML parsing passed.
- Workflow SVG SHA-256: `43836961684858aa5306886a6dc4e36adc4ec9ef01bab2f76335531c7ddc09b8`.
- `.github/workflows/tests.yml` exists and targets main pushes/pull requests on
  ubuntu-latest with Python **3.12**, using official checkout/setup-python @v7.
- It upgrades pip, installs the dev extra, compiles, runs pytest and the demo,
  regenerates the diagram, checks `git diff --exit-code -- assets/workflow.svg`,
  and verifies version consistency.
- The complete YAML was parsed with duplicate-key detection; event filters,
  runner, Python version, named steps and SVG diff command were checked. Its
  exact embedded version block compiled and executed locally.

**GitHub-hosted CI has not run.** Local execution above used Python 3.11.15;
it does not certify the future Python 3.12 / Ubuntu run. The badge targets the
intended workflow and is pending publication, not an unsupported success claim.
No secrets, GPU, external data or post-install application network is required.

## Public-content audit and scope

Public deliverables were scanned for Windows absolute paths, user-home paths,
local file URLs, credential assignments and private artifact extensions. No
private paths, credentials, API keys, tokens, real datasets, measured spectra,
model checkpoints or archives are included. All data are generated numeric
vectors at runtime; the workflow SVG is a generic programmatic architecture
figure, not dataset imagery. No real scientific metrics were copied or invented.

Dataset/result/checkpoint ignore rules are root-anchored so the source package
`src/multimodal_inference_demo/data/` remains included in Git.

Ignored environments, caches and logs may contain local installation paths;
they are excluded from public files and Git. Only this new repository was
written. The source scientific project and all three existing portfolio
repositories were untouched. No remote, push, tag or release is performed.

## Complete final public repository tree

Environment, logs, build metadata, bytecode and Git internals are excluded.

```text
.gitattributes
.github/workflows/tests.yml
.gitignore
LICENSE
README.md
assets/README.md
assets/generate_workflow.py
assets/workflow.svg
configs/example.yaml
docs/architecture.md
docs/portfolio_integration.md
docs/validation_report.md
examples/run_demo.py
pyproject.toml
src/multimodal_inference_demo/__init__.py
src/multimodal_inference_demo/acquisition/__init__.py
src/multimodal_inference_demo/acquisition/recommend.py
src/multimodal_inference_demo/data/__init__.py
src/multimodal_inference_demo/data/synthetic.py
src/multimodal_inference_demo/fusion/__init__.py
src/multimodal_inference_demo/fusion/mean.py
src/multimodal_inference_demo/inference/__init__.py
src/multimodal_inference_demo/inference/pipeline.py
src/multimodal_inference_demo/models/__init__.py
src/multimodal_inference_demo/models/classifier.py
src/multimodal_inference_demo/models/encoders.py
src/multimodal_inference_demo/provenance/__init__.py
src/multimodal_inference_demo/provenance/record.py
src/multimodal_inference_demo/reporting/__init__.py
src/multimodal_inference_demo/reporting/report.py
src/multimodal_inference_demo/uncertainty/__init__.py
src/multimodal_inference_demo/uncertainty/metrics.py
src/multimodal_inference_demo/validation/__init__.py
src/multimodal_inference_demo/validation/inputs.py
tests/conftest.py
tests/test_end_to_end.py
tests/test_inference.py
tests/test_models_fusion.py
tests/test_provenance_reporting.py
tests/test_recommendation.py
tests/test_synthetic.py
tests/test_uncertainty.py
```

## Known limitations

- This is a small independent synthetic integration layer, not a trained model,
  real textile classifier, full acquisition engine or production application.
- Projections and prototype geometry are known by construction. Class signal
  strengths differ deliberately. Combining arbitrary noisy views is not
  guaranteed to improve confidence, entropy, correctness or generalization.
- Confidence is uncalibrated softmax model certainty. True labels are retained
  only for controlled evaluation, never inference.
- One observation per modality is modeled. Multiple images/crops per modality,
  train/validation/test enforcement, calibration, confidence intervals and real
  lineage auditing belong in a validated external integration.
- The recommendation adapter uses supplied bounded gains and costs with lexical
  ties and confidence/utility stopping. It does not estimate Bayesian information
  gain, learn a policy or enforce a sequential total-cost budget.
- Provenance is deterministic for the same inputs/configuration/software stack;
  it is not a signature or proof of authenticity. Numerical results may vary
  across NumPy/platform versions, which is why NumPy version is recorded.
- The example YAML is illustrative, not a runtime dependency or automatic config
  loader. Runtime dependencies and conceptual portfolio boundaries stay separate.

## Local Git handoff

Implementation and validation completed before Git initialization. Inspect the
staged public file list, `git status` and `git diff --cached --check` before the
single local initial commit. After staging the SVG, regenerate it and run the
exact CI SVG diff command locally. The final response records the actual commit
hash/status; a report cannot embed its own commit hash without changing it.
No remote configuration, push, tag or release is part of this task.
