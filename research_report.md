# CrossFM Phase 0/1 report and Phase 2 protocol

Status: Phase 1 complete; Phase 2 baseline calibration pending. Queued, running, partial, or failed outputs are not evidence.

## Hypothesis and benchmark

The immediate gate tests the prerequisite that semantic and statistical backbones have complementary failure modes. Regime C uses six paired feature mechanisms. In context, five auxiliary pair differences are highly correlated with the stable premium–income difference; out of context they decorrelate. Metadata identifies the stable pair, while labels identify the episode-specific direction.

## Architecture and baselines

No recurrent CrossFM architecture is implemented. The repaired Phase 1 baselines are frozen Qwen2.5-1.5B with deterministic `FINAL: LOW/HIGH` generation, frozen TabICL v2 without metadata, and a declared non-learned diagnostic oracle. The oracle is only an identifiability check and is not a model result.

## Phase 1 validated result

The private Kaggle T4x2 protocol `crossfm-phase1-pilot-v2-constrained` completed 27/27 tasks with verified hashes. Mean accuracies were:

| Regime | Qwen only | TabICL only | Diagnostic oracle |
|---|---:|---:|---:|
| A | 0.9854 | 0.6583 | 1.0000 |
| B | 0.5104 | 0.9198 | 0.9302 |
| C | 0.5063 | 0.6104 | 0.9240 |

All Phase 1 gates passed. This supports proceeding to baseline calibration; it is not evidence for latent communication.

## Phase 2 frozen exploratory protocol

Protocol `crossfm-phase2-baselines-exploratory-v1-heldout` implements B0 through B6 while freezing both foundation models. B3 maps frozen Qwen task/schema embeddings to a latent choice among specialist feature views and makes one TabICL call at test time. B4 projects the frozen TabICL pre-decoder ICL state into four Qwen soft-prefix tokens. B5 supplies explicit numeric summaries in text. B6 applies the B3 selector to three specialist views, controlling for three specialist calls. The prediction ensemble weight is selected on validation log loss only.

Aliases are disjoint across train/validation/test, and Regime C holds out its interaction form. Adapter training examples remain well below 100,000, and total trainable parameters are hard-capped at five million. This run is exploratory and cannot be promoted to confirmatory evidence after inspection.
