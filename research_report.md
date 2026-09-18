# CrossFM Phase 0/1 report and Phase 2 protocol

Status: Phase 1 and exploratory Phase 2 baseline calibration complete. Queued, running, partial, or failed outputs are not evidence.

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

Protocol `crossfm-phase2-baselines-exploratory-v3-memory-safe-likelihood` implements B0 through B6 while freezing both foundation models. B3 maps frozen Qwen task/schema embeddings to a latent choice among specialist feature views and makes one TabICL call at test time. B4 projects the frozen TabICL pre-decoder ICL state into four Qwen soft-prefix tokens. B5 supplies explicit numeric summaries in text. B6 applies the B3 selector to three specialist views, controlling for three specialist calls. The prediction ensemble weight is selected on validation log loss only. LLM decisions use exact joint sequence likelihood for the complete `FINAL: LOW` and `FINAL: HIGH` verbalizers; no unconstrained generated text is parsed. The LM head is evaluated only at suffix-prediction positions to keep this exact calculation within T4 memory.

Aliases are disjoint across train/validation/test, and Regime C holds out its interaction form. Adapter training examples remain well below 100,000, and total trainable parameters are hard-capped at five million. This run is exploratory and cannot be promoted to confirmatory evidence after inspection.

## Phase 2 validated result

Private Kaggle kernel `tuktuai/crossfm-p2-full-v3-likelihood-t4x2` completed all 63 tasks under commit `74ac0ca`. Local validation checked 10,080 predictions, both worker summaries, every prediction/checkpoint hash, finite metrics, the config digest, and the wheel hash.

Mean accuracy across three held-out test seeds (standard deviation in parentheses):

| Method | A | B | C |
|---|---:|---:|---:|
| LLM only | 1.0000 (0.0000) | 0.5125 (0.0653) | 0.5208 (0.0036) |
| TabICL only | 0.6458 (0.0095) | 0.9292 (0.0180) | 0.5813 (0.0165) |
| Prediction ensemble | 0.9375 (0.0108) | 0.9292 (0.0157) | 0.5687 (0.0187) |
| One-way LLM to TFM | 0.4896 (0.0201) | 0.9292 (0.0180) | **0.8708 (0.0219)** |
| One-way TFM to LLM | 0.6292 (0.0732) | 0.9292 (0.0219) | 0.5500 (0.0488) |
| Textual/tool | 0.5188 (0.0063) | 0.4792 (0.0560) | 0.5125 (0.0108) |
| Three-call one-way control | 0.6375 (0.0217) | 0.9292 (0.0180) | **0.8708 (0.0219)** |

The validation-tuned ensemble used alpha 0.35 on the LLM probability. On C, the one-call LLM-to-TFM adapter beat the better individual model by 0.2896 mean accuracy; all three seed deltas were positive (0.2750, 0.2625, 0.3313). It selected the semantically correct held-out alias pair in all 60 C episodes. The three-call control did not improve on the one-call result. In A it selected the wrong singleton consistently, while the LLM-only baseline remained perfect; in B it selected the unrestricted full-table view in all episodes.

The learned adapters used 1,463,430 parameters total (599,557 LLM-to-TFM; 863,873 TFM-to-LLM). Peak allocated GPU memory was 14.72 GB. Rank runtimes were 3,544.9 and 2,335.0 seconds, or about 1.63 aggregate GPU-hours.

## Interpretation and stop decision

Phase 2 falsifies the claim that recurrent bidirectional communication is necessary for the current Regime C. A simple learned one-way semantic view selector already recovers most of the diagnostic oracle's capability and generalizes across held-out aliases and a held-out product mechanism. This is a useful negative result: implementing CrossFM-R on this benchmark would not provide a clean test of the central novelty claim.

Do not proceed to Phase 3 on the current C regime. First redesign C so that a single up-front choice of feature subset cannot solve it. The next benchmark should require specialist evidence to trigger a second, different semantic query—for example, episode-specific exceptions or conflicts where the relevant group itself changes conditional on the first statistical finding. That redesign must preserve A/B calibration, use a new exploratory protocol ID, and be frozen before inspecting outcomes.
