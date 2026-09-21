# CrossFM

This repository implements the Phase 0/1 falsifiability gate and the exploratory Phase 2 baseline calibration for CrossFM. It does **not** yet implement recurrent CrossFM.

The pilot asks whether a frozen language model and a frozen tabular foundation model exhibit the required A/B/C pattern before any communication architecture is built:

- A: semantic cues should favor the LLM.
- B: abundant anonymized statistics should favor the specialist.
- C: both individual models should be weak, while a non-learned diagnostic that combines the semantically identified feature pair with an empirically estimated direction should recover the signal.

The v1 pilot used `Qwen/Qwen3-0.6B` and exposed a degenerate numeric-label evaluation. The preserved v2 exploratory protocol uses deterministic, strictly parsed `FINAL: LOW/HIGH` generation from `Qwen/Qwen2.5-1.5B-Instruct` at commit `989aa798...`, plus `tabicl==2.2.0` using `tabicl-classifier-v2-20260212.ckpt`. Thresholds are frozen before each full Kaggle run.

Run local CPU-safe tests with `python -m pytest`. Kaggle bundles are produced by `scripts/build_kaggle_bundle.ps1` and validated before upload.

Phase 1 v2 passed all four preregistered gates on a private Kaggle T4x2 run. Phase 2 therefore freezes both backbones and calibrates the complete B0--B6 control family before any bidirectional recurrent model is built:

- constrained-generation Qwen-only and frozen TabICL-only controls;
- a validation-tuned prediction ensemble;
- a learned language-to-specialist view selector;
- a learned specialist-state-to-Qwen soft-prefix adapter;
- an explicit textual/tool evidence channel;
- a three-specialist-call compute-matched one-way control.

Phase 2 training, validation, and test use disjoint semantic aliases. Regime C additionally holds out the interaction family (difference during training, sum during validation, product during test). Both adapters together must remain below five million trainable parameters. Every Phase 2 result is explicitly exploratory and uses a new protocol ID.

The completed v3 exploratory run validates 63/63 tasks. Its most important result is negative for the need for recurrent CrossFM on the current benchmark: the one-way language-to-specialist selector reaches 87.08% on held-out Regime C, versus 58.13% for TabICL, 52.08% for Qwen, and 56.88% for their tuned prediction ensemble. The three-call compute-matched variant is identical at 87.08%. This benchmark therefore does not justify a bidirectional recurrent model yet; the next scientific step is a harder C regime in which a single semantic feature-view choice is insufficient.

Phase 2.5 implements that next gate as Regime C2. Four context-only routing markers encode one of 16 branches through their correlation signs. The natural-language codebook maps the branch to a semantic feature pair; the selected pair's transformation and direction must then be inferred statistically. Train, validation, and test use different aliases, codebook permutations, and pair transformations. A diagnostic two-stage oracle verifies recoverability, while all B0--B6 controls remain eligible to falsify the need for recurrent communication before CrossFM is implemented.

Phase 2.75 is a separate exploratory interface ablation. It leaves the Phase 2.5 run immutable and compares matched TFM-to-LLM adapters carrying either a hard one-hot route or a continuous posterior over all 16 route states. Both have an explicit residual gate around the adapter. Zero routing evidence produces a zero gate and therefore exactly reproduces the frozen LLM prediction; in the semantics-dominant A regime the tiny six-row context likewise triggers the preservation bypass. This phase tests communication-interface robustness only and is not evidence for recurrent CrossFM.

The completed Phase 2.75 full run validates 90/90 tasks. Soft residual routing exactly preserves A at 1.0000 and retains B at 0.9167 versus 0.9222 for TabICL, but reaches only 0.5000 on C2; the matched hard route reaches 0.5083. The adaptive diagnostic remains at 0.9222. Residual preservation is therefore adopted for Phase 3, while the remaining 42-point C2 gap provides a falsifiable target for recurrent communication.

Phase 3 begins with an engineering/overfit gate for a minimal cached-view CrossFM loop. Frozen Qwen embeddings and TabICL predictions are computed once; the backbones are then unloaded and a shared recurrent bridge performs one or two latent query/evidence rounds entirely from GPU-resident tensors. Test episodes are sharded evenly across both T4s. The comparison includes one round, two rounds, zero messages, and shuffled messages with identical bridge capacity. This cached response-bank implementation is a minimal architectural proxy, not yet direct conditioning inside TabICL, and its 16--17 cached specialist views are reported explicitly.

## Full-paper exploratory program

The real-data scale-up is implemented separately under `crossfm/fullpaper/` and
frozen in `configs/fullpaper.yaml`. It covers temporal customer-lapse datasets,
the mandatory BeyondArena transfer subset, tuned tabular baselines, two TFM
backbones, constrained sequence-level LLM likelihoods, both one-way directions,
CrossFM-AR, AR+, SMR, causal ablations, schema robustness, context budgets, and
resumable prediction-level artifacts. The H100 and compliant Kaggle author split
is documented in `docs/fullpaper_runbook.md`.

Local validation does not download models or run inference:

~~~bash
python -m pytest
python -m crossfm.fullpaper.cli plan --config configs/fullpaper.yaml \
  --profile author_b_kaggle_frontier_temporal \
  --output outputs/plan --world-size 2
~~~

The entire program remains exploratory and non-confirmatory. A completed process
is not a scientific result unless every expected task and referenced prediction
artifact validates against the frozen config and checksums.
