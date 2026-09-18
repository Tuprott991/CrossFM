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
