# CrossFM

This repository implements the Phase 0/1 falsifiability gate for CrossFM. It does **not** implement CrossFM itself.

The pilot asks whether a frozen language model and a frozen tabular foundation model exhibit the required A/B/C pattern before any communication architecture is built:

- A: semantic cues should favor the LLM.
- B: abundant anonymized statistics should favor the specialist.
- C: both individual models should be weak, while a non-learned diagnostic that combines the semantically identified feature pair with an empirically estimated direction should recover the signal.

Frozen models: `Qwen/Qwen3-0.6B` at commit `c1899de...` and `tabicl==2.2.0` using `tabicl-classifier-v2-20260212.ckpt`. The experiment is exploratory; its thresholds are frozen before the full Kaggle run.

Run local CPU-safe tests with `python -m pytest`. Kaggle bundles are produced by `scripts/build_kaggle_bundle.ps1` and validated before upload.

