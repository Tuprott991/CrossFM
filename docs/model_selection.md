# Phase 1 model selection

## Specialist: TabICL v2

Selected `tabicl==2.2.0`, checkpoint `tabicl-classifier-v2-20260212.ckpt` at Hugging Face revision `4dcd344e...`, and upstream source commit `0dbff3ec...`.

Reasons: released pretrained weights, permissive open implementation, probabilistic classification, accessible PyTorch source, and an upstream minimal architecture intended for experimentation. The model is loaded once per worker and uses FP16 without FlashAttention 3 on T4. Its documented pretraining range starts at 300 rows, so A/C are a deliberate low-data extrapolation and a risk disclosed in advance.

Rejected for this gate: nanoTabPFN is exceptionally modifiable but has no comparable ready-to-use pretrained classifier in its minimal distribution; pretraining it inside Phase 1 would confound backbone selection with prior design.

## Generalist v1: Qwen3-0.6B (superseded)

Selected `Qwen/Qwen3-0.6B`, Hugging Face commit `c1899de...`. It is Apache-2.0, below the 1.5B limit, exposes hidden states through Transformers, supports non-thinking chat mode, and fits comfortably beside experiment buffers on a 16 GB T4. Phase 1 uses normalized next-token probabilities for the single-token labels `0` and `1`; a preflight verifies hidden-state access for later phases.

The v1 run showed severe verbalizer collapse: the model predicted class 1 on under 1% of queries. Full response likelihood and semantic verbalizers did not repair the 0.6B checkpoint locally, while bounded thinking was too verbose for the evaluation contract.

## Generalist v2: Qwen2.5-1.5B-Instruct

The exploratory v2 protocol selects `Qwen/Qwen2.5-1.5B-Instruct` at Hugging Face commit `989aa798...`. Predictions use deterministic generation capped at eight tokens and must contain exactly one distinct `LOW` or `HIGH` label; malformed or ambiguous generations fail the run. Regime A uses metadata-selected semantic feature serialization because its context is deliberately non-identifying. A 40-query local gate reached 97.5% accuracy before the Kaggle smoke submission.
