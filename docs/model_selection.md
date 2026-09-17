# Phase 1 model selection

## Specialist: TabICL v2

Selected `tabicl==2.2.0`, checkpoint `tabicl-classifier-v2-20260212.ckpt` at Hugging Face revision `4dcd344e...`, and upstream source commit `0dbff3ec...`.

Reasons: released pretrained weights, permissive open implementation, probabilistic classification, accessible PyTorch source, and an upstream minimal architecture intended for experimentation. The model is loaded once per worker and uses FP16 without FlashAttention 3 on T4. Its documented pretraining range starts at 300 rows, so A/C are a deliberate low-data extrapolation and a risk disclosed in advance.

Rejected for this gate: nanoTabPFN is exceptionally modifiable but has no comparable ready-to-use pretrained classifier in its minimal distribution; pretraining it inside Phase 1 would confound backbone selection with prior design.

## Generalist: Qwen3-0.6B

Selected `Qwen/Qwen3-0.6B`, Hugging Face commit `c1899de...`. It is Apache-2.0, below the 1.5B limit, exposes hidden states through Transformers, supports non-thinking chat mode, and fits comfortably beside experiment buffers on a 16 GB T4. Phase 1 uses normalized next-token probabilities for the single-token labels `0` and `1`; a preflight verifies hidden-state access for later phases.
