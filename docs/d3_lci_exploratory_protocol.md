# D3 latent-codebook induction protocol

Protocol: `crossfm-align-fullpaper-exploratory-v10-lci`  
Classification: exploratory, non-confirmatory  
Dataset: RetailRocket temporal lapse/engagement cohort  
Seeds: 25101–25105 (not the seeds used to develop the hypothesis)

## Why this rerun exists

The earlier D3 run showed that CrossFM-AR is a diagnostic coordination protocol, not a general-purpose classifier. Its exact residual path placed almost all probability mass on an LLM whose D3 ranking was anti-informative. CrossFM-SMR performed better because its learned linear fusion could attenuate or invert unreliable frozen-model signals.

This observation does **not** establish that a latent codebook is useful. It motivates a falsifiable next hypothesis: real customer data may contain multiple response regimes for which a global linear fusion is insufficient.

The design is informed by learned discrete representations in [VQ-VAE](https://arxiv.org/abs/1711.00937), the known codebook-collapse failure mode documented by [SQ-VAE](https://arxiv.org/abs/2205.07547), soft routing in mixture-of-experts models, and vector-quantized recommendation representations in [VQ-Rec](https://arxiv.org/abs/2210.12316). It is not a generative VAE and should not be described as one.

## CrossFM-LCI

1. Obtain frozen response probabilities from the LLM and each semantic TFM view.
2. Encode each row's response geometry using view logits, disagreement, uncertainty, and semantic projections.
3. Induce a small codebook with K-means on pre-test rows only.
4. Pass a continuous distance posterior over codewords; the primary model never uses hard code IDs.
5. Fit a global SMR predictor plus code-conditioned residual interactions using convex logistic regression.
6. Select `K ∈ {1,2,4,8}`, posterior temperature, and regularization on the temporal validation split.
7. Require at least `0.002` validation log-loss improvement over `K=1` before accepting a multi-state codebook.
8. Refit the selected configuration on router plus validation rows, then evaluate once on test.
9. Blend the codebook residual with the `K=1` refit SMR fallback. The gate is exactly zero when validation rejects the codebook and also becomes zero for sufficiently out-of-distribution response states.

`K=1` is constructed to be exactly the same feature matrix as SMR. Therefore, it is the scientific null hypothesis, not merely a smaller version of a different model.

## Predeclared comparisons

- `crossfm_smr`: original router-only linear fusion.
- `crossfm_smr_refit`: validation-selected, pre-test-refit linear fusion; the fair capacity null.
- `crossfm_lci`: selected soft latent codebook.
- `ablate_lci_single_state`: forces `K=1`.
- `ablate_lci_no_semantic`: replaces semantic affinities with a uniform prior.
- `ablate_lci_hard`: uses nearest-code assignments instead of a soft posterior.
- `ablate_lci_shuffle_codes`: breaks the association between training rows and code assignments.
- `ablate_lci_fixed16`: tests the earlier unjustified fixed 16-state assumption.

## Decision rules

The dataset's declared primary metric remains average precision. ROC-AUC is a secondary ranking metric, and log loss measures calibration and is the selection criterion.

Evidence for latent regime induction requires all of the following:

1. CrossFM-LCI selects `K>1` in at least three of five seeds.
2. It improves mean test log loss over `crossfm_smr_refit` by at least `0.005` without reducing mean average precision.
3. The shuffled-code control loses the gain.
4. Effective code usage is greater than one and no selected code has negligible posterior mass.

Semantic cross-modal routing additionally requires `ablate_lci_no_semantic` to be worse. If it is not, any gain should be attributed to nonlinear response-regime fusion rather than semantic LLM↔TFM alignment.

If CrossFM-LCI falls back to `K=1`, ties SMR, or fails the controls, the correct conclusion is that D3 supports learned global calibration but not latent codebook induction.

## Leakage boundary

Test labels are not accepted by the selection/refit function and are used only by the metric layer after predictions are frozen. The validation split selects hyperparameters and the residual gate. This protocol was designed after inspecting the earlier D3 run, so even with new seeds its result remains exploratory and cannot serve as independent confirmation.
