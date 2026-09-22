# CrossFM-Align: Five-Day ICLR Full-Paper Scale-Up Plan

**Mode:** aggressive, evidence-first, five calendar days  
**Compute:** one H100 80 GB server plus compliant Kaggle T4x2 sessions  
**Application question:** **Which customers are likely to lapse next month?**

> Do not write “Which customers should lapse?” Prediction supports retention targeting, but deciding whom to contact is a causal-uplift problem requiring intervention data. This paper predicts next-period lapse risk.

## 1. Objective and scientific claim

The paper must establish more than another churn classifier:

> Frozen heterogeneous foundation models can be coordinated through an uncertainty-preserving statistical posterior, a semantic codebook, and a preservation residual—without fine-tuning either backbone—and this coordination improves robust next-period lapse prediction when semantic and statistical evidence are complementary.

Required evidence:

1. Mechanistic synthetic evidence under controlled semantic/statistical conflicts.
2. Real temporal evidence on customer-lapse prediction.
3. Comparisons with tuned GBDTs, modern tabular DL, TFMs, LLM-only, ensembles, and one-way fusion.
4. Robustness to aliases, anonymization, metadata corruption, low-data contexts, and backbone replacement.

### Working title

**Which Customers Will Lapse Next Month? Uncertainty-Preserving Alignment of Frozen Language and Tabular Foundation Models**

Method-first alternative:

**CrossFM-Align: Posterior–Codebook–Residual Routing Between Frozen Heterogeneous Foundation Models**

### Novelty boundary

Do not claim novelty for text embeddings, tool calling, generic adapters, extra inference calls, the number 16, or recurrence. The contribution is:

> **Posterior–Codebook–Residual Routing:** a statistical model communicates uncertainty over latent regimes; a semantic codebook marginalizes it into candidate-view weights; a preservation residual suppresses destructive routing when statistical structure is weak.

| Claim level | Allowed claim | Gate |
|---|---|---|
| L1 | Analytical alignment is sufficient on controlled tasks | Existing validated Phase 5/5.1 |
| L2 | Learned residual improves analytical alignment | Phase 5.2 accuracy or log-loss gain with causal ablations |
| L3 | CrossFM-Align transfers to real lapse prediction | Significant gain on at least 2 temporal datasets |
| L4 | CrossFM-Align is SOTA-competitive | Beats/ties strongest tuned baseline on at least 3 of 4 datasets |
| L5 | General heterogeneous-FM principle | Cross-backbone and cross-domain replication |

## 2. Frozen research questions

- **RQ1:** Does semantic schema knowledge plus local statistical evidence beat either alone?
- **RQ2:** Is a posterior and semantic codebook sufficient, or are learned weights needed?
- **RQ3:** Does preservation prevent negative transfer under weak structure?
- **RQ4:** Does the mechanism transfer under temporal, non-IID lapse evaluation?
- **RQ5:** Does it survive aliases, anonymization, partial metadata, and posterior noise?
- **RQ6:** Does it survive at least two TFMs and two LLM families?
- **RQ7:** Does it improve lift and calibration, not only AUROC?
- **RQ8:** Does the coordination mechanism transfer to standardized non-IID tasks selected independently of our customer-lapse datasets?

## 3. Full-paper architecture

Call the family **CrossFM-Align**. Report AR and AR+; select the primary variant globally on validation, never per test dataset.

### Real-data interface

At cutoff time \(t\):

1. Build views: recency, frequency, monetary value, engagement trend, payment/renewal friction, tenure/contract, product affinity, support behavior, and full table.
2. A frozen TFM produces one probability per view: the response bank.
3. A statistical sensor produces a soft reliability distribution using only pre-\(t\) data.
4. A frozen LLM embeds the task, schema, and view descriptions.
5. The codebook aligns statistical reliability with semantic views.
6. View probabilities are softly aggregated.
7. A preservation gate interpolates the routed result with a frozen base prediction.

Real data must use a factorized posterior over views; do not enumerate \(2^K\) states.

### Analytical and learned variants

AR combines semantic and statistical view logits, then applies softmax. AR+ keeps that route immutable and adds:

\[
\ell_v=\log(w_v^{AR}+\epsilon)/T_\theta+\alpha_\theta\Delta_\theta(v,x)
\]

Constraints:

- residual initialized at zero;
- KL regularization toward AR;
- no auxiliary prediction head;
- corrections modify view logits only;
- no test-label access;
- exact preservation remains when analytical confidence is zero.

### Global selection rule

Before final tests:

- Select AR+ only if mean validation log loss across development datasets improves by at least 0.01 and no dataset degrades by more than 0.005.
- Otherwise select AR.
- Report both in every result table.
- Do not use the held-out transfer dataset for selection.

## 4. Dataset program

| ID | Dataset | Domain | Target | Role |
|---|---|---|---|---|
| D1 | [KKBox Churn](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge) | Music subscription | Official next-period churn | Large-scale anchor |
| D2 | [Online Retail II](https://archive.ics.uci.edu/dataset/502/online%2Bretail) | Retail transactions | No purchase in next 30 days | Rolling temporal anchor |
| D3 | [RetailRocket](https://www.kaggle.com/retailrocket/ecommerce-dataset/home) | E-commerce behavior | No engagement/purchase in next 30 days | Sparse transfer |
| D4 | [Iranian Churn](https://archive.ics.uci.edu/ml/datasets/Iranian%2BChurn%2BDataset?TB_iframe=true&height=658.8&width=370.8) | Telecom | Official churn label | Small-data TFM stress |

Online Retail II has 1,067,371 transactions across two years. RetailRocket has about 2.76M events across 4.5 months but requires a repeat-engagement cohort; otherwise one-time visitors trivialize lapse.

### Mandatory standardized frontier track

The real lapse datasets remain the headline evidence. Add a fixed, secondary track from [BeyondArena](https://huggingface.co/datasets/TabArena/BeyondArena) to test whether CrossFM-Align transfers beyond custom cohort construction. BeyondArena is preferred over an IID-only suite because it provides official IID, temporal, and grouped splits. Freeze this list before running any method; do not replace a difficult dataset after observing results.

| ID | BeyondArena `unique_name` | Split | Size | Why it is included | Priority |
|---|---|---:|---:|---|---|
| B1 | `hotel_booking_demand` | Temporal | 81,418 x 31 | Customer cancellation, interpretable business schema, close to lapse risk | Mandatory |
| B2 | `kick` | Temporal | 72,983 x 32 | Manageable temporal business-risk shift outside churn | Mandatory |
| B3 | `emscad` | IID | 17,460 x 17 | Text-bearing semantic-schema test; probes whether the LLM channel adds value | Mandatory |
| B4 | `bank_customer_churn` | IID | 10,000 x 10 | Recognizable direct-churn reference and small-data TFM test | Mandatory |
| B5 | `ieee_fraud_detection` | Temporal | 590,540 x 435 | High-dimensional temporal scale stress | Stretch |
| B6 | `amex_non_iid_1m` | Grouped | 1,249,605 x 189 | Group-shift and largely opaque-feature negative control | Stretch |

Rules for this track:

- Use the official BeyondArena folds, target, metadata, and metric without redefining labels or splits.
- Run B1–B4 even if D3 or D4 is delayed. Run B5/B6 only after the lapse anchors and B1–B4 are complete.
- Preserve original feature names for the canonical condition. Use official/source descriptions only; do not invent label-informed descriptions.
- Treat B6 as a negative control: opaque features should reduce the semantic advantage. A null CrossFM gain there is not a failure if preservation prevents harm.
- Do not average B1–B6 into the customer-lapse headline. Report a separate standardized-transfer table and mean rank.
- Record the exact BeyondArena/Data Foundry revision, dataset UUID, checksum, and split IDs.

### Frontier-track method budget

The full real-data grid remains D1/D2. To fit the deadline, B1–B4 use a reduced but causally sufficient grid: CatBoost, strongest available standalone TFM, LLM-only, tuned probability ensemble, semantic one-way selector, CrossFM-AR, CrossFM-AR+, and no-semantic/shuffled-message ablations. B5/B6 use CatBoost, standalone TFM, ensemble, AR, and AR+ only.

[TabPFN-3](https://huggingface.co/Prior-Labs/tabpfn_3) is a frontier replication rather than the sole backbone. Use the official binary-classification checkpoint with its pinned repository revision and verified SHA-256 on D1, D2, and B1–B4. TabICLv2 remains the primary open backbone so the claim does not depend on one newly released system.

### Temporal labeling

For transactional datasets:

- Unit: customer at cutoff \(t\).
- History: previous 180 days, with 7/14/30/60/90-day features.
- Eligibility: at least two transactions or three engagement events on at least two days, plus activity in the previous 90 days.
- Positive label: no qualifying event in \((t,t+30]\).
- Drop cutoffs without a complete 30-day future window.
- Build every feature with an explicit as-of timestamp.
- Cluster statistical uncertainty by customer.

For RetailRocket, report engagement lapse and purchase lapse separately.

### Splits

- Train: earliest 60% of valid cutoffs.
- Validation: next 20%.
- Test: final 20%.
- Add a 30-day embargo between partitions.
- Add grouped-customer robustness for D2/D3.
- Use five stratified outer splits for D4 and label it small-data IID, not temporal.

### Immutable metadata

One YAML per dataset must contain target, horizon, descriptions, units, view membership, prohibited post-cutoff columns, source/license, and checksum. Metadata cannot include observed target correlations, model scores, or test-derived importance.

## 5. Evaluation conditions

Every primary dataset:

1. Canonical schema.
2. Unseen, manually validated aliases.
3. Anonymized columns \(x_1 \ldots x_d\).
4. Shuffled descriptions.
5. 25% and 50% missing descriptions.
6. Context budgets \(n \in \{32,128,512,2048\}\), where feasible.
7. Full supported context.

Headline results use canonical temporal splits.

## 6. Model matrix

Current strong references include [TabPFN v2](https://www.nature.com/articles/s41586-024-08328-6), [TabICL](https://arxiv.org/abs/2502.05564), [TabM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html), and the maintained [TabArena](https://github.com/autogluon/tabarena). Semantics-aware comparators include [ConTextTab](https://proceedings.neurips.cc/paper_files/paper/2025/file/d807e7678ba3afd3a904f4af52819e77-Paper-Conference.pdf) and [TabSTAR](https://arxiv.org/abs/2505.18125).

### Mandatory methods

| Family | Method | Protocol |
|---|---|---|
| Linear | Logistic regression | standardized numeric + one-hot; tune regularization |
| GBDT | CatBoost, LightGBM, XGBoost | equal 50-trial or 2 GPU-hour budget |
| AutoML | AutoGluon | best-quality, one hour per dataset |
| Tabular DL | TabM | official search space; 20 trials or 2 GPU-hours |
| TFM | TabPFN v2/v2.5 | official inference within limits |
| TFM | TabICLv2 | pinned official checkpoint |
| Frontier TFM | TabPFN-3 | pinned official Hugging Face binary checkpoint on D1/D2 and B1–B4 |
| LLM | Qwen2.5-7B-Instruct | constrained LOW/HIGH likelihood |
| Fusion | tuned probability ensemble | LLM + full-table TFM |
| One-way | semantic view selector | schema/LLM → TFM view |
| One-way | statistics → LLM | exact summaries, constrained likelihood |
| Proposed | CrossFM-AR | zero new routing parameters |
| Proposed | CrossFM-AR+ | anchored learned residual |
| Comparator | CrossFM-SMR | structured learned router |

Attempt official ConTextTab and TabSTAR during the first six hours. Keep only if official code loads, identical preprocessing is possible, license permits use, and one dataset completes within two hours. Document exclusions.

### Backbone replication

Full grid:

- Qwen2.5-7B-Instruct + TabICLv2.

Reduced D1/D2 grid:

- Qwen2.5-1.5B + TabICLv2;
- Llama-3.1-8B-Instruct + TabICLv2;
- Qwen2.5-7B + TabPFN v2/v2.5.

Reduced methods: TFM, LLM, ensemble, one-way selector, AR, AR+.

Standardized-transfer replication:

- B1–B4: Qwen2.5-7B + TabICLv2 reduced grid.
- B1–B4: revision-pinned TabPFN-3 replication for TFM, ensemble, AR, and AR+.
- B5/B6: one backbone only unless all mandatory cells are already validated.

### Fairness tracks

- **Matched context:** maximum 10,000 training rows for all methods.
- **Production scale:** each model receives its supported maximum/full data.

Never imply a 10k-row method beat a full-data GBDT unless both tracks are visible.

## 7. Synthetic mechanism grid

### S1 — Posterior noise

- Context sizes: 8, 16, 32, 64, 128.
- Noise multipliers: 0.5, 1.0, 1.5, 2.0.
- Hard, soft, calibrated soft, AR, AR+, SMR.
- Plot accuracy/log loss versus posterior entropy.

### S2 — Codebook corruption

- Corrupt 0%, 10%, 25%, 50% of bindings.
- Fixed wrong, missing, paraphrased, and undisclosed codebooks.
- Test whether AR+ degrades more gracefully than AR.

### S3 — Compositional scaling

- Marker count \(K \in \{2,4,6,8\}\).
- Enumerated joint posterior where feasible.
- Factorized Bernoulli, sparse top-\(m\), and continuous-vector alternatives.
- Report accuracy, runtime, and memory. This answers “why 16?”

### S4 — Semantic/statistical conflict

Vary conflict between semantic prior and empirical evidence. Plot posterior, semantic prior, aligned weights, gate, and final prediction. This becomes the main mechanism figure if clean.

### S5 — Preservation stress

Irrelevant schema, no-structure context, adversarial descriptions, uniform posterior, and corrupted response bank. CrossFM must revert to its declared fallback.

## 8. Metrics and statistics

Primary:

- PR-AUC;
- log loss;
- top-decile lift.

Secondary:

- AUROC, Brier score, adaptive ECE;
- recall/precision at 1%, 5%, 10%;
- calibration slope/intercept;
- latency, VRAM, parameters, GPU-hours, conceptual calls.

Inference:

- Save every customer-level prediction.
- Use 2,000 paired customer-cluster bootstrap replicates.
- Resample customers, not repeated cutoff rows.
- Report paired 95% CIs against the strongest baseline.
- Holm-correct primary comparisons within each dataset.
- Across datasets report mean rank and direction counts; four datasets do not justify universal significance.
- For B1–B6, retain the official benchmark metric and also report log loss for binary tasks when probabilities are available. Never pool their examples with the lapse datasets.

### GREEN gate

- Significant PR-AUC or log-loss gain on at least 2 datasets.
- No dataset loses more than 1 absolute PR-AUC point against the preserved base.
- Top-decile lift improves on at least 2 datasets.
- Preservation and message ablations behave causally.
- Alias performance retains the improvement.
- At least one backbone replacement reproduces the direction.
- On mandatory B1–B4, CrossFM-Align improves over the standalone TFM on at least 3 of 4 tasks by the official metric or log loss, with no material preservation failure. Label this supporting transfer evidence, not a new SOTA claim.

Use “operationally strong” only if CrossFM beats/ties the strongest tuned baseline on 3 of 4 datasets.

## 9. Leakage and contamination audit

- Hash raw files.
- Explicit as-of time for every feature.
- Reject post-cutoff fields.
- Independently reproduce KKBox labels.
- Remove identifiers from inputs.
- Hide dataset names from LLMs in primary conditions.
- Freeze aliases/descriptions before testing.
- Never estimate view reliability on final test.
- Handle refunds, cancellations, and missing IDs consistently.
- Emphasize temporal prediction and metadata perturbation because public datasets may appear in pretraining.

## 10. Compute plan

### Compliance

Use multiple Kaggle accounts only when they belong to distinct legitimate collaborators and usage complies with Kaggle terms. Do not create or rotate accounts to evade quotas. If only two researchers legitimately own accounts, use only those accounts and move overflow to the H100.

### H100 80 GB

Owns the critical path:

1. ETL and cohort creation.
2. Qwen/Llama embedding and likelihood caches.
3. TabICL/TabPFN response banks.
4. KKBox full-scale run.
5. Final bootstrap and plots.

Settings:

- BF16 autocast and TF32;
- pinned-memory loaders and Parquet/Arrow;
- static tokenization, schema embeddings, task embeddings, posterior features, LLM probabilities, and response-bank caching;
- unload backbones before adapter training;
- preserve 10 GB VRAM margin;
- never recompute a frozen cache for downstream ablations.

### Kaggle lanes

If eight compliant sessions are available:

| Lane | Job | Owner |
|---|---|---|
| K1 | Synthetic noise grid | Author B |
| K2 | Codebook corruption | Author B |
| K3 | Factorized-state scaling | Author B |
| K4 | Iranian full grid | Author B |
| K5 | Online Retail GBDT/TabM | Author A |
| K6 | Online Retail CrossFM ablations from cache | Author A |
| K7 | RetailRocket targets/cohort sensitivity | Author B |
| K8 | BeyondArena B1–B4 reduced grid | first free owner |

Run B5/B6 on the H100 only after D1/D2 caches finish. BeyondArena cache keys additionally include dataset UUID and official split/repeat/fold IDs.

Cache keys must contain dataset checksum, cohort ID, split hash, feature/schema revisions, model IDs/revisions, preprocessing revision, commit, and config digest.

## 11. Two-author split

### Author A — models and primary empirical lead

- H100 and frozen caches.
- Factorized router and AR+.
- D1/D2 primary experiments.
- Backbone replication.
- Architecture/mechanism figures.
- Methods and Results writing.

### Author B — data, baselines, independent verification

- Independent temporal cohort implementation.
- Leakage/count/checksum audit.
- GBDTs, AutoGluon, TabM, LLM-only, one-way baselines.
- D3/D4 experiments.
- BeyondArena B1–B4 reduced grid and official-split verification.
- Synthetic grids.
- Independent bootstrap and headline-table reconstruction.
- Datasets, Evaluation, Limitations writing.

Cross-checks:

- B verifies A’s D1/D2 manifests.
- A verifies B’s D3/D4 labels.
- Both independently regenerate the headline table.
- Any discrepancy blocks final numbers.

## 12. Five-day schedule

### Day 1 — freeze data and strong baselines

- Hours 0–2: freeze protocol, registry, datasets, checksums, labels, owners.
- Hours 0–2: also pin the BeyondArena/Data Foundry revision and B1–B6 UUIDs; no result-driven substitutions.
- Hours 2–8: build cohorts, leakage tests, views, metadata; launch GBDTs/TabM.
- Hours 8–18: create LLM/TFM caches; launch synthetic and small-data lanes.

Exit: three valid manifests, temporal audits passed, one strong baseline per dataset, primary cache running.

### Day 2 — primary CrossFM runs

- Finish D1/D2 response banks.
- Run AR, AR+, SMR, ensemble, one-way.
- Run canonical three-seed grid.
- Run matched/full tracks.
- Start D3/D4 CrossFM.
- Start B1–B4 from official folds; launch B5/B6 only if the primary cache critical path is clear.

Kill rule: if CrossFM improves validation log loss on no dataset, stop model-family expansion and prepare a qualified/negative transfer result.

### Day 3 — causal ablations and robustness

- Lock canonical final tests read-only.
- Alias, anonymization, shuffled/partial metadata.
- Zero/shuffle/uniform/hard/no-anchor/random residual.
- Qwen scale and TFM replacement on D1/D2.
- Complete B1–B4 reduced grid and one verified TabPFN-3 backbone replication.
- Complete S1–S5.
- Start paired bootstraps.

Exit: headline table, mechanism figure, backbone replication, final claim level.

### Day 4 — consolidate and write

- No new architecture after noon.
- Rerun only missing headline cells.
- Freeze results at one commit/manifest.
- Produce comparison, robustness, mechanism, calibration/lift, and compute figures.
- Write main paper and complete appendix.

### Day 5 — hostile review and submission

- Swap sections and review adversarially.
- Rebuild headline table from clean checkout.
- Map every number to prediction/config hashes.
- Remove unsupported claims.
- State cached-response-bank, contamination, disclosed-codebook, causal, and domain limitations.
- Finalize abstract, checklist, and anonymized artifact.

## 13. Registry and artifact contract

Create experiments/full_paper_registry.csv:

~~~text
experiment_id,owner,protocol_id,status,dataset,dataset_checksum,split_hash,
method,llm_id,llm_revision,tfm_id,tfm_revision,schema_condition,
context_budget,seed,config_digest,git_commit,trainable_params,gpu_type,
gpu_hours,prediction_path,prediction_sha256,checkpoint_sha256,notes
~~~

Statuses: PLANNED, RUNNING, COMPLETE_UNVALIDATED, VALIDATED, FAILED, EXCLUDED_WITH_REASON. Only VALIDATED enters the paper.

## 14. Mandatory ablations

- LLM and TFM only;
- tuned ensemble;
- both one-way directions;
- AR and full AR+;
- temperature, route-only, reliability-only;
- no anchor and shuffled response features;
- hard, soft, uniform posterior;
- random/zero messages;
- gate disabled;
- compute- and parameter-matched one-way;
- 1/2/3 rounds where relevant;
- canonical, alias, anonymized, and corrupted metadata.

## 15. Main figures and tables

1. Horizontal CrossFM-Align architecture.
2. Posterior → codebook → view weights → gate.
3. Semantic/statistical conflict trajectory.
4. Temporal lapse performance and top-decile lift.
5. Dataset/cohort table.
6. Primary comparison table.
7. Causal ablation and robustness table.
8. Parameters, GPU-hours, latency, VRAM.

Appendix: all seeds/CIs, hyperparameters, views, calibration, matched/full tracks, scaling, failures, prompts, licenses, and revisions.

## 16. Reviewer attacks

| Attack | Evidence required |
|---|---|
| “Just mixture-of-experts.” | No-anchor learned router versus posterior-codebook router and exact preservation |
| “Codebook gives away answer.” | Real factorized views and corrupted/missing/undisclosed codebooks |
| “Why 16?” | \(K=2,4,6,8\) factorized scaling |
| “GBDTs solve churn.” | Tuned GBDTs, AutoGluon, full-data track, lift |
| “LLM memorized dataset.” | Hidden dataset name, temporal labels, aliases/anonymization |
| “Just ensembling.” | Tuned ensemble plus message shuffle/zero |
| “Learning unnecessary.” | Report AR honestly; justify AR+ only by held-out gain |
| “Adapter predicts directly.” | No auxiliary head; view-logit corrections only |
| “Not causal.” | Risk prediction only, not intervention targeting |
| “Not true recurrence.” | Explicit cached-bank limitation; claim alignment/routing |

## 17. Stop rules

Stop architecture escalation if AR+ never improves validation log loss, response shuffling does not matter, no-anchor matches anchor, tuned GBDTs erase gains, or temporal tests reverse IID gains.

Exclude a dataset if leakage cannot be ruled out, positive rate is below 1%, fewer than 500 eligible customers remain, horizon is censored, license/access is unclear, or clean reproduction fails.

Exclude a comparator with reason after six engineering hours or two GPU-hours after successful installation. Never silently omit a strong baseline.

## 18. Minimum defensible submission

- validated synthetic causal evidence;
- at least three real datasets, two temporal;
- all four mandatory BeyondArena transfer tasks, reported separately from the lapse datasets;
- tuned GBDT, TabM, TFM, LLM, ensemble, one-way, AR, AR+;
- customer-level predictions and paired CIs;
- schema robustness;
- one cross-backbone replication;
- compute accounting;
- explicit negative Phase 4/5.1 findings;
- honest cached-response-bank limitation.

If real transfer fails, reframe:

> Structured analytical alignment solves controlled complementarity, while learned heterogeneous message passing does not reliably transfer to realistic temporal lapse prediction.

## 19. Immediate next twelve hours

1. Freeze dataset manifests and owners.
2. Implement a shared customer-cutoff builder with 180-day history, 30-day horizon, and 30-day embargo.
3. Produce D2/D3 cohort-count reports before training.
4. Start D1/D2 CatBoost and LightGBM.
5. Start H100 schema/view and TFM response caches.
6. Launch S1–S3 on independent Kaggle lanes.
7. Create the registry; Phase 5.2 remains RUNNING, not evidence.
8. Freeze AR-versus-AR+ selection.
9. Draft Introduction and Related Work while caches run.
10. Pin and prefetch only BeyondArena B1–B4; validate official fold indices and licenses before model execution.

The critical path is **data integrity → immutable caches → primary temporal results**. Adapter training is not the bottleneck.
