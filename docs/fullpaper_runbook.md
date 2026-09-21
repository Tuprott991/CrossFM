# CrossFM-Align Full-Paper Runbook

This runbook executes the frozen exploratory protocol in `configs/fullpaper.yaml`.
It does not publish datasets, kernels, or results. All Kaggle assets must remain
private unless every underlying dataset license permits publication.

## Responsibility split

### Author A — H100 80 GB

Author A owns the scientific critical path:

1. Validate and checksum the D1 KKBox and D2 Online Retail cohort tables.
2. Run `author_a_h100_primary`.
3. Run `author_a_h100_robustness` after the canonical caches validate.
4. Run `author_a_h100_scale` only after the primary profiles finish.
5. Freeze the AR-versus-AR+ global selection using validation results only.

~~~bash
python -m pip install -e . -r requirements-fullpaper.txt
CROSSFM_PROFILE=author_a_h100_primary \
CROSSFM_DATA_ROOT=/data/crossfm/fullpaper \
CROSSFM_OUTPUT=/results/crossfm/author_a_h100_primary \
bash scripts/run_fullpaper_h100.sh
~~~

H100 execution uses BF16 autocast, TF32 matmul, one frozen-backbone cache per
dataset/split/seed/schema/budget, and a 7B Qwen checkpoint. Downstream AR, AR+,
SMR, rounds, and ablations reuse those caches.

### Author B — Kaggle T4x2

Author B independently verifies official folds and runs:

- `author_b_kaggle_frontier_temporal`: B1 hotel cancellation and B2 vehicle risk;
- `author_b_kaggle_frontier_semantic`: B3 text-rich fraud and B4 bank churn;
- `author_b_kaggle_d4`: Iranian churn and small-context robustness;
- `author_b_kaggle_d3`: RetailRocket transfer.

Author B also owns the CPU-safe controlled mechanism grids in
`configs/fullpaper_synthetic.yaml`: S1 posterior noise, S2 codebook corruption,
S3 compositional scaling, S4 semantic/statistical conflict, and S5 preservation.
They can be sharded across two workers without loading a backbone:

~~~bash
python -m crossfm.fullpaper.synthetic_stress \
  --config configs/fullpaper_synthetic.yaml --suite s1_posterior_noise \
  --output outputs/synthetic --rank 0 --world-size 2
python -m crossfm.fullpaper.synthetic_stress \
  --config configs/fullpaper_synthetic.yaml --suite s1_posterior_noise \
  --output outputs/synthetic --rank 1 --world-size 2
python -m crossfm.fullpaper.synthetic_stress \
  --config configs/fullpaper_synthetic.yaml --suite s1_posterior_noise \
  --output outputs/synthetic --world-size 2 --aggregate
~~~

Repeat for `s2_codebook_corruption`, `s3_compositional_scaling`,
`s4_semantic_statistical_conflict`, and `s5_preservation`. These grids are
mechanism stress tests; the frozen-backbone Phase 5.2 results remain the source
of evidence about actual LLM/TFM behavior.

Each Kaggle job starts two long-lived workers with one isolated T4 each. The
workers synchronize only at stage boundaries: response banks, LLM caches, then
evaluation. T4 jobs use the pinned 1.5B Qwen checkpoint in FP16. They are a
backbone-scale replication, not a substitute for the H100 7B headline run.

Build one immutable private bundle per profile:

~~~powershell
./scripts/build_fullpaper_bundle.ps1 `
  -Profile author_b_kaggle_frontier_temporal `
  -KaggleOwner <legitimate-author-account> `
  -KernelSlug crossfm-frontier-temporal-v1 `
  -DataOwner <legitimate-data-owner> `
  -DataSlug crossfm-fullpaper-data
~~~

Upload the private bundle dataset, verify its file list, and only then push the
kernel. The build intentionally refuses a dirty Git tree.

## Kaggle account compliance

Kaggle lanes may be distributed among distinct, legitimate collaborators using
their own accounts and quotas. Do not use alternate accounts belonging to one
person, rotate credentials, or coordinate sessions to evade limits. Record the
actual owner and kernel version in the experiment registry. If compliant quota
is unavailable, serialize the lanes or move them to the H100.

## Required data layout

~~~text
data/fullpaper/
  kkbox/customer_cutoffs.parquet
  online_retail/online_retail_ii.parquet
  retailrocket/events.csv
  iranian_churn/iranian_churn.csv
~~~

`kkbox/customer_cutoffs.parquet` is deliberately an audited input rather than an
implicit label reconstruction. It must contain binary `target`, immutable
`split` (`train`, `validation`, `test`), and only pre-cutoff features. The cohort
producer must record cutoff, label horizon, raw-file hashes, and the independent
label-reproduction audit in its manifest. This prevents a convenient but
scientifically different “no transaction” target from being mislabeled as the
official KKBox definition.

Online Retail II and RetailRocket cohorts are built by the runner with explicit
history windows, 30-day horizons, eligibility rules, as-of timestamps, and
temporal embargoes. BeyondArena datasets use the official test folds; validation
is carved only from each official training fold.

## Local non-inference gates

~~~bash
python -m pytest
python -m crossfm.fullpaper.cli plan \
  --config configs/fullpaper.yaml \
  --profile author_b_kaggle_frontier_temporal \
  --output outputs/plan \
  --world-size 2
python -m compileall crossfm scripts
~~~

These checks validate logic and packaging, not model availability or GPU
correctness. Every remote run must still pass its doctor step and one full task
before expensive expansion.

## Completion contract

A profile is complete only when `summary.json` reports the full expected task
count, every task record and referenced array passes SHA-256 validation, all
metrics are finite, and the Kaggle/H100 process exits successfully. Partial
records are resumable but are never scientific evidence.
