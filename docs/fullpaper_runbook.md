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

Only `author_a_h100_primary` performs AR-versus-AR+ selection. Its D1/D2
canonical validation decision is frozen before robustness or scale results are
opened. The latter summaries are deliberately marked `evaluation_only` and must
not be used to change the selected architecture.

Run all three resumable profiles in the required order with one command:

~~~bash
CROSSFM_DATA_ROOT=/data/crossfm/fullpaper \
CROSSFM_OUTPUT_ROOT=/results/crossfm \
bash scripts/run_all_fullpaper_h100.sh
~~~

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

Every task record includes wall time, declared backbone-call count, logical round
count, and peak allocated accelerator memory. `summary.json` reports cumulative
task time, allocated accelerator-hours, and profile peak memory. Cache tasks are
included, so compute accounting does not disappear behind response-bank reuse.
Router calibration is capped at 2,048 label-isolated rows per seed/context cell;
this bounds LLM cache construction on million-row datasets without changing the
TFM's declared 10k/full fit context or the fixed validation/test evaluation sets.

Use one operating-system/GPU session for one H100. The supported parallelism is
one heavyweight worker during `response_bank` and `llm_cache`; launching two 7B
or TFM workers against the same device duplicates weights and makes OOM behavior
non-deterministic. The three profiles are logical, resumable run units, not three
simultaneous GPU jobs. If a scheduler imposes short wall limits, invoke the
existing runner separately for each profile and let its task records resume; do
not overlap GPU stages. Aggregation is CPU-only and may run after its profile's
three stages complete.

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

The four-author fleet launcher reads `KAGGLE_ACCESS_TOKEN_1` through
`KAGGLE_ACCESS_TOKEN_4` from the ignored repository `.env` using
`load_dotenv()`. It removes all four source variables from each child process
and passes only the selected account as `KAGGLE_API_TOKEN`. Run:

~~~powershell
python scripts/run_kaggle_fleet.py preflight
python scripts/run_kaggle_fleet.py launch
python scripts/run_kaggle_fleet.py status
python scripts/run_kaggle_fleet.py monitor --poll-seconds 120
python scripts/run_kaggle_fleet.py download
~~~

The initial launch uses one T4x2 notebook per account (eight T4s total) and
retains the second allowed session for bounded recovery. The accounts run D4,
frontier-temporal, frontier-semantic, and D3 respectively. Every bundle and
kernel remains private.

## Required data layout

~~~text
data/fullpaper/
  kkbox/customer_cutoffs.parquet
  kkbox/cohort_manifest.json
  online_retail/online_retail_ii.parquet
  events.csv
  Customer Churn.csv
~~~

For Kaggle, D3 attaches the original `retailrocket/ecommerce-dataset` source and
D4 attaches `alinoranianesfahani/iranian-churn-dataset`; the data root is the
mounted dataset directory. RetailRocket is CC BY-NC-SA 4.0. Iranian Churn must
be attributed to the canonical UCI release (DOI `10.24432/C5JW3Z`, CC BY 4.0)
even when the Kaggle mirror is used for transport.

`kkbox/customer_cutoffs.parquet` is deliberately an audited input rather than an
implicit label reconstruction. It must contain binary `target`, immutable
`split` (`train`, `validation`, `test`), stable `customer_id`, and only pre-cutoff
features. `cohort_manifest.json` must use schema `crossfm-cohort-manifest-v1`,
match the table SHA-256, identify every raw-file SHA-256, fix the horizon to 30
days, and mark the independent label-reproduction and feature as-of audits as
`passed`. The runner rejects anything weaker. This prevents a convenient but
scientifically different “no transaction” target from being mislabeled as the
official KKBox definition.

The v7 H100 release is a new exploratory protocol; immutable v6 artifacts are
never reused. Its third beat is a confidence-scaled soft-attention replay learned
only from router-split residuals, not a fresh backbone call. The
`llm_to_tfm_repeated_routing` arm repeats one-way routing three times over the
same response bank and is only a routing diagnostic. `learned_summary_fusion` is
logistic fusion over LLM and TFM summaries; it is not latent TFM-to-LLM injection.
The one-way LLM-to-TFM arm and AR share identical cached backbone outputs, so
their backbone compute is matched and routing latency is reported separately.

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
