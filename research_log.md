# Research log

| Observation | Hypothesis | Experiment | Result | Interpretation | Next experiment |
|---|---|---|---|---|---|
| TabICL v2 documents pretraining on 300–48k rows. | Its inductive bias should dominate Qwen on the 384-row anonymized nonlinear regime but may extrapolate poorly to 10–28 rows. | Frozen Phase 1 A/B/C pilot. | Pending. | Pending. | Run engineering smoke, then frozen exploratory pilot only if smoke is complete. |
| V1 Qwen3-0.6B predicted class 1 on under 1% of queries; sequence likelihood and LOW/HIGH verbalizers also collapsed locally. | The checkpoint/interface is too weak for a credible semantic baseline. | Test bounded generation and the next protocol-approved model size. | Qwen2.5-1.5B follows strict `FINAL: LOW/HIGH`; redesigned A reached 97.5% on a 40-query local gate. | V1 cannot test semantic complementarity; v2 is a new exploratory protocol, not a confirmatory retry. | Run private v2 engineering smoke, then the frozen v2 pilot if all artifact gates pass. |
