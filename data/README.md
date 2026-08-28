# Data

## What the models are trained on

Everything in this repository derives from the **Society of Actuaries Individual
Life Experience Committee (ILEC) 2012–2019** mortality experience dataset,
published by the SOA under Research. The ILEC data is distributed by the SOA;
this repository does not redistribute it. What is provided here are two small
derived files, both free of personally identifiable information.

**These are aggregated exposure cells, not individual policies.** Each row is a
cell of homogeneous risk characteristics carrying a summed exposure and a summed
death count. This distinction governs how every result in the submission should
be read — a "prediction" is an expected death count for a cell, never a
prediction about a person. The review committee pressed on exactly this point,
and it is the single most important caveat in the work.

## Files

| File | Committed? | Size | Purpose |
|---|---|---|---|
| `full_table_ae.json` | yes | 2.4 KB | Actual/expected aggregates over the **entire** 45,501,036-cell table, grouped by smoker status, sex, observation year and attained-age band. Source for Figure 1 and for the population A/E. |
| `soa_sample.parquet` | **no** — fetched | 5.94 MB | The 131,922-cell modelling extract. Every model in the notebook trains on this. |
| `sample/soa_sample_small.parquet` | yes | 245 KB | A 3,992-row CI fixture. **Not a statistical sample** — see below. |
| `CHECKSUMS.txt` | yes | — | SHA-256 for both fetched files. |

### Getting `soa_sample.parquet`

```bash
python scripts/download_data.py
```

It downloads from the public Hugging Face dataset
[`pramodmisra/soa-ml281-sample`](https://huggingface.co/datasets/pramodmisra/soa-ml281-sample)
— no account, no token — and verifies the SHA-256 against `CHECKSUMS.txt` before
the file is used. A mismatch is a hard error, not a warning. The notebook performs
the same fetch-and-verify itself, so running it in Colab needs no separate step.

Verified 2026-08-24: the Hugging Face copies are byte-identical to the files that
produced the published figures and results.

### The CI fixture is not a sample

`sample/soa_sample_small.parquet` exists so the pipeline can execute with no
network access, in CI or on a plane. Death-bearing cells are **deliberately
over-represented** — 37.5% of fixture rows carry a death, against 3.9% in the real
extract — so that the copula, SHAP and out-of-time steps all have enough signal to
run. Numbers computed from it are meaningless as actuarial results, which is why
smoke-mode output is written to `.smoke/` and never to `results/`.

Regenerate it deterministically (seed 42):

```bash
python scripts/download_data.py
python scripts/make_sample.py
```

## Columns

| Column | Type | Notes |
|---|---|---|
| `Observation_Year` | int | 2012–2019 |
| `Issue_Age`, `Attained_Age`, `Duration` | int | Policy-year structure of the cell |
| `Sex` | str | `M`, `F` |
| `Smoker_Status` | str | `NS`, `S`, `U` (unknown) |
| `Insurance_Plan` | str | `Term`, `Perm`, `UL`, `ULSG`, `VL`, `VLSG`, `Other` |
| `Amount_Exposed`, `Policies_Exposed` | float | Exposure, by face amount and by policy count |
| `Death_Count` | int | **Model target.** Deaths observed in the cell |
| `Death_Claim_Amount` | float | Claim dollars, used for the severity leg of the copula |
| `ExpDth_VBT2015wMI_Cnt` | float | Expected deaths under VBT 2015 with mortality improvement — the actuarial benchmark and the offset in the Poisson models |
| `Log_Amount_Exposed`, `Log_Policies_Exposed` | float | Log exposure, as model features |
| `Expected_Mortality_Rate` | float | The VBT rate; the feature removed in the Q1 ablation |

No nulls in either file; `tests/test_data_contract.py` asserts this along with the
schema, the study window and the category levels.

## Licensing

The ILEC dataset is the SOA's; consult the SOA's terms for the underlying data
rather than treating these derived files as an independent grant. The **code** in
this repository is GPL-3.0 (see `LICENSE`). The derived extract is published as a
public Hugging Face dataset to make the submission reproducible, contains no
personally identifiable information, and is not offered as a substitute for
obtaining the ILEC data from the SOA.

## How the two files were produced

The aggregation that turned the ILEC table into these two files ran against a
private data warehouse and is therefore **documented rather than executable
here** — see [`../docs/DATA_PROVENANCE.md`](../docs/DATA_PROVENANCE.md) for the
query logic, the age banding and the sampling step. Everything downstream of that
point, which is the entire analysis, runs from these public files.
