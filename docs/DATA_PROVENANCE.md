# Data provenance and the limits of reproducibility

This document exists so that a reader can tell exactly which parts of the ML281
submission they can re-run themselves, and which parts they have to take on
documentation.

## The chain

```
SOA ILEC 2012-2019 experience study          (obtained from the SOA)
        |
        |  loaded into a data warehouse, one row per aggregated exposure cell
        v
45,501,036 exposure cells
        |
        +--> GROUP BY aggregation  ------------->  data/full_table_ae.json   (2.4 KB, committed)
        |    actual vs VBT2015 expected deaths     Figure 1, population A/E
        |    by smoker status, sex, year, age band
        |
        +--> random extract, seed 42  ---------->  data/soa_sample.parquet   (5.94 MB, fetched)
             131,922 cells, 15 columns             every model in the notebook
```

**Step 1 (obtaining ILEC) and step 2 (the warehouse aggregation) are not
executable from this repository.** They ran against a private warehouse. Steps 3
and 4 — everything the notebook does — run entirely from the two public files.

## What the aggregation did

Generically, and independent of any particular warehouse:

```sql
-- Population A/E aggregates -> full_table_ae.json
-- Repeated once per grouping key: Smoker_Status, Sex, Observation_Year, age_band.
SELECT
    <grouping_key>                              AS k,
    SUM(Death_Count)                            AS deaths,
    SUM(ExpDth_VBT2015wMI_Cnt)                  AS exp,
    SUM(Death_Count) / SUM(ExpDth_VBT2015wMI_Cnt) AS ae
FROM <ilec_2012_2019_table>
GROUP BY k;
```

`age_band` buckets `Attained_Age` as `<40`, `40-49`, `50-59`, `60-69`, `70-79`,
`80+`. The modelling extract adds the two log-exposure columns
(`LN(GREATEST(Amount_Exposed, 1))` and the policy-count equivalent) and
`Expected_Mortality_Rate` as `ExpDth_VBT2015wMI_Cnt / NULLIF(Policies_Exposed, 0)`.

Because the aggregates in `full_table_ae.json` are computed over the whole table
while the models train on the extract, Figure 1 describes the full population even
though no model ever sees 45 million rows. Cross-checking the four groupings
against each other is a test in `tests/test_data_contract.py`: all four must sum
to the same population totals, and to the same A/E of 1.0764.

## The reference environment

The published figures and results were produced with:

| | |
|---|---|
| Python | 3.11.9 |
| OS | Windows 11 (`Windows-10-10.0.26200-SP0`) |
| numpy | 2.4.2 |
| pandas | 2.3.3 |
| scipy | 1.17.0 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| pyarrow | 23.0.1 |
| shap | 0.51.0 |
| seed | 42, propagated to every split, every estimator and the copula RNG |

`requirements.txt` pins exactly this set, and every run records what it actually
resolved into `results/environment.json`, printing a warning when it differs.

## Limits of reproducibility — read this before reporting a mismatch

1. **Tree ensembles and SHAP are not guaranteed bit-identical across library
   versions or platforms.** Seeding is thorough and complete, but XGBoost's
   histogram construction and thread scheduling can shift low-order digits
   between builds. Expect agreement to roughly three decimal places, not to the
   last bit. The verified reproduction on the reference environment agrees with
   every published value to within ±0.0005 except where noted below.

2. **The copula TVaR is Monte-Carlo noisy and is the least stable number in the
   submission.** `TVaR99 / mean` measures a 1-in-100 tail from a finite
   simulation. Measured on the published extract:

   | Configuration | TVaR99 / mean |
   |---|---|
   | 2,500 sims, seed 42 (as shipped) | 1.1979 |
   | 2,500 sims, seed 1 | 1.1804 |
   | 2,500 sims, seed 7 | 1.1909 |
   | 5,000 sims, quantile interpolation | 1.1837 |

   The report quotes approximately 1.18x. The shipped seed-42 configuration gives
   1.198. Both sit inside the sampling band above; the ratio should be read as
   "roughly 1.18–1.20", not as a three-decimal constant. `copula.rho`, by
   contrast, is computed from the data rather than simulated and reproduces
   exactly at -0.3659.

3. **On Colab, `numpy`, `scipy` and `pandas` are left at their preinstalled
   versions.** Re-pinning numpy inside a live kernel is what produces
   `ImportError: cannot import name '_center'`, so the notebook pins only the
   modelling libraries there and records the resolved versions instead. For an
   exact-environment run, use a local virtualenv and `requirements.txt`.

4. **Population counts.** The committed aggregates sum to 4,552,008 deaths against
   4,229,014.7 expected. The report quotes 4,552,009 and 4,229,015; the A/E of
   1.0764 is identical either way. The 45,501,036-cell table size is a property of
   the source data and is reported, not recomputed, here.

## What is deliberately absent

The submission's earlier draft included an optional cell that regenerated the data
from the warehouse. It has been removed rather than published: it named a private
GCP project and dataset, it could not run for anybody outside that project, and
its output is exactly the two files this repository already provides in verified
form. The SQL logic it contained is reproduced above, genericized.

The submission report, the response to the review committee and the practitioner
takeaways are published by the SOA, not mirrored here. This repository is the
code, the data path and the numbers.
