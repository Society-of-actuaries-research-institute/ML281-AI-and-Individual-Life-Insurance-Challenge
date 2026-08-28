# ML281 — Code Package and Review Response

**Submission:** ML281 — *From Detection to Attribution: A Comprehensive AI
Framework for Life Insurance Mortality Surveillance*
**Author:** Victoria Zhang, FSA, FCIA

This folder contains the code behind the paper and a written response to the
review committee's comments. It is meant to be easy to read, run, and check.

---

## 1. What's in this folder

```
ML281_Zhang_response/
├── README.md          ← this file (plain-English guide)
├── email_reply.txt    ← my written reply to the committee's five comments
└── code/              ← all the analysis scripts
    ├── config.py
    ├── chunk_to_parquet.py
    ├── phase1_trend_detection.py
    ├── phase2_early_warning.py
    ├── phase3_predictive_models.py
    ├── phase3_deep_learning.py
    ├── phase4_causal_analysis.py
    ├── run_all_phases.py
    └── requirements.txt
```

The data file itself (the ILEC 2012–2019 extract) is not bundled here because
of its size, but it is the public SOA ILEC dataset. The scripts expect to find
it in a `data/` folder next to `code/` (see "How to run" below).

---

## 2. What each script does

The work is split into four phases. Each phase answers one practical question
an actuary would ask, and each is a separate, runnable script.

| Script | Paper section | In plain English |
| --- | --- | --- |
| `phase1_trend_detection.py` | §4.1 | Which groups of policyholders are getting worse? Calculates actual-to-expected (A/E) mortality by group and year, and runs a Mann-Kendall test to see which trends are real, not noise. |
| `phase2_early_warning.py` | §4.2–4.3 | When did the deterioration start, and could we have caught it sooner? Runs Bayesian change-point detection, then compares an AI-style alert (z-score) against a traditional A/E threshold to measure how much earlier the AI alert fires. |
| `phase3_predictive_models.py` | §4.4 | How well can we predict mortality rates? Trains and compares the gradient-boosting models (XGBoost, LightGBM, CatBoost) and the Explainable Boosting Machine (EBM), plus a Poisson baseline. This is the script that produces Table 3. |
| `phase3_deep_learning.py` | §4.4 | The two deep-learning models (TabNet and FT-Transformer) from the same comparison. Kept separate because they need a GPU. |
| `phase4_causal_analysis.py` | §4.5–4.6 | How confident are we, and what caused the change? Builds conformal prediction intervals (uncertainty) and runs the synthetic-control analysis that estimates the opioid crisis impact. |

Two helper files support the above:

- `config.py` — sets the folder paths. Nothing to edit; paths are worked out
  relative to the code folder.
- `chunk_to_parquet.py` — a one-time step that converts the large raw text file
  into smaller per-year files so the model scripts load faster.
- `run_all_phases.py` — a convenience script that runs every phase in the right
  order.

---

## 3. How to run it

You need Python 3.10 or 3.11 and the ILEC data file.

**Step 1 — set up the folders.** Put this `code/` folder and a `data/` folder
side by side:

```
your_working_folder/
├── code/     ← the scripts from this package
└── data/     ← the ILEC text file goes here
    └── ILEC_2012_19 - 20240429.txt
```

**Step 2 — install the libraries.**

```bash
cd your_working_folder
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r code/requirements.txt
```

**Step 3 — prepare the data (one time).**

```bash
cd code
python chunk_to_parquet.py        # converts the raw text file into per-year files
```

**Step 4 — run the analysis.**

```bash
# Everything, in order:
python run_all_phases.py

# Or skip the GPU deep-learning step:
python run_all_phases.py --skip-dl

# Or run just one phase (e.g. Phase 1):
python run_all_phases.py --phase 1

# Quick test of Phase 1 on a small sample of rows:
python run_all_phases.py --phase 1 --sample 500000
```

Each phase can also be run on its own, e.g. `python phase1_trend_detection.py`.

**Where results go.** By default, results are written to a `results/` folder
next to `code/`. To run a check without overwriting anything, point the results
somewhere else with an environment variable:

```bash
SOA_RESULTS_DIR=../results_check python run_all_phases.py
```

---

## 4. A few honest notes

- **Run order matters.** Phase 2 and Phase 4 use the A/E table that Phase 1
  produces, and the deep-learning script adds its results to the table the
  gradient-boosting script produces. `run_all_phases.py` handles this for you.
- **The deep-learning step is the heavy one.** TabNet and FT-Transformer are
  best run on a GPU. Everything else runs fine on a normal CPU.
- **Small numeric differences are expected.** Results depend on library
  versions, random seeds, and whether you use a GPU or CPU. The numbers should
  match the paper closely, but not to the last digit.


---

*Questions are welcome — contact details are in the email reply.*
