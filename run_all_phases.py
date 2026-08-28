#!/usr/bin/env python3
"""
SOA AI Challenge (V17) - Full Reproduction Pipeline
===================================================

Runs all phases of the analysis in dependency order, each as a standalone
subprocess. This reproduces the results behind the paper:

    "From Detection to Attribution: A Comprehensive AI Framework for
     Life Insurance Mortality Surveillance" (V17 Final)

Dependency order:
    phase1  ->  phase2         (phase2 reads results/phase1_ae_ratios.csv)
    phase1  ->  phase4         (phase4 reads results/phase1_ae_ratios.csv)
    phase3_predictive_models  ->  phase3_deep_learning
        (deep learning merges results/models/model_comparison_poisson.csv)

Usage:
    python run_all_phases.py                 # run every phase
    python run_all_phases.py --phase 1       # run a single phase
    python run_all_phases.py --skip-dl       # skip the GPU deep-learning phase
    python run_all_phases.py --sample 200000 # phase 1 quick test on a row sample

Notes:
  * Inputs and outputs are resolved relative to this folder (soa_ai_v2/),
    so the project is portable — no absolute paths.
  * A full run is compute-heavy. Phase 1 streams the 12 GB raw text file;
    phase 3 trains gradient-boosting + (optionally) deep-learning models.
  * The deep-learning phase (TabNet, FT-Transformer) benefits from a GPU.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
# Phases honor SOA_RESULTS_DIR; mirror it here so the summary points at the
# right place. Child phases inherit this env var through subprocess.
RESULTS_DIR = Path(os.environ.get("SOA_RESULTS_DIR") or (PROJECT_DIR / "results"))

# (key, script, human label)
PHASES = [
    ("1",   "phase1_trend_detection.py",   "Phase 1 - Trend Detection (Mann-Kendall)"),
    ("2",   "phase2_early_warning.py",     "Phase 2 - Change Point + Early Warning (BOCPD)"),
    ("3",   "phase3_predictive_models.py", "Phase 3 - Predictive Models (Poisson/Tweedie GBMs + EBM)"),
    ("3dl", "phase3_deep_learning.py",     "Phase 3b - Deep Learning (TabNet, FT-Transformer)"),
    ("4",   "phase4_causal_analysis.py",   "Phase 4 - Conformal Prediction + Synthetic Control"),
]


def run_script(script, extra_args):
    cmd = [sys.executable, str(CODE_DIR / script), *extra_args]
    print("\n" + "=" * 80)
    print(f" RUN: {' '.join(cmd)} ".center(80, "="))
    print("=" * 80 + "\n", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(CODE_DIR))
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAILED] {script} exited with code {result.returncode} after {dt:.0f}s")
        sys.exit(result.returncode)
    print(f"\n[OK] {script} finished in {dt:.0f}s")


def main():
    ap = argparse.ArgumentParser(description="Reproduce all phases of the SOA AI (V17) analysis.")
    ap.add_argument("--phase", choices=[p[0] for p in PHASES],
                    help="Run only this phase (default: run all in order).")
    ap.add_argument("--skip-dl", action="store_true",
                    help="Skip the GPU deep-learning phase (3dl).")
    ap.add_argument("--sample", type=int, default=None,
                    help="Row sample size for Phase 1 (quick test).")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    selected = [p for p in PHASES if (args.phase is None or p[0] == args.phase)]
    if args.skip_dl:
        selected = [p for p in selected if p[0] != "3dl"]

    print("SOA AI Challenge (V17) reproduction")
    print(f"Project root : {PROJECT_DIR}")
    print(f"Results dir  : {RESULTS_DIR}")
    print("Phases       : " + ", ".join(p[0] for p in selected))

    overall = time.time()
    for key, script, label in selected:
        print(f"\n>>> {label}")
        extra = [str(args.sample)] if (key == "1" and args.sample is not None) else []
        run_script(script, extra)

    print("\n" + "=" * 80)
    print(f" ALL DONE in {time.time() - overall:.0f}s — outputs in {RESULTS_DIR} ".center(80, "="))
    print("=" * 80)


if __name__ == "__main__":
    main()
