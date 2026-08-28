"""Execute the reproduction notebook headlessly and assert it produced its outputs.

This is what CI runs (with ``ML281_SMOKE=1``) and it is also the easiest way to do
a full local reproduction without opening Jupyter::

    python scripts/download_data.py
    python scripts/run_notebook.py                 # full run, ~4 minutes
    ML281_SMOKE=1 python scripts/run_notebook.py   # fast smoke run

An executed copy is written to ``results/run_report.html`` so a reader can see the
expected output without running anything.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient.exceptions import CellExecutionError
from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

from ml281_paths import FIGURES, figures_dir, repo_root, results_dir, smoke_mode

NOTEBOOK_PATH = "notebooks/ML281_reproduce.ipynb"
#: Hard ceiling per cell, seconds. The multi-model CV cell is the slow one.
CELL_TIMEOUT_SECONDS = 1800

#: Ledger files every run must produce.
EXPECTED_RESULTS = (
    "environment.json",
    "full_table_ae_summary.json",
    "ablation.json",
    "generalization.json",
    "multimodel.json",
    "poisson_offset.json",
    "copula.json",
    "out_of_time.json",
)


def execute(notebook_path: Path, write_html: bool) -> nbformat.NotebookNode:
    """Run every cell of the notebook in the repo root working directory."""
    notebook = nbformat.read(notebook_path, as_version=4)
    processor = ExecutePreprocessor(timeout=CELL_TIMEOUT_SECONDS, kernel_name="python3")
    try:
        processor.preprocess(notebook, {"metadata": {"path": str(repo_root())}})
    except CellExecutionError as exc:
        raise SystemExit(f"\nNotebook execution FAILED:\n{exc}") from exc

    if write_html:
        html_body, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
        out_path = results_dir() / "run_report.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html_body, encoding="utf-8")
        print(f"wrote {out_path.relative_to(repo_root())}")
    return notebook


def check_outputs() -> None:
    """Fail if any expected figure or ledger file is missing."""
    missing_figures = [f for f in FIGURES if not (figures_dir() / f).exists()]
    missing_results = [f for f in EXPECTED_RESULTS if not (results_dir() / f).exists()]
    if missing_figures or missing_results:
        raise SystemExit(
            "Notebook ran but did not produce all outputs.\n"
            f"  missing figures: {missing_figures or 'none'}\n"
            f"  missing results: {missing_results or 'none'}"
        )
    print(f"all {len(FIGURES)} figures and {len(EXPECTED_RESULTS)} result files present")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="skip writing results/run_report.html (CI uses this)",
    )
    args = parser.parse_args(argv)

    notebook_path = repo_root() / NOTEBOOK_PATH
    if not notebook_path.exists():
        raise SystemExit(f"notebook not found: {notebook_path}")

    mode = "SMOKE (shrunken, not the published results)" if smoke_mode() else "FULL"
    print(f"Executing {NOTEBOOK_PATH} -- mode: {mode}")
    started = time.monotonic()
    execute(notebook_path, write_html=not args.no_html)
    print(f"notebook finished in {time.monotonic() - started:.1f}s")

    check_outputs()
    return 0


if __name__ == "__main__":
    sys.exit(main())
