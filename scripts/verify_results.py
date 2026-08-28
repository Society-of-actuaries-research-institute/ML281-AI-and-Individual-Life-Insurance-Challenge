"""Compare a freshly generated results ledger against the published values.

Reads ``tests/published_reference.json`` and checks each entry against the JSON
files in ``results/``. Prints a table and exits non-zero if anything is outside
tolerance, so it works as both a human report and a CI gate::

    python scripts/verify_results.py

Run this only after a FULL notebook run. Smoke output lives in ``.smoke/`` and is
not comparable to the published numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ml281_paths import published_results_dir, repo_root

REFERENCE_PATH = "tests/published_reference.json"


def load_reference() -> dict[str, Any]:
    path = repo_root() / REFERENCE_PATH
    if not path.exists():
        raise SystemExit(f"missing reference file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def lookup(key: str) -> float:
    """Resolve a dotted ``file.field[.subfield]`` key against results/."""
    parts = key.split(".")
    ledger_path = published_results_dir() / f"{parts[0]}.json"
    if not ledger_path.exists():
        raise SystemExit(
            f"missing {ledger_path.relative_to(repo_root())} -- run "
            "`python scripts/run_notebook.py` first (full mode, not smoke)."
        )
    node: Any = json.loads(ledger_path.read_text(encoding="utf-8"))
    for part in parts[1:]:
        if not isinstance(node, dict) or part not in node:
            raise SystemExit(f"key {key!r} not found in the results ledger")
        node = node[part]
    if not isinstance(node, (int, float)):
        raise SystemExit(f"key {key!r} resolved to {type(node).__name__}, expected a number")
    return float(node)


def check_scalars(reference: dict[str, Any]) -> list[str]:
    """Check every dotted-key entry. Returns a list of failure descriptions."""
    failures: list[str] = []
    print(f"{'metric':<52}{'published':>11}{'actual':>11}{'delta':>10}  status")
    print("-" * 92)
    for check in reference["checks"]:
        actual = lookup(check["key"])
        published = float(check["published"])
        delta = actual - published
        ok = abs(delta) <= float(check["tolerance"])
        print(
            f"{check['label'][:52]:<52}{published:>11.4f}{actual:>11.4f}"
            f"{delta:>+10.4f}  {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(
                f"{check['label']}: published {published}, got {actual} "
                f"(tolerance {check['tolerance']})"
            )
    return failures


def check_multimodel(reference: dict[str, Any]) -> list[str]:
    """Check the per-model cross-validated R2 values."""
    ledger_path = published_results_dir() / "multimodel.json"
    if not ledger_path.exists():
        raise SystemExit(f"missing {ledger_path}")
    by_model = {
        row["model"]: row for row in json.loads(ledger_path.read_text(encoding="utf-8"))["models"]
    }
    failures: list[str] = []
    print("\nMulti-model 5-fold CV R2")
    print("-" * 92)
    for entry in reference["multimodel_cv_r2"]:
        name = entry["model"]
        if name not in by_model:
            failures.append(f"{name}: absent from results/multimodel.json")
            continue
        actual = float(by_model[name]["cv_r2_mean"])
        published = float(entry["published"])
        delta = actual - published
        ok = abs(delta) <= float(entry["tolerance"])
        print(
            f"{name[:52]:<52}{published:>11.4f}{actual:>11.4f}"
            f"{delta:>+10.4f}  {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(f"{name}: published {published}, got {actual}")
    return failures


def warn_on_environment_drift() -> None:
    """Print the environment record so a mismatch is visible in the log."""
    env_path = published_results_dir() / "environment.json"
    if not env_path.exists():
        return
    env = json.loads(env_path.read_text(encoding="utf-8"))
    if env.get("smoke_mode"):
        raise SystemExit(
            "results/ was produced in SMOKE mode. Re-run without ML281_SMOKE "
            "before verifying against the published values."
        )
    drift = env.get("version_drift") or {}
    if drift:
        print("\nNOTE: this ledger was produced with non-reference package versions:")
        for package, versions in drift.items():
            print(f"  {package}: reference {versions['reference']}, running {versions['running']}")
        print("  Small deviations below are expected; see docs/DATA_PROVENANCE.md.")


def main() -> int:
    reference = load_reference()
    warn_on_environment_drift()
    failures = check_scalars(reference) + check_multimodel(reference)

    print()
    if failures:
        print(f"{len(failures)} check(s) OUTSIDE TOLERANCE:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All published values reproduced within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
