"""Assert the committed results ledger still matches the published values.

This is the test that protects the paper: if someone regenerates ``results/`` with
a different environment, a changed dataset, or an edited notebook and the numbers
move, CI fails rather than the drift going unnoticed.

Reference values and per-metric tolerances live in
``tests/published_reference.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ml281_paths import published_results_dir, repo_root

REFERENCE = json.loads(
    (Path(repo_root()) / "tests" / "published_reference.json").read_text(encoding="utf-8")
)

SCALAR_CHECKS = REFERENCE["checks"]
MULTIMODEL_CHECKS = REFERENCE["multimodel_cv_r2"]


def _ledger(name: str) -> dict[str, Any]:
    path = published_results_dir() / f"{name}.json"
    if not path.exists():
        pytest.skip(f"results/{name}.json absent; run scripts/run_notebook.py")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(dotted_key: str) -> float:
    parts = dotted_key.split(".")
    node: Any = _ledger(parts[0])
    for part in parts[1:]:
        assert part in node, f"{dotted_key}: {part!r} missing from the ledger"
        node = node[part]
    return float(node)


def test_ledger_is_not_a_smoke_run() -> None:
    """Smoke numbers must never be committed as results."""
    environment = _ledger("environment")
    assert environment["smoke_mode"] is False, (
        "results/ was generated in smoke mode; regenerate with a full run"
    )
    assert environment["input_file"] == "soa_sample.parquet"
    assert environment["n_cells"] == 131_922
    assert environment["seed"] == 42


def test_ledger_records_the_verified_input() -> None:
    environment = _ledger("environment")
    assert environment["input_sha256"] == (
        "9bcd09b4ef7b25e8c5a9dfe8a6723502b1cd5541af2b8c5b49a5d004793c2e9a"
    ), "the ledger was produced from a different input file"


@pytest.mark.parametrize("check", SCALAR_CHECKS, ids=[c["key"] for c in SCALAR_CHECKS])
def test_published_scalar_reproduces(check: dict[str, Any]) -> None:
    actual = _resolve(check["key"])
    published = float(check["published"])
    tolerance = float(check["tolerance"])
    assert abs(actual - published) <= tolerance, (
        f"{check['label']}: published {published}, ledger {actual} "
        f"(tolerance {tolerance})"
    )


@pytest.mark.parametrize(
    "check", MULTIMODEL_CHECKS, ids=[c["model"] for c in MULTIMODEL_CHECKS]
)
def test_multimodel_cv_r2_reproduces(check: dict[str, Any]) -> None:
    models = {row["model"]: row for row in _ledger("multimodel")["models"]}
    assert check["model"] in models, f"{check['model']} missing from results/multimodel.json"
    actual = float(models[check["model"]]["cv_r2_mean"])
    published = float(check["published"])
    assert abs(actual - published) <= float(check["tolerance"]), (
        f"{check['model']}: published {published}, ledger {actual}"
    )


def test_out_of_time_ranking_is_preserved() -> None:
    """The report's headline claim: static VBT2015 is closest to A/E = 1.0."""
    ledger = _ledger("out_of_time")
    distances = {
        "vbt_static": abs(ledger["ae_vbt_static"] - 1.0),
        "poisson_vbt_offset": abs(ledger["ae_poisson_vbt_offset"] - 1.0),
        "xgboost_anchored": abs(ledger["ae_xgboost_anchored"] - 1.0),
        "xgboost_no_vbt": abs(ledger["ae_xgboost_no_vbt"] - 1.0),
    }
    assert min(distances, key=distances.get) == "vbt_static", (
        f"out-of-time ranking changed: {distances}"
    )


def test_vbt_ablation_direction_is_preserved() -> None:
    """Removing the VBT feature must not reduce test R2 -- the Q1 finding."""
    ledger = _ledger("ablation")
    assert ledger["test_r2_without_vbt"] >= ledger["test_r2_with_vbt"], (
        "the ablation finding reversed: removing VBT now hurts performance"
    )


def test_vbt_offset_improves_on_squared_error_deviance() -> None:
    """The Q4 finding: a proper count model has lower Poisson deviance."""
    ledger = _ledger("poisson_offset")
    assert ledger["vbt_offset"]["test_poisson_dev"] < ledger["submitted_squared_error_deviance"]
    assert ledger["exposure_offset"]["test_poisson_dev"] < ledger["submitted_squared_error_deviance"]
