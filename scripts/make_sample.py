"""Build the small committed CI fixture from the full sample.

The fixture at ``data/sample/soa_sample_small.parquet`` exists so that the
pipeline can run end to end -- in CI, or on a laptop with no network -- without
downloading the 5.9 MB input. It is a **test fixture, not a statistical sample**:
death-bearing exposure cells are deliberately over-represented so that the
copula, SHAP and out-of-time steps all have enough signal to execute. Numbers
produced from it are meaningless as actuarial results.

Regenerate with::

    python scripts/download_data.py
    python scripts/make_sample.py

Deterministic: seed 42, so the committed file is byte-reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from ml281_paths import SEED, data_dir, set_seed

#: Death-bearing cells to keep (of ~5,100 available).
N_DEATH_ROWS: int = 1_500
#: Zero-death cells to keep, stratified across year x smoker status.
N_ZERO_ROWS: int = 2_500
#: Fixture must cover the full study window and every category level.
EXPECTED_YEARS: int = 8
EXPECTED_SMOKER_LEVELS: int = 3
EXPECTED_PLANS: int = 7


def _stratified_take(frame: pd.DataFrame, keys: list[str], total: int) -> pd.DataFrame:
    """Take ~``total`` rows from ``frame``, spread evenly across ``keys`` groups.

    At least one row per group, so rare strata never vanish from the fixture.
    """
    groups = frame.groupby(keys, observed=True)
    per_group = max(1, total // max(1, groups.ngroups))
    parts = [
        group.sample(n=min(per_group, len(group)), random_state=SEED)
        for _, group in groups
    ]
    return pd.concat(parts, ignore_index=True)


def build_sample(source: pd.DataFrame) -> pd.DataFrame:
    """Return the CI fixture: over-sampled deaths plus stratified zero rows."""
    deaths = source[source["Death_Count"] > 0]
    zeros = source[source["Death_Count"] == 0]

    death_part = _stratified_take(deaths, ["Observation_Year"], N_DEATH_ROWS)
    zero_part = _stratified_take(
        zeros, ["Observation_Year", "Smoker_Status"], N_ZERO_ROWS
    )

    fixture = pd.concat([death_part, zero_part], ignore_index=True)
    # Shuffle so nothing downstream can depend on the deaths-first ordering.
    fixture = fixture.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return fixture[list(source.columns)]


def check_coverage(fixture: pd.DataFrame) -> None:
    """Fail loudly if the fixture cannot exercise every pipeline branch."""
    assert len(fixture) > 0, "fixture is empty"
    assert fixture["Observation_Year"].nunique() == EXPECTED_YEARS, (
        f"expected {EXPECTED_YEARS} observation years, "
        f"got {fixture['Observation_Year'].nunique()}"
    )
    assert fixture["Smoker_Status"].nunique() == EXPECTED_SMOKER_LEVELS, (
        "fixture is missing a smoker-status level"
    )
    assert fixture["Insurance_Plan"].nunique() == EXPECTED_PLANS, (
        f"expected {EXPECTED_PLANS} insurance plans, "
        f"got {fixture['Insurance_Plan'].nunique()}"
    )
    # Out-of-time step needs both sides of the 2017/2018 boundary.
    assert (fixture["Observation_Year"] <= 2017).sum() > 0, "no pre-2018 rows"
    assert (fixture["Observation_Year"] >= 2018).sum() > 0, "no 2018-2019 rows"
    # Copula step needs cells with both a death and a claim amount.
    usable_copula = ((fixture["Death_Count"] > 0) & (fixture["Death_Claim_Amount"] > 0)).sum()
    assert usable_copula >= 500, f"only {usable_copula} copula-usable cells; need >= 500"
    assert fixture.isna().sum().sum() == 0, "fixture contains nulls"


def main() -> int:
    set_seed()
    source_path = data_dir() / "soa_sample.parquet"
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} not found. Run `python scripts/download_data.py` first."
        )

    source = pd.read_parquet(source_path)
    fixture = build_sample(source)
    check_coverage(fixture)

    out_path = data_dir() / "sample" / "soa_sample_small.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_parquet(out_path, index=False)

    deaths = int((fixture["Death_Count"] > 0).sum())
    print(
        f"wrote {out_path.relative_to(Path.cwd()) if out_path.is_relative_to(Path.cwd()) else out_path}\n"
        f"  rows: {len(fixture):,} (source {len(source):,})\n"
        f"  death-bearing cells: {deaths:,} ({deaths / len(fixture):.1%} vs "
        f"{(source['Death_Count'] > 0).mean():.1%} in source -- over-sampled on purpose)\n"
        f"  size: {out_path.stat().st_size / 1e3:.0f} KB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
