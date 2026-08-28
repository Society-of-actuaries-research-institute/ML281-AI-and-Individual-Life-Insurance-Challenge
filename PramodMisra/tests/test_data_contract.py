"""Data-contract tests for the ML281 inputs.

These assert the shape of the data the notebook relies on, so a silently changed
upstream file fails loudly instead of producing quietly different numbers.

The committed fixture (``data/sample/soa_sample_small.parquet``) is always
checked, which is what runs in CI. The full sample is checked too when present --
download it with ``python scripts/download_data.py``.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from ml281_paths import SEED, data_dir, read_checksums, sha256_file

FIXTURE = "data/sample/soa_sample_small.parquet"
FULL_SAMPLE = "data/soa_sample.parquet"

EXPECTED_COLUMNS = [
    "Observation_Year",
    "Issue_Age",
    "Attained_Age",
    "Duration",
    "Sex",
    "Smoker_Status",
    "Insurance_Plan",
    "Amount_Exposed",
    "Policies_Exposed",
    "Death_Count",
    "Death_Claim_Amount",
    "ExpDth_VBT2015wMI_Cnt",
    "Log_Amount_Exposed",
    "Log_Policies_Exposed",
    "Expected_Mortality_Rate",
]

#: Row count of the full published sample. A change here invalidates the results.
FULL_SAMPLE_ROWS = 131_922


def _available_frames() -> list[tuple[str, pd.DataFrame]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    for relative in (FIXTURE, FULL_SAMPLE):
        path = data_dir().parent / relative
        if path.exists():
            frames.append((relative, pd.read_parquet(path)))
    return frames


FRAMES = _available_frames()
FRAME_IDS = [name for name, _ in FRAMES]


@pytest.fixture(scope="module")
def full_sample() -> pd.DataFrame:
    path = data_dir() / "soa_sample.parquet"
    if not path.exists():
        pytest.skip("full sample absent; run `python scripts/download_data.py`")
    return pd.read_parquet(path)


def test_fixture_is_committed() -> None:
    """CI must be able to run the pipeline with no network access."""
    assert (data_dir() / "sample" / "soa_sample_small.parquet").exists(), (
        "the committed CI fixture is missing; regenerate with scripts/make_sample.py"
    )


@pytest.mark.parametrize(("name", "frame"), FRAMES, ids=FRAME_IDS)
def test_schema_is_exact(name: str, frame: pd.DataFrame) -> None:
    assert list(frame.columns) == EXPECTED_COLUMNS, f"{name}: column set or order changed"


@pytest.mark.parametrize(("name", "frame"), FRAMES, ids=FRAME_IDS)
def test_no_missing_values(name: str, frame: pd.DataFrame) -> None:
    assert int(frame.isna().sum().sum()) == 0, f"{name}: contains nulls"


@pytest.mark.parametrize(("name", "frame"), FRAMES, ids=FRAME_IDS)
def test_study_window_is_2012_to_2019(name: str, frame: pd.DataFrame) -> None:
    years = frame["Observation_Year"]
    assert int(years.min()) == 2012, f"{name}: study window starts at {years.min()}"
    assert int(years.max()) == 2019, f"{name}: study window ends at {years.max()}"
    # The out-of-time split needs both sides of the 2017/2018 boundary.
    assert int((years <= 2017).sum()) > 0, f"{name}: no training-era rows"
    assert int((years >= 2018).sum()) > 0, f"{name}: no out-of-time rows"


@pytest.mark.parametrize(("name", "frame"), FRAMES, ids=FRAME_IDS)
def test_counts_and_exposures_are_non_negative(name: str, frame: pd.DataFrame) -> None:
    for column in ("Death_Count", "Amount_Exposed", "Policies_Exposed",
                   "Death_Claim_Amount", "ExpDth_VBT2015wMI_Cnt"):
        assert int((frame[column] < 0).sum()) == 0, f"{name}: negative {column}"


@pytest.mark.parametrize(("name", "frame"), FRAMES, ids=FRAME_IDS)
def test_categorical_levels_present(name: str, frame: pd.DataFrame) -> None:
    assert set(frame["Sex"].unique()) == {"M", "F"}, f"{name}: unexpected Sex levels"
    assert set(frame["Smoker_Status"].unique()) == {"NS", "S", "U"}, (
        f"{name}: unexpected Smoker_Status levels"
    )
    assert frame["Insurance_Plan"].nunique() == 7, f"{name}: expected 7 insurance plans"


@pytest.mark.parametrize(("name", "frame"), FRAMES, ids=FRAME_IDS)
def test_copula_step_has_usable_cells(name: str, frame: pd.DataFrame) -> None:
    """Figure 6 needs cells carrying both a death and a claim amount."""
    usable = int(((frame["Death_Count"] > 0) & (frame["Death_Claim_Amount"] > 0)).sum())
    assert usable > 100, f"{name}: only {usable} copula-usable cells"


@pytest.mark.parametrize(("name", "frame"), FRAMES, ids=FRAME_IDS)
def test_splits_are_disjoint(name: str, frame: pd.DataFrame) -> None:
    """The 60/20/20 split the notebook builds must not leak between partitions."""
    index = list(range(len(frame)))
    train_ids, temp_ids = train_test_split(index, test_size=0.4, random_state=SEED)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=SEED)
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(frame)


def test_full_sample_row_count(full_sample: pd.DataFrame) -> None:
    assert len(full_sample) == FULL_SAMPLE_ROWS


def test_full_sample_checksum_matches_manifest() -> None:
    path = data_dir() / "soa_sample.parquet"
    if not path.exists():
        pytest.skip("full sample absent; run `python scripts/download_data.py`")
    expected = read_checksums()["data/soa_sample.parquet"]
    assert sha256_file(path) == expected, "full sample does not match data/CHECKSUMS.txt"


def test_full_table_aggregates_are_consistent() -> None:
    """Every grouping in full_table_ae.json must sum to the same population."""
    payload = json.loads((data_dir() / "full_table_ae.json").read_text(encoding="utf-8"))
    totals = {
        key: (sum(row["deaths"] for row in rows), sum(row["exp"] for row in rows))
        for key, rows in payload.items()
    }
    deaths = {round(d) for d, _ in totals.values()}
    expected = {round(e) for _, e in totals.values()}
    assert len(deaths) == 1, f"groupings disagree on total deaths: {totals}"
    assert len(expected) == 1, f"groupings disagree on total expected: {totals}"

    total_deaths, total_expected = next(iter(totals.values()))
    assert round(total_deaths / total_expected, 4) == 1.0764, "full-table A/E changed"


def test_full_table_checksum_matches_manifest() -> None:
    expected = read_checksums()["data/full_table_ae.json"]
    assert sha256_file(data_dir() / "full_table_ae.json") == expected
