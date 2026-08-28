"""
Chunking tool: Convert 12GB tab-separated ILEC data into partitioned Parquet files.
Reads in chunks, optimizes dtypes, and writes per-year parquet files for fast downstream use.
"""
import pandas as pd
import os
import sys
import time
from pathlib import Path

# Configuration — resolved relative to this file (soa_ai_v2/code/)
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
PARQUET_DIR = DATA_DIR / "parquet"
RAW_DATA_FILE = DATA_DIR / "ILEC_2012_19 - 20240429.txt"

CHUNK_SIZE = 500_000  # rows per chunk

# Optimized dtypes to minimize memory
DTYPE_MAP = {
    "Observation_Year": "int16",
    "Age_Ind": "category",
    "Sex": "category",
    "Smoker_Status": "category",
    "Insurance_Plan": "category",
    "Issue_Age": "int16",
    "Duration": "int16",
    "Face_Amount_Band": "category",
    "Issue_Year": "int16",
    "Attained_Age": "int16",
    "SOA_Antp_Lvl_TP": "category",
    "SOA_Guar_Lvl_TP": "category",
    "SOA_Post_Lvl_Ind": "category",
    "Slct_Ult_Ind": "category",
    "Preferred_Indicator": "category",
    "Number_of_Pfd_Classes": "category",
    "Preferred_Class": "category",
    "Amount_Exposed": "float32",
    "Policies_Exposed": "float32",
    "Death_Claim_Amount": "float32",
    "Death_Count": "float32",
    "ExpDth_VBT2015_Cnt": "float32",
    "ExpDth_VBT2015_Amt": "float32",
    "ExpDth_VBT2015wMI_Cnt": "float32",
    "ExpDth_VBT2015wMI_Amt": "float32",
    "Cen2MomP1wMI_Amt": "float64",
    "Cen2MomP2wMI_Amt": "float64",
    "Cen3MomP1wMI_Amt": "float64",
    "Cen3MomP2wMI_Amt": "float64",
    "Cen3MomP3wMI_Amt": "float64",
}

# Column names
COL_NAMES = [
    'Observation_Year', 'Age_Ind', 'Sex', 'Smoker_Status', 'Insurance_Plan',
    'Issue_Age', 'Duration', 'Face_Amount_Band', 'Issue_Year', 'Attained_Age',
    'SOA_Antp_Lvl_TP', 'SOA_Guar_Lvl_TP', 'SOA_Post_Lvl_Ind', 'Slct_Ult_Ind',
    'Preferred_Indicator', 'Number_of_Pfd_Classes', 'Preferred_Class',
    'Amount_Exposed', 'Policies_Exposed', 'Death_Claim_Amount', 'Death_Count',
    'ExpDth_VBT2015_Cnt', 'ExpDth_VBT2015_Amt', 'ExpDth_VBT2015wMI_Cnt',
    'ExpDth_VBT2015wMI_Amt', 'Cen2MomP1wMI_Amt', 'Cen2MomP2wMI_Amt',
    'Cen3MomP1wMI_Amt', 'Cen3MomP2wMI_Amt', 'Cen3MomP3wMI_Amt'
]


def check_raw_data():
    """Check if raw data file exists."""
    if not RAW_DATA_FILE.exists():
        print(f"ERROR: Raw data file not found: {RAW_DATA_FILE}")
        return False
    print(f"Raw data file found: {RAW_DATA_FILE}")
    print(f"Size: {RAW_DATA_FILE.stat().st_size / 1e9:.2f} GB")
    return True


def convert_to_parquet():
    if not check_raw_data():
        sys.exit(1)

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    year_buffers = {}
    total_rows = 0
    start = time.time()

    print(f"Reading from: {RAW_DATA_FILE}")
    print(f"Output to: {PARQUET_DIR}")
    print(f"Chunk size: {CHUNK_SIZE:,} rows")
    print()

    reader = pd.read_csv(
        RAW_DATA_FILE,
        sep="\t",
        names=COL_NAMES,
        dtype=DTYPE_MAP,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        skiprows=1,  # Skip header row
    )

    for chunk_idx, chunk in enumerate(reader):
        total_rows += len(chunk)
        elapsed = time.time() - start
        rate = total_rows / elapsed if elapsed > 0 else 0
        print(f"  Chunk {chunk_idx}: {total_rows:,} rows processed ({rate:,.0f} rows/sec)")

        for year, group in chunk.groupby("Observation_Year"):
            if year not in year_buffers:
                year_buffers[year] = []
            year_buffers[year].append(group)

            # Write when buffer gets large enough (every ~2M rows per year)
            total_buffered = sum(len(g) for g in year_buffers[year])
            if total_buffered >= 2_000_000:
                _flush_year(year, year_buffers[year])
                year_buffers[year] = []

    # Flush remaining
    for year, groups in year_buffers.items():
        if groups:
            _flush_year(year, groups)

    elapsed = time.time() - start
    print(f"\nDone! {total_rows:,} rows in {elapsed:.1f}s")
    print(f"Parquet files in: {PARQUET_DIR}")
    for f in sorted(PARQUET_DIR.iterdir()):
        if f.suffix == ".parquet":
            print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


def _flush_year(year, groups):
    out_path = PARQUET_DIR / f"ilec_{year}.parquet"
    df = pd.concat(groups, ignore_index=True)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"    Flushed year {year}: {len(df):,} rows -> {out_path}")


if __name__ == "__main__":
    convert_to_parquet()
