"""
Configuration for SOA AI Challenge project
"""
from pathlib import Path

# Base directories — resolved relative to this file (soa_ai_v2/code/config.py)
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
PARQUET_DIR = DATA_DIR / "parquet"
OUTPUT_DIR = PROJECT_DIR / "results"

# Data files
RAW_DATA_FILE = DATA_DIR / "ILEC_2012_19 - 20240429.txt"

# Create directories
PARQUET_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def check_raw_data():
    """Check if raw data file exists."""
    if not RAW_DATA_FILE.exists():
        print(f"ERROR: Raw data file not found: {RAW_DATA_FILE}")
        return False
    print(f"Raw data file found: {RAW_DATA_FILE}")
    print(f"Size: {RAW_DATA_FILE.stat().st_size / 1e9:.2f} GB")
    return True
