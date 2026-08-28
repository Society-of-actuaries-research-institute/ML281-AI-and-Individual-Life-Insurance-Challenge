"""Shared path, checksum and seed helpers for the ML281 reproduction scripts.

Kept deliberately tiny and dependency-free (stdlib only) so it imports cleanly in
CI, in a bare virtualenv, and from inside the notebook.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

SEED: int = 42

#: Public, credential-free Hugging Face dataset holding the model inputs.
HF_BASE: str = (
    "https://huggingface.co/datasets/pramodmisra/soa-ml281-sample/resolve/main/"
)

#: Files fetched by ``scripts/download_data.py``, in the order they are checked.
DATA_FILES: tuple[str, ...] = ("soa_sample.parquet", "full_table_ae.json")

#: Figures the pipeline must produce. CI asserts all seven exist.
FIGURES: tuple[str, ...] = (
    "fig1_eda_ae_segments.png",
    "fig2_xgb_performance.png",
    "fig3_multimodel.png",
    "fig4_ablation_offset.png",
    "fig5_shap.png",
    "fig6_copula_aggregate.png",
    "fig7_out_of_time.png",
)


def repo_root() -> Path:
    """Return the repository root, resolved from this file's location."""
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    return repo_root() / "data"


def output_root() -> Path:
    """Where the notebook writes figures and results for the current mode.

    Smoke runs are diverted to ``.smoke/`` so a fast CI run can never overwrite
    the published figures and results committed at the repo root.
    """
    return repo_root() / ".smoke" if smoke_mode() else repo_root()


def results_dir() -> Path:
    return output_root() / "results"


def figures_dir() -> Path:
    return output_root() / "figures"


def published_results_dir() -> Path:
    """The committed ledger, regardless of smoke mode."""
    return repo_root() / "results"


def published_figures_dir() -> Path:
    """The committed figures, regardless of smoke mode."""
    return repo_root() / "figures"


def smoke_mode() -> bool:
    """True when ``ML281_SMOKE`` is set to a truthy value.

    Smoke mode shrinks the expensive steps so CI finishes in a couple of minutes
    against the committed sample. It does NOT reproduce the published numbers --
    leave the variable unset for that.
    """
    return os.environ.get("ML281_SMOKE", "").strip().lower() in {"1", "true", "yes"}


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Return the hex SHA-256 of ``path``, read in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums() -> dict[str, str]:
    """Parse ``data/CHECKSUMS.txt`` into {relative path: sha256}."""
    checksums_path = data_dir() / "CHECKSUMS.txt"
    if not checksums_path.exists():
        raise FileNotFoundError(f"missing checksum manifest: {checksums_path}")
    entries: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel_path = line.partition("  ")
        if not digest or not rel_path:
            raise ValueError(f"malformed checksum line: {line!r}")
        entries[rel_path.strip()] = digest.strip()
    return entries


def set_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy RNGs and ``PYTHONHASHSEED``."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:  # numpy absent in a docs-only environment
        return
    np.random.seed(seed)
