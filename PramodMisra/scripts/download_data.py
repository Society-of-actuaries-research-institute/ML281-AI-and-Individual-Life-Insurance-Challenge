"""Fetch the ML281 model inputs from the public Hugging Face dataset.

Idempotent: a file that is already present and matches ``data/CHECKSUMS.txt`` is
left alone. A file that is present but does NOT match is re-downloaded once and
then re-verified; a second mismatch is a hard error rather than a warning.

Usage
-----
    python scripts/download_data.py
    python scripts/download_data.py --force     # re-download even if valid

Requires no credentials. Roughly 6 MB over the network on a cold run.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from ml281_paths import DATA_FILES, HF_BASE, data_dir, read_checksums, sha256_file

#: Bound on download attempts per file (initial try + one retry).
MAX_ATTEMPTS: int = 2
#: Network timeout per request, seconds.
TIMEOUT_SECONDS: int = 120


def _download(url: str, destination: Path) -> None:
    """Download ``url`` to ``destination`` via a temporary file."""
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            temp_path.write_bytes(response.read())
    except urllib.error.URLError as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"download failed for {url}: {exc}") from exc
    temp_path.replace(destination)


def fetch_one(filename: str, expected_sha256: str, force: bool) -> str:
    """Ensure ``filename`` exists in ``data/`` and matches its checksum.

    Returns a short status string: ``"cached"``, ``"downloaded"`` or
    ``"repaired"``.
    """
    destination = data_dir() / filename
    if destination.exists() and not force:
        if sha256_file(destination) == expected_sha256:
            return "cached"
        print(f"  checksum mismatch on existing {filename}; re-downloading")

    status = "repaired" if destination.exists() else "downloaded"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        _download(HF_BASE + filename, destination)
        actual = sha256_file(destination)
        if actual == expected_sha256:
            return status
        print(
            f"  attempt {attempt}/{MAX_ATTEMPTS}: sha256 mismatch for {filename}\n"
            f"    expected {expected_sha256}\n    actual   {actual}"
        )
    raise RuntimeError(
        f"{filename} failed checksum verification after {MAX_ATTEMPTS} attempts. "
        "Do not use this file; the download is corrupt or the upstream dataset changed."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if checksums already match"
    )
    args = parser.parse_args(argv)

    data_dir().mkdir(parents=True, exist_ok=True)
    checksums = read_checksums()

    for filename in DATA_FILES:
        key = f"data/{filename}"
        if key not in checksums:
            raise KeyError(f"no checksum recorded for {key} in data/CHECKSUMS.txt")
        print(f"{filename}:", end=" ")
        status = fetch_one(filename, checksums[key], force=args.force)
        size_mb = (data_dir() / filename).stat().st_size / 1e6
        print(f"{status} ({size_mb:.2f} MB, sha256 verified)")

    print("\nAll inputs verified. Run the notebook to reproduce the results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
