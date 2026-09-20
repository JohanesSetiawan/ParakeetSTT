"""Bounded-memory hashing utilities for checkpoint artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


# =============================================================================
# Reproducibility hashing
# =============================================================================
# Checkpoint files can exceed available memory. Hashing therefore reads fixed
# blocks and never materializes another copy of the source artifact.
# =============================================================================


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Hash a file in bounded memory and return its lowercase SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while block := input_file.read(block_size):
            digest.update(block)
    return digest.hexdigest()
