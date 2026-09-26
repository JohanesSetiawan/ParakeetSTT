"""
Crash-safe file replacement shared by checkpoint, marker, and CSV writers.

Writes go to a temporary file in the destination directory, are flushed and
fsynced, then atomically renamed over the target. An interrupted write can
leave a stray ``.tmp`` file but never a truncated target.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_text_atomic(path: Path, text: str) -> None:
    """
    Replace ``path`` with ``text`` (UTF-8) in one atomic rename.

    Args:
        path: Destination file. Its parent directory is created when missing.
        text: Complete file content. Newlines are written verbatim.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            temporary_file.write(text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        except BaseException:
            temporary_file.close()
            temporary_path.unlink(missing_ok=True)
            raise

    try:
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, payload: Any) -> None:
    """Serialize ``payload`` as sorted, indented ASCII JSON and replace ``path``."""

    text = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    write_text_atomic(path, text)
