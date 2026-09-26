"""
CLI for manifest-aware checkpoint download and automatic PyTorch conversion.

Run from the repository root with:

    venv\\Scripts\\python.exe -m src.commands.prepare_checkpoint

The destination and network policy come from config.toml ([paths] weights_dir
and [checkpoint]). Matching source artifacts and model.pth are reused; only
missing or invalid files are downloaded, and stale conversion output is
regenerated after all source artifacts pass identity validation.
"""

from __future__ import annotations

import argparse
import logging
import sys

from ..checkpoint.orchestration import prepare_checkpoint
from ..configuration.settings import load_settings
from ..runtime.logging_setup import configure_run_logging


# A literal name: under `python -m` __name__ is "__main__", outside the "src"
# logger that owns the run log file, so lines and tracebacks would be lost.
logger = logging.getLogger("src.commands.prepare_checkpoint")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line; every setting lives in config.toml."""

    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run acquisition and conversion checks; return the process exit code."""

    parse_arguments(argv)
    settings = load_settings()
    run_id, log_path = configure_run_logging(settings.paths.log_dir, settings.logging.level)
    print(f"Run id: {run_id}")
    print(f"Log file: {log_path}")

    try:
        result = prepare_checkpoint(
            checkpoint_dir=settings.paths.weights_dir,
            checkpoint_settings=settings.checkpoint,
            progress_callback=print,
        )
    except Exception as error:
        logger.exception("checkpoint preparation failed")
        print(f"Error: {error}", file=sys.stderr)
        print(f"Details: {log_path} (run {run_id})", file=sys.stderr)
        return 1

    print("Checkpoint preparation complete")
    print(f"Directory: {result.checkpoint_dir}")
    for download in result.downloads:
        print(
            f"File: {download.filename}, action: {download.action}, "
            f"bytes: {download.size_bytes}, sha256: {download.sha256}"
        )
        logger.info("file=%s action=%s sha256=%s", download.filename, download.action, download.sha256)
    print(
        f"model.pth: {result.conversion.action}, "
        f"bytes: {result.conversion.output_size_bytes}, "
        f"sha256: {result.conversion.output_sha256}"
    )
    logger.info("model.pth action=%s sha256=%s", result.conversion.action, result.conversion.output_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
