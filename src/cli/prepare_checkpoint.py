"""
CLI for manifest-aware checkpoint download and automatic PyTorch conversion.

Run from the repository root with:

    venv\\Scripts\\python.exe -m src.cli.prepare_checkpoint

Matching source artifacts and model.pth are reused. Only missing or invalid
source files are downloaded, and stale conversion output is regenerated after
all source artifacts pass identity validation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..checkpoint import prepare_checkpoint
from ..config import DEFAULT_WEIGHTS_DIR


def setup_logging() -> logging.Logger:
    """Configure plain-text console logging for interactive progress."""

    logger = logging.getLogger("checkpoint_preparation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def parse_arguments() -> argparse.Namespace:
    """Parse the configurable checkpoint destination directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
        help="Destination directory for source files and model.pth",
    )
    return parser.parse_args()


def main() -> None:
    """Run acquisition, conversion freshness checks, and terminal reporting."""

    arguments = parse_arguments()
    logger = setup_logging()
    result = prepare_checkpoint(
        checkpoint_dir=arguments.checkpoint_dir,
        progress_callback=logger.info,
    )

    print("Checkpoint preparation complete")
    print(f"Directory: {result.checkpoint_dir}")
    print("Files:")
    for download in result.downloads:
        print(
            f"  {download.filename}: {download.action}, "
            f"{download.size_bytes} bytes, sha256={download.sha256}"
        )
    print(
        f"model.pth: {result.conversion.action}, "
        f"{result.conversion.output_size_bytes} bytes, "
        f"sha256={result.conversion.output_sha256}"
    )


if __name__ == "__main__":
    main()
