"""
Manifest-aware checkpoint downloader for the local Parakeet artifacts.

This module owns network retrieval, per-file validation, and atomic replacement.
It does not know how model tensors are parsed or converted. A local manifest is
stored beside the checkpoint and records URL, byte size, and SHA-256 for each
successfully downloaded file. Existing files with matching manifest evidence are
preserved. Missing or mismatching files are downloaded individually.

The downloader uses only Python standard-library networking and filesystem APIs.
No cache, package manager, model hub SDK, or third-party downloader is needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..configuration.config import SOURCE_WEIGHTS_FILENAME
from ..configuration.settings import CheckpointSettings
from ..runtime.filesystem import write_json_atomic


logger = logging.getLogger(__name__)


# =============================================================================
# Download specification and manifest
# =============================================================================
# The URLs, sizes, and digests below are pinned on purpose. They identify the
# exact upstream revision this runtime was verified against (token parity with
# the reference implementation); changing the source without changing the pins
# is rejected by the identity check, which is the point. They are an integrity
# invariant, not a user setting, so they stay in code.
# =============================================================================


@dataclass(frozen=True)
class DownloadSpec:
    """One checkpoint artifact and its remote URL."""

    filename: str
    url: str
    expected_size_bytes: int
    expected_sha256: str | None = None
    expected_git_blob_sha1: str | None = None


@dataclass(frozen=True)
class FileEvidence:
    """Measured local file evidence used to decide whether redownload is needed."""

    filename: str
    size_bytes: int
    sha256: str
    url: str
    remote_content_length: int | None = None
    remote_etag: str | None = None
    git_blob_sha1: str | None = None


@dataclass(frozen=True)
class DownloadResult:
    """Outcome for one artifact without hiding whether it was reused or fetched."""

    filename: str
    action: str
    path: str
    size_bytes: int
    sha256: str
    url: str


CHECKPOINT_DOWNLOADS: tuple[DownloadSpec, ...] = (
    DownloadSpec(
        "config.json",
        "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/config.json?download=true",
        1153,
        expected_git_blob_sha1="969c65410a736e4a3922f968b3481b2418aba786",
    ),
    DownloadSpec(
        "generation_config.json",
        "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/generation_config.json?download=true",
        289,
        expected_git_blob_sha1="e368b292e2ad5232aae2a3e45f4761097e5fc216",
    ),
    DownloadSpec(
        SOURCE_WEIGHTS_FILENAME,
        "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/model.safetensors?download=true",
        2508311120,
        expected_sha256="3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11",
    ),
    DownloadSpec(
        "processor_config.json",
        "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/processor_config.json?download=true",
        392,
        expected_git_blob_sha1="7acffca89b7dd12e0ebcd085c7181c91cae6575c",
    ),
    DownloadSpec(
        "tokenizer.json",
        "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/tokenizer.json?download=true",
        1159960,
        expected_git_blob_sha1="a10a554cf61e38108756650018d5781ae9675c1d",
    ),
    DownloadSpec(
        "tokenizer_config.json",
        "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/tokenizer_config.json?download=true",
        290,
        expected_git_blob_sha1="8a31c60ee8ce7b81b999061d3a233845cf8bd033",
    ),
)

MANIFEST_FILENAME = "download_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


# =============================================================================
# Bounded file evidence
# =============================================================================
# Hashing streams the file in fixed-size blocks. The large safetensors payload is
# never duplicated in memory just to decide whether it matches the manifest.
# =============================================================================


def measure_file(
    path: Path,
    url: str,
    block_size: int = 1024 * 1024,
    remote_content_length: int | None = None,
    remote_etag: str | None = None,
) -> FileEvidence:
    """Return size and SHA-256 evidence for an existing local file."""

    digest = hashlib.sha256()
    git_blob_digest = hashlib.sha1()
    file_size = path.stat().st_size
    git_blob_digest.update(f"blob {file_size}\0".encode("ascii"))
    size_bytes = 0

    with path.open("rb") as input_file:
        while block := input_file.read(block_size):
            digest.update(block)
            git_blob_digest.update(block)
            size_bytes += len(block)

    return FileEvidence(
        filename=path.name,
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
        url=url,
        remote_content_length=remote_content_length,
        remote_etag=remote_etag,
        git_blob_sha1=git_blob_digest.hexdigest(),
    )


def load_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    """Load a valid manifest or return an empty manifest for first use."""

    manifest_path = checkpoint_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "files": {}}

    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Download manifest must be a JSON object: {manifest_path}")

    files = value.get("files")
    if value.get("schema_version") != MANIFEST_SCHEMA_VERSION or not isinstance(files, dict):
        raise ValueError(f"Unsupported or malformed download manifest: {manifest_path}")

    return value


def write_manifest(checkpoint_dir: Path, manifest: dict[str, Any]) -> None:
    """Atomically persist the per-file download manifest."""

    write_json_atomic(checkpoint_dir / MANIFEST_FILENAME, manifest)


# =============================================================================
# Per-file matching and download
# =============================================================================
# A file is reusable only when it exists and its recorded URL, byte count, and
# digest all match. If no manifest evidence exists, remote Content-Length is used
# as a cheap preflight check; the file is still downloaded when its identity cannot
# be proven locally. This avoids silently trusting an unverified legacy file.
# =============================================================================


def _manifest_entry_matches(
    path: Path,
    spec: DownloadSpec,
    entry: Any,
    remote_content_length: int | None,
    remote_etag: str | None,
) -> FileEvidence | None:
    """Return evidence when a manifest entry proves the file is unchanged."""

    if not path.is_file() or not isinstance(entry, dict):
        return None
    if entry.get("url") != spec.url:
        return None
    if not isinstance(entry.get("size_bytes"), int) or not isinstance(entry.get("sha256"), str):
        return None

    evidence = measure_file(
        path,
        spec.url,
        remote_content_length=remote_content_length,
        remote_etag=remote_etag,
    )
    if evidence.size_bytes != entry["size_bytes"] or evidence.sha256 != entry["sha256"]:
        return None
    if not _evidence_matches_spec(evidence, spec):
        return None

    return evidence


def _remote_metadata(spec: DownloadSpec, timeout_seconds: float) -> dict[str, Any]:
    """Read stable HTTP identity metadata when the server provides it."""

    request = urllib.request.Request(spec.url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            content_length = response.headers.get("Content-Length")
            etag = response.headers.get("ETag")
            repository_commit = response.headers.get("X-Repo-Commit")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as error:
        # Metadata is only a cheap preflight; the download and hash checks
        # still run, so an unreachable HEAD is logged, not fatal.
        logger.info("HEAD %s unavailable: %s", spec.filename, error)
        return {
            "content_length": None,
            "etag": None,
            "repository_commit": None,
        }

    try:
        parsed_content_length = int(content_length) if content_length is not None else None
    except ValueError:
        parsed_content_length = None

    return {
        "content_length": parsed_content_length,
        "etag": etag,
        "repository_commit": repository_commit,
    }


def _evidence_matches_spec(evidence: FileEvidence, spec: DownloadSpec) -> bool:
    """Require measured local content to match official repository identity."""

    if evidence.size_bytes != spec.expected_size_bytes:
        return False
    if spec.expected_sha256 is not None and evidence.sha256 != spec.expected_sha256:
        return False
    if (
        spec.expected_git_blob_sha1 is not None
        and evidence.git_blob_sha1 != spec.expected_git_blob_sha1
    ):
        return False

    return True


def _download_to_temporary(
    spec: DownloadSpec,
    checkpoint_dir: Path,
    settings: CheckpointSettings,
    progress_callback: Callable[[str], None] | None,
    remote_content_length: int | None,
    remote_etag: str | None,
) -> FileEvidence:
    """Stream one URL to a same-directory temporary file and atomically replace it."""

    target_path = checkpoint_dir / spec.filename
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timeout_seconds = settings.request_timeout_seconds
    block_size = settings.stream_block_bytes
    download_attempts = settings.download_attempts

    last_network_error: Exception | None = None
    for attempt in range(1, download_attempts + 1):
        if attempt > 1:
            time.sleep(settings.retry_backoff_seconds * (attempt - 1))
        digest = hashlib.sha256()
        size_bytes = 0
        response_content_length: str | None = None

        # Close the named temporary handle before networking. Windows will not
        # reliably unlink an open named file after a socket exception.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=checkpoint_dir,
            prefix=f"{spec.filename}.",
            suffix=".part",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        request = urllib.request.Request(
            spec.url,
            headers={"User-Agent": "javanese-exp-v2-checkpoint-downloader"},
        )
        # One progress line per whole percent keeps a 2.5 GB transfer to about
        # 100 lines instead of one line per streamed block.
        last_reported_percent = -1
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                response_content_length = response.headers.get("Content-Length")
                with temporary_path.open("wb") as output_file:
                    while block := response.read(block_size):
                        output_file.write(block)
                        digest.update(block)
                        size_bytes += len(block)
                        percent = min(100, size_bytes * 100 // spec.expected_size_bytes)
                        if progress_callback is not None and percent > last_reported_percent:
                            last_reported_percent = percent
                            progress_callback(
                                f"{spec.filename}: {size_bytes} / {spec.expected_size_bytes} bytes, "
                                f"Progress: {percent} percent "
                                f"(attempt {attempt} / {download_attempts})"
                            )
                    output_file.flush()
                    os.fsync(output_file.fileno())
        except (urllib.error.URLError, OSError) as error:
            temporary_path.unlink(missing_ok=True)
            last_network_error = error
            logger.warning(
                "%s: attempt %d/%d failed: %s",
                spec.filename,
                attempt,
                download_attempts,
                error,
            )
            if progress_callback is not None:
                progress_callback(
                    f"{spec.filename}: attempt {attempt}/{download_attempts} failed: {error}"
                )
            continue
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        break
    else:
        if last_network_error is None:
            raise RuntimeError(f"Download failed without an error for {spec.filename}")
        raise last_network_error

    if remote_content_length is not None and size_bytes != remote_content_length:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded size mismatch for {spec.filename}: "
            f"expected {remote_content_length}, got {size_bytes}"
        )
    if response_content_length is not None and int(response_content_length) != size_bytes:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"HTTP Content-Length mismatch for {spec.filename}: "
            f"header {response_content_length}, got {size_bytes}"
        )

    evidence = FileEvidence(
        filename=spec.filename,
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
        url=spec.url,
        remote_content_length=remote_content_length,
        remote_etag=remote_etag,
        git_blob_sha1=None,
    )
    # Git blob identity includes a size-prefixed header and therefore requires
    # one bounded second pass over the temporary file. LFS files use SHA-256 and
    # are already proven by the streaming digest above.
    if spec.expected_git_blob_sha1 is not None:
        measured_temporary = measure_file(
            temporary_path,
            spec.url,
            block_size=block_size,
            remote_content_length=remote_content_length,
            remote_etag=remote_etag,
        )
        evidence = measured_temporary

    if not _evidence_matches_spec(evidence, spec):
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded content identity mismatch for {spec.filename}"
        )

    # Atomic replacement happens only after the complete temporary payload has
    # passed remote and response size validation. A failed transfer therefore
    # cannot destroy the preceding valid target file.
    temporary_path.replace(target_path)

    return evidence


def ensure_checkpoint_files(
    checkpoint_dir: Path,
    settings: CheckpointSettings,
    specifications: tuple[DownloadSpec, ...] = CHECKPOINT_DOWNLOADS,
    progress_callback: Callable[[str], None] | None = None,
) -> list[DownloadResult]:
    """
    Ensure every requested checkpoint artifact exists and matches its manifest.

    Existing matching files are preserved. Only missing or mismatching files are
    fetched. The manifest is updated after each successful file, so an interrupted
    multi-file download can resume without repeating completed files.

    Args:
        checkpoint_dir: Destination directory under ``weights``.
        settings: Timeout, retry, backoff, and block size from config.toml.
        specifications: Ordered remote artifact specifications.
        progress_callback: Optional callback for plain-text progress messages.

    Returns:
        One result for each specification, with action ``reused`` or ``downloaded``.

    Raises:
        urllib.error.URLError: When a required download cannot be fetched.
        ValueError: When the manifest is malformed or a downloaded file is empty.
    """

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(checkpoint_dir)
    results: list[DownloadResult] = []

    for spec in specifications:
        target_path = checkpoint_dir / spec.filename
        existing_entry = manifest["files"].get(spec.filename)
        remote_metadata = _remote_metadata(spec, settings.request_timeout_seconds)
        remote_content_length = remote_metadata["content_length"]
        remote_etag = remote_metadata["etag"]
        if (
            remote_content_length is not None
            and remote_content_length != spec.expected_size_bytes
        ):
            raise ValueError(
                f"Remote size changed for {spec.filename}: expected "
                f"{spec.expected_size_bytes}, got {remote_content_length}"
            )
        evidence = _manifest_entry_matches(
            target_path,
            spec,
            existing_entry,
            remote_content_length,
            remote_etag,
        )

        if evidence is not None:
            action = "reused"
        else:
            # Existing files from before the manifest feature can be retained
            # when the server proves their byte length. Their SHA-256 is measured
            # now and becomes the local integrity baseline for future runs.
            if (
                target_path.is_file()
                and remote_content_length is not None
                and target_path.stat().st_size == remote_content_length
            ):
                evidence = measure_file(
                    target_path,
                    spec.url,
                    remote_content_length=remote_content_length,
                    remote_etag=remote_etag,
                )
                if _evidence_matches_spec(evidence, spec):
                    action = "bootstrapped"
                else:
                    evidence = _download_to_temporary(
                        spec,
                        checkpoint_dir,
                        settings,
                        progress_callback,
                        remote_content_length,
                        remote_etag,
                    )
                    action = "downloaded"
            else:
                if progress_callback is not None:
                    progress_callback(
                        f"{spec.filename}: downloading; "
                        f"remote_content_length={remote_content_length}"
                    )
                evidence = _download_to_temporary(
                    spec,
                    checkpoint_dir,
                    settings,
                    progress_callback,
                    remote_content_length,
                    remote_etag,
                )
                if evidence.size_bytes == 0:
                    raise ValueError(f"Downloaded file is empty: {target_path}")
                action = "downloaded"

        manifest["files"][spec.filename] = {
            "url": evidence.url,
            "size_bytes": evidence.size_bytes,
            "sha256": evidence.sha256,
            "remote_content_length": evidence.remote_content_length,
            "remote_etag": evidence.remote_etag,
        }
        write_manifest(checkpoint_dir, manifest)
        results.append(
            DownloadResult(
                filename=evidence.filename,
                action=action,
                path=str(target_path),
                size_bytes=evidence.size_bytes,
                sha256=evidence.sha256,
                url=evidence.url,
            )
        )

    return results
