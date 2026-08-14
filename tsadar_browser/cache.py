"""Disk cache for MLflow artifacts.

A run is immutable once its MLflow status is terminal, so terminal-run
artifacts are cached indefinitely and only evicted to stay under
``CACHE_MAX_GB`` (least-recently-used first). Artifacts of a still-running run
are fetched fresh every time -- they may still be rewritten.

Cache keys are ``<run_id>/<artifact_path>``, which doubles as the on-disk
layout, so the cache is inspectable with ``ls``.
"""

import logging
import os
import shutil
import tempfile
import threading
from collections import Counter
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class UnsafeArtifactPath(ValueError):
    """Raised for artifact paths that would escape the cache directory."""


def sanitize_artifact_path(artifact_path: str) -> str:
    """Validate an artifact path and return it normalized to forward slashes.

    Artifact paths arrive from the URL, so they are untrusted. A ``..`` segment
    is rejected outright -- that is the only construct that can escape the cache
    root. A leading slash is merely stripped rather than rejected: once
    relativized it lands harmlessly under the run's cache directory (and MLflow
    will 404 it), so there is nothing to defend against.
    """
    if not artifact_path or not artifact_path.strip():
        raise UnsafeArtifactPath("artifact path must not be empty")

    normalized = artifact_path.replace("\\", "/").strip("/")
    segments = [segment for segment in normalized.split("/") if segment not in ("", ".")]
    if not segments:
        raise UnsafeArtifactPath("artifact path must not be empty")

    if any(segment == ".." for segment in segments):
        raise UnsafeArtifactPath(f"artifact path must not contain '..': {artifact_path!r}")
    if ":" in normalized:
        raise UnsafeArtifactPath(f"artifact path must not contain a drive or scheme: {artifact_path!r}")

    return "/".join(segments)


class ArtifactCache:
    """LRU-evicting disk cache keyed by ``run_id`` and artifact path."""

    def __init__(self, root: Path, max_bytes: int):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self._global_lock = threading.Lock()
        # Per-key download locks, reference counted so the mapping does not grow
        # for the lifetime of the process. A lock is dropped once no thread is
        # waiting on it; the next request for that key makes a fresh one.
        self._key_locks: dict[str, threading.Lock] = {}
        self._key_waiters: Counter[str] = Counter()

    def _claim_key_lock(self, key: str) -> threading.Lock:
        with self._global_lock:
            self._key_waiters[key] += 1
            return self._key_locks.setdefault(key, threading.Lock())

    def _release_key_lock(self, key: str) -> None:
        with self._global_lock:
            self._key_waiters[key] -= 1
            if self._key_waiters[key] <= 0:
                del self._key_waiters[key]
                self._key_locks.pop(key, None)

    # -- layout ---------------------------------------------------------------

    def path_for(self, run_id: str, artifact_path: str) -> Path:
        """Absolute cache path for an artifact, with traversal guarded twice.

        The sanitizer rejects ``..`` up front; the resolved-prefix check below
        catches anything that still manages to escape (e.g. via a symlinked
        cache root).
        """
        safe_run_id = sanitize_artifact_path(run_id)
        if "/" in safe_run_id:
            raise UnsafeArtifactPath(f"run id must be a single path segment: {run_id!r}")

        candidate = (self.root / safe_run_id / sanitize_artifact_path(artifact_path)).resolve()
        root = self.root.resolve()
        if not candidate.is_relative_to(root):
            raise UnsafeArtifactPath(f"artifact path escapes the cache root: {artifact_path!r}")
        return candidate

    # -- reads ----------------------------------------------------------------

    def get(self, run_id: str, artifact_path: str) -> Path | None:
        """Return the cached file, marking it recently used, or None on a miss."""
        target = self.path_for(run_id, artifact_path)
        if not target.is_file():
            return None
        # mtime is the recency signal for eviction: atime is unreliable on
        # noatime/relatime mounts, which is the common container default.
        try:
            os.utime(target, None)
        except OSError:  # pragma: no cover - best effort recency bump
            logger.debug("could not bump mtime for %s", target)
        return target

    def fetch(
        self,
        run_id: str,
        artifact_path: str,
        downloader: Callable[[Path], Path],
        cacheable: bool = True,
    ) -> Path:
        """Return a local path for an artifact, downloading it if necessary.

        ``downloader`` is handed a scratch directory and must return the path of
        the file it wrote there. The file is moved into place only once the
        download is complete, so an interrupted fetch never leaves a truncated
        cache entry behind. When ``cacheable`` is False (non-terminal run) the
        artifact is re-downloaded every time -- it may still be rewritten.
        """
        target = self.path_for(run_id, artifact_path)

        if cacheable:
            hit = self.get(run_id, artifact_path)
            if hit is not None:
                return hit

        # One download per key: concurrent requests for the same cold artifact
        # would otherwise each pull the whole file from S3.
        key = f"{run_id}/{artifact_path}"
        key_lock = self._claim_key_lock(key)
        try:
            with key_lock:
                if cacheable:
                    hit = self.get(run_id, artifact_path)
                    if hit is not None:
                        return hit

                target.parent.mkdir(parents=True, exist_ok=True)
                # Stage inside the cache root so the final move is same-filesystem
                # and therefore atomic.
                with tempfile.TemporaryDirectory(dir=self.root) as scratch:
                    downloaded = Path(downloader(Path(scratch)))
                    if not downloaded.is_file():
                        raise FileNotFoundError(f"downloader did not produce a file for {artifact_path!r}")
                    os.replace(downloaded, target)
        finally:
            self._release_key_lock(key)

        # Enforce the cap on both paths. Non-cacheable fetches still write into
        # the cache root, so skipping eviction here let the directory grow past
        # CACHE_MAX_GB until the next cacheable fetch happened to clean up.
        # `protect` keeps the file we are about to hand to the caller from being
        # evicted out from under the response that is about to stream it.
        self.evict_if_needed(protect=target)
        return target

    # -- eviction -------------------------------------------------------------

    def _entries(self) -> list[tuple[Path, int, float]]:
        entries: list[tuple[Path, int, float]] = []
        if not self.root.exists():
            return entries
        for path in self.root.rglob("*"):
            if path.is_file():
                try:
                    stat = path.stat()
                except OSError:  # pragma: no cover - raced with eviction
                    continue
                entries.append((path, stat.st_size, stat.st_mtime))
        return entries

    def stats(self) -> tuple[int, int]:
        """Return ``(entry_count, total_bytes)``."""
        entries = self._entries()
        return len(entries), sum(size for _, size, _ in entries)

    def evict_if_needed(self, protect: Path | None = None) -> int:
        """Delete least-recently-used files until under the size cap.

        Returns the number of bytes freed. ``protect`` is never evicted, which is
        how a just-fetched artifact survives long enough to be streamed: once
        Starlette has the file open, unlinking it is harmless on POSIX because
        the descriptor keeps the data alive, but there is a window between
        :meth:`fetch` returning and the response opening the file where deleting
        it would produce a spurious 404.

        A single artifact larger than the whole cap therefore leaves the cache
        over its limit rather than evicting the file that is about to be served.
        That is logged rather than silently tolerated.
        """
        with self._global_lock:
            entries = self._entries()
            total = sum(size for _, size, _ in entries)
            if total <= self.max_bytes:
                return 0

            protected = protect.resolve() if protect is not None else None
            freed = 0
            for path, size, _ in sorted(entries, key=lambda item: item[2]):
                if protected is not None and path.resolve() == protected:
                    continue
                try:
                    path.unlink()
                except OSError:  # pragma: no cover - raced with another evictor
                    continue
                freed += size
                total -= size
                if total <= self.max_bytes:
                    break

            logger.info("evicted %d bytes from artifact cache", freed)
            if total > self.max_bytes:
                logger.warning(
                    "artifact cache still %d bytes over its %d byte cap after eviction; "
                    "a single artifact may exceed CACHE_MAX_GB",
                    total - self.max_bytes,
                    self.max_bytes,
                )
            return freed

    def clear(self) -> None:
        with self._global_lock:
            if self.root.exists():
                shutil.rmtree(self.root)
