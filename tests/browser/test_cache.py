"""Artifact cache: path safety, hits, and LRU eviction."""

import os
import time

import pytest

from tsadar_browser.cache import ArtifactCache, UnsafeArtifactPath, sanitize_artifact_path


@pytest.mark.parametrize(
    "path",
    [
        "../../etc/passwd",
        "binary/../../../etc/passwd",
        "..",
        "",
        "   ",
        "C:/windows/system32",
        "..\\..\\windows",
    ],
)
def test_sanitize_rejects_traversal(path):
    with pytest.raises(UnsafeArtifactPath):
        sanitize_artifact_path(path)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("plots/fit_and_data.png", "plots/fit_and_data.png"),
        ("/binary/ele_fit_and_data.nc", "binary/ele_fit_and_data.nc"),
        ("./csv/learned_parameters.csv", "csv/learned_parameters.csv"),
        ("a//b///c.txt", "a/b/c.txt"),
        # An absolute-looking path is relativized, not rejected: it cannot
        # escape the cache root once the leading slash is gone.
        ("/etc/passwd", "etc/passwd"),
    ],
)
def test_sanitize_normalizes(path, expected):
    assert sanitize_artifact_path(path) == expected


def test_relativized_absolute_path_stays_inside_the_cache_root(cache):
    resolved = cache.path_for("run-1", "/etc/passwd")
    assert resolved.is_relative_to(cache.root.resolve())


def test_path_for_rejects_multi_segment_run_id(cache):
    with pytest.raises(UnsafeArtifactPath):
        cache.path_for("run/../other", "plots/a.png")


def test_fetch_caches_then_hits(cache):
    calls = []

    def downloader(scratch):
        calls.append(scratch)
        target = scratch / "a.png"
        target.write_bytes(b"payload")
        return target

    first = cache.fetch("run-1", "plots/a.png", downloader, cacheable=True)
    assert first.read_bytes() == b"payload"
    assert len(calls) == 1

    second = cache.fetch("run-1", "plots/a.png", downloader, cacheable=True)
    assert second == first
    assert len(calls) == 1, "a cached artifact must not be re-downloaded"


def test_fetch_does_not_cache_non_terminal_runs(cache):
    """A running run may still rewrite its artifacts, so every read re-fetches."""
    calls = []

    def downloader(scratch):
        calls.append(1)
        target = scratch / "a.png"
        target.write_bytes(b"v%d" % len(calls))
        return target

    cache.fetch("run-1", "plots/a.png", downloader, cacheable=False)
    result = cache.fetch("run-1", "plots/a.png", downloader, cacheable=False)
    assert len(calls) == 2
    assert result.read_bytes() == b"v2"


def test_fetch_raises_when_downloader_produces_nothing(cache):
    with pytest.raises(FileNotFoundError):
        cache.fetch("run-1", "plots/a.png", lambda scratch: scratch / "missing.png", cacheable=True)


def test_eviction_removes_least_recently_used_first(tmp_path):
    cache = ArtifactCache(root=tmp_path / "cache", max_bytes=300)

    def write(name, payload):
        def downloader(scratch):
            target = scratch / name
            target.write_bytes(payload)
            return target

        return cache.fetch("run-1", name, downloader, cacheable=True)

    old = write("old.bin", b"x" * 200)
    # Age the first entry so mtime ordering is unambiguous.
    past = time.time() - 600
    os.utime(old, (past, past))

    new = write("new.bin", b"y" * 200)

    assert not old.exists(), "the oldest entry should have been evicted"
    assert new.exists()

    entries, total = cache.stats()
    assert entries == 1
    assert total <= 300


def test_stats_on_missing_root(tmp_path):
    cache = ArtifactCache(root=tmp_path / "absent", max_bytes=100)
    assert cache.stats() == (0, 0)
    assert cache.evict_if_needed() == 0
