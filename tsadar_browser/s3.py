"""Fetch artifact bytes straight from S3, bypassing MLflow's artifact repository.

Artifacts live in ``s3://public-ergodic-continuum/<experiment_id>/<run_id>/artifacts``
and the run's own ``artifact_uri`` already names that prefix, so once the run is
in hand there is nothing MLflow can tell us that we need. Going direct removes a
tracking-server round trip (and its retry/backoff budget) from every artifact
read -- which the lineout scrubber does once per step -- and means a slow or
degraded tracking server no longer stalls plots whose bytes are sitting in S3.

The fallback to ``MlflowClient.download_artifacts`` is kept for artifact stores
that are not S3 at all: a local ``file://`` ``mlruns`` directory in development,
and the fake client the tests run against.

Credentials are ambient -- an AWS profile locally, the task role in deployment --
exactly as they were when mlflow's ``S3ArtifactRepository`` read them.
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

#: S3 error codes that mean 'this object is not there', as opposed to a transport
#: or permission failure. Mapped to FileNotFoundError so the route answers 404.
MISSING_CODES = frozenset({"404", "NoSuchKey", "NoSuchBucket", "NotFound"})


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def parse_s3_uri(artifact_uri: str | None) -> tuple[str, str] | None:
    """Split an ``s3://bucket/prefix`` URI into its parts.

    Returns ``None`` for anything that is not S3 -- a ``file://`` store, a bare
    path, or a missing URI -- which is the caller's signal to fall back to MLflow
    rather than an error.
    """
    if not artifact_uri:
        return None

    parsed = urlparse(artifact_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        return None

    return parsed.netloc, parsed.path.strip("/")


def object_key(prefix: str, artifact_path: str) -> str:
    """Join an artifact prefix and a run-relative artifact path into a key.

    ``artifact_path`` has already been through
    :func:`~tsadar_browser.cache.sanitize_artifact_path`, so it carries no ``..``
    segment that could climb out of the run's prefix.
    """
    if not prefix:
        return artifact_path
    return f"{prefix}/{artifact_path}"


class S3ArtifactReader:
    """Downloads artifact objects with boto3.

    The client is built on first use, not at construction: importing this module
    (and starting the app) must not require credentials, since ``/api/health``
    has to answer even when nothing else works.
    """

    def __init__(self, client=None):
        self._client = client
        self._lock = threading.Lock()

    def client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import boto3  # imported lazily so a missing dep is a fetch error, not an import error

                    self._client = boto3.client("s3")
        return self._client

    def download(self, artifact_uri: str, artifact_path: str, destination_dir: Path) -> Path:
        """Fetch one artifact into ``destination_dir`` and return its local path.

        Raises :class:`FileNotFoundError` when the object does not exist and
        :class:`OSError` for anything else, which is what the artifact route
        already turns into a 404 and a 502 respectively.
        """
        parsed = parse_s3_uri(artifact_uri)
        if parsed is None:
            raise ValueError(f"not an S3 artifact URI: {artifact_uri!r}")

        bucket, prefix = parsed
        location = S3Location(bucket=bucket, key=object_key(prefix, artifact_path))

        destination_dir.mkdir(parents=True, exist_ok=True)
        target = destination_dir / Path(artifact_path).name

        try:
            self.client().download_file(location.bucket, location.key, str(target))
        except Exception as exc:  # botocore's exception hierarchy is translated below
            raise self._translate(exc, location) from exc

        return target

    @staticmethod
    def _translate(exc: Exception, location: S3Location) -> Exception:
        """Turn a botocore failure into the errors the routes already handle.

        Matched on the error code rather than the exception class: botocore
        raises the same ``ClientError`` for a missing key and for access denied,
        and those are a 404 and a 502 respectively.
        """
        code = None
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            code = str(response.get("Error", {}).get("Code", "")) or None

        if code in MISSING_CODES:
            return FileNotFoundError(f"no such artifact: {location.uri}")
        if code is not None:
            # AccessDenied, SlowDown, a bad region -- upstream problems, not the
            # caller's, so the route should answer 502 rather than 404.
            return OSError(f"could not read {location.uri} ({code}): {exc}")
        if isinstance(exc, FileNotFoundError):
            return exc
        return OSError(f"could not read {location.uri}: {exc}")
