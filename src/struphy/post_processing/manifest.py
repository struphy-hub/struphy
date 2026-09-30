"""Manifest fingerprints, processing option comparison and the processing lock."""

import hashlib
import json
import os
import time
from contextlib import contextmanager

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

MANIFEST_SCHEMA_VERSION = 1
LOCK_NAME = ".post_processing.lock"


def source_fingerprint(path_out: str) -> str:
    """Fingerprint the raw run files that determine post-processing products."""
    digest = hashlib.sha256()
    for name in ("run_metadata.json", "data/data_proc0.hdf5"):
        path = os.path.join(path_out, name)
        if not os.path.exists(path):
            continue
        stat = os.stat(path)
        digest.update(name.encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        if name != "data/data_proc0.hdf5":
            with open(path, "rb") as stream:
                digest.update(stream.read())
    return digest.hexdigest()


def normalize_options(**options) -> dict:
    """JSON-comparable processing options, as stored in the manifest."""
    celldivide = options.get("celldivide")
    if celldivide is not None:
        options["celldivide"] = [int(celldivide)] * 3 if isinstance(celldivide, int) else [int(c) for c in celldivide]
    return options


def is_processed(path_out: str, options: dict | None = None) -> bool:
    """Whether ``path_out`` holds complete post-processing of its current raw output.

    With ``options``, the stored processing options must match as well, so a request for
    different products (e.g. ``physical=True``) is never answered with stale ones.
    """
    path = os.path.join(path_out, "post_processing", "manifest.json")
    try:
        with open(path) as stream:
            manifest = json.load(stream)
    except (OSError, ValueError):
        return False
    return (
        manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION
        and manifest.get("status") == "complete"
        and manifest.get("source_fingerprint") == source_fingerprint(path_out)
        and (options is None or manifest.get("options") == normalize_options(**options))
    )


@contextmanager
def processing_lock(path_out: str, *, poll: float = 0.2):
    """Hold the right to post-process ``path_out``, waiting while another process holds it.

    Separate jobs or scripts may start processing the same run at once; they take turns
    here. Within one MPI job, rank 0 holds the lock on behalf of the job, both when it
    processes serially and when every rank processes in parallel. The lock is a POSIX record lock on a file next to the
    products, which the operating system releases if its holder dies. File systems without
    such locks (some Lustre or NFS mounts) fall back to creating the file exclusively; a
    process killed while holding that leaves the file behind, and it must be removed by hand.
    """
    path = os.path.join(path_out, LOCK_NAME)
    with open(path, "a") as stream:
        if _record_lock(stream):
            try:
                yield
            finally:
                fcntl.lockf(stream, fcntl.LOCK_UN)
            return
    held = path + ".held"
    while True:
        try:
            os.close(os.open(held, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            time.sleep(poll)
    try:
        yield
    finally:
        os.remove(held)


def _record_lock(stream) -> bool:
    """Take an exclusive POSIX record lock on ``stream``, blocking; False where unsupported."""
    if fcntl is None:
        return False
    try:
        fcntl.lockf(stream, fcntl.LOCK_EX)
    except OSError:
        return False
    return True
