"""Manifest fingerprints and processing option comparison."""

import hashlib
import json
import os


MANIFEST_SCHEMA_VERSION = 1


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
