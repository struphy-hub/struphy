"""Build and vendor a static site; internet is needed only during this step.

First run npm ci --prefix browser, then python browser/prepare.py.
"""

import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from build import build

HERE = Path(__file__).resolve().parent
SITE = HERE / "site"
VENDOR = SITE / "vendor"


def download(url, target, sha256):
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == sha256:
        return
    print(f"Downloading {target.name}", flush=True)
    for attempt in range(3):
        try:
            with urlopen(url, timeout=60) as response:
                data = response.read()
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"Checksum mismatch: {url}")
    target.write_bytes(data)


def main():
    runtime = HERE / "node_modules/pyodide"
    if not runtime.exists():
        raise SystemExit("First run npm ci --prefix browser")
    VENDOR.mkdir(parents=True, exist_ok=True)
    for source in runtime.iterdir():
        if source.suffix in (".js", ".mjs", ".wasm", ".zip", ".json"):
            shutil.copy2(source, VENDOR / source.name)
    version = json.loads((runtime / "package.json").read_text())["version"]
    lock = json.loads((runtime / "pyodide-lock.json").read_text())["packages"]
    needed = set()

    def include(name):
        if name not in needed:
            needed.add(name)
            for dependency in lock[name]["depends"]:
                include(dependency)

    for name in ("numpy", "scipy", "micropip"):
        include(name)
    downloads = []
    for name in sorted(needed):
        package = lock[name]
        filename = package["file_name"]
        downloads.append(
            (f"https://cdn.jsdelivr.net/pyodide/v{version}/full/{filename}", VENDOR / filename, package["sha256"])
        )
    extras = []
    for name, pinned in (("cunumpy", "0.1.4"), ("array-api-compat", "1.15.0")):
        with urlopen(f"https://pypi.org/pypi/{name}/{pinned}/json", timeout=60) as response:
            metadata = json.load(response)
        wheel = next(p for p in metadata["urls"] if p["filename"].endswith("py3-none-any.whl"))
        downloads.append((wheel["url"], VENDOR / wheel["filename"], wheel["digests"]["sha256"]))
        extras.append("vendor/" + wheel["filename"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: download(*item), downloads))
    wheel = build(HERE / "dist")
    shutil.copy2(wheel, SITE / wheel.name)
    (SITE / "wheel.json").write_text(json.dumps({"wheel": wheel.name, "dependencies": extras, "pyodide": version}))
    size = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"Static site ready: {SITE} ({size / 1024**2:.1f} MiB)")
    print("Serve: python -m http.server 8765 --directory browser/site")


if __name__ == "__main__":
    main()
