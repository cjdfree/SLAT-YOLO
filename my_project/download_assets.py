# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Download versioned dataset or checkpoint archives and verify their SHA256 digests."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Download selected assets from the repository release and extract inside the project."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("datasets", "checkpoints", "all"))
    args = parser.parse_args()
    assets = json.loads((ROOT / "my_project/provenance/release_assets.json").read_text())
    cache = ROOT / "my_project/downloads"
    cache.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        if args.kind != "all" and asset["kind"] != args.kind:
            continue
        path = cache / asset["name"]
        if not path.exists():
            partial = path.with_suffix(".part")
            print(f"Downloading {asset['name']} ({asset['bytes'] / 1e6:.1f} MB)", flush=True)
            urllib.request.urlretrieve(asset["url"], partial)
            partial.replace(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != asset["sha256"]:
            raise ValueError(f"Checksum mismatch for {path}. Remove the cached archive and download it again.")
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (ROOT / member.filename).resolve()
                if ROOT not in target.parents:
                    raise ValueError(f"Unsafe archive path: {member.filename}")
            archive.extractall(ROOT)
        print(f"Verified and extracted {asset['name']}")


if __name__ == "__main__":
    main()
