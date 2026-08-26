#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download every dataset used in the paper and write it into `datasets/`.

The repository does not have to carry the data. This script fetches each
dataset from its public repository (PMLB, OpenML, UCI), applies the
transformation documented in `datasets/manifest.json`, restores the row order
and the column formatting of the files used in the experiments, and writes the
result into `datasets/`.

    python scripts/fetch_datasets.py

The twelve files so produced are **byte-identical** to the ones used for
Table 7 and for the benchmarks: every one of them is verified against the
MD5/SHA-256 digests of the manifest before the script reports success, and the
script exits with a non-zero status if any file fails. Nothing is written
unless every requested dataset has been rebuilt and verified, so a failed run
never leaves `datasets/` in a half-updated state.

This matters because the row order is part of the experimental setup: the
notebook and the benchmark harness split the data with
``train_test_split(..., shuffle=True, stratify=y, random_state=42)``, whose
outcome is seeded but depends on the order of the input rows. Reordering the
rows changes the accuracies of Table 7. Ten of the eleven CSV files simply
keep the order of their public source; `car` is a stable partition of it,
recorded in the manifest.

Options
-------
--outdir DIR        write elsewhere than `datasets/` (the digests are still
                    checked)
--datasets A B ...  restrict to these dataset names
--skip-skin         do not fetch Skin Segmentation (245057 rows, ~3 MB
                    archive); it is only needed by `run_benchmarks.py B`
--only-skin         fetch Skin Segmentation and nothing else
--keep-existing     leave files that are already present and already correct
--force             re-download even the files that are already correct

Requirements
------------
Network access, `pandas`, and `scikit-learn` for the two OpenML sources
(`pip install -r requirements.txt` covers both).

Sources and licences are documented in `datasets/README.md`; the data are not
covered by the MIT licence of this repository.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
sys.path.insert(0, str(DATASETS))

import prepare_datasets as P  # noqa: E402  (needs the path above)

USER_AGENT = "pgm-complexity-repo/1.0 (dataset download script)"


def write(target: Path, raw: bytes) -> None:
    """Write `raw` to `target`, clearing a read-only bit if one is in the way."""
    try:
        target.write_bytes(raw)
    except PermissionError:
        if not target.exists():
            raise
        target.chmod(target.stat().st_mode | 0o200)
        target.write_bytes(raw)


# --------------------------------------------------------------------------
# Skin Segmentation
# --------------------------------------------------------------------------
def fetch_skin(spec: dict, outdir: Path, force: bool, keep: bool) -> bool:
    """Download Skin_NonSkin.txt from UCI. Returns True on success."""
    target = outdir / spec["file"]
    expected = spec["n_rows"]

    def matches(raw: bytes) -> bool:
        found = P.digests(raw)
        return found["md5"] == spec["md5"] and found["sha256"] == spec["sha256"]

    if target.exists() and not force:
        if matches(target.read_bytes()):
            print(f"{'skin_segmentation':<20}{expected:>7}{spec['n_features']:>6}"
                  f"{spec['n_classes']:>5}  already correct")
            return True
        if keep:
            print(f"{'skin_segmentation':<20}{'':>18}  PRESENT BUT DIFFERENT "
                  f"(--keep-existing)")
            return False

    url = spec["source"]["url"]
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        member = spec["file"] if spec["file"] in names else names[0]
        data = archive.read(member)

    rows = sum(1 for line in io.BytesIO(data) if line.strip())
    if not matches(data):
        print(f"{'skin_segmentation':<20}{rows:>7}{'':>11}  DIGEST MISMATCH "
              f"(expected {expected} rows, got {rows})")
        return False

    write(target, data)
    print(f"{'skin_segmentation':<20}{rows:>7}{spec['n_features']:>6}"
          f"{spec['n_classes']:>5}  downloaded, digests OK")
    return True


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=None,
                        help="destination directory (default: datasets/)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="restrict to these dataset names")
    parser.add_argument("--skip-skin", action="store_true",
                        help="do not fetch Skin Segmentation")
    parser.add_argument("--only-skin", action="store_true",
                        help="fetch Skin Segmentation and nothing else")
    parser.add_argument("--keep-existing", action="store_true",
                        help="leave the files that are already correct")
    parser.add_argument("--force", action="store_true",
                        help="re-download the files that are already correct")
    args = parser.parse_args()

    manifest = P.load_manifest()
    outdir = Path(args.outdir).resolve() if args.outdir else DATASETS
    outdir.mkdir(parents=True, exist_ok=True)

    names = [] if args.only_skin else (args.datasets or list(manifest["datasets"]))
    unknown = [n for n in names if n not in manifest["datasets"]]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}; "
                     f"available: {', '.join(manifest['datasets'])}")

    print(f"{'dataset':<20}{'rows':>7}{'feat':>6}{'cls':>5}  status")
    print("-" * 60)

    built: dict[Path, bytes] = {}
    failures = 0

    for name in names:
        spec = manifest["datasets"][name]
        target = outdir / spec["file"]

        if target.exists() and not args.force:
            found = P.file_digests(target)
            if found["md5"] == spec["md5"] and found["sha256"] == spec["sha256"]:
                print(f"{name:<20}{spec['n_rows']:>7}{spec['n_features']:>6}"
                      f"{spec['n_classes']:>5}  already correct")
                continue
            if args.keep_existing:
                print(f"{name:<20}{'':>18}  PRESENT BUT DIFFERENT "
                      f"(--keep-existing)")
                failures += 1
                continue

        try:
            raw = P.materialize(name, spec)
        except Exception as exc:  # noqa: BLE001 - report every failure
            print(f"{name:<20}{'':>18}  FAILED: {exc}")
            failures += 1
            continue

        found = P.digests(raw)
        if found["md5"] != spec["md5"] or found["sha256"] != spec["sha256"]:
            print(f"{name:<20}{'':>18}  DIGEST MISMATCH "
                  f"(rebuilt md5 {found['md5'][:12]}...)")
            failures += 1
            continue

        built[target] = raw
        print(f"{name:<20}{spec['n_rows']:>7}{spec['n_features']:>6}"
              f"{spec['n_classes']:>5}  rebuilt, digests OK")

    if failures:
        print("-" * 60)
        print(f"{failures} dataset(s) could not be reproduced; nothing was "
              f"written to {outdir}.")
        print("The sources may have been re-released, or the network may have "
              "truncated a download; re-run, and see datasets/README.md for "
              "the documented transformations.")
        return 1

    for target, raw in built.items():
        write(target, raw)

    if not args.skip_skin:
        skin = manifest["not_redistributed"]["skin_segmentation"]
        try:
            if not fetch_skin(skin, outdir, args.force, args.keep_existing):
                failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"{'skin_segmentation':<20}{'':>18}  FAILED: {exc}")
            failures += 1

    print("-" * 60)
    if built:
        print(f"{len(built)} file(s) written to {outdir}.")
    if failures:
        print("Skin Segmentation could not be fetched; the eleven small "
              "datasets are in place and Component A of the harness can run.")
        return 1
    print("Every dataset is in place and matches the manifest byte for byte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
