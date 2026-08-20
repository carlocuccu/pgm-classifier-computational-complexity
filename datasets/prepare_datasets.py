#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check or rebuild the dataset files of this repository.

The eleven CSV files shipped in this directory are the exact inputs of the
experiments reported in the paper. Each one is a redistributed copy, or a
documented derivation, of a public dataset; `manifest.json` records the source,
the licence and the transformation of every file.

Two modes are available.

``check`` (default)
    Recompute the digests of the shipped CSV files and compare them with
    `manifest.json`. This is the check to run before reproducing any result.

        python datasets/prepare_datasets.py check

``rebuild``
    Download each source from its public repository, re-apply the documented
    transformation and compare the outcome with the shipped file. The
    comparison is made on a row-order-independent digest, because the row
    order of the shipped files is not part of the transformation (it is,
    however, part of the experimental setup: the stratified split of the
    notebook and of the benchmark harness is seeded but order-dependent, so
    reproducing the reported numbers requires the shipped files).

        python datasets/prepare_datasets.py rebuild --outdir /tmp/rebuilt

    Requires network access, `pandas`, and `scikit-learn` for the two OpenML
    sources.

The transformations are, in all cases, a composition of three operations:

1. *label-rank encoding* -- a nominal column is replaced by the rank of its
   value in the ascending lexicographic order of that column's distinct
   labels (`car` only);
2. *row removal* -- rows that cannot be amplitude-encoded, i.e. whose feature
   vector is identically zero (`analcatdata_dmft`, `car`);
3. *class relabelling* -- the class column is mapped onto the consecutive
   integers 0..l-1, preserving the order of the original labels (all files;
   a no-op where the source is already 0-based).

plus, for `cleveland-nominal` only, the restoration of the 1-based coding of
the original Cleveland database on the columns `cp` and `slope`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"

USER_AGENT = "pgm-complexity-repo/1.0 (dataset preparation script)"


# --------------------------------------------------------------------------
# digests
# --------------------------------------------------------------------------
def file_digests(path: Path) -> dict:
    raw = path.read_bytes()
    return {"md5": hashlib.md5(raw).hexdigest(),
            "sha256": hashlib.sha256(raw).hexdigest()}


def content_digest(frame) -> str:
    """Row-order-independent digest of a numeric table.

    Each value is formatted with ``%.10g``, the values of a row are joined by
    tabs, the rows are sorted lexicographically and the resulting text is
    hashed with SHA-256.
    """
    rows = ["\t".join(format(float(v), ".10g") for v in row)
            for row in frame.to_numpy()]
    rows.sort()
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------
def _get(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_pmlb(url: str):
    """Read a PMLB `.tsv.gz` release into a data frame."""
    import pandas as pd

    return pd.read_csv(io.BytesIO(gzip.decompress(_get(url))), sep="\t")


def fetch_openml(data_id: int):
    """Read an OpenML release into a data frame of raw (string) labels."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    frame = bunch.frame.copy()
    for column in frame.columns:
        frame[column] = frame[column].astype(str)
    return frame


# --------------------------------------------------------------------------
# transformations
# --------------------------------------------------------------------------
def label_rank_encode(series):
    """Rank of each value in the ascending lexicographic order of the labels."""
    labels = sorted(series.unique())
    codes = {label: rank for rank, label in enumerate(labels)}
    return series.map(codes)


def relabel_consecutive(series):
    """Map the class labels onto 0..l-1, preserving their order."""
    labels = sorted(series.unique())
    return series.map({label: index for index, label in enumerate(labels)})


def drop_zero_feature_rows(frame):
    """Drop the rows whose feature vector (all columns but the last) is zero."""
    features = frame.iloc[:, :-1]
    return frame.loc[(features != 0).any(axis=1)].reset_index(drop=True)


def build(name: str, spec: dict):
    """Rebuild one dataset from its public source. Returns a data frame."""
    import pandas as pd

    source = spec["source"]

    if source["repository"] == "PMLB":
        frame = fetch_pmlb(source["url"])
        # PMLB stores the class in a column named `target`; move it last.
        target = "target" if "target" in frame.columns else frame.columns[-1]
        features = [c for c in frame.columns if c != target]
        frame = frame[features + [target]]

    elif source["repository"] == "OpenML":
        frame = fetch_openml(source["data_id"])
        target = "class" if "class" in frame.columns else frame.columns[-1]
        features = [c for c in frame.columns if c != target]
        frame = frame[features + [target]]

        if name == "car":
            for column in frame.columns:
                frame[column] = label_rank_encode(frame[column])
        elif name == "cleveland-nominal":
            expected = spec["columns"]
            if list(frame.columns) != expected:
                raise RuntimeError(
                    f"{name}: unexpected column order {list(frame.columns)}, "
                    f"expected {expected}")
            for column in frame.columns:
                frame[column] = frame[column].astype(float).astype(int)
            for column in spec["shift_plus_one"]:
                frame[column] = frame[column] + 1
        else:
            raise RuntimeError(f"{name}: no OpenML rule defined")

    else:
        raise RuntimeError(f"{name}: unknown repository {source['repository']!r}")

    frame = frame.astype(float)

    if spec["rule"] in ("drop_zero_feature_rows", "label_rank_encode_and_drop_rows"):
        frame = drop_zero_feature_rows(frame)

    frame.iloc[:, -1] = relabel_consecutive(frame.iloc[:, -1])

    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def cmd_check(args) -> int:
    manifest = load_manifest()
    failures = 0
    print(f"{'dataset':<20}{'rows':>7}{'feat':>6}{'cls':>5}  digest")
    print("-" * 58)
    for name, spec in manifest["datasets"].items():
        path = HERE / spec["file"]
        if not path.exists():
            print(f"{name:<20}{'':>18}  MISSING")
            failures += 1
            continue
        digests = file_digests(path)
        ok = (digests["md5"] == spec["md5"]
              and digests["sha256"] == spec["sha256"])
        failures += 0 if ok else 1
        print(f"{name:<20}{spec['n_rows']:>7}{spec['n_features']:>6}"
              f"{spec['n_classes']:>5}  {'OK' if ok else 'DIGEST MISMATCH'}")

    skin = HERE / manifest["not_redistributed"]["skin_segmentation"]["file"]
    print("-" * 58)
    print(f"Skin_NonSkin.txt: {'present' if skin.exists() else 'absent'} "
          f"(not redistributed; run download_skin_segmentation.py to fetch it)")

    if failures:
        print(f"\n{failures} file(s) do not match the manifest.")
    else:
        print("\nAll shipped dataset files match the manifest.")
    return 1 if failures else 0


def cmd_rebuild(args) -> int:
    manifest = load_manifest()
    outdir = Path(args.outdir).resolve() if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    names = args.datasets or list(manifest["datasets"])
    failures = 0
    for name in names:
        spec = manifest["datasets"][name]
        try:
            frame = build(name, spec)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{name:<20} FETCH/BUILD FAILED: {exc}")
            failures += 1
            continue

        digest = content_digest(frame)
        ok = digest == spec["content_sha256"]
        shape_ok = (frame.shape[0] == spec["n_rows"]
                    and frame.shape[1] - 1 == spec["n_features"])
        failures += 0 if (ok and shape_ok) else 1
        status = "OK" if ok else ("SHAPE MISMATCH" if not shape_ok
                                  else "CONTENT MISMATCH")
        print(f"{name:<20}{frame.shape[0]:>7}{frame.shape[1] - 1:>6}  {status}")

        if outdir:
            target = outdir / spec["file"]
            frame.to_csv(target, header=False, index=False)

    if outdir:
        print(f"\nRebuilt files written to {outdir}")
    if failures:
        print(f"{failures} dataset(s) could not be reproduced from the source.")
    else:
        print("Every dataset was reproduced from its public source.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p_check = sub.add_parser("check", help="verify the shipped CSV files")
    p_check.set_defaults(func=cmd_check)

    p_rebuild = sub.add_parser(
        "rebuild", help="rebuild the CSV files from their public sources")
    p_rebuild.add_argument("--outdir", default=None,
                           help="write the rebuilt files here (default: only "
                                "compare, write nothing)")
    p_rebuild.add_argument("--datasets", nargs="+", default=None,
                           help="restrict to these dataset names")
    p_rebuild.set_defaults(func=cmd_rebuild)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        args = parser.parse_args(["check"])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
