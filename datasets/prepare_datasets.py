#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify, or rebuild from their public sources, the dataset files of this repository.

The eleven CSV files of this directory are the exact inputs of the experiments
reported in the paper. Each one is a redistributed copy, or a documented
derivation, of a public dataset; `manifest.json` records the source, the
licence, the transformation, the row order and the column formatting of every
file, which together determine it down to the byte.

This module is both a command-line tool and the library used by
`scripts/fetch_datasets.py`, which downloads the datasets instead of taking
them from the repository.

Two commands are available.

``check`` (default)
    Recompute the digests of the CSV files present in this directory and
    compare them with `manifest.json`. This is the check to run before
    reproducing any result; it needs no network access.

        python datasets/prepare_datasets.py check

``rebuild``
    Download each source from its public repository, re-apply the documented
    transformation, restore the documented row order and column formatting,
    and compare the outcome with `manifest.json`. With ``--outdir`` the
    rebuilt files are written out; the files so produced are byte-identical to
    the ones shipped here, so they reproduce the reported numbers exactly.

        python datasets/prepare_datasets.py rebuild --outdir /tmp/rebuilt

    Requires network access, `pandas`, and `scikit-learn` for the two OpenML
    sources. To populate `datasets/` directly, use
    ``python scripts/fetch_datasets.py``, which also fetches Skin Segmentation.

The transformations are, in all cases, a composition of three operations:

1. *label-rank encoding* -- a nominal column is replaced by the rank of its
   value in the ascending lexicographic order of that column's distinct
   labels (`car` only);
2. *row removal* -- rows whose feature vector is identically zero
   (`analcatdata_dmft`, `car`);
3. *class relabelling* -- the class column is mapped onto the consecutive
   integers 0..l-1, preserving the order of the original labels (all files;
   a no-op where the source is already 0-based).

plus, for `cleveland-nominal` only, the 1-based coding of the original
Cleveland database on the columns `cp` and `slope`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"

USER_AGENT = "pgm-complexity-repo/1.0 (dataset preparation script)"


# --------------------------------------------------------------------------
# digests
# --------------------------------------------------------------------------
def file_digests(path: Path) -> dict:
    return digests(path.read_bytes())


def digests(raw: bytes) -> dict:
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
def _get(url: str, timeout: int = 180) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _lfs_url(url: str) -> str:
    """Direct Git-LFS media URL for a `github.com/<o>/<r>/raw/<ref>/<path>` link.

    PMLB stores its releases with Git LFS. The `github.com/.../raw/...` link
    recorded in the manifest is the citable one and resolves the LFS object,
    but it is a redirect and some networks block it; `media.githubusercontent`
    serves the same bytes directly.
    """
    if "github.com/" not in url:
        return url
    tail = url.split("github.com/", 1)[1].replace("/raw/", "/", 1)
    return "https://media.githubusercontent.com/media/" + tail


def fetch_pmlb(url: str):
    """Read a PMLB `.tsv.gz` release into a data frame."""
    import pandas as pd

    candidates = [_lfs_url(url), url]
    errors = []
    for candidate in candidates:
        try:
            payload = _get(candidate)
        except urllib.error.URLError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        if payload.startswith(b"version https://git-lfs"):
            errors.append(f"{candidate}: Git-LFS pointer instead of the file")
            continue
        return pd.read_csv(io.BytesIO(gzip.decompress(payload)), sep="\t")
    raise RuntimeError("could not download the source; " + "; ".join(errors))


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
    """Rebuild one dataset from its public source, in the order of the source.

    Returns a data frame of floats. The row order is the one of the public
    release; `apply_row_order` turns it into the order of the shipped file.
    """
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
            # Some OpenML clients return every nominal attribute as a 0-based
            # category index; others return the codes of the original
            # database. Shift only in the first case.
            for column in spec["shift_plus_one"]:
                if int(frame[column].min()) == 0:
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
# row order and rendering
# --------------------------------------------------------------------------
def apply_row_order(frame, spec: dict):
    """Reorder the rows of a rebuilt frame into the order of the shipped file.

    `row_order` is either the string ``"source"``, meaning that the shipped
    file keeps the order of the public release, or a ``stable_partition``
    record listing the source indices that the shipped file moves to the end
    (`car` only). Both blocks of a partition keep their relative order.
    """
    order = spec.get("row_order", "source")
    if order == "source":
        return frame
    if isinstance(order, dict) and order.get("kind") == "stable_partition":
        tail = list(order["tail"])
        moved = set(tail)
        if any(i < 0 or i >= len(frame) for i in moved):
            raise RuntimeError("row_order: index out of range")
        head = [i for i in range(len(frame)) if i not in moved]
        return frame.iloc[head + tail].reset_index(drop=True)
    raise RuntimeError(f"unsupported row_order {order!r}")


def render_csv(frame, spec: dict) -> bytes:
    """Serialize a frame exactly as the shipped CSV file.

    `column_types` records, per column, whether the shipped file writes it as
    an integer or as a decimal; the files use CRLF line endings.
    """
    import pandas as pd

    types = spec["column_types"]
    if len(types) != frame.shape[1]:
        raise RuntimeError(f"column_types has {len(types)} entries for "
                           f"{frame.shape[1]} columns")
    out = pd.DataFrame(frame.to_numpy(), columns=range(frame.shape[1]))
    for index, kind in enumerate(types):
        out[index] = out[index].astype(int if kind == "int" else float)
    return out.to_csv(header=False, index=False,
                      lineterminator="\r\n").encode()


def materialize(name: str, spec: dict) -> bytes:
    """Download, transform, order and serialize one dataset. Returns CSV bytes."""
    frame = build(name, spec)
    if frame.shape[0] != spec["n_rows"] or frame.shape[1] - 1 != spec["n_features"]:
        raise RuntimeError(
            f"unexpected shape {frame.shape[0]}x{frame.shape[1] - 1}, "
            f"expected {spec['n_rows']}x{spec['n_features']}")
    if content_digest(frame) != spec["content_sha256"]:
        raise RuntimeError("the rebuilt content does not match the manifest")
    return render_csv(apply_row_order(frame, spec), spec)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def cmd_check(args) -> int:
    manifest = load_manifest()
    directory = Path(args.directory).resolve() if args.directory else HERE
    failures = 0
    print(f"{'dataset':<20}{'rows':>7}{'feat':>6}{'cls':>5}  digest")
    print("-" * 58)
    for name, spec in manifest["datasets"].items():
        path = directory / spec["file"]
        if not path.exists():
            print(f"{name:<20}{'':>18}  MISSING")
            failures += 1
            continue
        found = file_digests(path)
        ok = (found["md5"] == spec["md5"] and found["sha256"] == spec["sha256"])
        failures += 0 if ok else 1
        print(f"{name:<20}{spec['n_rows']:>7}{spec['n_features']:>6}"
              f"{spec['n_classes']:>5}  {'OK' if ok else 'DIGEST MISMATCH'}")

    skin = directory / manifest["not_redistributed"]["skin_segmentation"]["file"]
    print("-" * 58)
    print(f"Skin_NonSkin.txt: {'present' if skin.exists() else 'absent'} "
          f"(not redistributed; run scripts/fetch_datasets.py to fetch it)")

    if failures:
        print(f"\n{failures} file(s) do not match the manifest.")
        print("Run `python scripts/fetch_datasets.py` to rebuild them from "
              "their public sources.")
    else:
        print("\nAll dataset files match the manifest.")
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
            raw = materialize(name, spec)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{name:<20} FAILED: {exc}")
            failures += 1
            continue

        found = digests(raw)
        ok = (found["md5"] == spec["md5"] and found["sha256"] == spec["sha256"])
        failures += 0 if ok else 1
        print(f"{name:<20}{spec['n_rows']:>7}{spec['n_features']:>6}  "
              f"{'OK' if ok else 'DIGEST MISMATCH'}")

        if outdir:
            (outdir / spec["file"]).write_bytes(raw)

    if outdir:
        print(f"\nRebuilt files written to {outdir}")
    if failures:
        print(f"{failures} dataset(s) could not be reproduced from the source.")
    else:
        print("Every dataset was reproduced, byte for byte, from its public source.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p_check = sub.add_parser("check", help="verify the CSV files on disk")
    p_check.add_argument("--directory", default=None,
                         help="verify the files in this directory "
                              "(default: datasets/)")
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
