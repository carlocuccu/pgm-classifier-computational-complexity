"""Download and prepare every dataset used in the paper.

This is the only file needed to obtain the data: run it and `datasets/` is
filled with the twelve files that `notebooks/table7.ipynb` and
`run_benchmarks.py` read.

    python scripts/fetch_datasets.py

Each dataset is downloaded from its public repository, transformed as
documented below, written with the row order and the column formatting of the
files actually used in the experiments, and verified against the MD5/SHA-256
digest recorded here. The result is **byte-identical** to the data behind
Table 7 and Table 8, so the reported numbers reproduce exactly. Nothing is
written unless every requested dataset has been rebuilt and verified, so a
failed run never leaves `datasets/` half-populated.

    python scripts/fetch_datasets.py --check     # verify what is on disk, offline

Options
-------
--outdir DIR        write elsewhere than `datasets/`
--datasets A B ...  restrict to these dataset names
--skip-skin         leave out Skin Segmentation (245057 rows), which only
                    `run_benchmarks.py B` needs
--only-skin         fetch Skin Segmentation and nothing else
--force             re-download the files that are already correct
--check             verify the files on disk against the digests, without
                    downloading anything
--sources           print the source, licence and transformation of every file

Requirements: network access, `pandas`, and `scikit-learn` for the OpenML
sources -- all three are in `requirements.txt`.


Sources and licences
--------------------
The code of this repository is MIT-licensed; the data are not covered by that
licence and carry the terms of their own sources.

PMLB -- Penn Machine Learning Benchmarks, MIT-licensed. `analcatdata_dmft`,
    `balance_scale`, `cloud`, `confidence`, `ecoli`, `haberman`, `iris`,
    `led7`, `new_thyroid`.
    Romano, J.D., Le, T.T., La Cava, W., Gregg, J.T., Goldberg, D.J.,
    Chakraborty, P., Ray, N.L., Himmelstein, D., Fu, W., Moore, J.H. (2021).
    PMLB v1.0: an open source dataset collection for benchmarking machine
    learning methods. Bioinformatics 38(3), 878-880.
    https://github.com/EpistasisLab/pmlb
    The releases are stored with Git LFS; they are fetched here through
    `media.githubusercontent.com`, which serves the file itself rather than
    the LFS pointer that the plain raw URL returns.

OpenML -- CC BY 4.0. `car` (id 21) and `cleveland-nominal` (id 40711); also
    the mirror of Skin Segmentation (id 1502) used when UCI is unreachable.
    Vanschoren, J., van Rijn, J.N., Bischl, B., Torgo, L. (2013). OpenML:
    networked science in machine learning. SIGKDD Explorations 15(2), 49-60.

UCI Machine Learning Repository -- CC BY 4.0. Skin Segmentation (id 229).
    Bhatt, R., Dhall, A. (2012). Skin Segmentation. UCI Machine Learning
    Repository. https://doi.org/10.24432/C5T30C

Several of these datasets originate from UCI and are redistributed by PMLB and
OpenML; please cite the original sources listed on the corresponding pages.


Transformations
---------------
Every CSV file is header-less, comma-separated, CRLF-terminated, with the
integer class label in the last column taking the consecutive values 0..l-1.
The transformations are compositions of four elementary operations.

*Label-rank encoding* -- a nominal column is replaced by the rank of its value
in the ascending lexicographic order of that column's distinct labels. Applies
to `car` only, whose attributes are all nominal: `buying` becomes high->0,
low->1, med->2, vhigh->3, and the class acc->0, good->1, unacc->2, vgood->3.

*Removal of zero-feature rows* -- the rows whose feature vector is identically
zero are dropped: ten rows in `analcatdata_dmft` (797 -> 787), a single
repeated feature vector carrying several different class labels, and one row
in `car` (1728 -> 1727). Neither file keeps a zero-feature row afterwards.
The filter is applied to these two datasets only: `led7`, for instance, keeps
the all-zero patterns of its source.

*Class relabelling* -- the class column is mapped onto the consecutive integers
0..l-1, preserving the order of the original labels. This turns the PMLB labels
{0,1,4,5,7} of `ecoli` into {0,1,2,3,4}, {1,2} of `haberman` into {0,1} and
{1,2,3} of `new_thyroid` into {0,1,2}. Where the source labels are already
0..l-1 the operation is the identity.

*Restoration of the 1-based nominal coding* (`cleveland-nominal` only) -- the
columns are, in order, sex, cp, fbs, restecg, exang, slope, thal, class. The
columns `cp` and `slope` carry the coding of the original Cleveland database,
cp in 1..4 and slope in 1..3; some OpenML clients return every nominal
attribute as a 0-based category index, in which case those two columns are
shifted by one here.


Row order
---------
The row order is not a detail of presentation. The notebook and the harness
split the data with `train_test_split(..., shuffle=True, stratify=y,
random_state=42)`, whose outcome is seeded but depends on the order of the
input rows: the same rows in a different order give different accuracies --
0.900 instead of 0.967 on `iris`, 0.818 instead of 0.909 on `ecoli`.

Ten of the eleven CSV files simply keep the order of their public release
(`"row_order": "source"`); the rows removed by the zero-feature filter drop
out, leaving the remaining ones in place. `car.csv` is a *stable partition* of
that order: the 1353 rows whose source index is not listed in `CAR_TAIL` come
first, then the 374 rows listed there, each block in the order of the source.

`column_types` records, per column, whether the file writes it as an integer
or as a decimal. It affects nothing but the bytes, and is what allows the
digests below to be checked on the file rather than on the parsed values.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import sys
import urllib.error
import urllib.request
import warnings
import zipfile
import zipfile as _zipfile  # noqa: F401  (kept for readability below)
from pathlib import Path

from pgm_complexity.data.specs import CAR_TAIL, SKIN

USER_AGENT = "pgm-complexity-repo/1.0 (dataset preparation script)"
TIMEOUT = 300


# ==========================================================================
# digests
# ==========================================================================
def digests(raw: bytes) -> dict:
    return {
        "md5": hashlib.md5(raw).hexdigest(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def matches(raw: bytes, spec: dict) -> bool:
    found = digests(raw)
    return found["md5"] == spec["md5"] and found["sha256"] == spec["sha256"]


def content_digest(frame) -> str:
    """Row-order-independent digest: the rows, sorted, hashed.

    Each value is formatted with '%.10g', the values of a row are joined by
    tabs, the rows are sorted lexicographically and the text is hashed. It
    identifies the content of a file up to a permutation of its rows, and so
    separates a transformation error from an ordering error.
    """
    rows = [
        "\t".join(format(float(v), ".10g") for v in row) for row in frame.to_numpy()
    ]
    rows.sort()
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


# ==========================================================================
# download
# ==========================================================================
def get(url: str, timeout: int = TIMEOUT) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def lfs_url(url: str) -> str:
    """Direct Git-LFS media URL for a `github.com/<o>/<r>/raw/<ref>/<path>` link."""
    if "github.com/" not in url:
        return url
    tail = url.split("github.com/", 1)[1].replace("/raw/", "/", 1)
    return "https://media.githubusercontent.com/media/" + tail


def fetch_pmlb(url: str):
    """Read a PMLB `.tsv.gz` release into a data frame."""
    import pandas as pd

    errors = []
    for candidate in (lfs_url(url), url):
        try:
            payload = get(candidate)
        except (urllib.error.URLError, OSError) as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        if payload.startswith(b"version https://git-lfs"):
            errors.append(f"{candidate}: Git-LFS pointer instead of the file")
            continue
        return pd.read_csv(io.BytesIO(gzip.decompress(payload)), sep="\t")
    raise RuntimeError("download failed; " + "; ".join(errors))


def fetch_openml(data_id: int, as_string: bool = True):
    """Read an OpenML release into a data frame."""
    from sklearn.datasets import fetch_openml as _fetch

    with warnings.catch_warnings():
        # OpenML flags version 1 of `car` as inactive; it is the release the
        # paper uses, and the one the digests below correspond to.
        warnings.filterwarnings("ignore", message=".*is inactive.*")
        bunch = _fetch(data_id=data_id, as_frame=True, parser="auto")
    frame = bunch.frame.copy()
    if as_string:
        for column in frame.columns:
            frame[column] = frame[column].astype(str)
    return frame


# ==========================================================================
# transformations
# ==========================================================================
def label_rank_encode(series):
    """Rank of each value in the ascending lexicographic order of the labels."""
    labels = sorted(series.unique())
    return series.map({label: rank for rank, label in enumerate(labels)})


def relabel_consecutive(series):
    """Map the class labels onto 0..l-1, preserving their order."""
    labels = sorted(series.unique())
    return series.map({label: index for index, label in enumerate(labels)})


def drop_zero_feature_rows(frame):
    """Drop the rows whose feature vector (all columns but the last) is zero."""
    features = frame.iloc[:, :-1]
    return frame.loc[(features != 0).any(axis=1)].reset_index(drop=True)


def build(name: str, spec: dict):
    """Download one dataset and transform it, in the order of the source."""
    if spec["repository"] == "PMLB":
        frame = fetch_pmlb(spec["url"])
        # PMLB stores the class in a column named `target`; move it last.
        target = "target" if "target" in frame.columns else frame.columns[-1]
        frame = frame[[c for c in frame.columns if c != target] + [target]]

    elif spec["repository"] == "OpenML":
        frame = fetch_openml(spec["data_id"])
        target = "class" if "class" in frame.columns else frame.columns[-1]
        frame = frame[[c for c in frame.columns if c != target] + [target]]

        if name == "car":
            for column in frame.columns:
                frame[column] = label_rank_encode(frame[column])
        elif name == "cleveland-nominal":
            if list(frame.columns) != spec["columns"]:
                raise RuntimeError(
                    f"unexpected column order {list(frame.columns)}, "
                    f"expected {spec['columns']}"
                )
            for column in frame.columns:
                frame[column] = frame[column].astype(float).astype(int)
            for column in spec["shift_plus_one"]:
                if int(frame[column].min()) == 0:
                    frame[column] = frame[column] + 1
        else:
            raise RuntimeError(f"{name}: no OpenML rule defined")

    else:
        raise RuntimeError(f"unknown repository {spec['repository']!r}")

    frame = frame.astype(float)

    if spec["rule"] in ("drop_zero_feature_rows", "label_rank_encode_and_drop_rows"):
        frame = drop_zero_feature_rows(frame)

    frame.iloc[:, -1] = relabel_consecutive(frame.iloc[:, -1])
    return frame.reset_index(drop=True)


# ==========================================================================
# row order and rendering
# ==========================================================================
def apply_row_order(frame, spec: dict):
    """Reorder a transformed frame into the order of the file used in the paper."""
    order = spec["row_order"]
    if order == "source":
        return frame
    if order == "car_partition":
        moved = set(CAR_TAIL)
        if max(moved) >= len(frame):
            raise RuntimeError("CAR_TAIL index out of range")
        head = [i for i in range(len(frame)) if i not in moved]
        return frame.iloc[head + CAR_TAIL].reset_index(drop=True)
    raise RuntimeError(f"unsupported row_order {order!r}")


def render_csv(frame, spec: dict) -> bytes:
    """Serialize a frame exactly as the file used in the paper."""
    import pandas as pd

    types = spec["column_types"]
    if len(types) != frame.shape[1]:
        raise RuntimeError(
            f"column_types has {len(types)} entries for {frame.shape[1]} columns"
        )
    out = pd.DataFrame(frame.to_numpy(), columns=range(frame.shape[1]))
    for index, kind in enumerate(types):
        out[index] = out[index].astype(int if kind == "int" else float)
    return out.to_csv(header=False, index=False, lineterminator="\r\n").encode()


def materialize(name: str, spec: dict) -> bytes:
    """Download, transform, order and serialize one dataset."""
    frame = build(name, spec)
    if frame.shape[0] != spec["n_rows"] or frame.shape[1] - 1 != spec["n_features"]:
        raise RuntimeError(
            f"unexpected shape {frame.shape[0]}x{frame.shape[1] - 1}, "
            f"expected {spec['n_rows']}x{spec['n_features']}"
        )
    if content_digest(frame) != spec["content_sha256"]:
        raise RuntimeError(
            "the transformed content does not match the "
            "expected digest (the source may have been "
            "re-released)"
        )
    return render_csv(apply_row_order(frame, spec), spec)


def materialize_skin() -> bytes:
    """Download Skin Segmentation, from UCI or from its OpenML mirror."""
    errors = []
    try:
        payload = get(SKIN["url"], timeout=600)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            member = SKIN["file"] if SKIN["file"] in names else names[0]
            return archive.read(member)
    except Exception as exc:  # noqa: BLE001 - fall back to the mirror
        errors.append(f"UCI: {exc}")

    try:
        frame = fetch_openml(SKIN["mirror_data_id"], as_string=False)
        rows = frame.to_numpy().astype(float).astype(int)
        return (
            "\r\n".join("\t".join(str(v) for v in row) for row in rows) + "\r\n"
        ).encode()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"OpenML mirror: {exc}")

    raise RuntimeError("; ".join(errors))


# ==========================================================================
# output
# ==========================================================================
def write(target: Path, raw: bytes) -> None:
    """Write `raw` to `target`, clearing a read-only bit if one is in the way."""
    try:
        target.write_bytes(raw)
    except PermissionError:
        if not target.exists():
            raise
        target.chmod(target.stat().st_mode | 0o200)
        target.write_bytes(raw)


def row(name: str, spec: dict, status: str) -> str:
    return (
        f"{name:<20}{spec['n_rows']:>7}{spec['n_features']:>6}"
        f"{spec['n_classes']:>5}  {status}"
    )


def blank_row(name: str, status: str) -> str:
    return f"{name:<20}{'':>18}  {status}"


def check_environment() -> None:
    """Fail once, clearly, if the interpreter cannot do the work.

    A CPython built without libffi has no `_ctypes`, and numpy and scipy fail
    to import on it. Without this check the same message is repeated for every
    dataset, which reads like eleven download failures rather than one broken
    interpreter.
    """
    try:
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"this interpreter cannot import the libraries the script needs: {exc}\n"
            f"  python:  {sys.executable}\n"
            f"  version: {sys.version.split()[0]}\n"
            "\nA 'No module named _ctypes' here means the interpreter was built "
            "without libffi,\nwhich numpy and scipy need. Let uv provide one "
            "instead:\n\n"
            "    rm -rf .venv && uv python install 3.12 && uv sync --group dev\n"
        ) from exc
