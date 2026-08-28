#!/usr/bin/env python3
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

import argparse
import gzip
import hashlib
import io
import sys
import urllib.error
import urllib.request
import warnings
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "datasets"

USER_AGENT = "pgm-complexity-repo/1.0 (dataset preparation script)"
TIMEOUT = 300


# ==========================================================================
# What to fetch, and what the result must look like
# ==========================================================================
SOURCES = {
    "analcatdata_dmft": {
        "file": "analcatdata_dmft.csv",
        "repository": "PMLB",
        "pmlb_name": "analcatdata_dmft",
        "url": "https://github.com/EpistasisLab/pmlb/raw/master/datasets/analcatdata_dmft/analcatdata_dmft.tsv.gz",
        "licence": "MIT",
        "rule": "drop_zero_feature_rows",
        "n_rows": 787,
        "n_features": 4,
        "n_classes": 6,
        "column_types": ["int", "int", "int", "int", "int"],
        "row_order": "source",
        "md5": "f88b41295eb5ab253b1c6616b596cb4f",
        "sha256": "7657e562bddaf00bcb0b5e7cfe08b50586d24ce74b7680893cdc9cdd1fd66afa",
        "content_sha256": "87b0ec1f740978e350568f08e7307d6cd3cebfe60485023fbf213b74bf165908",
    },
    "balance-scale": {
        "file": "balance-scale.csv",
        "repository": "PMLB",
        "pmlb_name": "balance_scale",
        "url": "https://github.com/EpistasisLab/pmlb/raw/master/datasets/balance_scale/balance_scale.tsv.gz",
        "licence": "MIT",
        "rule": "identity",
        "n_rows": 625,
        "n_features": 4,
        "n_classes": 3,
        "column_types": ["int", "int", "int", "int", "int"],
        "row_order": "source",
        "md5": "5f33cc3ffdec0ef717b6dc1be7ac7fa7",
        "sha256": "9cf658909c5731aed7382597f9b1480882bd6aad551c79c1a158f78759cc5096",
        "content_sha256": "c380d15874981f522ea0b58d3a4948945d120bfed0cdd63402103664e0a516b2",
    },
    "car": {
        "file": "car.csv",
        "repository": "OpenML",
        "data_id": 21,
        "url": "https://www.openml.org/d/21",
        "licence": "CC BY 4.0",
        "rule": "label_rank_encode_and_drop_rows",
        "n_rows": 1727,
        "n_features": 6,
        "n_classes": 4,
        "column_types": ["int", "int", "int", "int", "int", "int", "int"],
        "row_order": "car_partition",
        "md5": "97bf85b5072577e87a02cacb66c8eb9a",
        "sha256": "53247ba06a74d3e4be76fc184fe1282ac4eeaddbc8938d62bcd41393625154a2",
        "content_sha256": "84f8af219178156cd1e6e1dfbbf0f0843ae653d9fab1243efe3eaf61d762ec73",
    },
    "cleveland-nominal": {
        "file": "cleveland-nominal.csv",
        "repository": "OpenML",
        "data_id": 40711,
        "url": "https://www.openml.org/d/40711",
        "licence": "CC BY 4.0",
        "rule": "restore_one_based_codes",
        "n_rows": 303,
        "n_features": 7,
        "n_classes": 5,
        "column_types": ["int", "int", "int", "int", "int", "int", "int", "int"],
        "row_order": "source",
        "columns": ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "class"],
        "shift_plus_one": ["cp", "slope"],
        "md5": "1e99b951f3537a5f0bfa1a6815c675ee",
        "sha256": "e942973e80c9d1715fdb15c3aadfcb46bcaeedd39f106a168f3e790d4908fd85",
        "content_sha256": "5139e308619fd636d01c5a55f4f25827170be9237f28540ceea1d23341de5c5e",
    },
    "cloud": {
        "file": "cloud.csv",
        "repository": "PMLB",
        "pmlb_name": "cloud",
        "url": "https://github.com/EpistasisLab/pmlb/raw/master/datasets/cloud/cloud.tsv.gz",
        "licence": "MIT",
        "rule": "identity",
        "n_rows": 108,
        "n_features": 7,
        "n_classes": 4,
        "column_types": [
            "float",
            "int",
            "float",
            "float",
            "float",
            "float",
            "float",
            "int",
        ],
        "row_order": "source",
        "md5": "6bcffdfbed79a79e70a16e66dd217430",
        "sha256": "d03ab81f7b789d40ebce609516c2414a8dd876757e6b95eb492c065ed96bb2c3",
        "content_sha256": "91bea7a07750b03a592be37dfdf2569d413b6a29adde65a3e8fc67b9c1c98f8c",
    },
    "confidence": {
        "file": "confidence.csv",
        "repository": "PMLB",
        "pmlb_name": "confidence",
        "url": "https://github.com/EpistasisLab/pmlb/raw/master/datasets/confidence/confidence.tsv.gz",
        "licence": "MIT",
        "rule": "identity",
        "n_rows": 72,
        "n_features": 3,
        "n_classes": 6,
        "column_types": ["float", "float", "float", "int"],
        "row_order": "source",
        "md5": "a208a47427e686367e02f9032a680a79",
        "sha256": "de19d6852644c51ce8618c24c51ad7ba08705d1f1a1d1eab7747271b9cd7e96f",
        "content_sha256": "778bd53ab4b168877634ece9b0230b1cef2c2ce3956865b5562abd09fef0d38b",
    },
    "ecoli": {
        "file": "ecoli.csv",
        "repository": "PMLB",
        "pmlb_name": "ecoli",
        "url": "https://github.com/EpistasisLab/pmlb/raw/master/datasets/ecoli/ecoli.tsv.gz",
        "licence": "MIT",
        "rule": "relabel",
        "n_rows": 327,
        "n_features": 7,
        "n_classes": 5,
        "column_types": [
            "float",
            "float",
            "float",
            "float",
            "float",
            "float",
            "float",
            "int",
        ],
        "row_order": "source",
        "md5": "e2d1f30ef256664f51882edb65517ae1",
        "sha256": "de5e79ed372fa5bfb68d5c3dd345b9a45f0fce5162a3d1ac5ba96a254274c331",
        "content_sha256": "b83811f42d068a1a16dadcdc35695ffc94f2f524b69057eae71a5e4dce2a39b9",
    },
    "haberman": {
        "file": "haberman.csv",
        "repository": "PMLB",
        "pmlb_name": "haberman",
        "url": "https://github.com/EpistasisLab/pmlb/raw/master/datasets/haberman/haberman.tsv.gz",
        "licence": "MIT",
        "rule": "relabel",
        "n_rows": 306,
        "n_features": 3,
        "n_classes": 2,
        "column_types": ["float", "int", "float", "int"],
        "row_order": "source",
        "md5": "2536765265b0a7098c255bfb21838a74",
        "sha256": "381181bdd266ffe4d275e6fc622a30561eb38673d3a1bff9e581d52b6519a38b",
        "content_sha256": "0c3d1f3c56d44afcee421d68eed98cc5a0d288d333bc8eb7cae06d189b9b9947",
    },
    "iris": {
        "file": "iris.csv",
        "repository": "PMLB",
        "pmlb_name": "iris",
        "url": "https://github.com/EpistasisLab/pmlb/raw/master/datasets/iris/iris.tsv.gz",
        "licence": "MIT",
        "rule": "identity",
        "n_rows": 150,
        "n_features": 4,
        "n_classes": 3,
        "column_types": ["float", "float", "float", "float", "int"],
        "row_order": "source",
        "md5": "e84cb3fd6f5c33a17c004010328dd158",
        "sha256": "00772f5609aaa4e42f4842fb5379e0c9a227b8f1bda52cd66cf5b0a492b553fb",
        "content_sha256": "1039fbfe1ba1a986dd138a6bf5e6f79cf746641be4f7ad820fbe85a584f888f9",
    },
    "led7": {
        "file": "led7.csv",
        "repository": "PMLB",
        "pmlb_name": "led7",
        "url": "https://github.com/EpistasisLab/pmlb/raw/master/datasets/led7/led7.tsv.gz",
        "licence": "MIT",
        "rule": "identity",
        "n_rows": 3200,
        "n_features": 7,
        "n_classes": 10,
        "column_types": ["int", "int", "int", "int", "int", "int", "int", "int"],
        "row_order": "source",
        "md5": "c359e41fec1df6831c2b0de34f3f23b6",
        "sha256": "4ebd85d45f4eb4e17ef6fbce868c202243599fd138b2fd4370cdd59f800d6c29",
        "content_sha256": "3c7df48735248317607ce3fc5670afa0f97aa23d1edb9d60f6eef26e9310ffbd",
    },
    "new-thyroid": {
        "file": "new-thyroid.csv",
        "repository": "PMLB",
        "pmlb_name": "new_thyroid",
        "url": "https://github.com/EpistasisLab/pmlb/raw/master/datasets/new_thyroid/new_thyroid.tsv.gz",
        "licence": "MIT",
        "rule": "relabel",
        "n_rows": 215,
        "n_features": 5,
        "n_classes": 3,
        "column_types": ["float", "float", "float", "float", "float", "int"],
        "row_order": "source",
        "md5": "eddcbb606349436fe306ee46e5d93fb7",
        "sha256": "b1cce1c6594dc58b5509c027480806be8d24c1adc2e4ecccb9b8167f633c28b3",
        "content_sha256": "3d3bef03fac5eab192c6d9375e1c495207883f8ef50f024681ea3466449a79af",
    },
}

CAR_TAIL = [
    7,
    16,
    23,
    29,
    46,
    49,
    54,
    56,
    59,
    60,
    61,
    62,
    66,
    67,
    74,
    77,
    79,
    84,
    89,
    90,
    92,
    94,
    97,
    101,
    103,
    114,
    116,
    120,
    122,
    124,
    125,
    127,
    128,
    130,
    132,
    135,
    145,
    152,
    154,
    157,
    161,
    172,
    187,
    189,
    193,
    195,
    197,
    210,
    213,
    233,
    237,
    239,
    240,
    241,
    245,
    247,
    249,
    258,
    259,
    262,
    263,
    266,
    268,
    270,
    277,
    292,
    295,
    298,
    313,
    324,
    325,
    329,
    330,
    338,
    340,
    345,
    346,
    351,
    358,
    359,
    360,
    363,
    372,
    374,
    377,
    379,
    381,
    382,
    388,
    389,
    391,
    409,
    410,
    413,
    418,
    420,
    421,
    431,
    434,
    436,
    439,
    442,
    445,
    448,
    454,
    455,
    461,
    466,
    474,
    478,
    484,
    486,
    487,
    491,
    496,
    499,
    500,
    504,
    508,
    509,
    518,
    533,
    534,
    546,
    551,
    559,
    562,
    563,
    571,
    574,
    579,
    587,
    588,
    589,
    591,
    599,
    600,
    619,
    620,
    626,
    636,
    641,
    645,
    652,
    655,
    658,
    660,
    667,
    675,
    683,
    688,
    691,
    702,
    710,
    712,
    713,
    715,
    720,
    731,
    733,
    742,
    748,
    752,
    754,
    759,
    761,
    768,
    769,
    772,
    773,
    774,
    791,
    792,
    793,
    796,
    800,
    801,
    804,
    806,
    817,
    818,
    819,
    821,
    828,
    835,
    836,
    846,
    847,
    849,
    854,
    859,
    862,
    866,
    882,
    883,
    896,
    899,
    900,
    913,
    919,
    921,
    930,
    931,
    935,
    937,
    951,
    953,
    955,
    959,
    972,
    974,
    976,
    979,
    982,
    984,
    994,
    998,
    1008,
    1011,
    1013,
    1017,
    1023,
    1025,
    1026,
    1030,
    1035,
    1037,
    1042,
    1047,
    1048,
    1051,
    1055,
    1058,
    1059,
    1061,
    1065,
    1073,
    1076,
    1083,
    1092,
    1094,
    1095,
    1098,
    1101,
    1103,
    1105,
    1106,
    1113,
    1118,
    1130,
    1131,
    1136,
    1146,
    1150,
    1153,
    1179,
    1180,
    1184,
    1185,
    1187,
    1188,
    1189,
    1191,
    1194,
    1207,
    1212,
    1220,
    1221,
    1222,
    1224,
    1228,
    1230,
    1235,
    1238,
    1240,
    1243,
    1247,
    1258,
    1264,
    1271,
    1278,
    1282,
    1283,
    1299,
    1312,
    1313,
    1323,
    1326,
    1328,
    1336,
    1338,
    1350,
    1351,
    1365,
    1367,
    1380,
    1382,
    1394,
    1396,
    1399,
    1400,
    1405,
    1416,
    1425,
    1427,
    1431,
    1433,
    1437,
    1440,
    1442,
    1447,
    1450,
    1453,
    1454,
    1458,
    1461,
    1465,
    1468,
    1471,
    1473,
    1475,
    1478,
    1479,
    1480,
    1483,
    1489,
    1490,
    1494,
    1500,
    1513,
    1514,
    1516,
    1519,
    1523,
    1528,
    1533,
    1534,
    1541,
    1545,
    1551,
    1552,
    1553,
    1555,
    1558,
    1571,
    1575,
    1576,
    1583,
    1587,
    1597,
    1599,
    1606,
    1610,
    1623,
    1630,
    1631,
    1632,
    1636,
    1647,
    1655,
    1656,
    1659,
    1663,
    1670,
    1671,
    1687,
    1692,
    1693,
    1699,
    1703,
    1705,
    1713,
    1714,
    1724,
]

SKIN = {
    "file": "Skin_NonSkin.txt",
    "name": "skin_segmentation",
    "url": "https://archive.ics.uci.edu/static/public/229/skin+segmentation.zip",
    "mirror_data_id": 1502,
    "licence": "CC BY 4.0",
    "n_rows": 245057,
    "n_features": 3,
    "n_classes": 2,
    "bytes": 3400818,
    "md5": "64ec30a3d91338593ecb62583526b93e",
    "sha256": "e30c0a845385dcc95a45c45ed263465674a49638e98ef740afd520769c7714a4",
}


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


# ==========================================================================
# commands
# ==========================================================================
def cmd_sources() -> int:
    print(f"{'dataset':<20}{'rows':>7}{'feat':>6}{'cls':>5}  {'licence':<12}source")
    print("-" * 96)
    for name, spec in SOURCES.items():
        origin = (
            f"PMLB {spec['pmlb_name']}"
            if spec["repository"] == "PMLB"
            else f"OpenML id {spec['data_id']}"
        )
        print(row(name, spec, f"{spec['licence']:<12}{origin}"))
    print(
        row(
            SKIN["name"],
            SKIN,
            f"{SKIN['licence']:<12}UCI id 229 (mirror: OpenML id "
            f"{SKIN['mirror_data_id']})",
        )
    )
    print("-" * 96)
    print(
        "Transformations, row order and citations: see the docstring of this "
        "file\n(`python scripts/fetch_datasets.py --help`) and "
        "`docs/datasets.md`."
    )
    return 0


def cmd_check(outdir: Path, names: list, want_skin: bool) -> int:
    failures = 0
    print(f"{'dataset':<20}{'rows':>7}{'feat':>6}{'cls':>5}  digest")
    print("-" * 60)
    for name in names:
        spec = SOURCES[name]
        target = outdir / spec["file"]
        if not target.exists():
            print(blank_row(name, "MISSING"))
            failures += 1
        elif matches(target.read_bytes(), spec):
            print(row(name, spec, "OK"))
        else:
            print(blank_row(name, "DIGEST MISMATCH"))
            failures += 1

    if want_skin:
        target = outdir / SKIN["file"]
        if not target.exists():
            print(blank_row(SKIN["name"], "MISSING"))
            failures += 1
        elif matches(target.read_bytes(), SKIN):
            print(row(SKIN["name"], SKIN, "OK"))
        else:
            print(blank_row(SKIN["name"], "DIGEST MISMATCH"))
            failures += 1

    print("-" * 60)
    if failures:
        print(
            f"{failures} file(s) missing or altered. Run "
            f"`python scripts/fetch_datasets.py` to (re)build them."
        )
        return 1
    print("Every file is present and matches the expected digest.")
    return 0


def cmd_fetch(outdir: Path, names: list, want_skin: bool, force: bool) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{'dataset':<20}{'rows':>7}{'feat':>6}{'cls':>5}  status")
    print("-" * 60)

    built: dict = {}
    failures = 0

    for name in names:
        spec = SOURCES[name]
        target = outdir / spec["file"]

        if target.exists() and not force and matches(target.read_bytes(), spec):
            print(row(name, spec, "already correct"))
            continue

        try:
            raw = materialize(name, spec)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(blank_row(name, f"FAILED: {exc}"))
            failures += 1
            continue

        if not matches(raw, spec):
            print(
                blank_row(
                    name, f"DIGEST MISMATCH (got md5 {digests(raw)['md5'][:12]}...)"
                )
            )
            failures += 1
            continue

        built[target] = raw
        print(row(name, spec, "rebuilt, digests OK"))

    if failures:
        print("-" * 60)
        print(
            f"{failures} dataset(s) could not be reproduced; nothing was "
            f"written to {outdir}."
        )
        return 1

    for target, raw in built.items():
        write(target, raw)

    skin_failed = False
    if want_skin:
        target = outdir / SKIN["file"]
        if target.exists() and not force and matches(target.read_bytes(), SKIN):
            print(row(SKIN["name"], SKIN, "already correct"))
        else:
            try:
                raw = materialize_skin()
                if not matches(raw, SKIN):
                    raise RuntimeError(
                        f"digest mismatch (got md5 {digests(raw)['md5'][:12]}...)"
                    )
                write(target, raw)
                print(row(SKIN["name"], SKIN, "downloaded, digests OK"))
            except Exception as exc:  # noqa: BLE001
                print(blank_row(SKIN["name"], f"FAILED: {exc}"))
                skin_failed = True

    print("-" * 60)
    if built:
        print(f"{len(built)} file(s) written to {outdir}.")
    if skin_failed:
        print(
            "Skin Segmentation could not be fetched from UCI nor from its "
            "OpenML mirror.\nThe eleven small datasets are in place: "
            "Table 7 and Component A of the harness\ncan run; only "
            "`run_benchmarks.py B` needs this file."
        )
        return 1
    print("Every dataset is in place and matches its digest byte for byte.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--outdir", default=None, help="destination directory (default: datasets/)"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None, help="restrict to these dataset names"
    )
    parser.add_argument(
        "--skip-skin", action="store_true", help="do not fetch Skin Segmentation"
    )
    parser.add_argument(
        "--only-skin",
        action="store_true",
        help="fetch Skin Segmentation and nothing else",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download the files that are already correct",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the files on disk, without downloading",
    )
    parser.add_argument(
        "--sources",
        action="store_true",
        help="print the source and licence of every file",
    )
    args = parser.parse_args()

    if args.sources:
        return cmd_sources()

    if args.only_skin and args.skip_skin:
        parser.error("--only-skin and --skip-skin are mutually exclusive")

    outdir = Path(args.outdir).resolve() if args.outdir else DEFAULT_OUTDIR
    names = [] if args.only_skin else (args.datasets or list(SOURCES))
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        parser.error(
            f"unknown dataset(s): {', '.join(unknown)}; available: {', '.join(SOURCES)}"
        )
    want_skin = not args.skip_skin and (args.only_skin or args.datasets is None)

    if args.check:
        return cmd_check(outdir, names, want_skin)
    return cmd_fetch(outdir, names, want_skin, args.force)


if __name__ == "__main__":
    sys.exit(main())
