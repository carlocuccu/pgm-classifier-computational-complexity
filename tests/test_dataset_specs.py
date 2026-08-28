"""The dataset specifications, and the files they describe.

scripts/fetch_datasets.py carries everything needed to rebuild the twelve
datasets byte for byte: the source, the transformation, the row order, the
column formatting and the digests. Those declarations have to be internally
consistent before they can be trusted to reconstruct anything, and the files on
disk -- when they are there -- have to match them.

The tests that reach the network are marked `network` and skipped by default.
"""

from __future__ import annotations

import csv
import re

import pytest

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX32 = re.compile(r"^[0-9a-f]{32}$")


def specs(datasets_script):
    return datasets_script.SOURCES


def test_the_eleven_small_datasets_are_declared(datasets_script):
    assert len(specs(datasets_script)) == 11
    assert set(specs(datasets_script)) == {
        "analcatdata_dmft",
        "balance-scale",
        "car",
        "cleveland-nominal",
        "cloud",
        "confidence",
        "ecoli",
        "haberman",
        "iris",
        "led7",
        "new-thyroid",
    }


def test_every_spec_is_complete(datasets_script):
    required = {
        "file",
        "repository",
        "url",
        "licence",
        "rule",
        "n_rows",
        "n_features",
        "n_classes",
        "column_types",
        "row_order",
        "md5",
        "sha256",
        "content_sha256",
    }
    for name, spec in specs(datasets_script).items():
        assert required <= set(spec), f"{name} is missing {required - set(spec)}"
        assert spec["repository"] in ("PMLB", "OpenML"), name
        assert HEX32.match(spec["md5"]), name
        assert HEX64.match(spec["sha256"]), name
        assert HEX64.match(spec["content_sha256"]), name


def test_column_types_cover_the_features_and_the_label(datasets_script):
    """One entry per column, and every entry is int or float."""
    for name, spec in specs(datasets_script).items():
        types = spec["column_types"]
        assert len(types) == spec["n_features"] + 1, name
        assert set(types) <= {"int", "float"}, name
        assert types[-1] == "int", f"{name}: the label column is an integer"


def test_row_order_is_either_the_source_or_the_car_partition(datasets_script):
    orders = {name: spec["row_order"] for name, spec in specs(datasets_script).items()}
    assert orders.pop("car") == "car_partition"
    assert set(orders.values()) == {"source"}, "only car departs from the source order"


def test_the_car_partition_is_a_valid_index_set(datasets_script):
    """CAR_TAIL selects rows to move to the end: distinct, in range, ordered."""
    tail = datasets_script.CAR_TAIL
    n_rows = specs(datasets_script)["car"]["n_rows"]

    assert len(tail) == len(set(tail)), "repeated index"
    assert min(tail) >= 0 and max(tail) < n_rows, "index out of range"
    assert tail == sorted(tail), "the tail keeps the order of the source"
    assert len(tail) == 374
    assert n_rows - len(tail) == 1353


def test_skin_segmentation_is_declared_but_not_redistributed(datasets_script):
    skin = datasets_script.SKIN
    assert skin["n_rows"] == 245057
    assert HEX32.match(skin["md5"]) and HEX64.match(skin["sha256"])
    assert "mirror_data_id" in skin, "UCI is not reachable from every network"


# --------------------------------------------------------------------------
# The files themselves, when the datasets directory has been populated
# --------------------------------------------------------------------------
def _present(datasets_script, repo_root):
    return [
        (name, repo_root / "datasets" / spec["file"], spec)
        for name, spec in specs(datasets_script).items()
        if (repo_root / "datasets" / spec["file"]).exists()
    ]


def test_files_on_disk_match_their_digests(datasets_script, repo_root):
    present = _present(datasets_script, repo_root)
    if not present:
        pytest.skip("datasets/ is empty; run scripts/fetch_datasets.py")

    for name, path, spec in present:
        assert datasets_script.matches(path.read_bytes(), spec), name


def test_files_on_disk_have_the_declared_shape(datasets_script, repo_root):
    present = _present(datasets_script, repo_root)
    if not present:
        pytest.skip("datasets/ is empty; run scripts/fetch_datasets.py")

    for name, path, spec in present:
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == spec["n_rows"], name
        assert len(rows[0]) == spec["n_features"] + 1, name

        labels = {int(r[-1]) for r in rows}
        assert labels == set(range(spec["n_classes"])), (
            f"{name}: labels are not the consecutive integers 0..l-1"
        )


@pytest.mark.network
def test_rebuild_from_the_public_sources(datasets_script):
    """The transformation still reproduces every file, byte for byte."""
    for name, spec in specs(datasets_script).items():
        raw = datasets_script.materialize(name, spec)
        assert datasets_script.matches(raw, spec), name
