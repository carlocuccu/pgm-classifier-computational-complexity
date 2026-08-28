"""The three things one does with the datasets: list, verify, rebuild.

Separated from `build`, which knows how to produce a file, and from the command
line, which only parses arguments. What lives here is the reporting: what to
print, when to refuse to write, and what exit status to return.
"""

from __future__ import annotations

from pathlib import Path

from pgm_complexity.data.build import (
    blank_row,
    check_environment,
    digests,
    matches,
    materialize,
    materialize_skin,
    row,
    write,
)
from pgm_complexity.data.specs import SKIN, SOURCES


# ==========================================================================
# commands
# ==========================================================================
def sources() -> int:
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
        "file\n(`pgm data fetch --help`) and "
        "`docs/datasets.md`."
    )
    return 0


def check(outdir: Path, names: list, want_skin: bool) -> int:
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


def fetch(outdir: Path, names: list, want_skin: bool, force: bool) -> int:
    check_environment()
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
