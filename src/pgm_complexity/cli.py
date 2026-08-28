"""The command line of the project.

    pgm data fetch | check | sources
    pgm bench a | b | all | selftest | replot
    pgm paper numbers

Two things about this module are deliberate and easy to undo by accident.

It pins the BLAS and OpenMP thread count *before* importing anything that
imports numpy, because BLAS reads that setting at load time. Every heavy import
therefore happens inside the command functions, not at the top of the file.

And it stays thin: it parses arguments and calls into the package. Nothing here
decides anything the library does not already know how to do.
"""

from __future__ import annotations

import argparse
import sys

from pgm_complexity.threads import pin_threads

# Before numpy exists anywhere in the process.
N_THREADS = pin_threads()


# ---------------------------------------------------------------------------
# pgm data
# ---------------------------------------------------------------------------
def _data_fetch(args) -> int:
    from pathlib import Path

    from pgm_complexity import DATA_DIR
    from pgm_complexity.data import commands

    outdir = Path(args.outdir).resolve() if args.outdir else DATA_DIR
    names = [] if args.only_skin else (args.datasets or list(commands.SOURCES))
    unknown = [n for n in names if n not in commands.SOURCES]
    if unknown:
        raise SystemExit(
            f"unknown dataset(s): {', '.join(unknown)}; "
            f"available: {', '.join(commands.SOURCES)}"
        )
    want_skin = not args.skip_skin and (args.only_skin or args.datasets is None)
    return commands.fetch(outdir, names, want_skin, args.force)


def _data_check(args) -> int:
    from pathlib import Path

    from pgm_complexity import DATA_DIR
    from pgm_complexity.data import commands

    outdir = Path(args.outdir).resolve() if args.outdir else DATA_DIR
    names = args.datasets or list(commands.SOURCES)
    return commands.check(outdir, names, want_skin=args.datasets is None)


def _data_sources(args) -> int:
    from pgm_complexity.data import commands

    return commands.sources()


# ---------------------------------------------------------------------------
# pgm bench
# ---------------------------------------------------------------------------
def _bench(args) -> int:
    import json

    from pgm_complexity import OUT_DIR
    from pgm_complexity.bench.io import env_info

    OUT_DIR.mkdir(exist_ok=True)
    print(f"[env] {json.dumps(env_info())}")

    if args.what == "replot":
        from pgm_complexity.figures import replot_B

        return replot_B()

    if args.what == "selftest":
        from pgm_complexity.bench.selftest import selftest

        selftest(args)
        return 0

    from pgm_complexity.bench.components import component_A, component_B

    if args.what in ("a", "all"):
        component_A(args)
    if args.what in ("b", "all"):
        component_B(args)
    return 0


# ---------------------------------------------------------------------------
# pgm paper
# ---------------------------------------------------------------------------
def _paper(args) -> int:
    from pathlib import Path

    from pgm_complexity import OUT_DIR
    from pgm_complexity.paper import component_a_report, component_b_report, load

    results = Path(args.results).resolve() if args.results else OUT_DIR
    if not (results / "componentA.csv").exists():
        raise SystemExit(
            f"no deposited run in {results}: run `pgm bench all` first, "
            "or point --results at one."
        )
    a, a_meta, b, b_meta = load(results)
    component_a_report(a, a_meta)
    component_b_report(b, b_meta)
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pgm",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="group", required=True)

    # -- data ---------------------------------------------------------------
    data = sub.add_parser("data", help="download and verify the datasets")
    data_sub = data.add_subparsers(dest="action", required=True)

    fetch = data_sub.add_parser(
        "fetch", help="download every dataset and rebuild it byte for byte"
    )
    fetch.add_argument("--outdir", default=None, help="default: datasets/")
    fetch.add_argument("--datasets", nargs="+", default=None)
    fetch.add_argument("--skip-skin", action="store_true")
    fetch.add_argument("--only-skin", action="store_true")
    fetch.add_argument("--force", action="store_true")
    fetch.set_defaults(func=_data_fetch)

    check = data_sub.add_parser(
        "check", help="verify the files on disk against their digests, offline"
    )
    check.add_argument("--outdir", default=None)
    check.add_argument("--datasets", nargs="+", default=None)
    check.set_defaults(func=_data_check)

    sources = data_sub.add_parser(
        "sources", help="print the source and licence of every file"
    )
    sources.set_defaults(func=_data_sources)

    # -- bench --------------------------------------------------------------
    from pgm_complexity.config import DEFAULT_A, TABLE7_C

    bench = sub.add_parser("bench", help="run the measurements")
    bench.add_argument(
        "what",
        choices=["a", "b", "all", "selftest", "replot"],
        help="a: per-dataset timings · b: the crossover sweep · "
        "selftest: the pipeline with numpy stand-ins · "
        "replot: redraw Figure 4 from the deposited run",
    )
    bench.add_argument("--reps", type=int, default=5)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--threads", type=int, default=1)
    bench.add_argument(
        "--datasets", nargs="+", default=DEFAULT_A, choices=sorted(TABLE7_C)
    )
    bench.add_argument(
        "--include-cpgm",
        action="store_true",
        help="also run the explicit c-PGM where d_enc^c is small",
    )
    bench.add_argument("--cpgm-dim-limit", type=int, default=4100)
    bench.add_argument("--mem-limit-gb", type=float, default=8.0)
    bench.add_argument("--nmax", type=int, default=8000)
    bench.add_argument("--sweep", nargs="+", type=int, default=None)
    bench.set_defaults(func=_bench)

    # -- paper --------------------------------------------------------------
    paper = sub.add_parser("paper", help="derive the quoted figures from a run")
    paper_sub = paper.add_subparsers(dest="action", required=True)
    numbers = paper_sub.add_parser(
        "numbers", help="every figure the empirical section quotes"
    )
    numbers.add_argument("--results", default=None, help="default: results_benchmark/")
    numbers.set_defaults(func=_paper)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
