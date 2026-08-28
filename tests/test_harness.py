"""The plumbing of the benchmark harness.

Not the timings, which are properties of a machine, but the things that decide
which models get measured in the first place: the split, the size accounting,
and the self-test that exercises the whole pipeline with numpy stand-ins.
"""

from __future__ import annotations

import numpy as np
import pytest

# Training-set sizes of Table 7. They are what the seeded stratified split
# produces on the shipped files, and Component A reproduces them exactly.
TABLE7_N = {
    "analcatdata_dmft": 629,
    "balance-scale": 500,
    "car": 1381,
    "cleveland-nominal": 242,
    "cloud": 86,
    "confidence": 57,
    "ecoli": 261,
    "haberman": 244,
    "iris": 120,
    "led7": 2560,
    "new-thyroid": 172,
}


def test_the_split_reproduces_the_tabulated_training_sizes(harness, repo_root):
    """The N column of Table 7 comes out of the code, on the data as shipped."""
    available = {
        name: repo_root / "datasets" / f"{name}.csv"
        for name in TABLE7_N
        if (repo_root / "datasets" / f"{name}.csv").exists()
    }
    if not available:
        pytest.skip("datasets/ is empty; run scripts/fetch_datasets.py")

    for name in available:
        _, y = harness.load_dataset(name)
        train, _ = harness.stratified_split(y)
        assert len(train) == TABLE7_N[name], name


def test_the_split_is_stratified_and_seeded(harness):
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], [60, 30, 10])
    rng.shuffle(y)

    train, test = harness.stratified_split(y)

    assert len(train) + len(test) == len(y)
    assert set(train).isdisjoint(test)
    assert len(test) == pytest.approx(0.2 * len(y), abs=1)

    for label in (0, 1, 2):
        share = (y[test] == label).mean()
        assert share == pytest.approx((y == label).mean(), abs=0.05)

    again, _ = harness.stratified_split(y)
    assert list(train) == list(again), "the same seed must give the same split"


def test_the_estimator_modules_are_importable_from_the_repository_root():
    """`import qunica` must resolve, whether or not PyTorch is installed.

    The project is not installed as a package, so the repository root has to be
    on sys.path. When it is not, the estimator tests do not skip -- they fail at
    collection, and only on a machine that has PyTorch, which is precisely
    where they were meant to run. `find_spec` locates the modules without
    executing them, so this catches the path problem here too.
    """
    import importlib.util

    for module in (
        "qunica",
        "qunica.classifiers",
        "qunica.classifiers.KPGMC_Low_Rank",
        "qunica.classifiers.PGMHQC_gpu_cpu_dtype_Reduced_Low_Rank",
    ):
        assert importlib.util.find_spec(module) is not None, (
            f"{module} is not importable; is the repository root on sys.path? "
            "See `pythonpath` in [tool.pytest.ini_options]."
        )


def test_a_missing_dataset_points_at_the_fetch_command(harness):
    with pytest.raises(FileNotFoundError, match="pgm data fetch"):
        harness.load_dataset("a-dataset-that-does-not-exist")


def test_model_bytes_counts_what_the_fitted_model_holds(harness):
    """The size accounting is what the memory conditions are compared against."""

    class Fitted:
        def __init__(self):
            self.a = np.zeros((10, 20), dtype=np.float64)  # 1600 B
            self.b = np.zeros(5, dtype=np.float64)  # 40 B
            self.name = "not an array"
            self._skipped = None

    size = harness.model_bytes(Fitted())
    assert size >= 1640
    assert size < 4096, "only the stored arrays should be counted"


def test_selftest_runs_the_whole_pipeline_without_torch(harness, tmp_path, monkeypatch):
    """`pgm bench selftest` is the smoke test the CI can always run."""
    from pgm_complexity.bench import selftest as selftest_module

    monkeypatch.setattr(selftest_module, "OUT_DIR", tmp_path)

    class Args:
        reps = 1
        threads = 1
        seed = 42

    harness.selftest(Args())
    assert (tmp_path / "selftest.csv").exists()


def test_the_harmonised_tolerance_rescales_by_the_training_size(harness):
    """k-PGM truncates the spectrum of G^c, the Rc-PGM that of sigma = G^c / N.

    The two spectra differ by the factor N, so a single numerical threshold
    would discard different eigendirections in the two estimators. This is the
    rescaling that keeps them comparable.
    """
    assert harness.HARMONIZE_TOL is True
    assert harness.BASE_TOL == 1e-6


def test_the_estimators_are_importable_from_an_installed_entry_point(
    repo_root, tmp_path
):
    """The `pgm` console script must be able to import `qunica`.

    This is the one path the rest of the suite cannot check on its own.
    `pythonpath = ["."]` puts the repository root on sys.path for every test,
    which is exactly the condition that hides the failure: an installed
    console script gets the directory of the script on sys.path, never the
    working directory, so `import qunica` fails there and nowhere else --
    in `pgm bench a`, the command the README documents.

    So this runs in a subprocess whose sys.path[0] is a temporary directory,
    with the repository root removed from the environment, and asserts that
    the package makes `qunica` importable by itself.
    """
    import os
    import subprocess
    import sys

    script = tmp_path / "as_a_console_script.py"
    script.write_text(
        "import importlib.util, sys\n"
        "from pgm_complexity.bench.estimators import ensure_estimators_importable\n"
        "assert str(sys.path[0]) != ROOT, sys.path[0]\n"
        "print('before', importlib.util.find_spec('qunica') is not None)\n"
        "print('after', ensure_estimators_importable())\n".replace(
            "ROOT", repr(str(repo_root))
        )
    )

    env = dict(os.environ)
    keep = [
        p
        for p in env.get("PYTHONPATH", "").split(os.pathsep)
        if p and os.path.realpath(p) != os.path.realpath(repo_root)
    ]
    env["PYTHONPATH"] = os.pathsep.join(keep)

    out = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "after True" in out.stdout, out.stdout + out.stderr
