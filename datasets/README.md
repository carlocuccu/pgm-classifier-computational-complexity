# Datasets

Twelve datasets are used in the paper: eleven small benchmarks, which appear in
Table 7 and in Component A of the benchmark harness, and Skin Segmentation,
which is used in Component B.

## Getting the data

None of the twelve files has to be shipped with the code. Running

```bash
python scripts/fetch_datasets.py
```

downloads every dataset from its public repository, applies the transformation
documented below, restores the row order and the column formatting of the files
used in the experiments, writes the result into this directory and verifies
each file against the MD5/SHA-256 digests of `manifest.json`. The outcome is
**byte-identical** to the data used for Table 7 and for the benchmarks, so the
reported numbers reproduce exactly. Nothing is written unless every file has
been rebuilt and verified.

If the CSV files are already present, the same command reports them as
`already correct` and downloads nothing; `--force` re-downloads them,
`--skip-skin` leaves out the one large file, and `--datasets NAME ...`
restricts the run. To verify files that are already on disk, without network
access, use

```bash
python datasets/prepare_datasets.py check
```

## File format

The eleven CSV files are **header-less**, comma-separated, CRLF-terminated,
and the **last column is the integer class label**, taking the consecutive
values `0 .. l-1`. This is the layout that `notebooks/table7.ipynb` and
`run_benchmarks.py` read with `pandas.read_csv(path, header=None)`.

`manifest.json` is the machine-readable counterpart of this page: for every
file it records the source, its licence, the transformation applied to it, the
resulting shape, the MD5/SHA-256 digests, and the two keys that make a
downloaded copy byte-identical to the one used in the experiments —
`row_order`, described below, and `column_types`, which says whether each
column is written as an integer or as a decimal.

## Sources and transformations

| File | Rows | Feat. | Classes | Source | Licence | Transformation |
|---|---:|---:|---:|---|---|---|
| `analcatdata_dmft.csv` | 787 | 4 | 6 | PMLB `analcatdata_dmft` | MIT | drop zero-feature rows (797 → 787) |
| `balance-scale.csv` | 625 | 4 | 3 | PMLB `balance_scale` | MIT | none |
| `car.csv` | 1727 | 6 | 4 | OpenML `car` (id 21) | CC BY 4.0 | label-rank encoding; drop zero-feature row (1728 → 1727) |
| `cleveland-nominal.csv` | 303 | 7 | 5 | OpenML `cleveland-nominal` (id 40711) | CC BY 4.0 | restore the 1-based coding of `cp` and `slope` |
| `cloud.csv` | 108 | 7 | 4 | PMLB `cloud` | MIT | none |
| `confidence.csv` | 72 | 3 | 6 | PMLB `confidence` | MIT | none |
| `ecoli.csv` | 327 | 7 | 5 | PMLB `ecoli` | MIT | relabel classes to `0..4` |
| `haberman.csv` | 306 | 3 | 2 | PMLB `haberman` | MIT | relabel classes to `0..1` |
| `iris.csv` | 150 | 4 | 3 | PMLB `iris` | MIT | none |
| `led7.csv` | 3200 | 7 | 10 | PMLB `led7` | MIT | none |
| `new-thyroid.csv` | 215 | 5 | 3 | PMLB `new_thyroid` | MIT | relabel classes to `0..2` |
| `Skin_NonSkin.txt` | 245057 | 3 | 2 | UCI Skin Segmentation (id 229) | CC BY 4.0 | **not redistributed**, see below |

The transformations are compositions of four elementary operations.

**Label-rank encoding.** A nominal column is replaced by the rank of its value
in the ascending lexicographic order of that column's distinct labels. This
applies to `car` only, whose attributes are all nominal; for instance
`buying` becomes `high → 0`, `low → 1`, `med → 2`, `vhigh → 3`, and the class
becomes `acc → 0`, `good → 1`, `unacc → 2`, `vgood → 3`.

**Removal of zero-feature rows.** The rows whose feature vector is identically
zero are dropped: ten rows in `analcatdata_dmft` — a single repeated feature
vector carrying several different class labels — and one row in `car`. In both
files this filter leaves no zero-feature row behind. It is applied to these two
datasets only: `led7`, for instance, keeps the all-zero patterns of its source.

**Class relabelling.** The class column is mapped onto the consecutive
integers `0 .. l-1`, preserving the order of the original labels. This is what
turns the PMLB labels `{0,1,4,5,7}` of `ecoli` into `{0,1,2,3,4}`, `{1,2}` of
`haberman` into `{0,1}`, and `{1,2,3}` of `new_thyroid` into `{0,1,2}`. Where
the source labels are already `0 .. l-1` the operation is the identity.

**Restoration of the 1-based nominal coding** (`cleveland-nominal` only). The
columns of the file are, in order, `sex`, `cp`, `fbs`, `restecg`, `exang`,
`slope`, `thal`, `class`. The columns `cp` and `slope` carry the coding of the
original Cleveland database, `cp ∈ {1,2,3,4}` and `slope ∈ {1,2,3}`; some
OpenML clients return every nominal attribute as a 0-based category index, in
which case those two columns are shifted by one. All other columns are copied
verbatim.

## Row order

The row order is not a detail of presentation: the notebook and the harness
split the data with `train_test_split(..., shuffle=True, stratify=y,
random_state=42)`, whose outcome is seeded but depends on the order of the
input rows. A file with the same rows in a different order gives different
accuracies in Table 7 — on `iris`, for instance, 0.900 instead of 0.967.

The order is therefore recorded in `manifest.json`, under `row_order`, and
restored by `scripts/fetch_datasets.py`:

- ten of the eleven files keep the order of their public release
  (`"row_order": "source"`); the rows removed by the zero-feature filter simply
  drop out, leaving the remaining ones in place;
- `car.csv` is a *stable partition* of that order: 1353 rows first, then the
  374 rows whose source indices `manifest.json` lists, each block in the order
  of the source.

`manifest.json` also keeps `content_sha256`, a row-order-independent digest of
the same content, which identifies a file up to a permutation of its rows.

## Rebuilding the files from their sources

`scripts/fetch_datasets.py` is the command to use. `prepare_datasets.py` also
exposes the rebuild as a check that writes nothing:

```bash
python datasets/prepare_datasets.py rebuild                    # verify only
python datasets/prepare_datasets.py rebuild --outdir /tmp/rebuilt
```

Both paths run the same code and produce the same bytes.

## Skin Segmentation

`Skin_NonSkin.txt` is never committed, because of its size. It is fetched by
`scripts/fetch_datasets.py` along with everything else, or on its own with

```bash
python datasets/download_skin_segmentation.py     # = fetch_datasets.py --only-skin
```

which downloads the archive from the UCI Machine Learning Repository and writes
the tab-separated, header-less file that `run_benchmarks.py B` expects (three
integer features, last column the class label in `{1,2}`).

## Licences and citation

The code of this repository is MIT-licensed (see `../LICENSE`); the data files
are not covered by that licence and carry the terms of their sources.

- **PMLB** — Penn Machine Learning Benchmarks, MIT-licensed.
  Romano, J.D., Le, T.T., La Cava, W., Gregg, J.T., Goldberg, D.J., Chakraborty,
  P., Ray, N.L., Himmelstein, D., Fu, W., Moore, J.H. (2021). *PMLB v1.0: an
  open source dataset collection for benchmarking machine learning methods.*
  Bioinformatics 38(3), 878–880. https://github.com/EpistasisLab/pmlb
- **OpenML** — datasets distributed under CC BY 4.0.
  Vanschoren, J., van Rijn, J.N., Bischl, B., Torgo, L. (2013). *OpenML:
  networked science in machine learning.* SIGKDD Explorations 15(2), 49–60.
  `car`: https://www.openml.org/d/21 · `cleveland-nominal`:
  https://www.openml.org/d/40711
- **UCI Machine Learning Repository** — CC BY 4.0.
  Bhatt, R., Dhall, A. (2012). *Skin Segmentation.* UCI Machine Learning
  Repository. https://doi.org/10.24432/C5T30C

Several of these datasets originate from the UCI Machine Learning Repository
and are redistributed by PMLB and OpenML; please cite the original sources
listed on the corresponding PMLB/OpenML pages when reusing them.
