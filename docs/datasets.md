# Datasets

Twelve datasets are used in the paper: eleven small benchmarks, which appear in
Table 7 and in Component A of the benchmark harness, and Skin Segmentation,
which is used in Component B.

**None of them is carried in this repository.** `datasets/` is empty until it is
filled by

```bash
pgm data fetch
```

That script is self-contained: it holds the source, the licence, the
transformation, the row order, the column formatting and the digest of every
file, and needs nothing else in `datasets/` to do its work. It downloads each
dataset from PMLB, OpenML or UCI, applies the transformation, writes the result
into `datasets/` and verifies it against the recorded MD5/SHA-256 digests. What
it writes is **byte-identical** to the data used for Table 7 and for the
benchmarks, so the reported numbers reproduce exactly. Nothing is written unless
every file has been rebuilt and verified.

This page is the human-readable summary; `pgm data fetch
--help` prints the same information from the script itself, and `--sources`
prints the source table below.

Useful options:

| command | effect |
|---|---|
| `pgm data fetch` | download everything into `datasets/` |
| `pgm data check` | verify the files on disk, offline |
| `pgm data fetch --skip-skin` | leave out the one large file |
| `pgm data fetch --only-skin` | fetch Skin Segmentation alone |
| `pgm data fetch --force` | re-download files already present |
| `pgm data sources` | print sources and licences |

## File format

The eleven CSV files are **header-less**, comma-separated, CRLF-terminated,
and the **last column is the integer class label**, taking the consecutive
values `0 .. l-1`. This is the layout that `notebooks/table7.ipynb` and
the harness read with `pandas.read_csv(path, header=None)`.
`Skin_NonSkin.txt` is tab-separated and header-less, with three integer
features and the class label in `{1,2}`.

## Sources and transformations

| File | Rows | Feat. | Classes | Source | Licence | Transformation |
|---|---:|---:|---:|---|---|---|
| `analcatdata_dmft.csv` | 787 | 4 | 6 | PMLB `analcatdata_dmft` | MIT | drop zero-feature rows (797 → 787) |
| `balance-scale.csv` | 625 | 4 | 3 | PMLB `balance_scale` | MIT | none |
| `car.csv` | 1727 | 6 | 4 | OpenML `car` (id 21) | CC BY 4.0 | label-rank encoding; drop zero-feature row (1728 → 1727) |
| `cleveland-nominal.csv` | 303 | 7 | 5 | OpenML `cleveland-nominal` (id 40711) | CC BY 4.0 | 1-based coding of `cp` and `slope` |
| `cloud.csv` | 108 | 7 | 4 | PMLB `cloud` | MIT | none |
| `confidence.csv` | 72 | 3 | 6 | PMLB `confidence` | MIT | none |
| `ecoli.csv` | 327 | 7 | 5 | PMLB `ecoli` | MIT | relabel classes to `0..4` |
| `haberman.csv` | 306 | 3 | 2 | PMLB `haberman` | MIT | relabel classes to `0..1` |
| `iris.csv` | 150 | 4 | 3 | PMLB `iris` | MIT | none |
| `led7.csv` | 3200 | 7 | 10 | PMLB `led7` | MIT | none |
| `new-thyroid.csv` | 215 | 5 | 3 | PMLB `new_thyroid` | MIT | relabel classes to `0..2` |
| `Skin_NonSkin.txt` | 245057 | 3 | 2 | UCI Skin Segmentation (id 229) | CC BY 4.0 | none |

PMLB releases are stored with Git LFS, so they are fetched through
`media.githubusercontent.com`, which serves the file itself rather than the LFS
pointer that the plain raw URL returns. Skin Segmentation is downloaded from
UCI; if `archive.ics.uci.edu` is unreachable — it is not always resolvable from
every network — the script falls back to the OpenML mirror (id 1502), which
gives the same file byte for byte.

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
accuracies in Table 7 — on `iris`, 0.900 instead of 0.967; on `ecoli`, 0.818
instead of 0.909.

The order is therefore part of what the script restores:

- ten of the eleven files keep the order of their public release
  (`"row_order": "source"`); the rows removed by the zero-feature filter simply
  drop out, leaving the remaining ones in place;
- `car.csv` is a *stable partition* of that order: the 1353 rows whose source
  index is not in `CAR_TAIL` come first, then the 374 rows listed there, each
  block in the order of the source.

## Verification

Every file is checked twice. `content_sha256` is a row-order-independent digest
— the rows formatted with `%.10g`, sorted and hashed — and catches a
transformation error; the MD5 and SHA-256 of the file itself catch an ordering
or formatting error on top of that. Both are recorded in the script, and a
mismatch aborts the run before anything is written.

To re-verify files already on disk, without network access:

```bash
pgm data check
```

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
  https://www.openml.org/d/40711 · Skin Segmentation mirror:
  https://www.openml.org/d/1502
- **UCI Machine Learning Repository** — CC BY 4.0.
  Bhatt, R., Dhall, A. (2012). *Skin Segmentation.* UCI Machine Learning
  Repository. https://doi.org/10.24432/C5T30C

Several of these datasets originate from the UCI Machine Learning Repository
and are redistributed by PMLB and OpenML; please cite the original sources
listed on the corresponding PMLB/OpenML pages when reusing them.
