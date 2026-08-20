# Datasets

Twelve datasets are used in the paper: eleven small benchmarks, which appear in
Table 7 and in Component A of the benchmark harness, and Skin Segmentation,
which is used in Component B.

## File format

The eleven CSV files in this directory are **header-less**, comma-separated,
and the **last column is the integer class label**, taking the consecutive
values `0 .. l-1`. This is the layout that `notebooks/table7.ipynb` and
`run_benchmarks.py` read with `pandas.read_csv(path, header=None)`.

`manifest.json` is the machine-readable counterpart of this page: for every
file it records the source, its licence, the transformation applied to it, the
resulting shape and the MD5/SHA-256 digests. Verify the files with

```bash
python datasets/prepare_datasets.py check
```

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
`slope`, `ca`, `class`. The OpenML release stores every nominal attribute as a
0-based category index; `cp` and `slope` are shifted back to the coding of the
original Cleveland database, `cp ∈ {1,2,3,4}` and `slope ∈ {1,2,3}`.

## Rebuilding the files from their sources

```bash
python datasets/prepare_datasets.py rebuild --outdir /tmp/rebuilt
```

downloads each source, re-applies the transformation above and compares the
result with the shipped file. The comparison is made on a row-order-independent
digest (`content_sha256` in `manifest.json`), because the row order of the
shipped files is not part of the transformation.

The row order does, however, matter for the reported numbers: the notebook and
the harness use `train_test_split(..., shuffle=True, stratify=y,
random_state=42)`, whose outcome is seeded but depends on the order of the
input rows. **Reproducing Table 7 and the benchmarks requires the CSV files as
shipped here**, not a rebuilt copy.

## Skin Segmentation

`Skin_NonSkin.txt` is not included, because of its size. Fetch it with

```bash
python datasets/download_skin_segmentation.py
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
