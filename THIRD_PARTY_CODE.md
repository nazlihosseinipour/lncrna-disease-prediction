# Third-party code provenance

## lncRNA_CIBCB2025

- Upstream repository: <https://github.com/fkenjinakano/lncRNA_CIBCB2025>
- Upstream owner: F. Kenji Nakano
- Local path: `lncRNA_CIBCB2025-main/`
- Upstream revision reviewed: `2e7cd9794b4d1d999faff6bbe5d30622390b43ac`
  (`updates`, 2026-04-15)
- Adapted fork:
  <https://github.com/nazlihosseinipour/lncRNA_CIBCB2025/tree/agent/internship-adaptations>
- Adaptation branch: `agent/internship-adaptations`
- Adaptation commit: `9bfab69` (`Add internship experiment adaptations`)
- Relationship: locally adapted research code used by this project's
  structured-output lncRNA–disease association experiments

The upstream repository supplies the original RFLDA and IPCARF
implementations, dataset-construction utilities, evaluation utilities, and
supporting research assets.

## Local adaptations

The copy used by this project differs from the reviewed upstream revision in
the following substantive ways:

- `IPCARF/ipcarf.py`
  - uses the experiment-scale estimator and component-search settings;
  - sanitizes non-finite feature values;
  - constrains the PCA component grid to valid dimensions;
  - falls back to randomized PCA when IncrementalPCA fails.
- `RFLDA/rflda.py`

## Pse-in-One RNA property values

`config/pse_in_one_rna_dcc_properties.csv` contains the 22 RNA dinucleotide
property vectors distributed with Pse-in-One 1.0.6. They were exported from
`Pse-in-One/data/dirna.data` in upstream commit
`bd4f39e3d509a4b9367f0f657317e243fca6f11b` and retain the upstream GPL-3.0
license. Upstream source: https://github.com/bioinformatics-hitsz/Pse-in-One
  - supports feature sets with fewer than 50 columns;
  - guarantees that Random Forest `max_features` is at least one.
- `parse_results/evaluate.py`
  - supports fixed-threshold evaluation;
  - handles labels with only one observed class safely.
- `RF/`
  - adds the Random Forest baseline used by this project.
- `utils/__init__.py`
  - makes the upstream utilities importable by the project runners.

These adaptations should not be interpreted as changes authored by the
upstream repository owner. Consult the upstream repository for the original
implementation and its history.

## Reproducible checkout

The active experiment scripts expect the model implementation at
`lncRNA_CIBCB2025-main/`. If the directory needs to be reconstructed, check
out the adapted fork at the recorded commit:

```bash
git clone --branch agent/internship-adaptations \
  https://github.com/nazlihosseinipour/lncRNA_CIBCB2025.git \
  lncRNA_CIBCB2025-main
git -C lncRNA_CIBCB2025-main checkout 9bfab69
```

Do not substitute the upstream `main` branch directly: it does not contain the
experiment-specific adaptations listed above.
