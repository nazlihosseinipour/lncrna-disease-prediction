# Nazli V2 827-feature feasibility audit — 2026-10-08

## Scope and measured population

This is a data and feature-schema audit. **V2 model training remains deferred**. The canonical corrected V2 table contains **5,338 RNA rows** and **436 disease outcomes**. Of those rows, **4,130 have usable sequences** and **1,208 do not**.

The defensible V2 predictor matrix has 4,130 rows and **542 sequence features**: 462 DCC descriptors, 16 normalized 2-mers, and 64 normalized 3-mers. Its RNA IDs and order exactly match the sequence-available population.

## What happens if the V1 apparent construction is copied

The V1 raw source has 285 disease columns, while V2 has 436. Appending all V2 disease outcomes to the 542 sequence descriptors would produce **978 columns**, not 827. Selecting only 285 of the 436 outcomes would discard 151 solely to reach a target width.

These are **association-derived** inputs: in a within-V2 experiment they are the same RNA-disease outcomes the model is asked to predict. Adding them to `X` creates direct **label leakage**. The fact that 542 + 285 equals 827 does not provide a valid feature method.

V1 and V2 share only **20 exact disease names**. Name normalization gives **73 shared disease concepts**. These yield widths of 562 or 615, respectively, and neither defines an 827-column V2 method.

## Decision

Use the 542 sequence-only matrix for a future controlled V1/V2 comparison after that experiment is approved. Appending V2 association outcomes is invalid, and selecting arbitrary columns merely to total 827 is **not an 827-feature paper reproduction**. Felipe's released 827 artifact remains a V1 positive control; it does not supply a reproducible, leakage-free V2 feature recipe.

No V2 model was trained by this audit and no source artifact was modified.
