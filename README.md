# FBM Multi-Modal Experiment Toolkit

This repository contains a lightweight experiment scaffold for FBM image and WL/electrical measurement multi-label classification.

It implements the data contracts and evaluation utilities from `plan.md`:

- FBM intensity normalization for 0..8 graded images
- Measurement mapping validation without assuming `MSR_*` suffix order is physical order
- Multi-label metrics, class-wise threshold search, class-pair metrics, and synthetic-to-real gap reports
- Synthetic FBM composition modes
- Active-learning ranking for unlabeled chip review
- CLI helpers for mapping coverage and review candidate selection

## Install

Use the system Python available in this workspace:

```bash
python3 -m pip install -e .
```

## Run Tests

```bash
PYTHONPATH=src python3 -m pytest -q
```

## CLI Examples

Validate that `measurement_map.csv` covers all `MSR_*` columns in a chip manifest:

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli validate-map \
  --manifest data/manifest.csv \
  --measurement-map data/measurement_map.csv
```

Rank unlabeled chips for engineering review:

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli rank-unlabeled \
  --candidates outputs/unlabeled_predictions.csv \
  --labels defect_a,defect_b,defect_c \
  --target-labels defect_a,defect_b \
  --budget 100 \
  --output outputs/review_queue.csv
```

Expected candidate probability columns:

- `prob_<label>`
- optional `image_prob_<label>`
- optional `tabular_prob_<label>`

## Measurement Map

`measurement_map.csv` should include:

```csv
feature_name,measurement_condition,measurement_type,wl_index,physical_region,physical_order
MSR_000,read,leakage,100,top,2
MSR_001,read,leakage,0,bottom,0
```

`physical_order` is metadata for position-aware experiments. The code does not infer physical order from the `MSR_*` suffix.
