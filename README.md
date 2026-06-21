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

Evaluate experiment conditions against the current subset-accuracy targets:

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli evaluate-conditions \
  --predictions outputs/condition_predictions.csv \
  --labels defect_a,defect_b,defect_c \
  --output outputs/condition_summary.csv \
  --single-target 0.8 \
  --composite-target 0.6 \
  --kpi-target 0.65
```

Evaluate one CSV per condition by glob. When a file has no `condition` column, the file stem becomes the condition name:

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli evaluate-conditions \
  --prediction-glob "outputs/conditions/*.csv" \
  --labels defect_a,defect_b,defect_c \
  --output outputs/condition_summary.csv \
  --threshold-grid 0.30,0.35,0.40,0.45,0.50,0.55,0.60 \
  --single-target 0.8 \
  --composite-target 0.6 \
  --kpi-target 0.65
```

Sweep scalar thresholds and keep the best threshold per condition:

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli evaluate-conditions \
  --predictions outputs/condition_predictions.csv \
  --labels defect_a,defect_b,defect_c \
  --output outputs/condition_summary.csv \
  --threshold-grid 0.30,0.35,0.40,0.45,0.50,0.55,0.60 \
  --single-target 0.8 \
  --composite-target 0.6 \
  --kpi-target 0.65
```

Expected prediction columns:

- `condition`: experiment condition name, such as `image_only_synth`, `late_fusion`, or `mapped_fusion`
- `chip_id`
- `eval_group`: `real_single`, `real_composite`, or `synthetic_composite`
- `true_<label>` for every label
- `prob_<label>` for every label

The evaluator reports:

- single-defect subset accuracy
- real composite subset accuracy
- synthetic composite subset accuracy
- synthetic-to-real composite gap
- `single_subset_accuracy * composite_subset_accuracy`
- selected `threshold`
- pass/fail flags for single, composite, KPI, and all targets
- required composite accuracy at the observed single accuracy to reach the KPI target

Note that the individual minimums alone do not imply the product target:

```text
0.8 * 0.6 = 0.48
```

To reach a product KPI of `0.65`, a condition with single subset accuracy `0.8` needs composite subset accuracy at least `0.8125`.

## Measurement Map

`measurement_map.csv` should include:

```csv
feature_name,measurement_condition,measurement_type,wl_index,physical_region,physical_order
MSR_000,read,leakage,100,top,2
MSR_001,read,leakage,0,bottom,0
```

`physical_order` is metadata for position-aware experiments. The code does not infer physical order from the `MSR_*` suffix.
