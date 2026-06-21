# Fusion evaluation report

- labels: edge_ring, center_blob, leak_top, leak_bottom
- heads present: image_only, tabular_only, fusion

## Subset accuracy by head x eval_group  (acc [CI] / support)

| head | real_composite | real_single | synthetic_composite |
|---|---|---|---|
| image_only | 0.650 [0.52,0.76] / 60 | 0.700 [0.63,0.76] / 180 | 0.660 [0.58,0.73] / 150 |
| tabular_only | 0.500 [0.38,0.62] / 60 | 1.000 [0.98,1.00] / 180 | — |
| fusion | 0.883 [0.78,0.94] / 60 | 1.000 [0.98,1.00] / 180 | — |

## KPI product per head  (single x composite, real only)

| head | single | composite | KPI product | composite support |
|---|---|---|---|---|
| image_only | 0.700 | 0.650 | 0.455 | 60 |
| tabular_only | 1.000 | 0.500 | 0.500 | 60 |
| fusion | 1.000 | 0.883 | 0.883 | 60 |

## Fusion gain

- fusion_kpi: 0.883
- best_unimodal_kpi: 0.500
- gain: 0.383
- gain_over_image_only: 0.428
- gain_over_tabular_only: 0.383

## Modality-collapse diagnostic (real composite)

- n_composite: 60.000
- image_only_subset_acc: 0.650
- fusion_subset_acc: 0.883
- fusion_gain_over_image: 0.233
- tabular_rescue_candidates: 6.000
- fusion_followed_tabular: 4.000
- fusion_follow_rate: 0.667
- fusion_regressions: 0.000

## Identity-class slice (electrical-only classes)

- identity labels: leak_top, leak_bottom
- n: 143
- image_only: acc=0.476 / support=143
- tabular_only: acc=0.860 / support=143
- fusion: acc=0.951 / support=143
- tabular_minus_image: 0.385

_No warnings._
