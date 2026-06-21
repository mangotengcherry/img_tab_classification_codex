# Fusion evaluation report

- labels: edge_ring, center_blob, leak_top, leak_bottom
- heads present: image_only, tabular_only, fusion

## Subset accuracy by head x eval_group  (acc [CI] / support)

| head | real_composite | real_single | synthetic_composite |
|---|---|---|---|
| image_only | 0.583 [0.46,0.70] / 60 | 0.722 [0.65,0.78] / 180 | 0.580 [0.50,0.66] / 150 |
| tabular_only | 0.550 [0.42,0.67] / 60 | 1.000 [0.98,1.00] / 180 | — |
| fusion | 0.967 [0.89,0.99] / 60 | 0.989 [0.96,1.00] / 180 | — |

## KPI product per head  (single x composite, real only)

| head | single | composite | KPI product | composite support |
|---|---|---|---|---|
| image_only | 0.722 | 0.583 | 0.421 | 60 |
| tabular_only | 1.000 | 0.550 | 0.550 | 60 |
| fusion | 0.989 | 0.967 | 0.956 | 60 |

## Fusion gain

- fusion_kpi: 0.956
- best_unimodal_kpi: 0.550
- gain: 0.406
- gain_over_image_only: 0.535
- gain_over_tabular_only: 0.406

## Modality-collapse diagnostic (real composite)

- n_composite: 60.000
- image_only_subset_acc: 0.583
- fusion_subset_acc: 0.967
- fusion_gain_over_image: 0.383
- tabular_rescue_candidates: 17.000
- fusion_followed_tabular: 15.000
- fusion_follow_rate: 0.882
- fusion_regressions: 0.000

## Identity-class slice (electrical-only classes)

- identity labels: leak_top, leak_bottom
- n: 141
- image_only: acc=0.468 / support=141
- tabular_only: acc=0.879 / support=141
- fusion: acc=0.972 / support=141
- tabular_minus_image: 0.411

_No warnings._
