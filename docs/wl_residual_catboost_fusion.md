# WL Residual Map + CatBoost Fusion Extension

This extension keeps the existing FBM fusion experiment intact and adds three
compatible surfaces:

- WL raw measurements -> high-side residual map tensors.
- Synthetic composite WL maps from parent residual maps using max residuals,
  union masks, and a source-count channel.
- Offline one-vs-rest CatBoost logits that feed fusion as direct logits.

Pseudo-labeling remains scaffolded only. The default config keeps it disabled.

## Config

Use `configs/wl_residual_catboost_fusion.yaml` as the reference patch. The repo
does not currently have a global config loader, so this file documents the
intended experiment parameters for scripts and notebooks.

## Fit WL Residual Normalizer

```bash
PYTHONPATH=src python3 - <<'PY'
import pandas as pd
from fbm_multimodal.wl_residual_map import WLResidualMapTensorizer

measurements = pd.read_csv("data/wl_measurements.csv")
tensorizer = WLResidualMapTensorizer(num_wl_bins=20, clip_max=10.0)
tensorizer.fit(measurements)
tensorizer.save("outputs/wl_residual_tensorizer.json")
maps = tensorizer.transform(measurements)
tensorizer.save_tensor_cache(maps, "outputs/wl_residual_maps.npz")
print(f"fit_samples={len(tensorizer.fit_sample_ids_)} maps={len(maps)}")
PY
```

If `split`, `eval_group`, or `is_synthetic` columns are present, `fit()` uses
only train real rows. Validation, test, and synthetic rows are excluded from the
median/IQR baseline.

Reload later with:

```python
from fbm_multimodal.wl_residual_map import WLResidualMapTensorizer

tensorizer = WLResidualMapTensorizer.load("outputs/wl_residual_tensorizer.json")
maps = WLResidualMapTensorizer.load_tensor_cache("outputs/wl_residual_maps.npz")
```

## Compose Synthetic WL Maps

```bash
PYTHONPATH=src python3 - <<'PY'
from fbm_multimodal.synthetic_wl_map import compose_synthetic_wl_map
from fbm_multimodal.wl_residual_map import DEFAULT_WL_CHANNELS

# parent_a_map and parent_b_map are [C, B, T] tensors from the fitted tensorizer.
synthetic_map = compose_synthetic_wl_map(parent_a_map, parent_b_map, channel_spec=DEFAULT_WL_CHANNELS)
print(synthetic_map.shape)
PY
```

Synthetic maps should be used with low auxiliary weights such as `0.2`. Do not
generate synthetic raw tabular rows.

## Train CatBoost OOF Logits

```bash
PYTHONPATH=src python3 -m fbm_multimodal.training.train_catboost_oof \
  --features data/tabular_features.csv \
  --labels data/labels.csv \
  --label-columns ERS_0,ERS_1,ERS_2 \
  --output-dir outputs/catboost_logits \
  --sample-id-column sample_id \
  --split-column split \
  --synthetic-column is_synthetic \
  --num-folds 5
```

The trainer uses real train samples only. Train logits are out-of-fold; valid
and test logits use fold-ensemble average predictions. CatBoost is optional at
package import time, but real training requires installing `catboost`.

The output directory contains logit tables, `metadata.json`, `warnings.txt`,
and pickled fold models under `models/class_<c>_fold_<k>.pkl`. If parquet
support is unavailable, the trainer writes CSV fallbacks next to the intended
parquet paths.

## Train Fusion Baseline

```bash
PYTHONPATH=src python3 examples/run_fusion_experiment.py
```

The current end-to-end example is still the original dependency-light baseline.
Use `ClasswiseGatedResidualFusion` when a pipeline has FBM logits, optional WL
logits, and optional CatBoost logits:

```python
from fbm_multimodal.fusion.model import ClasswiseGatedResidualFusion

combiner = ClasswiseGatedResidualFusion(num_classes=len(label_names))
fusion_logits = combiner.combine_logits(
    fbm_logits,
    wl_logits=wl_logits,
    has_wl_map=has_wl_map,
    catboost_logits=catboost_logits,
    has_catboost_logits=has_catboost_logits,
)
```

For this repo's dependency-light numpy path, `WLResidualCatBoostFusionMLP`
trains FBM and WL MLP heads and uses CatBoost logits directly in the class-wise
residual fusion formula:

```python
from fbm_multimodal.fusion.model import WLResidualCatBoostFusionMLP

model = WLResidualCatBoostFusionMLP(hidden=32, epochs=150, seed=42)
model.fit(
    fbm_features,
    wl_maps,                 # [N, C, B, T]
    y_multilabel,
    has_wl_map=has_wl_map,
    wl_loss_weight=wl_loss_weight,
    catboost_logits=catboost_logits,
    has_catboost_logits=has_catboost_logits,
)
heads = model.predict_heads(
    fbm_features,
    wl_maps,
    has_wl_map=has_wl_map,
    catboost_logits=catboost_logits,
    has_catboost_logits=has_catboost_logits,
)
```

## Evaluate Fusion

```bash
PYTHONPATH=src python3 -m fbm_multimodal.fusion \
  --predictions reports/fusion_predictions.csv \
  --labels edge_ring,center_blob,leak_top,leak_bottom \
  --identity-labels leak_top,leak_bottom
```

Official metrics remain real-only: `real_single`, `real_composite`, and their
combined `real_all` view. Synthetic rows are auxiliary diagnostics only.

Use `run_leakage_checks()` from `fbm_multimodal.fusion.fusion_eval` to flag:

- WL baselines fitted outside train real samples.
- CatBoost train logits that are not OOF.
- Synthetic rows included in CatBoost training.
- Synthetic rows marked as official metric groups.
- Pseudo-labeling enabled by mistake.
