# FBM + WL Residual Map + CatBoost Fusion Experiment Plan

Last updated: 2026-07-01

## Summary

이 브랜치의 현재 기준은 기존 FBM fusion 구조를 유지하면서 tabular 경로를 두 갈래로 확장하는 것이다.

- FBM image -> FBM encoder/head
- WL raw measurement -> train-real-only high-side residual map -> WL map encoder/head
- raw EDS/tabular scalar features -> one-vs-rest CatBoost -> OOF/fold-ensemble logits
- FBM logits + WL logits + CatBoost logits -> class-wise gated residual fusion

중요한 비목표는 그대로 유지한다.

- synthetic raw 417-feature tabular row를 만들지 않는다.
- validation/test로 WL median/IQR baseline을 fit하지 않는다.
- CatBoost train prediction은 in-fold prediction이 아니라 OOF만 쓴다.
- pseudo-labeling은 scaffold만 있고 기본값은 off다.
- official metric에는 synthetic sample을 넣지 않는다.

구현 기준 config는 `configs/wl_residual_catboost_fusion.yaml`, 상세 사용법은
`docs/wl_residual_catboost_fusion.md`와 `docs/real_dataset_quickstart.md`를 따른다.

## Current Implementation Map

| Area | Current file | Status |
|---|---|---|
| WL residual tensorizer | `src/fbm_multimodal/wl_residual_map.py` | implemented |
| synthetic WL map composer | `src/fbm_multimodal/synthetic_wl_map.py` | implemented |
| CatBoost OOF logits | `src/fbm_multimodal/training/train_catboost_oof.py` | implemented |
| pairwise top-K pseudo-label scaffold | `src/fbm_multimodal/pseudo_labeling/pairwise_topk.py` | implemented, default off |
| fusion dataset masks | `src/fbm_multimodal/fusion/data.py` | implemented |
| gated residual fusion / numpy model | `src/fbm_multimodal/fusion/model.py` | implemented |
| fusion evaluation + leakage checks | `src/fbm_multimodal/fusion/fusion_eval.py` | implemented |
| EDS mapping + WL measurement conversion | `src/fbm_multimodal/eds_mapping.py`, `src/fbm_multimodal/cli.py` | implemented |
| real FBM/EDS loader smoke path | `src/fbm_multimodal/fusion/real_data.py` | implemented |
| full real training pipeline | project-specific integration | TODO |

## Data Contract

### FBM Tensor Dataset

Recommended local layout:

```text
data/raw/fbm_tensor/
  fbm_images.npy
  fbm_manifest.csv
  label_map.json
```

`fbm_manifest.csv` minimum columns:

```csv
row_idx,sample_id,split,eval_group,label_ERS_0,label_ERS_1
0,CHIP_000001,train,real_single,1,0
```

### EDS Tabular Dataset

Recommended local layout:

```text
data/raw/eds_tabular/eds_tabular.csv
data/metadata/eds_test_item_wordline_map.csv
```

`eds_tabular.csv` is wide-form:

```csv
sample_id,split,eval_group,label_ERS_0,label_ERS_1,EDS_RD_WL000,EDS_GLOBAL_IDDQ
CHIP_000001,train,real_single,1,0,12.4,5.1
```

`eds_test_item_wordline_map.csv` is feature-level metadata:

```csv
feature_name,eds_step,eds_item,wordline_position,value_direction,include_in_catboost,notes
EDS_RD_WL000,READ,RD_LEAK,0,high_bad,1,single WL feature
EDS_GLOBAL_IDDQ,IDDQ,IDDQ_TOTAL,,high_bad,1,global scalar
```

Rules:

- `wordline_position` exists -> feature can enter WL residual maps.
- blank `wordline_position` + `include_in_catboost=1` -> scalar CatBoost-only feature.
- `value_direction=low_bad` is sign-flipped before residualization so all WL residuals remain high-side.
- label columns and metadata columns are not CatBoost features.

## Residual Definition

For raw measurement `x`, test method `t`, and WL bin `b`:

```text
R = max(0, (x - median_train_real(t, b)) / (IQR_train_real(t, b) + eps))
```

Default tensor shape:

```text
[C, B, T]
C = mean_residual, max_residual, std_residual, observed_mask, count_ratio, source_count_norm
B = WL bins, default 20
T = discovered train test methods unless configured
```

Fit scope is train real only. If IQR is zero/missing, fallback order is test-method IQR -> global IQR -> 1.0.

## Synthetic WL Policy

Synthetic composite WL maps are composed from parent residual maps:

```text
value channels      = max(parent_a, parent_b)
observed_mask       = observed_a OR observed_b
source_count_norm   = (observed_a + observed_b) / 2
default loss weight = 0.2
```

This is an auxiliary training signal only. It is not a synthetic raw tabular row.

## CatBoost OOF Policy

CatBoost is trained offline as one binary classifier per class.

- Train rows: real train only.
- Synthetic rows: excluded.
- Train logits: OOF only.
- Validation/test logits: fold-ensemble mean probability, converted to logit.
- Output directory: `outputs/catboost_logits/`.
- Artifacts: train/eval logit tables, `metadata.json`, `warnings.txt`, and pickled fold models under `models/`.

If parquet support is unavailable, the trainer writes CSV fallbacks and records a warning.

## Fusion Model Policy

Current implemented fusion formula:

```text
fusion_logits =
    fbm_logits
    + has_wl_map * gate_wl[class] * wl_logits
    + has_catboost_logits * gate_cat[class] * catboost_logits
```

`gate_wl` and `gate_cat` are class-wise, not global modality gates. CatBoost logits are direct offline logits; the neural model does not train CatBoost itself.

Default loss intent:

| Sample type | FBM loss | WL loss | Fusion loss | CatBoost neural loss |
|---|---:|---:|---:|---:|
| real_single | 1.0 | 1.0 | 1.0 | 0 |
| real_composite | 1.0 | 1.0 | 1.0 | 0 |
| synthetic_composite image-only | 1.0 | 0 | 0 | 0 |
| synthetic_composite with synthetic WL | 1.0 | 0.2 | 0.2 | 0 |

## Evaluation Policy

Official groups:

- `real_single`
- `real_composite`
- `real_all` = real_single + real_composite

Synthetic rows are auxiliary diagnostics only. Fusion gain is computed from real single/composite KPI, not synthetic rows.

Required leakage checks:

1. WL baseline fit sample IDs are a subset of train real sample IDs.
2. CatBoost train logits metadata says `train_prediction_mode=oof`.
3. CatBoost metadata says synthetic rows were excluded.
4. Synthetic rows are not marked as official metric rows.
5. Pseudo-labeling remains disabled unless explicitly enabled.

## Experiment Sequence

### E0. Existing Baseline Reproduction

Run the current numpy FBM + tabular fusion demo and condition evaluator.

Purpose:

- Preserve the previous baseline behavior.
- Confirm report generation and tests still pass.

### E1. CatBoost Logit Branch

Use FBM logits + CatBoost OOF/fold-ensemble logits. Do not use WL residual maps.

Purpose:

- Measure scalar EDS/tabular information gain.
- Verify OOF train logits and synthetic exclusion.

### E2. WL Residual Map Branch

Use FBM + WL residual map + CatBoost logits on real samples. Do not use synthetic WL maps yet.

Purpose:

- Check whether WL-profile patterns add information beyond scalar CatBoost features.
- Inspect missing coverage and source/count channels.

### E3. Synthetic WL Residual Map

Use FBM synthetic composites plus synthetic WL residual maps from parent max/union composition.

Purpose:

- Test whether low-weight synthetic WL helps scarce real composite performance without raw tabular generation.

### E4. Synthetic WL Weight Ablation

Sweep:

```text
synthetic_wl_weight = 0.0 / 0.1 / 0.2 / 0.3
synthetic_fusion_weight = 0.0 / 0.1 / 0.2
```

Purpose:

- Select a conservative auxiliary weight based on real composite validation, not synthetic validation.

### E5. Pseudo-labeling Off-State Check

Keep:

```yaml
pseudo_labeling:
  enabled: false
```

Purpose:

- Confirm no unlabeled loader or pseudo-label sample injection is called.
- Keep `select_pairwise_topk()` tested for future use only.

## Commands

Validate EDS mapping:

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli validate-eds-map \
  --eds data/raw/eds_tabular/eds_tabular.csv \
  --mapping data/metadata/eds_test_item_wordline_map.csv \
  --label-columns label_ERS_0,label_ERS_1
```

Build long-form WL measurements:

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli build-wl-measurements \
  --eds data/raw/eds_tabular/eds_tabular.csv \
  --mapping data/metadata/eds_test_item_wordline_map.csv \
  --output data/interim/wl_measurements.csv
```

Fit/cache WL residual maps:

```bash
PYTHONPATH=src python3 - <<'PY'
import pandas as pd
from fbm_multimodal.wl_residual_map import WLResidualMapTensorizer

measurements = pd.read_csv("data/interim/wl_measurements.csv")
tensorizer = WLResidualMapTensorizer(num_wl_bins=20, clip_max=10.0)
tensorizer.fit(measurements)
tensorizer.save("data/interim/wl_residual_tensorizer.json")
maps = tensorizer.transform(measurements)
tensorizer.save_tensor_cache(maps, "data/interim/wl_maps.npz")
print(len(tensorizer.fit_sample_ids_), len(maps))
PY
```

Train CatBoost OOF logits:

```bash
PYTHONPATH=src python3 -m fbm_multimodal.training.train_catboost_oof \
  --features data/raw/eds_tabular/eds_tabular.csv \
  --labels data/raw/eds_tabular/eds_tabular.csv \
  --label-columns label_ERS_0,label_ERS_1 \
  --output-dir outputs/catboost_logits \
  --sample-id-column sample_id \
  --split-column split \
  --synthetic-column is_synthetic \
  --num-folds 5
```

Run tests:

```bash
PYTHONPATH=src python3 -m pytest -q
```

## Acceptance Criteria

- Existing baseline tests remain green.
- WL tensors are `[C, B, T]` and fit baseline from train real only.
- Synthetic WL maps use union mask and source_count channel.
- CatBoost train logits are OOF and synthetic rows are excluded.
- Fusion path can consume FBM features, WL maps, and CatBoost logits with masks.
- Pseudo-labeling default remains off.
- Official metrics remain real-only: `real_single`, `real_composite`, `real_all`.

## Assumptions

- Real raw data is not committed; local files live under `data/raw/`.
- The repo still uses a dependency-light numpy fusion implementation for smoke tests.
- CatBoost is optional; install `catboost` only for real CatBoost training.
- Parquet output requires `pyarrow` or `fastparquet`; otherwise CSV fallback is expected.
