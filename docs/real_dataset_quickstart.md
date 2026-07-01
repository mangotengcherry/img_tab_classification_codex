# Real Dataset Quickstart

팀원이 실제 데이터만 교체해서 첫 smoke test를 돌리는 절차입니다. 실제 원본 데이터는 커밋하지 말고 `data/raw/`와 `data/metadata/` 아래에 로컬로 둡니다.

## 1. 파일 배치

```text
data/raw/fbm_tensor/
  fbm_images.npy
  fbm_manifest.csv
  label_map.json

data/raw/eds_tabular/
  eds_tabular.csv

data/metadata/
  eds_test_item_wordline_map.csv
```

`fbm_manifest.csv` 최소 컬럼:

```csv
row_idx,sample_id,split,eval_group,label_ERS_0,label_ERS_1
0,CHIP_000001,train,real_single,1,0
```

`eds_tabular.csv` 최소 컬럼:

```csv
sample_id,split,eval_group,label_ERS_0,label_ERS_1,EDS_RD_WL000,EDS_GLOBAL_IDDQ
CHIP_000001,train,real_single,1,0,12.4,5.1
```

`eds_test_item_wordline_map.csv` 최소 컬럼:

```csv
feature_name,eds_step,eds_item,wordline_position,value_direction,include_in_catboost,notes
EDS_RD_WL000,READ,RD_LEAK,0,high_bad,1,
EDS_GLOBAL_IDDQ,IDDQ,IDDQ_TOTAL,,high_bad,1,global scalar
```

## 2. EDS Mapping 검증

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli validate-eds-map \
  --eds data/raw/eds_tabular/eds_tabular.csv \
  --mapping data/metadata/eds_test_item_wordline_map.csv \
  --label-columns label_ERS_0,label_ERS_1
```

확인할 것:

- `mapped_features`: mapping table row 수
- `wl_map_features`: WL residual map으로 변환될 feature 수
- `catboost_features`: CatBoost scalar input 후보 수
- `low_bad_features`: sign flip이 필요한 feature 수

## 3. WL Measurements 생성

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli build-wl-measurements \
  --eds data/raw/eds_tabular/eds_tabular.csv \
  --mapping data/metadata/eds_test_item_wordline_map.csv \
  --output data/interim/wl_measurements.csv
```

생성 결과는 long-form입니다.

```csv
sample_id,split,eval_group,test_method,wordline,value,feature_name,test_item,is_synthetic
CHIP_000001,train,real_single,READ,0,12.4,EDS_RD_WL000,RD_LEAK,False
```

## 4. FBM + EDS Join Smoke Test

```bash
PYTHONPATH=src python3 - <<'PY'
from fbm_multimodal.fusion.real_data import load_fbm_tensor_dataset, load_eds_tabular, build_fusion_manifest

images, fbm_manifest, label_map = load_fbm_tensor_dataset("data/raw/fbm_tensor")
eds = load_eds_tabular("data/raw/eds_tabular/eds_tabular.csv")
fusion_manifest = build_fusion_manifest(
    fbm_manifest,
    eds,
    label_columns=label_map["label_columns"],
)

print("images:", images.shape)
print("samples:", len(fusion_manifest))
print(fusion_manifest[["sample_id", "has_fbm_image", "has_eds_tabular", "has_wl_map", "has_catboost_logits"]].head())
PY
```

## 5. WL Residual Tensorizer / Cache

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
print("fit_samples:", len(tensorizer.fit_sample_ids_))
print("maps:", len(maps))
PY
```

`fit()`은 `split`, `eval_group`, `is_synthetic` column이 있으면 train real row만 baseline에 사용합니다.

## 6. CatBoost OOF Logits

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

Train logits는 OOF만 저장되고, validation/test는 fold ensemble 평균으로 생성됩니다. Synthetic row는
CatBoost training에서 제외됩니다.

## 7. Full Test Suite

```bash
PYTHONPATH=src python3 -m pytest -q
```

현재 환경에서 parquet engine이 없으면 CatBoost OOF 테스트가 `.parquet` 대신 `.csv` fallback warning을 낼 수 있습니다. 이는 실패가 아니라 optional dependency warning입니다.
