# Real Dataset Layout

실제 데이터 파일은 이 저장소에 커밋하지 않는 것을 권장합니다. 아래 위치와 파일명을 표준으로 맞추면 Roo 또는 후속 작업자가 loader/refactor 작업을 일관되게 진행할 수 있습니다.

```text
data/
  raw/
    fbm_tensor/
      fbm_images.npy              # or fbm_images.pt, shape [N,H,W] or [N,C,H,W]
      fbm_manifest.csv            # row_idx, sample_id, split, label columns
      label_map.json              # class index/name mapping
    eds_tabular/
      eds_tabular.parquet         # preferred; CSV also OK
      eds_tabular.csv             # optional fallback
  metadata/
    eds_test_item_wordline_map.csv
    split_manifest.csv            # optional canonical split override
  interim/
    wl_measurements.parquet       # generated long-form measurement table
    wl_residual_tensorizer.json   # generated train-only WL baseline
    wl_maps.npz                   # generated sample_id -> [C,B,T] WL tensors
    fusion_manifest.parquet       # generated joined FBM/EDS manifest
```

핵심 원칙:

- 모든 데이터셋은 같은 `sample_id`로 join 가능해야 합니다.
- `split`은 `train`, `valid`, `test` 중 하나로 고정합니다.
- label column은 FBM manifest와 EDS tabular에서 같은 이름을 사용합니다.
- EDS test item -> WL 위치 매핑은 `data/metadata/eds_test_item_wordline_map.csv`에 한 row per EDS feature column으로 작성합니다.
- validation/test row는 WL residual baseline median/IQR 계산에 절대 사용하지 않습니다.
