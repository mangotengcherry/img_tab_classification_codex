# FBM Multi-Modal Classification Experiment Plan

## Summary

- 목표는 FBM image와 WL/electrical tabular feature를 함께 사용해 단일 불량, 실제 중첩 불량, 이미지상 유사하지만 전기 특성으로만 구분되는 불량을 안정적으로 분류하는 것이다.
- 기존 image-only + synthetic composite 학습은 유지하되, synthetic-to-real gap을 별도 지표로 관리한다.
- `MSR_000` ~ `MSR_200` 번호 순서는 물리 WL 순서로 가정하지 않는다. 별도 mapping table로 feature와 WL/측정조건/물리 위치를 연결한다.
- 최종 KPI는 `single subset acc * composite subset acc`를 유지하고, 보조 지표로 per-class F1/recall, class-pair subset acc, real-vs-synthetic gap, label cost 대비 성능 상승량을 본다.
- 목표값 사이에는 수학적 긴장점이 있다. `single=0.8`, `composite=0.6`만 만족하면 product는 `0.48`이므로, product `0.65`를 달성하려면 둘 중 하나가 더 높아야 한다. 예를 들어 single이 `0.8`이면 composite은 최소 `0.8125`가 필요하다.

## Key Experiments

### E0: Evaluation Frame

- split은 1차로 `chip random split`을 사용한다.
- test set은 `real single`, `real composite`, `synthetic composite`를 분리한다.
- 모델 선택은 synthetic validation이 아니라 real validation 성능 기준으로 한다.
- threshold는 `0.5 fixed`, `class-wise threshold`, `class-pair-sensitive threshold`를 비교한다.
- 조건별 최종 판정은 `condition`, `eval_group`, `true_<label>`, `prob_<label>` 형식의 prediction CSV를 `evaluate-conditions` CLI에 넣어 수행한다.

### E1: Image-Only Baseline

- 입력 image는 `128x46`, intensity `0~8`을 `/8` 정규화한다.
- 기존 CNN, spatial attention CNN, synthetic composite 학습 조건을 재현한다.
- 좌우 flip은 유지 후보로 둔다.
- 상하 flip은 물리 위치 의미를 바꿀 수 있으므로, 전기 feature mapping과 함께 뒤집을 수 있는 경우에만 별도 ablation으로 평가한다.
- synthetic holdout 성능과 real composite 성능 차이를 class-pair별로 기록한다.

### E2: Tabular-Only Baseline

- `MSR_000` ~ `MSR_200`은 단순 순서 feature가 아니라 measurement feature set으로 취급한다.
- mapping table을 사용해 각 MSR feature에 `wl_index`, `physical_region`, `measurement_condition`, `measurement_type`을 붙인다.
- baseline은 LightGBM/CatBoost 또는 sklearn one-vs-rest, MLP, metadata-aware MLP, WL-position-aware 1D encoder를 비교한다.
- 이미지상 유사하지만 LTI/leakage의 상/중/하 위치로 구분되는 class group을 별도 evaluation slice로 둔다.

### E3: Multi-Modal Fusion

- 1차 추천 구조는 `image encoder + mapped-tabular encoder + fusion head`이다.
- tabular encoder는 raw MSR 순서를 쓰지 않고, mapping table 기반으로 WL/region/condition embedding을 주입한다.
- 모델은 `image-only head`, `tabular-only head`, `fusion head`를 함께 출력한다.
- synthetic image는 image branch와 image auxiliary head 학습에만 사용한다.
- tabular/fusion head는 real labeled data 중심으로 학습하고, synthetic에는 tabular를 임의 생성하지 않는다.
- late fusion calibrator도 비교한다: image model logits + tabular model logits + metadata를 입력으로 label별 calibration 모델을 학습한다.

### E4: Synthetic-to-Real Gap Reduction

- 합성 방식은 `max`, `clipped sum`, `weighted saturating sum`을 비교한다.
- 현재 사용하는 합성-원본 유사도 threshold는 real composite validation 기준으로 sweep한다.
- synthetic sample은 real embedding과 가까운 정도에 따라 loss weight를 다르게 주는 실험을 포함한다.
- 보고서에는 `synthetic composite acc`, `real composite acc`, `gap`을 class-pair별로 기록한다.

### E5: Label Cost Reduction

- unlabeled 수천~수만 chip에 대해 teacher ensemble로 pseudo-label 후보를 만든다.
- 라벨 요청 우선순위는 `high-confidence target class`, `image-tabular disagreement`, `high uncertainty`, `embedding cluster representative`를 섞어 선정한다.
- pseudo-label은 high-confidence 샘플만 낮은 loss weight로 학습에 포함한다.
- validation/test에는 pseudo-label을 절대 포함하지 않는다.
- active learning simulation으로 random sampling 대비 KPI 목표치에 도달하는 라벨 수를 비교한다.

## Interfaces And Artifacts

### Data Manifest

Chip 단위 manifest는 다음 컬럼을 가진다.

- `chip_id`
- `image_path`
- `label_vector` 또는 defect label columns
- `is_real`
- `is_synthetic`
- `is_pseudo_labeled`
- `label_cardinality`
- `wafer_position`
- `split`
- `MSR_*` feature columns

### Measurement Mapping Table

권장 파일명은 `measurement_map.csv`이다.

필수 컬럼:

- `feature_name`
- `measurement_condition`
- `measurement_type`

위치 정보 컬럼:

- `wl_index`
- `physical_region`
- `physical_order`

규칙:

- `feature_name`은 `MSR_000` 같은 원천 feature명과 정확히 매칭한다.
- `wl_index`는 실제 WL 위치가 확인된 경우에만 사용한다.
- `physical_region`은 최소 `top`, `middle`, `bottom`, `unknown` 중 하나로 둔다.
- `physical_order`는 raw feature 정렬용이 아니라 위치 metadata로만 사용한다.

### Model Outputs

모든 모델은 label별로 다음 값을 저장한다.

- `probability`
- `binary_prediction`
- `threshold`
- `uncertainty`

Multi-modal 모델은 추가로 다음 값을 저장한다.

- `image_only_probability`
- `tabular_only_probability`
- `fusion_probability`
- `human_review_flag`

### Reports

실험 리포트는 다음 항목을 포함한다.

- single/composite subset acc
- KPI product
- per-class F1/recall
- class-pair acc
- threshold table
- synthetic-to-real gap table
- label cost curve

## Test Plan

- image intensity가 `0~8` 범위에서 `/8`로 정규화되는지 확인한다.
- `MSR_*` feature가 raw 번호 순서로 물리 위치 처리되지 않는지 확인한다.
- `measurement_map.csv`의 feature coverage를 계산하고, mapping 누락 feature 비율을 기록한다.
- real test set에 synthetic sample이나 pseudo-label sample이 섞이지 않는지 확인한다.
- 상하 flip ablation에서는 image와 mapped physical region이 함께 변환되는 조건과 비활성 조건을 분리한다.
- 모든 핵심 실험은 최소 3개 seed로 반복하고 평균, 표준편차, confidence interval을 기록한다.
- active learning은 동일 라벨 budget에서 random sampling, uncertainty sampling, disagreement sampling, cluster representative sampling을 비교한다.
- 조건 평가기는 `single >= 0.8`, `composite >= 0.6`, `single * composite >= 0.65`를 별도 gate로 보고, 개별 최소값은 만족하지만 product를 만족하지 못하는 조건을 실패로 분류한다.

## Assumptions

- `MSR_000` ~ `MSR_200` 번호 자체는 물리 WL 순서를 의미하지 않는다.
- feature와 WL/측정조건을 연결하는 mapping table은 제공 가능하다.
- synthetic tabular는 생성하지 않는다.
- 실제 중첩 데이터가 적으므로, 단일 subset acc와 중첩 subset acc 외에 class-pair별 결과와 synthetic-to-real gap을 반드시 함께 판단한다.
- 1차 평가는 `chip random split`으로 진행하고, wafer/lot/time metadata가 확보되면 추가 stress test로 확장한다.
