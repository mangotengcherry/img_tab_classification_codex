# FBM 멀티모달 실험 도구

이 저장소는 FBM image와 WL/electrical measurement tabular feature를 함께 사용하는 multi-label classification 실험을 위한 경량 도구 모음입니다.

`plan.md`에 정리된 현재 실험 계약과 평가 유틸리티를 코드로 제공합니다. 현재 기준 구조는
**FBM image + WL high-side residual map + CatBoost OOF logits + class-wise gated residual fusion**입니다.

- `0..8` grade로 표현된 FBM image intensity 정규화
- `MSR_*` suffix 순서를 물리 위치로 가정하지 않는 measurement mapping 검증
- multi-label metric, class-wise threshold 탐색, class-pair metric, synthetic-to-real gap 리포트
- synthetic FBM 합성 모드
- unlabeled chip review를 위한 active-learning ranking
- mapping coverage, review candidate 선정, 조건별 성능 평가용 CLI
- EDS feature-to-WL mapping 검증, long-form WL measurement 생성, WL residual map/CatBoost fusion scaffold

## 설치

현재 workspace에서 사용할 수 있는 Python 환경 기준으로 설치합니다.

```bash
python3 -m pip install -e .
```

## 테스트 실행

```bash
PYTHONPATH=src python3 -m pytest -q
```

## CLI 예시

chip manifest의 모든 `MSR_*` column이 `measurement_map.csv`에 매핑되어 있는지 확인합니다.

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli validate-map \
  --manifest data/manifest.csv \
  --measurement-map data/measurement_map.csv
```

unlabeled chip 중 엔지니어 review를 요청할 후보를 우선순위화합니다.

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli rank-unlabeled \
  --candidates outputs/unlabeled_predictions.csv \
  --labels defect_a,defect_b,defect_c \
  --target-labels defect_a,defect_b \
  --budget 100 \
  --output outputs/review_queue.csv
```

candidate prediction CSV에 기대하는 probability column은 다음과 같습니다.

- `prob_<label>`
- 선택 column: `image_prob_<label>`
- 선택 column: `tabular_prob_<label>`

현재 subset accuracy 목표 기준으로 실험 조건별 성능을 평가합니다.

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli evaluate-conditions \
  --predictions outputs/condition_predictions.csv \
  --labels defect_a,defect_b,defect_c \
  --output outputs/condition_summary.csv \
  --single-target 0.8 \
  --composite-target 0.6 \
  --kpi-target 0.65
```

조건별 CSV가 따로 있을 때는 glob으로 한 번에 평가할 수 있습니다. 파일에 `condition` column이 없으면 파일명이 condition 이름으로 사용됩니다.

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

scalar threshold 후보를 sweep하고, condition별로 가장 좋은 threshold를 선택합니다.

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

prediction에 `seed` 같은 repeated run column이 있으면 run별 summary와 condition-level aggregate summary를 함께 저장합니다.

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli evaluate-conditions \
  --prediction-glob "outputs/conditions/*.csv" \
  --labels defect_a,defect_b,defect_c \
  --output outputs/condition_by_seed.csv \
  --report-output outputs/condition_report.md \
  --run-column seed \
  --aggregate-output outputs/condition_aggregate.csv \
  --threshold-grid 0.30,0.35,0.40,0.45,0.50,0.55,0.60 \
  --single-target 0.8 \
  --composite-target 0.6 \
  --kpi-target 0.65
```

aggregate report에는 평균, 표준편차, 최소값, 최대값, 모든 run이 목표를 만족했는지 여부가 포함됩니다.
Markdown report에는 PASS/FAIL, 추천 condition, 선택 threshold, 목표를 만족하지 못했을 때의 KPI gap이 기록됩니다.

CI 또는 서비스 투입 가능 여부를 gate로 판단하려면 `--fail-on-miss`를 추가합니다. 이 옵션을 사용해도 CSV/Markdown artifact는 저장되며, single/composite/KPI 목표를 모두 만족하는 condition이 없으면 exit code `2`로 종료됩니다.

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli evaluate-conditions \
  --prediction-glob "outputs/conditions/*.csv" \
  --labels defect_a,defect_b,defect_c \
  --output outputs/condition_summary.csv \
  --report-output outputs/condition_report.md \
  --threshold-grid 0.30,0.35,0.40,0.45,0.50,0.55,0.60 \
  --single-target 0.8 \
  --composite-target 0.6 \
  --kpi-target 0.65 \
  --fail-on-miss
```

repeated seed를 사용할 때는 `--require-all-runs`를 추가하면, 하나 이상의 condition이 모든 run에서 목표를 만족해야 gate가 통과됩니다.

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli evaluate-conditions \
  --prediction-glob "outputs/conditions/*.csv" \
  --labels defect_a,defect_b,defect_c \
  --output outputs/condition_by_seed.csv \
  --aggregate-output outputs/condition_aggregate.csv \
  --run-column seed \
  --threshold-grid 0.30,0.35,0.40,0.45,0.50,0.55,0.60 \
  --single-target 0.8 \
  --composite-target 0.6 \
  --kpi-target 0.65 \
  --fail-on-miss \
  --require-all-runs
```

실패 원인 분석을 위한 세부 slice도 별도 CSV로 저장할 수 있습니다.

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli evaluate-conditions \
  --predictions outputs/condition_predictions.csv \
  --labels defect_a,defect_b,defect_c \
  --output outputs/condition_summary.csv \
  --per-class-output outputs/condition_per_class.csv \
  --class-pair-output outputs/condition_class_pair.csv \
  --single-target 0.8 \
  --composite-target 0.6 \
  --kpi-target 0.65
```

prediction CSV에 기대하는 column은 다음과 같습니다.

- `condition`: `image_only_synth`, `catboost_logit_fusion`, `wl_residual_catboost_fusion` 같은 실험 조건 이름
- `chip_id`
- `eval_group`: `real_single`, `real_composite`, `synthetic_composite` 중 하나
- 모든 label에 대한 `true_<label>`
- 모든 label에 대한 `prob_<label>`, 또는 binary prediction만 있을 때는 `pred_<label>`

`--threshold-grid`는 `prob_<label>` column이 있을 때만 결과에 영향을 줍니다. `pred_<label>` 입력은 이미 binary prediction이므로 evaluator가 해당 값을 그대로 사용합니다.

evaluator가 리포트하는 항목은 다음과 같습니다.

- 단일 불량 subset accuracy
- 실제 중첩 불량 subset accuracy
- synthetic composite subset accuracy
- synthetic-to-real composite gap
- 선택 출력: per-class precision, recall, F1, positive support CSV
- 선택 출력: class-pair support와 subset accuracy CSV
- `single_subset_accuracy * composite_subset_accuracy`
- 선택된 `threshold`
- single, composite, KPI, 전체 목표에 대한 pass/fail flag
- 관측된 single accuracy에서 KPI target을 만족하기 위해 필요한 composite accuracy

개별 최소 목표를 만족한다고 해서 product KPI가 자동으로 만족되는 것은 아닙니다.

```text
0.8 * 0.6 = 0.48
```

product KPI `0.65`를 만족하려면, single subset accuracy가 `0.8`인 condition은 composite subset accuracy가 최소 `0.8125` 이상이어야 합니다.

## 멀티모달 Fusion 실험 (image + tabular)

> "이미지는 합성·증강이 가능하지만 tabular(전기 MSR)는 합성 불가인데, 둘을 fusion해 학습이 되는가?"
> 에 답하는, 바로 실행 가능한 end-to-end 실험입니다. 전체 리포트:
> **[reports/fusion_experiment_report.md](reports/fusion_experiment_report.md)** ·
> 설계 배경: [docs/multimodal_fusion_guide.md](docs/multimodal_fusion_guide.md) ·
> 평가기 사용법: [docs/fusion_eval_quickstart.md](docs/fusion_eval_quickstart.md)

핵심은 모델 크기가 아니라 **학습 스킴**입니다.

- **Loss masking** — synthetic(이미지만) 샘플은 image head만, real 샘플만 tabular/fusion head를 학습.
- **Modality dropout** — fusion이 (synthetic으로 풍부한) image 쪽으로 collapse하지 못하게 강제.
- 세 head(image-only / tabular-only / fusion)를 동시에 평가하고 collapse·정체성 슬라이스를 진단.
- **WL residual map + CatBoost logits** — raw tabular row를 합성하지 않고, WL residual map은 낮은 weight로만 합성 보조에 사용.
- WL residual map + CatBoost OOF logit branch 확장 가이드:
  [docs/wl_residual_catboost_fusion.md](docs/wl_residual_catboost_fusion.md)
- 실제 FBM tensor + EDS tabular 데이터 교체 quickstart:
  [docs/real_dataset_quickstart.md](docs/real_dataset_quickstart.md) ·
  상세 Roo handoff: [docs/real_dataset_roo_refactor_guide.md](docs/real_dataset_roo_refactor_guide.md)

### 실행

```bash
PYTHONPATH=src python3 examples/run_fusion_experiment.py      # 학습→평가→시각화 (numpy만, ~1초)
PYTHONPATH=src python3 -m fbm_multimodal.fusion \             # 예측 CSV만으로 진단 리포트
  --predictions reports/fusion_predictions.csv \
  --labels edge_ring,center_blob,leak_top,leak_bottom \
  --identity-labels leak_top,leak_bottom
```

실데이터로 바꾸려면 `examples/run_fusion_experiment.py`의 `generate_dataset()` 한 줄만 본인
loader로 교체하면 됩니다(컬럼 형식은 quickstart 참고). 같은 예측 CSV를 위 `evaluate-conditions`에도 넣을 수 있습니다.

### 결과 한눈에

| head | single acc | composite acc | **KPI product** |
|---|---|---|---|
| image_only | 0.70 | 0.65 | 0.45 |
| tabular_only | 1.00 | 0.50 | 0.50 |
| **fusion** | **1.00** | **0.88** | **0.88** |

fusion KPI **0.88** ≫ best unimodal **0.50** (gain **+0.38**). 정체성 클래스(이미지로 동일)에서
tabular가 image를 **+0.38** 앞서고, 진짜 ablation에서 tabular 기여 **+0.39** → fusion이 tabular를
실제로 사용(collapse 아님).

![dataset](reports/figures/01_dataset_overview.png)
![kpi](reports/figures/04_kpi_product.png)
![identity & collapse](reports/figures/05_identity_and_collapse.png)

## Measurement Map

`measurement_map.csv`는 다음 column을 포함하는 형태를 권장합니다.

```csv
feature_name,measurement_condition,measurement_type,wl_index,physical_region,physical_order
MSR_000,read,leakage,100,top,2
MSR_001,read,leakage,0,bottom,0
```

`physical_order`는 position-aware experiment를 위한 metadata입니다. 이 코드는 `MSR_*` suffix에서 물리 순서를 추론하지 않습니다.
