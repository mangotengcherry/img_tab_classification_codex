# Fusion Evaluation — 팀 사용 가이드 (데이터만 갈아끼우기)

> 모델 코드를 건드리지 않고, **예측 결과 CSV 한 장만** 표준 형식으로 만들면
> 3개 head 평가 + fusion gain + collapse 진단 + 정체성 슬라이스를 바로 얻습니다.
> 개념 배경은 [multimodal_fusion_guide.md](multimodal_fusion_guide.md).

현재 브랜치의 fusion 입력은 FBM, WL residual map, CatBoost logits까지 확장되어 있습니다. 이
평가기는 prediction CSV 기준으로 동작하므로, 모델 구현이 어떤 branch를 쓰든 최종적으로 아래
확률 column만 맞추면 됩니다. WL-only/CatBoost-only branch를 별도 비교하려면 condition별 CSV를
따로 만들거나, branch 확률을 `tabular_prob_*` 역할로 넣어 단일 tabular-side baseline으로 평가하세요.

이 평가기는 core의 `evaluate-conditions`(조건별 KPI gate)와 **충돌하지 않는 별도 모듈**입니다
(`src/fbm_multimodal/fusion/`). 같은 예측 CSV를 두 평가기에 모두 넣을 수 있습니다.

---

## 1. 30초 체험 (데모 데이터)

```bash
PYTHONPATH=src python3 examples/fusion_eval_example.py
```

작은 합성 데모로 리포트가 출력됩니다. image-only는 정체성 클래스에서 무너지고(KPI≈0.08),
tabular/fusion은 성공하며, collapse follow-rate가 1.0(건강)으로 나오는 걸 확인하세요.

---

## 2. 내 데이터로 바꾸기 — **딱 한 곳만** 수정

`examples/fusion_eval_example.py` 안의 REPLACE 블록을 본인 CSV 로드로 바꿉니다.

```python
# >>> REPLACE THIS BLOCK WITH YOUR REAL DATA <<<
# predictions = build_demo_predictions()
predictions = pd.read_csv("outputs/fusion_predictions.csv")   # ← 본인 예측 CSV
# --------------------------------------------------------------------------
```

그리고 위쪽의 `LABELS` / `IDENTITY_LABELS`를 실제 label 이름으로 맞추세요.

또는 코드를 안 건드리고 **CLI**로:

```bash
PYTHONPATH=src python3 -m fbm_multimodal.fusion \
  --predictions outputs/fusion_predictions.csv \
  --labels short,leak_top,leak_bottom \
  --identity-labels leak_top,leak_bottom \
  --output outputs/fusion_report.md \
  --json-output outputs/fusion_report.json
```

---

## 3. 예측 CSV 형식 (이게 전부)

**행 = chip 1개.** label이 `short, leak_top, leak_bottom`이라면:

| 컬럼 | 설명 | 필수 |
|---|---|:---:|
| `eval_group` | `real_single` / `real_composite` / `synthetic_composite` | ✅ |
| `true_<label>` | 정답 0/1 (label마다 한 컬럼) | ✅ |
| `image_prob_<label>` | image-only head 확률 [0,1] | head별 |
| `tabular_prob_<label>` | tabular-only head 확률 [0,1] | head별 |
| `fusion_prob_<label>` | fusion head 확률 [0,1] (`prob_<label>`도 허용) | head별 |
| `chip_id` | 식별자 | 권장 |
| `wafer_id` / `lot_id` | grouped 리포트용 | 선택 |

평가기 내부에는 `real_all = real_single + real_composite` aggregate가 자동으로 추가됩니다.
`real_all`은 official real-only summary이고, synthetic row는 포함하지 않습니다.

### 핵심 규칙 — 모달리티 비대칭은 **NaN으로 표현**

`synthetic_composite` 행은 tabular가 없으므로 `tabular_prob_*` / `fusion_prob_*`를 **비워둡니다(NaN)**.
평가기는 "한 head는, 그 head의 확률 컬럼이 모두 채워진 행에서만 평가"하므로,
**tabular/fusion head는 synthetic 행을 자동으로 건너뜁니다.** 즉 비대칭을 따로 코딩할 필요가 없습니다.

예시:

```csv
chip_id,eval_group,true_short,true_leak_top,true_leak_bottom,image_prob_short,image_prob_leak_top,image_prob_leak_bottom,tabular_prob_short,tabular_prob_leak_top,tabular_prob_leak_bottom,fusion_prob_short,fusion_prob_leak_top,fusion_prob_leak_bottom
R001,real_single,1,0,0,0.92,0.5,0.5,0.88,0.1,0.1,0.95,0.05,0.05
R014,real_composite,1,0,1,0.90,0.5,0.5,0.85,0.1,0.9,0.93,0.07,0.91
S210,synthetic_composite,0,1,1,0.6,0.55,0.6,,,,,,
```

> `image_prob_*`만 채우고 나머지를 NaN으로 두면 image-only 평가만, 세 head를 다 채우면 전체 평가가 됩니다.

---

## 4. 예측 CSV 만드는 법 (모델 쪽)

학습된 모델로 **test set을 추론**해서 위 형식으로 저장하면 됩니다. 의사코드:

```python
import pandas as pd

records = []
for chip in test_set:                      # real_single / real_composite / synthetic_composite 포함
    rec = {"chip_id": chip.id, "eval_group": chip.group}
    for k, label in enumerate(LABELS):
        rec[f"true_{label}"] = int(chip.y[k])
        rec[f"image_prob_{label}"] = float(image_head_prob[k])
        if chip.has_tabular:               # synthetic이면 False -> 채우지 않음(NaN)
            rec[f"tabular_prob_{label}"] = float(tabular_head_prob[k])
            rec[f"fusion_prob_{label}"]  = float(fusion_head_prob[k])
    records.append(rec)

pd.DataFrame(records).to_csv("outputs/fusion_predictions.csv", index=False)
```

이 CSV는 그대로 core 평가기에도 넣을 수 있습니다(`prob_<label>`을 fusion으로 alias):

```bash
PYTHONPATH=src python3 -m fbm_multimodal.cli evaluate-conditions ...   # KPI gate
PYTHONPATH=src python3 -m fbm_multimodal.fusion ...                    # fusion 진단
```

---

## 5. 리포트 읽는 법

- **Subset accuracy by head × eval_group** — head별·group별 정답률 + Wilson 신뢰구간 / support.
  `real_single`, `real_composite`, `real_all`, `synthetic_composite`가 분리됩니다.
  composite support가 작으면 CI가 넓게 나옵니다(그게 정상이고, 그래서 같이 봅니다).
- **KPI product per head** — `single × composite`. fusion이 unimodal보다 높아야 의미가 있습니다.
- **Fusion gain** — `fusion − best_unimodal`. **0 근처면 fusion이 굳이 필요 없다는 신호.**
- **Modality-collapse diagnostic** —
  - `fusion_follow_rate`: tabular가 구해준 케이스를 fusion이 따라간 비율. **낮으면 collapse.**
  - `fusion_gain_over_image ≤ 0`: tabular 브랜치가 기여를 안 함 → 경고.
- **Identity-class slice** — 전기로만 구분되는 클래스에서 `tabular_minus_image`가 **양수**여야
  정상. 음수면 정체성 가설이나 tabular feature/label을 재검토해야 합니다.
- **⚠️ Warnings** — 위 위험들을 자동 요약.

모델이 있으면 **진짜 ablation**도 가능합니다:

```python
from fbm_multimodal.fusion.fusion_eval import modality_contribution
out = modality_contribution(predict_fn, images, tabular, y_true, thresholds=0.5)
# tabular_contribution ≈ 0 이면 fusion이 tabular를 무시하는 것(collapse).
```

---

## 6. 권장 워크플로

1. test set을 `real_single` / `real_composite` / `synthetic_composite`로 나눠 예측 CSV 생성.
2. `python -m fbm_multimodal.fusion`으로 fusion 리포트 생성.
3. **Warnings부터** 확인 → collapse면 modality dropout / 2-stage freeze로, 정체성 슬라이스가
   나쁘면 tabular feature/label 정의부터 점검.
4. threshold는 **real validation에서 튜닝한 값**(`{label: thr}` dict)을 넣고 **test에서는 고정**.

---

## 7. 테스트

```bash
PYTHONPATH=src python3 -m pytest tests/test_fusion_eval.py -q
```

> 이 모듈은 core 모듈을 import하지 않는 **독립 서브패키지**라, `condition_eval` 등이 동시 개발로
> 바뀌어도 영향을 받지 않습니다. (그래서 작은 헬퍼 일부를 의도적으로 자체 구현했습니다 —
> 추후 안정화되면 core와 합쳐도 됩니다.)
