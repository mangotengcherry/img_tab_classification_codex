# Image + Tabular Fusion under Modality Asymmetry

> 이미지(FBM)는 합성·증강이 가능하지만 tabular(전기 MSR 측정)는 합성이 불가능한 상황에서,
> image와 tabular를 fusion해 학습하는 방법과 그 함정에 대한 참고 문서.
>
> 평가 코드: `src/fbm_multimodal/fusion/` · 사용법: `docs/fusion_eval_quickstart.md`

---

## 0. 결론

**가능합니다.** 단, "image+tabular를 한 head에 concat해서 전부 같이 학습"하는 순진한 방식은
안 됩니다. 각 데이터를 **자기가 valid한 경로로만** 흘려보내는 구조 + **missing-modality 학습**이
필요합니다.

진짜 제약은 "tabular를 못 만든다"가 아니라 두 가지입니다.

1. **모달리티 비대칭** — synthetic 샘플은 tabular가 없어 fusion/tabular 경로로 못 흐른다.
2. **데이터량 비대칭** — fusion이 실제로 배울 수 있는 (image ∧ tabular ∧ label) 샘플,
   특히 **real composite**가 극소수다.

---

## 1. 데이터 구조부터 정확히

| 샘플 종류 | image | tabular | label | 양 | 어떤 head를 학습시키나 |
|---|:---:|:---:|:---:|---|---|
| real single | ✓ | ✓ | ✓ | 많음 | image / tabular / fusion 전부 |
| **real composite** | ✓ | ✓ | ✓ | **극소수 ← 병목** | image / tabular / fusion 전부 |
| synthetic composite | ✓ | ✗ | ✓ | 많음 | **image head만** |
| unlabeled | ✓ | ✓ | ✗ | 매우 많음 | 사전학습(self-supervised), pseudo-label |

> Fusion head가 학습되려면 image·tabular·label이 **동시에** 있어야 하고 그건 real뿐입니다.
> 그래서 "fusion이 가능한가"는 결국 **scarce real(특히 composite)로 fusion을 어떻게 안정적으로
> 맞추느냐**의 문제로 환원됩니다.

---

## 2. 가능하게 만드는 4가지 방법 (안전한 순서)

### ① Loss masking — 각 샘플을 valid한 head로만 흘린다 *(학습 자체는 바로 가능)*

head를 3개(image-only / tabular-only / fusion) 두고, **각 샘플이 입력을 가진 head에만 loss를 준다.**

- synthetic composite (image만) → **image head loss만.** tabular·fusion head는 gradient 마스킹.
- real (image+tabular) → 3개 head 전부.

학습은 문제없이 돌아갑니다. synthetic이 fusion head를 오염시키지 않으면서 image encoder는
풍부하게 학습됩니다. "missing modality via loss masking"은 표준 기법입니다.

```python
# 개념용 의사코드 (프레임워크 무관)
img_logit, tab_logit, fus_logit = model(image, tabular)   # tabular 없으면 null 토큰
loss = bce(img_logit, y)                                   # image head: 항상
if has_tabular:                                            # synthetic이면 False
    loss = loss + bce(tab_logit, y) + bce(fus_logit, y)    # tabular/fusion head: real만
loss.backward()
```

### ② 2-stage + encoder freeze — *scarce real 문제의 가장 큰 레버*

적은 real composite로 **딥 encoder까지** 학습시키니 과적합합니다. 분리하세요.

1. **image encoder**: synthetic + unlabeled로 사전학습(self-supervised 또는 single-defect supervised).
2. **tabular encoder**: unlabeled + real tabular로 사전학습.
3. 그다음 **두 encoder는 freeze(또는 low-LR)** 하고 **작은 fusion head(선형~2층)만** scarce real로 학습.

→ 적은 paired 데이터가 딥 네트워크가 아니라 **저용량 head만** 맞추면 되니 과적합이 급감합니다.

### ③ Late fusion calibrator — *가장 데이터 효율적 (plan E3에 이미 있음)*

각 unimodal 모델을 독립 학습(image는 image+synthetic, tabular는 real) → 각자의 label별 확률을
뽑아 → **작은 결합기**만 real paired로 학습합니다.

```
p_fusion = g(p_image, p_tabular, metadata)   # g는 파라미터 극소(로지스틱/얕은 MLP)
```

- 파라미터가 극소라 scarce real에서도 안정적. **paired data가 정말 적을 때 1순위.**
- 단점: unimodal 확률이 담지 못한 cross-modal 상호작용은 못 배운다.
- ⚠️ 입력에 `abs_diff`만 넣으면 "둘 다 같은 방향으로 틀린" 가장 위험한 합의 오류를 못 거른다.
  metadata(region/condition, label_cardinality, 각 modality 신뢰도)와 **부호 보존 diff**를 넣어라.
- calibrator 학습용 split을 real validation 안에서 **별도로 분리**해 누수를 막아라.
  품질은 F1뿐 아니라 **calibration(ECE/reliability)** 으로도 봐라.

### ④ Modality dropout — *collapse 방어 + synthetic도 fusion에 흘리기*

학습 중 tabular(또는 image)를 확률 p로 **learned "missing" 토큰**으로 대체합니다.

- synthetic image-only 샘플도 fusion head를 통과시킬 수 있고,
- fusion이 한 모달리티로 무너지지 않으며,
- 추론 시 tabular가 noisy/결측이어도 견딘다.
- zeros 말고 **학습된 null 임베딩**을 써라.

### (보너스) ⑤ Residual / prior fusion

`fusion_logit = image_logit + Δ(tabular)` — image-only 예측(synthetic으로 잘 학습됨)에서 출발해
tabular가 **보정만** 하도록. 정체성 클래스("이미지 유사·전기로만 구분")가 바로 이 보정이
필요한 지점이고, 저용량 보정이라 데이터 효율적입니다.

---

## 3. 반드시 알아야 할 두 함정

### 함정 A — Fusion이 image로 collapse한다

image branch는 synthetic으로 강하게 학습되고 tabular 경로는 데이터가 굶주려서, fusion이
tabular를 무시해버립니다.

- **진단(모델 보유 시):** 추론에서 tabular를 null로 ablation → composite-acc 하락폭 측정.
  하락폭 ≈ 0이면 tabular 기여 0 = collapse. → `fusion.modality_contribution(predict_fn, ...)`
- **진단(예측표만 있을 때):** real composite에서 *image는 틀렸는데 tabular는 맞춘* 행("rescue
  candidate") 중 fusion이 맞춘 비율(**follow rate**). 낮으면 collapse. → `evaluate_fusion(...)`이
  자동 계산·경고.
- **방어:** modality dropout(④), 2-stage freeze(②), residual fusion(⑤).

### 함정 B — 정체성 클래스는 image 합성으로 **근본적으로** 해결 불가

정의상 그 클래스들은 이미지가 똑같고 전기로만 갈립니다. 따라서

- synthetic image를 아무리 늘려도 이 클래스 구분에 **전혀** 도움이 안 된다.
- 여기서 fusion은 **사실상 tabular 주도 분류**이며 **real tabular 양에 직접 묶인다.**
- "tabular 합성 불가"가 가장 아프게 작용하는 지점이 바로 여기.
- → 이 클래스는 **별도 슬라이스**로 떼서 fusion의 가치를 따로 측정해야 한다. 전체 평균에
  묻으면 안 된다. → `evaluate_fusion(..., identity_labels=[...])`이 슬라이스 + tabular 우위를 보고.

---

## 4. "왜 가짜 tabular를 만들지 않는가"는 옳은 결정

가짜 전기값을 만들면 비물리적 분포가 주입되고, 하필 **전기로 정의되는 정체성 클래스가 통째로
오염**됩니다. 이 비대칭은 "tabular를 합성해서 고칠 결함"이 아니라 **구조로 우회할 제약**입니다.
plan.md의 "synthetic tabular는 생성하지 않는다" 가정은 유지하세요.

---

## 5. 실무 권장 경로

1. **③ late calibrator** 를 안전 baseline으로 먼저 (scarce real에서도 동작).
2. real이 fusion head를 fit할 만큼 모이면 **② + ④** (frozen encoder + modality dropout) joint fusion.
3. 항상 **함정 A 진단**(tabular ablation / follow rate)으로 collapse 점검.
4. 평가는 **"image로 풀리는 클래스" vs "tabular로만 풀리는 정체성 클래스"** 를 분리 보고.
   fusion의 가치는 대부분 **후자 + disagreement 케이스**에 있다.

평가 메트릭은 단일 head가 아니라 **3개 head를 동시에** 보고, KPI(single×composite)는 head별로,
gain은 best-unimodal 대비로, collapse/identity는 별도 진단으로 봅니다 —
이 전부를 `src/fbm_multimodal/fusion/`이 해줍니다. 사용법은
[fusion_eval_quickstart.md](fusion_eval_quickstart.md).

---

## 부록 — 이 문서가 가정한 평가 지표 정의

- **single subset acc**: `eval_group == real_single` 행에서 전 label 동시 정답 비율.
- **composite subset acc**: `eval_group == real_composite` 행에서 전 label 동시 정답 비율.
- **KPI product**: `single_acc × composite_acc` (head별). plan.md의 헤드라인 KPI와 동일.
- **fusion gain**: `fusion KPI − max(image_only KPI, tabular_only KPI)`.
- **follow rate**: real composite에서 (image 오답 ∧ tabular 정답)인 행 중 fusion 정답 비율.
- **identity slice**: 정체성 label을 하나라도 가진 real 행에서의 head별 subset acc와
  `tabular_only − image_only` 우위.
