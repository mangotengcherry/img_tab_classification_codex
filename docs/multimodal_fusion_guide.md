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

현재 `experiment-plan-v2` 브랜치의 기준 구현은 generic tabular encoder가 아니라 다음 구조입니다.

```text
FBM image -> FBM encoder/head -> fbm_logits
WL raw measurement -> train-real-only high-side residual map -> WL head -> wl_logits
raw EDS/tabular scalar features -> CatBoost one-vs-rest OOF/fold-ensemble -> catboost_logits
fbm_logits + wl_logits + catboost_logits -> class-wise gated residual fusion
```

즉 "tabular를 합성하지 않는다"는 원칙은 유지하되, tabular 정보는 **CatBoost scalar branch**와
**WL residual profile branch**로 분리해서 쓴다. Synthetic composite에는 raw tabular row를 만들지
않고 parent residual maps의 `max`/union/source-count 합성만 낮은 weight로 사용한다.

진짜 제약은 "tabular를 못 만든다"가 아니라 두 가지입니다.

1. **모달리티 비대칭** — synthetic 샘플은 tabular가 없어 fusion/tabular 경로로 못 흐른다.
2. **데이터량 비대칭** — fusion이 실제로 배울 수 있는 (image ∧ tabular ∧ label) 샘플,
   특히 **real composite**가 극소수다.

---

## 1. 데이터 구조부터 정확히

| 샘플 종류 | image | WL residual map | CatBoost logits | label | 어떤 head를 학습시키나 |
|---|:---:|:---:|:---:|:---:|---|
| real single | ✓ | ✓ | ✓ | ✓ | FBM / WL / fusion |
| **real composite** | ✓ | ✓ | ✓ | ✓ | FBM / WL / fusion |
| synthetic composite image-only | ✓ | ✗ | ✗ | ✓ | FBM head만 |
| synthetic composite with synthetic WL | ✓ | 합성 map | ✗ | ✓ | FBM + WL/fusion 낮은 weight |
| unlabeled | ✓ | 가능 | 가능 | ✗ | 현재 pseudo-label default off |

> Fusion head의 official 성능은 image·실제 tabular 정보·label이 동시에 있는 real sample로만
> 판단합니다. Synthetic WL은 부족한 composite 학습을 보조하는 낮은 weight signal이지 official
> metric 대상이 아닙니다.

---

## 1.5 학술적 배경 — 문제 정식화와 관련 연구

이 절은 위 직관을 문헌에 비춰 정식화한다(참고문헌은 문서 끝).

**문제 정식화.** 각 chip을 `(xᴵ, xᵂ, zᶜ, y)`로 둔다 — `xᴵ` image, `xᵂ` WL residual map,
`zᶜ` CatBoost logits, `y ∈ {0,1}ᴸ` multi-label. tabular 관측 여부를 지시변수 `mᵀ ∈ {0,1}`로 두면, 본 문제의 결측은
**무작위가 아니다**: synthetic 샘플은 *항상* `mᵀ=0`이고 그 결측은 라벨·그룹과 상관된다.
이는 Rubin(1976)의 분류에서 **MNAR(Missing Not At Random)** 에 해당한다. 따라서 결측을
평균대치/생성모델로 메우면 추정량에 편향이 들어간다(특히 전기로 정의되는 정체성 클래스). 이것이
"가짜 raw tabular row를 만들지 않는다"의 **통계적** 근거다. 현재 합성은 raw row가 아니라
high-side residual map의 보수적 max composition으로 제한한다.

**언제 fusion이 이득인가(이론).** Huang et al.(NeurIPS 2021)은 "각 모달리티를 공통 잠재공간으로
인코딩 후 결합"하는 표준 fusion에서 다모달이 단일모달보다 **모집단 위험(population risk)이 작음을
증명**했다 — 직관은 다모달이 잠재표현을 더 정확히 추정한다는 것. 단 이 이득은 *paired 데이터가
충분*하고 *모달리티가 상호보완적*일 때 실현된다. 본 문제의 정체성 클래스는 image-only의 베이즈
오류가 0으로 못 내려가는(이미지가 동일) 전형적 경우이고 tabular가 그 정보를 보완하므로, 이론상
fusion 이득이 **가장 큰** 구간이다(§3 함정 B).

**왜 naive fusion이 오히려 더 나쁠 수 있는가.** Wang et al.(CVPR 2020)은 다모달 망이 *최고
단일모달보다 자주 못한* 현상을 보고하고 원인을 (1) 용량 증가로 인한 과적합, (2) 모달리티별
과적합/일반화 **속도 차이**로 규명했다(해법: Gradient Blending). Peng et al.(CVPR 2022, OGM-GE)과
Wu et al.(ICML 2022)은 학습이 **지배 모달리티로 쏠려(greedy)** 약한 모달리티가 under-optimized
된다고 보였다. 본 설계의 image-rich(synthetic) 상황은 이 "쏠림"이 구조적으로 image 쪽으로
일어나는 케이스이며, **modality dropout·2-stage freeze·collapse 진단**(§2④,②/§3 A)은 정확히 이
실패를 막는 장치다.

**결측 모달리티 학습 계열.** 학습·추론 양쪽에서 모달리티가 빠지는 상황은 Ma & Peng(SMIL,
AAAI 2021)이 정식 연구했고, 모달리티 무작위 제거(ModDrop, Neverova et al., TPAMI 2016)는 결측에
견고한 표현을 학습시키는 표준 기법이다. 본 문서의 **loss masking(①)·modality dropout(④)** 은 이
계열에 속한다. 융합 구조(early/late/intermediate)와 co-learning 분류는 Baltrušaitis et al.
(TPAMI 2019) taxonomy를 따른다 — late calibrator(③)=late fusion, 3-head joint=intermediate fusion.

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

현재 구현의 residual fusion은 class-wise gate를 둔 형태입니다.

```text
fusion_logits =
    fbm_logits
    + has_wl_map * gate_wl[class] * wl_logits
    + has_catboost_logits * gate_cat[class] * catboost_logits
```

CatBoost는 offline branch라 neural loss를 주지 않고, WL/fusion loss는 `has_wl_map`,
`wl_loss_weight`, synthetic mask에 의해 조절한다.

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
plan.md의 "synthetic raw tabular는 생성하지 않는다" 가정은 유지하세요. 단, raw row가 아니라
WL residual map은 parent residual의 `max`/union/source-count로만 낮은 weight auxiliary signal로
쓸 수 있다.

---

## 5. 실무 권장 경로

1. 기존 FBM baseline을 먼저 재현한다.
2. CatBoost OOF logits를 추가해 scalar tabular 이득을 분리 측정한다.
3. WL residual map branch를 추가해 WL profile 이득을 측정한다.
4. Synthetic WL residual map은 weight ablation으로만 도입한다.
5. 항상 **함정 A 진단**(tabular/WL/CatBoost ablation / follow rate)으로 collapse 점검.
6. 평가는 **"image로 풀리는 클래스" vs "tabular로만 풀리는 정체성 클래스"** 를 분리 보고.
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

### 평가의 통계적 엄밀성 (메모)

- **subset accuracy = exact-match ratio.** multi-label에서 가장 엄격한 지표(전 label 동시 정답).
  보조로 Hamming accuracy(label 평균)를 병기하면 부분 정답 경향을 본다.
- **KPI = single×composite (곱)** 은 두 비율 추정량의 곱이라, 작은 composite n에서 분산이 크고
  gameable하다. 점추정만 보지 말고 **두 항·n·신뢰구간**을 항상 동반한다(KPI 카드).
- **소표본 신뢰구간:** composite/class-pair는 support가 작아 정규근사가 깨진다 → **Wilson 구간**
  또는 그룹(wafer) 단위 **부트스트랩**을 쓴다. seed 반복(가중치 분산)과 표본 분산은 다른 출처임을
  구분한다.
- **확률 품질:** late calibrator·uncertainty 기반 의사결정은 확률 신뢰도에 의존하므로
  **calibration(ECE/reliability diagram; Guo et al. 2017)** 을 F1·acc와 별도로 본다.
- **누수 차단:** group split(§ 실험계획)·threshold tuning split 분리·라벨 정의 feature 배제는
  내적 타당성(internal validity)의 전제다.

---

## 참고문헌

1. W. Wang, D. Tran, M. Feiszli. "What Makes Training Multi-Modal Classification Networks Hard?"
   *CVPR* 2020. arXiv:1905.12681. — 다모달이 최고 단일모달보다 자주 못함; 모달리티별 과적합 속도 차이; Gradient Blending.
2. Y. Huang, C. Du, Z. Xue, X. Chen, H. Zhao, L. Huang. "What Makes Multi-modal Learning Better than
   Single (Provably)." *NeurIPS* 2021. arXiv:2106.04538. — 표준 fusion의 모집단 위험 감소 증명.
3. X. Peng, Y. Wei, A. Deng, D. Wang, D. Hu. "Balanced Multimodal Learning via On-the-fly Gradient
   Modulation (OGM-GE)." *CVPR* 2022 (oral). arXiv:2203.15332. — 지배 모달리티 쏠림과 균형화.
4. N. Wu, S. Jastrzebski, K. Cho, K. J. Geras. "Characterizing and Overcoming the Greedy Nature of
   Learning in Multi-modal DNNs." *ICML* 2022. — greedy(쏠림) 학습 규명.
5. M. Ma, X. Peng. "SMIL: Multimodal Learning with Severely Missing Modality." *AAAI* 2021.
   arXiv:2103.05677. — 학습·추론 결측 모달리티 정식 연구.
6. N. Neverova, C. Wolf, G. Taylor, F. Nebout. "ModDrop: Adaptive Multi-modal Gesture Recognition."
   *IEEE TPAMI* 2016. — 모달리티 dropout.
7. T. Baltrušaitis, C. Ahuja, L.-P. Morency. "Multimodal Machine Learning: A Survey and Taxonomy."
   *IEEE TPAMI* 2019. arXiv:1705.09406. — fusion/co-learning taxonomy.
8. C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. "On Calibration of Modern Neural Networks."
   *ICML* 2017. arXiv:1706.04599. — ECE/calibration.
9. D. B. Rubin. "Inference and Missing Data." *Biometrika* 63(3), 1976. — MCAR/MAR/MNAR.
10. B. Kim, Y.-S. Jeong, S. H. Tong, I.-K. Chang, M.-K. Jeong. "A Regularized SVD-based Approach for
    Failure Pattern Classification on Fail Bit Map in a DRAM Wafer." *IEEE Trans. Semiconductor
    Manufacturing* 28(1), 2015. — FBM 불량 패턴 도메인(이미지 패턴 참고용).
