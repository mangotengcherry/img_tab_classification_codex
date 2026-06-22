# FBM Image + Tabular Fusion 실험 계획 (v2)

> 이 문서는 루트 `plan.md`(초기 E0–E5 스케치)의 **fusion 부분을 학술적으로 재설계**한 실험
> 프로토콜이다. 멀티모달 학습 문헌과 다중 전문가 리뷰(평가 누수·circular labeling·소표본·collapse)
> 지적을 통합했다. 설계 근거·참고문헌은 [multimodal_fusion_guide.md](multimodal_fusion_guide.md).
>
> 핵심 원칙: **(1) 평가 무결성 먼저, (2) fusion 이득은 가설로 두고 통계적으로 검증, (3) 합성은
> image 경로에만, 가짜 tabular는 만들지 않음(MNAR).**

---

## 0. 목적과 범위

FBM image와 WL/전기(MSR) tabular를 결합해 단일·중첩·"이미지 유사·전기로만 구분(정체성)" 불량을
분류한다. 본 계획은 **fusion이 단일모달 대비 실제로 이득인지**, 그리고 **모달리티 비대칭(이미지
합성 가능 / tabular 합성 불가, 결측은 MNAR)** 에서 그 이득을 누수 없이 측정하는 절차를 정의한다.

---

## 1. 연구 질문과 가설

| RQ | 가설(H) | 측정 가능한 예측 | 판정 |
|---|---|---|---|
| RQ1 fusion이 KPI를 올리나 | H1: fusion KPI > 최고 단일모달 KPI | group-split에서 `fusion_gain`의 부트스트랩 CI 하한 > 0 | CI 하한 ≤ 0이면 fusion 채택 보류 |
| RQ2 정체성 클래스 | H2: 정체성 slice에서 `tabular−image` > 0, `fusion ≥ tabular` | slice subset acc 차이의 CI | tabular−image ≤ 0이면 가설/라벨 재검토 |
| RQ3 분리(synthetic=image only)가 collapse 없이 동작 | H3: tabular ablation 기여 > 0 AND follow rate ≥ 0.5 | `modality_contribution`, follow rate | collapse 신호 시 dropout/2-stage 강화 |
| RQ4 어떤 fusion이 데이터효율적 | H4: scarce real에서 late(③)·2-stage(②)가 naive joint보다 분산↓ | arm별 KPI 평균±CI, seed 분산 | 데이터량 구간별 최적 arm 보고 |
| RQ5 라벨 효율 | H5: AL이 random 대비 동일 KPI를 더 적은 라벨로 | label-cost curve(엔지니어-시간 축) | 한계효용 임계 이하 시 stop |

> H1/H2는 **사전 등록(pre-registered) 1차 가설**이다. 나머지는 탐색적. 비교는 사후 cherry-pick이
> 아니라 사전 정의된 arm 간 비교로만 한다.

---

## 2. 문제 정식화

각 chip `i`: `(xᴵ_i, xᵀ_i, y_i)`, `y_i ∈ {0,1}ᴸ`. tabular 관측 지시 `mᵀ_i`. synthetic이면 `mᵀ_i=0`이고
결측이 라벨·그룹과 상관 → **MNAR**(Rubin 1976) → 가짜 tabular 생성 금지(편향 회피).

모델은 세 head를 출력: `p^I=σ(f_I(xᴵ))`, `p^T=σ(f_T(xᵀ))`, `p^F=σ(f_F(xᴵ,xᵀ))`.
손실은 **각 샘플이 입력을 가진 head에만** (loss masking):
`L = BCE(p^I,y) + mᵀ·[BCE(p^T,y) + BCE(p^F,y)]` (+ class-imbalance 가중/asymmetric loss).
학습 시 modality dropout: real 샘플의 `xᵀ`를 확률 `ρ`로 learned null 토큰으로 치환.

평가 그룹: `real_single`(mᵀ=1, |y|=1), `real_composite`(mᵀ=1, |y|≥2), `synthetic_composite`(mᵀ=0, |y|≥2).

---

## 3. 데이터 설계와 분할 (최우선 — 누수 차단)

### 3.1 manifest 필수 컬럼
`chip_id, wafer_id, lot_id, tester_id, process_timestamp, eval_group, is_synthetic, is_pseudo_labeled,
label_cardinality, true_<label>…, MSR_000…MSR_200`. (현재 manifest엔 wafer_position만 있음 → lot/tester/time **승격 필수**.)

### 3.2 분할 (P0)
- **1차 평가 = wafer 단위 `GroupKFold`**(가능하면 lot). chip-random split은 **낙관적 상한
  참고치로만** 병기하고, 두 split의 KPI 격차 자체를 *누수 진단 지표*로 보고.
- **stress split**: `leave-one-lot-out`, 시간 순서 forward-chaining(과거→미래).
- threshold·model selection·test를 **3분할(또는 nested CV)** 로 분리. test에서 threshold 재튜닝 금지.

### 3.3 라벨 정의와 circular-labeling 차단 (P0)
- 각 class의 ground-truth 정의(어떤 MSR rule / bin code / PFA)를 **명문화**.
- 라벨 정의에 직접 쓰인 MSR feature는 tabular/fusion 입력에서 **배제**하거나 leakage-free 평가
  slice를 분리. (그렇지 않으면 tabular가 라벨 생성 규칙을 외워 fusion 우월성이 허상이 됨.)

### 3.4 가정 검증 게이트 (P0)
Phase 0에서 먼저 측정 후 조건부 진행:
- `measurement_map.csv` coverage < 70%(또는 wl_index 대부분 unknown) → WL-position-aware/region
  embedding 실험 **보류**, condition/type-only 축소 모델로 폴백.
- class-pair당 real composite < `N_min`(기본 30) → 해당 pair는 KPI에서 가중을 낮추고 **underpowered**로 격리.
- 측정 repeatability(같은 die retest grade·MSR 일관성)로 **라벨/feature 노이즈 상한** 정량화.

---

## 4. 실험 조건 (arms)

| arm | 모달리티 | 구조 | 학습 | 가이드 § |
|---|---|---|---|---|
| **A0** image-only | image(+synthetic) | CNN/attention | BCE | baseline |
| **A1** tabular-only | tabular(real) | GBM 또는 metadata-aware MLP | BCE | baseline |
| **A2** late calibrator | both | unimodal logits + metadata → 얕은 결합기 | real로 결합기만 | ③ |
| **A3** joint 3-head | both | image/tabular enc + fusion head | loss masking + modality dropout | ①④ |
| **A4** 2-stage frozen | both | enc 사전학습 후 freeze, 작은 fusion head | scarce real로 head만 | ② |
| **A5**(opt) balanced | both | A3 + Gradient Blending/OGM-GE | 모달리티 균형 | §1.5 |

**Ablation**: modality dropout on/off · null-token vs zeros · synthetic(image branch) on/off ·
fusion에서 tabular zero-ablation(기여 측정) · class-imbalance 손실(BCE vs asymmetric/focal).

---

## 5. 지표

**1차(primary)**
- **KPI = single subset acc × composite subset acc** (real, group-split), 부트스트랩 CI 동반.
- **fusion gain** = KPI(fusion) − max(KPI(A0), KPI(A1)), CI 동반.

**2차(secondary)**
- head×eval_group subset acc(+Wilson CI/support), per-class F1/recall.
- **정체성 slice**: `tabular−image`, `fusion−image`.
- **collapse 진단**: follow rate, tabular ablation 기여, fusion regressions.
- **calibration**: ECE/reliability(특히 A2와 uncertainty 기반 결정).
- **synthetic-to-real gap**: class-pair별 `synth_acc − real_acc`(real 절대수준 게이트와 함께 해석).
- **label-cost curve**: x축은 *엔지니어-시간/비용*, random/uncertainty/disagreement/cluster 비교.

각 지표는 **어떤 결정과 연결되는지** 사전 명시(예: 정체성 slice tabular−image≤0 → tabular feature/라벨 재검토).

---

## 6. 통계 분석 계획

- **반복**: 핵심 arm 최소 **5 seed**(가중치 초기화·shuffle 분산). 작은 composite에는 seed CI로
  불충분 → **wafer 그룹 단위 부트스트랩**을 1차 근거로.
- **KPI CI**: 곱의 분산이 작은 n에서 크므로 부트스트랩(그룹 리샘플)으로 산출. **composite CI 하한이
  목표 미만이면 곱이 높아도 배포 불가**(gaming 차단).
- **arm 간 비교**: paired 부트스트랩으로 fusion vs best-unimodal 차이의 CI/유의성.
- **다중비교**: 여러 arm·class-pair 동시 비교 시 **Benjamini–Hochberg** 보정 또는 사전 등록 가설로 제한.
- **소표본**: support < `N_min`인 class-pair는 underpowered로 표기, KPI 가중 축소/제외.

---

## 7. 타당성 위협과 통제

| 구분 | 위협 | 통제 |
|---|---|---|
| internal | wafer/lot 공간상관 누수 | wafer GroupKFold 1차화, random과의 격차 보고 (§3.2) |
| internal | circular labeling | 라벨 정의 feature 배제 / leakage-free slice (§3.3) |
| internal | threshold tuning 누수 | selection/tuning/test 3분할, test 재튜닝 금지 |
| internal | fusion이 image로 collapse | modality dropout·2-stage·ablation/follow-rate 진단 (§4,§5) |
| external | synthetic-to-real gap | gap을 class-pair별 분리 보고, 합성 물리 검증(트리플릿) |
| external | lot/tester/time drift | stress split(forward-chaining, leave-one-lot-out) |
| construct | KPI 곱 gameability | KPI 카드(두 항·n·CI) + composite CI 하한 게이트 |
| construct | subset acc 과엄격 | Hamming acc 병기 |
| statistical | 작은 composite n | Wilson/부트스트랩 CI, underpowered 격리 |
| statistical | 다중비교 | BH 보정 / 사전 등록 |

---

## 8. 성공 기준 (Go / No-Go)

목표 수치(루트 plan.md 계승): `single ≥ 0.8`, `composite ≥ 0.6`, `KPI = single×composite ≥ 0.65`.
단, **반드시 wafer group-split** 기준으로 판정하고 다음을 추가:

- **GO(채택)**: 위 KPI 게이트 통과 **그리고** `fusion_gain` CI 하한 > 0 **그리고** 정체성 slice
  `tabular−image` > 0 **그리고** collapse 신호 없음(follow rate ≥ 0.5 또는 tabular 기여 > 0)
  **그리고** composite acc CI 하한 ≥ 목표.
- **조건부**: KPI는 만족하나 fusion_gain CI가 0을 포함 → fusion 대신 **최고 단일모달**을 채택하고
  fusion은 추가 데이터 후 재평가.
- **NO-GO/보류**: chip-random에서만 통과(group split 급락), 또는 circular-labeling 미차단,
  또는 mapping coverage 게이트 미달.

---

## 9. 단계별 실행 (Phases)

**Phase 0 — 데이터 계약·가정 검증** *(진입 전 필수)*
- manifest에 lot/tester/time 승격, group-split 구현, 라벨 정의 명문화·feature 배제.
- mapping coverage·class-pair support·repeatability 측정 → §3.4 게이트로 Phase 1 범위 확정.
- *exit*: 전체 테스트 green + group-split·KPI 카드 산출 재현.

**Phase 1 — 단일모달 baseline** (A0, A1)
- group-split 기준 KPI baseline 확정, calibration·per-class 보고.
- *exit*: A0/A1 KPI와 CI 확보(= fusion_gain의 비교 기준).

**Phase 2 — fusion** (A2 → A3/A4, opt A5)
- A2(late) 안전 baseline → real이 충분하면 A3/A4. 매 arm마다 collapse 진단·정체성 slice 필수.
- *exit*: H1/H2 판정(부트스트랩 CI), 최적 arm 선정.

**Phase 3 — gap 축소 & 라벨 효율**
- 합성 mode(max/clipped/saturating) 물리 타당성: 정렬된 (단일A, 단일B, 실제 A+B) 트리플릿 분포 비교.
- AL 시뮬레이션(random/uncertainty/disagreement/cluster), label-cost curve.
- *exit*: gap 관리 정책·라벨 stop rule 확정.

---

## 10. 산출물

- 3-head 예측 CSV(`true_*`, `image_prob_*`, `tabular_prob_*`, `fusion_prob_*` = `prob_*` alias),
- KPI 카드(두 항·n·CI), head×group 표, fusion gain, 정체성 slice, collapse 진단, calibration,
  synthetic-to-real gap, label-cost curve, group-vs-random 격차.
- 재현성: manifest/measurement_map content hash, split seed 고정, run config·환경 lockfile 로깅.

---

## 11. 현재 구현과의 매핑 (무엇이 되어 있고, 무엇이 TODO인가)

| 항목 | 현재 repo | TODO |
|---|---|---|
| 3-head + loss masking + modality dropout (A3) | `fusion/model.py` (numpy) | torch CNN/transformer 인코더로 교체 |
| 평가기(KPI·head×group·fusion gain·collapse·identity·Wilson CI) | `fusion/fusion_eval.py` | 그룹 부트스트랩 CI, calibration(ECE) 추가 |
| KPI gate(single/composite/product) | `condition_eval.py`(main 계열) | group-split 입력으로 결합 |
| late calibrator (A2) | `features.build_late_fusion_frame` | metadata·부호보존 diff·전용 split |
| **group split (wafer/lot)** | 없음 | **Phase 0 1순위 구현** |
| circular-labeling 차단 | 없음 | 라벨 정의·feature 배제 규약 |
| 합성 물리 검증(트리플릿) | `synthetic.py`(픽셀 max/sum) | 분포 비교·양립불가 pair 제외 |
| AL(label-cost) | `active_learning.py` | 전략별 분리·비용축 시뮬레이션 |

> 즉, **모델·평가 골격은 존재**하고, v2의 핵심 추가는 **group split·circular-labeling 차단·통계적
> CI 게이트**(평가 무결성)와 **arm 간 통제 비교**다.

---

## 참고문헌

[multimodal_fusion_guide.md의 참고문헌](multimodal_fusion_guide.md#참고문헌)을 공유한다(Wang 2020;
Huang 2021; Peng 2022; Wu 2022; Ma & Peng 2021; Neverova 2016; Baltrušaitis 2019; Guo 2017;
Rubin 1976; Kim 2015).
