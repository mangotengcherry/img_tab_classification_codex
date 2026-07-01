# FBM Image + WL Residual Map + CatBoost Fusion 실험 계획 (v3)

> 이 문서는 루트 `plan.md`의 현재 구현 기준을 통계적 실험 프로토콜로 확장한 문서다.
> v2의 평가 무결성 원칙은 유지하되, 구현 기준은 **mapped-tabular encoder 중심**에서
> **WL high-side residual map + CatBoost OOF logit branch + class-wise gated residual fusion**으로
> 갱신했다. 설계 근거·참고문헌은 [multimodal_fusion_guide.md](multimodal_fusion_guide.md).
>
> 핵심 원칙: **(1) 평가 무결성 먼저, (2) fusion 이득은 가설로 두고 통계적으로 검증, (3) 합성은
> raw tabular row가 아니라 WL residual map에만 낮은 weight로 사용, (4) CatBoost train logits는 OOF만 사용.**

---

## 0. 목적과 범위

FBM image와 WL/전기(EDS/MSR) 정보를 결합해 단일·중첩·"이미지 유사·전기로만 구분(정체성)" 불량을
분류한다. 현재 tabular 사용 방식은 두 가지다.

- raw scalar EDS/tabular feature는 **CatBoost one-vs-rest OOF logits**로 변환해 fusion input으로 쓴다.
- WL 위치가 있는 raw measurement는 train-real baseline 대비 **high-side residual map tensor**로 변환한다.

본 계획은 **fusion이 단일모달/단일 branch 대비 실제로 이득인지**, 그리고 **모달리티 비대칭(FBM 합성
가능 / raw tabular 합성 금지 / WL residual map만 보수적 합성)** 에서 그 이득을 누수 없이 측정하는
절차를 정의한다.

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

각 chip `i`: `(xᴵ_i, xᵂ_i, zᶜ_i, y_i)`, `y_i ∈ {0,1}ᴸ`.

- `xᴵ`: FBM image.
- `xᵂ`: WL residual map tensor `[C,B,T]`.
- `zᶜ`: CatBoost class-wise logits.
- `mᵂ`, `mᶜ`: WL map/CatBoost logit availability masks.

WL residual:

```text
R = max(0, (x - median_train_real(test_method, wl_bin)) / (IQR_train_real(test_method, wl_bin) + eps))
```

Fusion logits:

```text
ell_fusion =
  ell_fbm
  + m_wl  * gate_wl[class]  * ell_wl
  + m_cat * gate_cat[class] * ell_catboost
```

Raw tabular row 합성은 MNAR 편향 때문에 금지한다. Synthetic composite에는 parent residual maps의
`max`/union/source-count 합성만 낮은 loss weight(기본 0.2)로 쓴다.

평가 그룹: `real_single`, `real_composite`, `real_all`, `synthetic_composite`.

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

| arm | 모달리티 | 구조 | 학습 | 목적 |
|---|---|---|---|---|
| **E0/A0** | FBM | 기존 FBM + synthetic image baseline | existing scheme | baseline 재현 |
| **E1/A1** | FBM + CatBoost logits | CatBoost one-vs-rest OOF logits direct fusion | CatBoost real-train only, fusion real only | scalar tabular 이득 |
| **E2/A2** | FBM + WL map + CatBoost | WL residual map encoder + CatBoost logits | WL baseline train-real only | WL profile 추가 이득 |
| **E3/A3** | E2 + synthetic WL | parent max/union/source-count WL synthesis | synthetic WL/fusion low weight | scarce composite 보완 |
| **E4/A4** | E3 weight sweep | synthetic WL/fusion weight ablation | validation real composite 기준 | 보수적 weight 선택 |
| **E5/A5** | pseudo-label scaffold | pairwise top-K selector only | default off | future 확장 검증 |

**Ablation**: modality dropout on/off · null-token vs zeros · synthetic(image branch) on/off ·
synthetic WL weight 0.0/0.1/0.2/0.3 · CatBoost direct-logit vs calibrated-logit · class-imbalance 손실(BCE vs asymmetric/focal).

---

## 5. 지표

**1차(primary)**
- **KPI = single subset acc × composite subset acc** (real, group-split), 부트스트랩 CI 동반.
- **fusion gain** = KPI(fusion) − max(KPI(A0), KPI(A1)), CI 동반.

**2차(secondary)**
- head×eval_group subset acc(+Wilson CI/support), per-class F1/recall.
- official real groups: `real_single`, `real_composite`, `real_all`.
- branch diagnostics: FBM, WL, CatBoost, Fusion where logits/probabilities exist.
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

**Phase 1 — baseline + CatBoost** (E0, E1)
- 기존 FBM baseline 재현 후 CatBoost OOF logits 추가.
- CatBoost train logits가 OOF인지, synthetic이 제외됐는지 metadata로 확인.
- *exit*: FBM-only와 FBM+CatBoost KPI/CI 확보.

**Phase 2 — WL residual map fusion** (E2)
- WL residual tensorizer를 train real only로 fit하고 real WL map branch를 추가.
- missing/source_count coverage를 함께 보고한다.
- *exit*: WL profile이 CatBoost scalar 대비 추가 이득을 주는지 판정.

**Phase 3 — synthetic WL + weight ablation** (E3, E4)
- parent map `max` + union mask + source_count synthetic WL을 낮은 weight로 사용.
- synthetic metric은 auxiliary only, 모델 선택은 real validation 기준.
- *exit*: synthetic WL/fusion weight 정책 확정.

**Phase 4 — pseudo-label future gate** (E5)
- 기본 off 유지. `pairwise_topk` 함수 테스트만 유지.
- unlabeled loader/sample injection은 enabled true일 때만 연결.
- *exit*: pseudo-label off 상태에서 학습/평가 경로가 변하지 않음 확인.

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
| FBM baseline + old fusion smoke path | `fusion/model.py`, `examples/run_fusion_experiment.py` | real loader 연결 |
| WL residual tensorizer | `wl_residual_map.py` | real WL cache pipeline 자동화 |
| synthetic WL composer | `synthetic_wl_map.py` | parent metadata 기반 batch composer |
| CatBoost OOF logits | `training/train_catboost_oof.py` | real feature column selection script와 연결 |
| gated residual fusion | `fusion/model.py::ClasswiseGatedResidualFusion`, `WLResidualCatBoostFusionMLP` | torch CNN/production model로 교체 |
| fusion dataset masks | `fusion/data.py` | real `FusionDataset` builder 연결 |
| EDS mapping CLI | `cli validate-eds-map`, `cli build-wl-measurements` | operator-facing validation report |
| 평가기(KPI·head×group·fusion gain·collapse·identity·Wilson CI·leakage checks) | `fusion/fusion_eval.py` | 그룹 부트스트랩 CI, calibration(ECE), explicit WL/CatBoost report export |
| pseudo-label pairwise top-K | `pseudo_labeling/pairwise_topk.py` | enabled true pipeline, diversity option |
| **group split (wafer/lot)** | planning only | **Phase 0 1순위 구현** |
| circular-labeling 차단 | planning only | 라벨 정의·feature 배제 규약 |

현재 repo는 **WL/CatBoost branch scaffold와 leakage-aware 평가 골격**까지 구현된 상태다. 다음 핵심은
실제 데이터 loader에서 `wl_maps`, `catboost_logits`, `has_*` masks를 안정적으로 생성하고, wafer/lot
group split과 circular-labeling 차단을 실험 프로토콜에 강제하는 것이다.

---

## 참고문헌

[multimodal_fusion_guide.md의 참고문헌](multimodal_fusion_guide.md#참고문헌)을 공유한다(Wang 2020;
Huang 2021; Peng 2022; Wu 2022; Ma & Peng 2021; Neverova 2016; Baltrušaitis 2019; Guo 2017;
Rubin 1976; Kim 2015).
