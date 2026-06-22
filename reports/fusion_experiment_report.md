# Image + Tabular Fusion 실험 리포트

> 재현 명령어: `PYTHONPATH=src python3 examples/run_fusion_experiment.py`
> 설계 배경: [../docs/multimodal_fusion_guide.md](../docs/multimodal_fusion_guide.md)
> 평가기 사용법: [../docs/fusion_eval_quickstart.md](../docs/fusion_eval_quickstart.md)

## 한 줄 결론

이번 데모 실험에서는 **image와 tabular를 함께 쓰는 fusion이 가장 좋았습니다.**

- 단일 불량 acc: `0.972`
- 중첩 불량 acc: `0.983`
- KPI product: `0.956`

목표였던 `single >= 0.8`, `composite >= 0.6`, `single * composite >= 0.65`를 모두 넘었습니다.

다만 이 수치는 실데이터가 아니라 **실제 문제 구조를 흉내 낸 합성 데이터**에서 나온 결과입니다.
따라서 이 리포트의 핵심은 "실제 KPI 달성 확정"이 아니라, **어떤 방식으로 평가하면 되는지**와
**image만으로 안 풀리는 케이스를 tabular가 어떻게 살리는지**를 보여주는 것입니다.

## 실험에서 확인한 패턴

데이터는 `128x46` FBM image를 사용하고, intensity는 `0~8` grade로 표현했습니다.

![dataset](figures/01_dataset_overview.png)

패턴은 네 가지로 구성했습니다.

- `edge_ring`: 가장자리 쪽이 강한 패턴
- `center_blob`: 가운데 영역이 강한 패턴
- `leak_top`, `leak_bottom`: FBM image에서는 거의 같은 세로 stripe로 보이는 패턴

특히 `leak_top`과 `leak_bottom`은 **이미지만 보면 구분하기 어렵게** 만들었습니다.
두 클래스는 tabular 전기값에서 top/bottom 영역 차이가 나야 구분됩니다.

아래 그림은 단일 4종과 가능한 2-label 중첩 6종을 모두 보여줍니다.
이번 보완에서 generator가 작은 데이터셋에서도 이 6개 중첩 조합을 빠뜨리지 않도록 수정했습니다.

![pattern gallery](figures/06_pattern_gallery.png)

## 도메인 패턴 보완

추가된 FBM 논문은 모델 방법론을 따라가기보다, **FBM에 어떤 모양의 불량 패턴이 자주 보이는지**를
확인하는 용도로 참고했습니다.

이번 보완에서는 synthetic image generator와 QA 그림에 다음 패턴을 반영했습니다.

- 랜덤하게 흩어진 낮은 grade 점
- 세로선 형태의 column 계열 패턴
- 짧은 가로선 형태의 row 계열 패턴
- 국소 block 또는 cluster
- 가장자리 ring 또는 periphery 패턴

아래 그림은 이 패턴들을 같은 `0~8` grade scale에서 비교한 것입니다.
두 번째 줄의 `grade >= 3` 화면은 사람이 구조를 빠르게 확인하기 위한 보조 보기일 뿐,
fusion 모델 입력이나 실험 방향을 논문 방식으로 바꾸지는 않습니다.

![domain pattern stress gallery](figures/07_domain_pattern_stress_gallery.png)

주의할 점은 논문의 `single-bit / non-single-bit` 구분이 여기서 말하는
`단일 label / 중첩 label`과 같은 뜻은 아니라는 것입니다.
논문은 FBM 안에 공간 패턴이 있는지 없는지를 보는 쪽에 가깝고,
우리 실험은 여러 불량 원인이 동시에 붙은 multi-label 분류를 다룹니다.
따라서 이 리포트에서는 논문 방법 자체보다, **패턴 모양을 더 현실적으로 만들고 확인하는 것**에만
반영했습니다.

## 왜 fusion이 필요한가

image-only와 tabular-only는 각각 강점이 다릅니다.

- image-only는 edge, center 같은 공간 패턴을 잘 봅니다.
- tabular-only는 `leak_top`, `leak_bottom`처럼 이미지가 비슷한 클래스를 잘 나눕니다.
- fusion은 둘의 장점을 같이 쓰기 때문에 중첩 불량에서 가장 안정적입니다.

## 성능 결과

![kpi](figures/04_kpi_product.png)

| head | single acc | composite acc | KPI product |
|---|---:|---:|---:|
| image_only | 0.700 | 0.533 | 0.373 |
| tabular_only | 1.000 | 0.517 | 0.517 |
| fusion | **0.972** | **0.983** | **0.956** |

해석은 단순합니다.

- image-only는 중첩에서 `0.533`으로 낮습니다.
- tabular-only는 단일은 완벽하지만 중첩에서 `0.517`에 머뭅니다.
- fusion은 단일 `0.972`, 중첩 `0.983`으로 둘 다 높습니다.
- best unimodal KPI가 `0.517`인데 fusion KPI는 `0.956`입니다. 차이는 `+0.439`입니다.

## image로만 안 되는 영역

![identity & collapse](figures/05_identity_and_collapse.png)

`leak_top`, `leak_bottom`처럼 이미지가 비슷한 클래스만 따로 보면 다음과 같습니다.

- image-only acc: `0.418`
- tabular-only acc: `0.865`
- fusion acc: `0.957`
- tabular가 image보다 `+0.447` 높음

즉, 이 유형은 synthetic image를 많이 늘려도 근본적으로 해결하기 어렵습니다.
전기값이 들어와야 분류가 됩니다.

## fusion이 tabular를 실제로 쓰는지 확인

fusion이 겉으로만 좋아 보이고 실제로는 image만 따라가면 위험합니다.
그래서 두 가지 확인을 했습니다.

1. real composite에서 image-only는 틀렸지만 tabular-only는 맞춘 샘플을 찾았습니다.
   이런 샘플 18개 중 fusion이 17개를 맞췄습니다. follow rate는 `0.944`입니다.
2. 모델 입력에서 tabular를 지우는 ablation을 했습니다.
   fusion acc가 `0.975`에서 `0.446`으로 내려갔습니다. tabular 기여는 `+0.529`입니다.

따라서 이번 실험에서는 fusion이 tabular를 무시하는 상태가 아닙니다.

## 실무 적용 시 주의점

- 이 결과는 합성 데이터 기반입니다. 실데이터에서는 noise, label 오류, lot/wafer 차이 때문에 수치가 낮아질 수 있습니다.
- 실데이터 평가는 random split만 보지 말고 wafer/lot/time 기준 split도 봐야 합니다.
- 실제 중첩 불량 support가 작으면 subset acc 하나만 보지 말고 confidence interval도 같이 봐야 합니다.
- synthetic tabular는 만들지 않는 편이 안전합니다. 없는 전기값을 임의로 만들면 오히려 tabular 기준이 흔들릴 수 있습니다.
- 팀 평가에서는 항상 세 가지를 같이 보세요: 전체 KPI, image로 비슷한 클래스 slice, fusion이 tabular를 실제로 쓰는지.

## 재현

```bash
PYTHONPATH=src python3 examples/run_fusion_experiment.py
PYTHONPATH=src python3 -m pytest tests/test_fusion_model.py tests/test_fusion_eval.py -q
```

주요 산출물:

- `reports/figures/01_dataset_overview.png`
- `reports/figures/06_pattern_gallery.png`
- `reports/figures/07_domain_pattern_stress_gallery.png`
- `reports/figures/04_kpi_product.png`
- `reports/figures/05_identity_and_collapse.png`
- `reports/fusion_predictions.csv`
- `reports/fusion_report.md`
- `reports/fusion_report.json`
- `reports/training_history.csv`
