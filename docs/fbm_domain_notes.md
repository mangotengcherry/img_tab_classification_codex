# FBM 패턴 도메인 노트

이 노트는 `papers/RSVD.pdf`에 나온 FBM 패턴 관찰을 현재 실험에 어떻게 반영했는지 정리한 문서입니다.

## 참고한 도메인 관점

- FBM grade는 failed cell 수를 직접 세는 값이라기보다, fail 강도를 단계로 나눈 값에 가깝습니다.
- 낮은 grade는 랜덤하게 흩어진 noise처럼 보일 수 있고, 높은 grade는 실제 공간 패턴을 더 잘 보여줍니다.
- 패턴 예시는 크게 랜덤하게 흩어진 점, 길게 이어진 세로선, 짧은 가로선, 국소 block으로 볼 수 있습니다.
- wafer edge나 periphery 쪽에서 강하게 나타나는 ring 형태도 실제 QA에서 따로 확인할 가치가 있습니다.

## 현재 repo에 반영한 내용

- `src/fbm_multimodal/fusion/fbm_patterns.py`
  - 세로선, 가로선, block, edge ring, 랜덤 scatter 패턴 painter
- `src/fbm_multimodal/fusion/data.py`
  - synthetic FBM 생성 시 낮은 grade 랜덤 scatter를 항상 배경으로 추가
  - edge, center, 세로 stripe 패턴을 위 painter로 생성
- `reports/figures/07_domain_pattern_stress_gallery.png`
  - 랜덤, 세로선, 가로선, block, edge ring 패턴을 같은 scale에서 비교하는 그림
  - 두 번째 줄의 `grade >= 3` 보기는 사람이 구조를 빠르게 확인하기 위한 보조 화면입니다.

## 해석할 때 주의할 점

논문에서 말하는 `single-bit / non-single-bit`는 이 repo의 `단일 label / 중첩 label`과 같은 뜻이 아닙니다.

- 논문 관점: FBM 안에 공간 패턴이 있는지 없는지에 가까움
- 현재 실험 관점: 한 chip에 여러 불량 원인이 동시에 붙었는지 보는 multi-label 분류

따라서 논문에서 본 방법을 모델에 그대로 넣기보다, **이미지 패턴의 모양을 더 현실적으로 만들고
팀원이 쉽게 확인하는 용도**로만 반영했습니다.
