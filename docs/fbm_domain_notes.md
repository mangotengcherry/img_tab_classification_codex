# FBM 패턴 도메인 노트

이 노트는 `papers/RSVD.pdf`에 나온 FBM 패턴 관찰을 현재 실험에 어떻게 반영했는지 정리한 문서입니다.

## 논문에서 가져온 관점

- FBM grade는 failed cell 수를 직접 세는 값이라기보다, fail 강도를 단계로 나눈 값에 가깝습니다.
- 낮은 grade는 랜덤하게 흩어진 noise처럼 보일 수 있고, 높은 grade는 실제 공간 패턴을 더 잘 보여줍니다.
- 논문은 grade `3` 이상만 남기는 이진화 화면으로 패턴을 먼저 확인합니다.
- 패턴 예시는 크게 랜덤하게 흩어진 점, 길게 이어진 세로선, 짧은 가로선, 국소 block으로 볼 수 있습니다.
- 선이나 block처럼 구조가 있는 패턴은 몇 개의 큰 구조로 설명되고, 랜덤하게 흩어진 패턴은 더 복잡하게 퍼져 보입니다.

## 현재 repo에 반영한 내용

- `src/fbm_multimodal/fusion/fbm_patterns.py`
  - grade `>=3` 이진화 helper
  - 세로선, 가로선, block, edge ring, 랜덤 scatter 패턴 painter
  - image feature에 원본 grade 화면, 이진화 화면, eigen-image norm(논문 Eq. 8) 16개를 함께 넣는 helper
  - `eigenimage_norm_features` / `effective_rank`: 구조 패턴은 low-rank(norm 빠르게 감쇠),
    랜덤 single-bit은 high-rank(느리게 감쇠) — 논문 Fig. 6의 판별 근거
  - `nonneg_factorize`: 논문 RSVD가 확장하는 비음수 분해(Lee–Seung) 계열의 robust 구현
- `src/fbm_multimodal/fusion/data.py`
  - synthetic FBM 생성 시 낮은 grade 랜덤 scatter를 항상 배경으로 추가
  - 실제 불량 패턴은 grade `>=3`에서 남도록 높은 grade로 생성
- `reports/figures/07_binarized_pattern_gallery.png`
  - 현재 단일/중첩 실험 패턴을 grade `>=3` 화면으로 다시 보여주는 진단 그림
- `reports/figures/08_paper_pattern_stress_gallery.png`
  - 랜덤, 세로선, 가로선, block 패턴이 이진화 후에도 어떻게 보이는지 확인하는 그림
- `reports/figures/09_eigenimage_spectrum.png`
  - 구조 패턴 vs 랜덤 single-bit의 eigen-image norm 감쇠 곡선(논문 Eq. 8 / Fig. 6)

## 해석할 때 주의할 점

논문에서 말하는 `single-bit / non-single-bit`는 이 repo의 `단일 label / 중첩 label`과 같은 뜻이 아닙니다.

- 논문 관점: FBM 안에 공간 패턴이 있는지 없는지에 가까움
- 현재 실험 관점: 한 chip에 여러 불량 원인이 동시에 붙었는지 보는 multi-label 분류

따라서 논문의 single/non-single 분류기를 그대로 라벨로 쓰지는 않되, 논문의 **이진화 + eigen-image
norm feature(Eq. 8)** 는 fusion 모델의 image feature로 실제 반영했고, **고강도 grade 진단**과
**대표 패턴 시각화**도 함께 제공합니다.
