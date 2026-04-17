# aptcommunity
서울시 공동주택(아파트) 커뮤니티 시설 공급 패턴 분석

import os

os.chdir('/content/aptcommunity')

# README.md 업데이트
readme = """# 아파트 커뮤니티 시설 공급 패턴 분석

## 프로젝트 개요
서울시 신축 아파트 단지의 커뮤니티 시설 공급 패턴을 분석하고,
인구 구조·세대 구성·주변 상권 데이터를 활용한 수요 간접 예측 모델을 구축합니다.

- **과목**: 데이터마이닝
- **작성자**: 양세미 (2024720536)
- **분석 대상**: 서울시 신축 아파트 단지 997개

---

## 분석 파이프라인

```python
수집 → 전처리(시설 분류) → JOIN → EDA → 모델 → SHAP → 시각화
```

---

## 데이터 소스

| 데이터 | 출처 | 규모 | 포함 여부 |
|--------|------|------|-----------|
| 세움터 건축HUB 복분양시설 | [건축HUB](https://www.hub.go.kr/portal/opn/tyb/idx-hspmslg-wlofc.do) | 5,441건 → 997단지 | ✅ data/output/ |
| 행안부 주민등록 인구통계 | [주민등록통계](https://jumin.mois.go.kr) | 327개 동 | ✅ data/raw/ |
| 행안부 세대원수별 세대수 | [주민등록통계](https://jumin.mois.go.kr) | 327개 동 | ✅ data/raw/ |
| 소상공인 상권정보 | [공공데이터포털](https://www.data.go.kr) | 252,430개 업소 | ❌ 용량 초과 (아래 참고) |

> **소상공인 상권정보 다운로드 방법**  
> data.go.kr → '소상공인 상권정보' 검색 → 서울시 파일 다운로드  
> 중분류 코드 필터링 후 사용 (상권코드.csv 참고)

---

## 폴더 구조
aptcommunity/
├── data/
│   ├── raw/                          # 원본 수집 데이터
│   │   ├── 서울_인구데이터_10년단위_3년치.csv
│   │   └── 서울_세대수_디테일전체.csv
│   └── output/                       # 전처리 완료 데이터
│       └── 복지분양시설_단지단위.csv
├── notebooks/
│   ├── APT_Community_Analysis.ipynb  # 전체 분석 노트북
│   └── charts/                       # EDA 시각화 차트
│       ├── chart1_facility_rate.png
│       ├── chart2_feature_dist.png
│       ├── chart3_corr_heatmap.png
│       ├── chart4_facility_comparison.png
│       ├── chart5_gu_heatmap.png
│       └── chart6_hypothesis_scatter.png
└── README.md

---

## 노트북 실행 방법

1. [Google Colab](https://colab.research.google.com)에서 `notebooks/APT_Community_Analysis.ipynb` 열기
2. `data/raw/` 및 `data/output/` 파일 Colab에 업로드
3. 섹션 1~4 순서대로 실행 (EDA까지 완료)
4. 섹션 5~6은 모델 학습 및 SHAP 분석 (진행 예정)

---

## 핵심 전처리 — 시설 분류 전략

세움터 데이터의 시설명이 '피트니스', '휘트니스', 'GX룸' 등 다양한 표기로 입력되어  
**200개 이상의 키워드를 직접 조사**하여 8개 유형으로 분류하였음.

| 시설 유형 | 보유 단지 | 보유율 |
|-----------|-----------|--------|
| 키즈시설 | 618개 | 61.9% |
| 시니어시설 | 550개 | 55.1% |
| 교육_학습시설 | 388개 | 38.9% |
| 운동시설 | 368개 | 36.9% |
| 공유오피스_회의 | 204개 | 20.4% |
| 라운지_휴게 | 168개 | 16.9% |

---

## EDA 주요 발견

- **1인가구 비율** 동별 편차 최대 (20%~83%) — 핵심 피처 예상
- **아동비율 ↑ → 키즈시설 ↑** (+0.19) — Q2 가설 방향 지지
- **고령비율 ↑ → 시니어시설** 관계가 예상과 반대 (-0.10) — 다변량 재검증 필요
- **1인가구비율 ↔ 3인이상비율** 강한 다중공선성 (-0.96) — 모델 설계 시 주의

---

## 향후 계획

- [ ] Random Forest 분류 모델 (시설 유형별 6개 이진 분류)
- [ ] SHAP 변수 기여도 분석 (Q1~Q4 가설 정량 검증)
- [ ] Folium 서울 25구 시설 보급률 지도
- [ ] Streamlit 대시보드 배포
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)

print('README 작성 완료')

