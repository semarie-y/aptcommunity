# 아파트 커뮤니티 시설 공급 패턴 분석

서울시 신축 아파트 단지의 커뮤니티 시설 공급 패턴을 분석하고, 인구 구조·세대 구성·주변 상권 데이터를 활용해 시설 공급 여부를 예측·해석하는 데이터마이닝 프로젝트입니다.

> **연구 전제**  
> 본 연구의 타겟은 실제 이용 수요가 아니라 건축 인허가상 커뮤니티 시설 공급 여부입니다. 따라서 결과는 “실제 수요 예측”이 아니라 **공급 패턴 예측**으로 해석합니다.

- **과목**: 데이터마이닝
- **작성자**: 양세미 (2024720536)
- **최종 분석 대상**: 서울시 신축 아파트 **1,123개 단지**
- **최종 기준 데이터**: `data/output/apt_final_1123.csv`
- **Streamlit 앱**: https://apt-community-analysis.streamlit.app/

---

## 1. 연구 질문과 최종 결론

| 가설 | 내용 | 최종 판정 | 핵심 해석 |
|---|---|---|---|
| Q1 | 1인가구비율↑ → 공유오피스·회의시설↑ | 기각 | 단변량 상관이 거의 0이고 SHAP 순위도 낮아 핵심 변수로 보기 어려움 |
| Q2 | 아동비율↑ → 키즈시설↑ | 지지 | 아동비율은 키즈시설 예측에서 SHAP 1위이며 단변량 방향도 일관됨 |
| Q3 | 고령비율↑ → 시니어시설↑ | 기각/재해석 | 고령비율은 시니어시설과 음의 방향을 보여 신축 단지 공급지와 고령 인구 밀집지의 지리적 불일치 가능성 |
| Q4 | 외부 카페·헬스장·학원·독서실↑ → 단지 내 유사 시설↓ | 부분 지지 | 헬스장·독서실은 약한 대체 가능성, 카페·학원은 보완/혼재 관계 |

---

## 2. 최종 분석 기준

| 항목 | 최종 값 |
|---|---:|
| 최종 분석 단지 수 | 1,123개 |
| 자치구 수 | 25개 |
| 단지키 중복 | 0개 |
| 모델 타겟 | 6개 시설 |
| 최종 모델 피처 | 18개 |
| 상권 매칭 | 법정동 직접 매칭 1,123개 |
| 모델 주 평가지표 | 보유 클래스 F1, positive-class F1 |
| 보조 지표 | Macro F1, Balanced Accuracy |

---

## 3. 데이터 소스

| # | 데이터 | 출처 | 규모 | GitHub 포함 여부 |
|---|---|---|---:|---|
| 1 | 주택인허가 복리분양시설 | 건축HUB/세움터 | 5,441건 | 일부 포함 |
| 2 | 행안부 주민등록 인구통계 | 주민등록 인구통계 | 2024년 329개 동 기준 | 포함 가능 |
| 3 | 행안부 세대원수별 세대수 | 주민등록 인구통계 | 2024년 329개 동 기준 | 포함 가능 |
| 4 | 소상공인시장진흥공단 상권정보 원본 | 공공데이터포털 | 서울시 534,978건 | 원본은 용량상 미포함 권장 |

### 상권 변수 정의

최종 분석에서는 필터링 파일이 아니라 **원본 상권정보 파일의 업종코드**를 기준으로 변수를 생성했습니다.

| 변수 | 기준 코드 | 설명 |
|---|---|---|
| 카페수 | `I21201` | 카페 |
| 헬스장수 | `R10307` | 헬스장 |
| 학원수 | `P10501` | 입시·교과학원 |
| 기타학원수 | `P106` | 기타 교육 전체 |
| 독서실수 | `R10202` | 독서실/스터디카페 |
| 사우나수 | `S20801` | 목욕탕/사우나 |

모델에는 상권 원 카운트 대신 `log1p` 변환한 `*_log` 변수를 사용했습니다.

---

## 4. 폴더 구조 권장안

```text
aptcommunity/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── 복지분양시설_서울_통합.csv
│   │   ├── 서울_인구데이터_10년단위_3년치.csv
│   │   └── 서울_세대수_디테일전체.csv
│   └── output/
│       ├── apt_final_1123.csv
│       ├── step7_target_summary.csv
│       ├── step9_best_model_summary.csv
│       ├── step10_shap_top3_summary.csv
│       └── step11_hypothesis_final_conclusion.csv
├── notebooks/
│   ├── APT_Community_Analysis.ipynb
│   └── charts/
│       ├── chart1_facility_rate.png
│       ├── chart2_feature_dist.png
│       ├── chart3_corr_heatmap.png
│       ├── chart4_facility_comparison.png
│       ├── chart5_gu_heatmap.png
│       ├── chart6_hypothesis_scatter.png
│       ├── chart8_shap_importance.png
│       ├── chart9_q4_direction.png
│       └── chart10_hypothesis_final_conclusion.png
├── reports/
│   └── 최종보고서_양세미_v4.docx
└── results/
    ├── tables/
    └── model_results/
```

---

## 5. 최종 타겟 보유율

| 시설 유형 | 보유 단지 수 | 보유율 |
|---|---:|---:|
| 키즈시설 | 621 | 55.3% |
| 시니어시설 | 510 | 45.4% |
| 운동시설 | 377 | 33.6% |
| 교육_학습시설 | 351 | 31.3% |
| 공유오피스_회의 | 197 | 17.5% |
| 라운지_휴게 | 176 | 15.7% |

편의_생활시설은 보유 단지 수가 적어 최종 모델 타겟에서 제외했습니다.

---

## 6. 모델링 요약

- **모델 비교**: Dummy majority baseline, Logistic Regression, Random Forest
- **검증 방식**: 5-Fold Stratified Cross Validation
- **주 지표**: 보유 클래스 F1
- **보조 지표**: Macro F1, Balanced Accuracy, Precision/Recall
- **클래스 불균형 대응**: `class_weight='balanced'`

| 시설 유형 | 보유율 | 최종 모델 | 보유 클래스 F1 | Macro F1 | 해석 |
|---|---:|---|---:|---:|---|
| 운동시설 | 33.6% | Logistic Regression | 0.423 | 0.526 | 유의미 |
| 교육_학습시설 | 31.3% | Logistic Regression | 0.424 | 0.545 | 유의미 |
| 키즈시설 | 55.3% | Random Forest | 0.580 | 0.547 | 보조 지표 중심 해석 필요 |
| 시니어시설 | 45.4% | Random Forest | 0.534 | 0.565 | 유의미 |
| 공유오피스_회의 | 17.5% | Logistic Regression | 0.295 | 0.467 | 낮은 보유율로 제한 |
| 라운지_휴게 | 15.7% | Random Forest | 0.289 | 0.544 | 선택시설 특성상 제한 |

---

## 7. SHAP 해석 요약

| 시설 유형 | 최종 모델 | SHAP Top 3 변수 |
|---|---|---|
| 운동시설 | Logistic Regression | 카페수(log) > 서남권 > 2인가구비율 |
| 교육_학습시설 | Logistic Regression | 아동비율 > 학원수(log) > 헬스장수(log) |
| 키즈시설 | Random Forest | 아동비율 > 2인가구비율 > 사우나수(log) |
| 시니어시설 | Random Forest | 아동비율 > 2인가구비율 > 3인이상비율 |
| 공유오피스_회의 | Logistic Regression | 카페수(log) > 사우나수(log) > 동북권 |
| 라운지_휴게 | Random Forest | 3인이상비율 > 1인가구비율 > 아동비율 |

> SHAP 평균 절댓값은 변수 영향력의 크기입니다. 방향성은 feature-SHAP 상관과 분위별 보유율 분석을 함께 보아야 합니다.

---

## 8. 실행 방법

### Colab 분석 재현

1. `notebooks/APT_Community_Analysis.ipynb` 열기
2. `data/raw/`의 인구·세대·복리분양시설 CSV 업로드
3. 원본 상권정보 파일 `02 소상공인시장진흥공단_상가(상권)정보_서울_202512.csv`는 용량상 GitHub에 올리지 말고 Colab에 직접 업로드
4. 노트북의 1~12단계를 순서대로 실행
5. 생성된 최종 파일 중 필요한 파일만 GitHub에 업로드

### Streamlit 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit 앱은 `data/output/apt_final_1123.csv`가 있으면 최종 데이터 기준으로 실행하고, 없으면 기존 `apt_final_v3_biz.csv`를 임시로 사용합니다.

---

## 9. GitHub 업로드 시 주의

업로드 권장:

- `data/output/apt_final_1123.csv`
- `data/output/step7_target_summary.csv`
- `data/output/step9_best_model_summary.csv`
- `data/output/step10_shap_top3_summary.csv`
- `data/output/step11_hypothesis_final_conclusion.csv`
- `notebooks/charts/chart1~chart10*.png`
- `reports/최종보고서_양세미_v4.docx`

업로드 비권장:

- 원본 상권정보 대용량 CSV
- Colab 임시 산출물 전체
- `models_step9/*.pkl` 모델 파일, 꼭 필요한 경우에만 업로드

---

## 10. 다음 정리 작업

- [ ] 최종 보고서 v4 업로드
- [ ] `apt_final_1123.csv` 업로드
- [ ] 새 EDA/SHAP 차트 업로드
- [ ] 기존 `apt_final_v3_biz.csv`는 legacy 파일로 유지 또는 삭제
- [ ] Streamlit 앱에서 최종 데이터 기준 화면 확인
