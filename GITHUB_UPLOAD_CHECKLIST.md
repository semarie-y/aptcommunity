# 최종 GitHub 업로드 체크리스트

아래 순서대로 업로드하면 최종 보고서 기준과 저장소 기준을 맞출 수 있습니다.

## 1. 반드시 업로드할 파일

### data/output

- [ ] `apt_final_1123.csv`
- [ ] `step7_target_summary.csv`
- [ ] `step7_demo_summary.csv`
- [ ] `step7_biz_summary.csv`
- [ ] `step9_best_model_summary.csv`
- [ ] `step9_f1_pos_pivot.csv`
- [ ] `step9_f1_macro_pivot.csv`
- [ ] `step10_shap_top3_summary.csv`
- [ ] `step11_q4_direction_summary.csv`
- [ ] `step11_hypothesis_final_conclusion.csv`

### notebooks/charts

- [ ] `chart1_facility_rate.png`
- [ ] `chart2_feature_dist.png`
- [ ] `chart3_corr_heatmap.png`
- [ ] `chart4_facility_comparison.png`
- [ ] `chart5_gu_heatmap.png`
- [ ] `chart6_hypothesis_scatter.png`
- [ ] `chart8_shap_importance.png`
- [ ] `chart9_q4_direction.png`
- [ ] `chart10_hypothesis_final_conclusion.png`

### reports

- [ ] `최종보고서_양세미_v4.docx`
- [ ] 필요 시 `최종보고서_양세미_v4.pdf`

## 2. 업로드하지 말 것

- [ ] `02 소상공인시장진흥공단_상가(상권)정보_서울_202512.csv`
- [ ] `소상공인 상권정보_서울_필터.csv`
- [ ] `models_step9/*.pkl`
- [ ] `final_report_assets.zip`
- [ ] Colab 임시 파일 전체

## 3. 기존 파일 처리

기존 파일은 바로 삭제하지 말고, 최종 제출 전까지는 legacy 자료로 유지하는 것을 권장합니다.

- `apt_final_v3_biz.csv`: 기존 Streamlit 앱 임시 기준 데이터
- `apt_final_v2_jumin.csv`: 중간 분석용 인구·세대 결합 데이터
- `복지분양시설_단지단위.csv`: 초기 시설 집계 데이터

## 4. 최종 기준 문구

저장소와 보고서에서 아래 표현으로 통일합니다.

- 최종 분석 단지 수: 1,123개
- 단지키 기준 집계, 단지키 중복 0개
- 상권 변수: 원본 소상공인 상권정보 업종코드 기준
- 모델 주 지표: 보유 클래스 F1
- 보조 지표: Macro F1, Balanced Accuracy
- Q4 결론: 부분 지지, 시설별 대체·보완 관계 혼재

## 5. Streamlit 정리 전 확인

- `data/output/apt_final_1123.csv` 업로드 완료
- `notebooks/charts/`의 최종 차트 업로드 완료
- `README.md` 기준과 실제 파일명이 일치하는지 확인
