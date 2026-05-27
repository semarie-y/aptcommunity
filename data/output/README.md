# data/output

최종 보고서와 Streamlit 앱에서 사용하는 전처리 완료 데이터 파일을 저장하는 폴더입니다.

## 최종 기준 파일

| 파일 | 설명 | 필수 여부 |
|---|---|---|
| `apt_final_1123.csv` | 최종 분석 데이터셋. 단지키 기준 서울시 신축 아파트 1,123개 단지. | 필수 |
| `step7_target_summary.csv` | 시설별 보유 단지 수와 보유율 요약. | 권장 |
| `step9_best_model_summary.csv` | 시설별 최종 채택 모델과 성능 요약. | 권장 |
| `step10_shap_top3_summary.csv` | 시설별 SHAP Top 3 변수 요약. | 권장 |
| `step11_hypothesis_final_conclusion.csv` | Q1~Q4 최종 결론표. | 권장 |

## Legacy 파일

기존 중간분석 및 Streamlit 초기 버전에서 사용한 `apt_final_v3_biz.csv`, `apt_final_v2_jumin.csv`, `복지분양시설_단지단위.csv` 등은 필요하면 유지할 수 있으나, 최종 보고서 기준 데이터는 `apt_final_1123.csv`입니다.

## 업로드하지 않는 파일

원본 소상공인 상권정보 파일은 용량이 크므로 GitHub에 올리지 않습니다. Colab에서 직접 업로드하여 재현합니다.
