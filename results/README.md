# results

모델링·SHAP·가설 검증 결과표를 정리하는 폴더입니다.

## 권장 하위 폴더

```text
results/
├── tables/
└── model_results/
```

## tables 권장 파일

- `table_00_final_overview.csv`
- `table_01_facility_rate.csv`
- `table_02_model_performance_summary.csv`
- `table_05_shap_top3_summary.csv`
- `table_07_q4_direction_summary.csv`
- `table_08_hypothesis_final_conclusion.csv`

## model_results 권장 파일

- `step9_model_all_results.csv`
- `step9_best_model_summary.csv`
- `step9_f1_pos_pivot.csv`
- `step9_f1_macro_pivot.csv`
- `step10_shap_importance_long.csv`
- `step10_hypothesis_shap_summary.csv`

모델 pickle 파일은 용량과 재현성 관리 문제 때문에 기본적으로 업로드하지 않습니다. 필요할 때만 별도 업로드합니다.
