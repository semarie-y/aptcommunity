
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np


# =========================
# 기본 설정
# =========================

st.set_page_config(
    page_title="아파트 커뮤니티 시설 공급 패턴 분석",
    page_icon="🏢",
    layout="wide"
)

DATA_PATH = Path("data/output/apt_final_v3_biz.csv")
CHART_DIR = Path("notebooks/charts")

TARGETS = [
    "운동시설",
    "교육_학습시설",
    "키즈시설",
    "시니어시설",
    "공유오피스_회의",
    "라운지_휴게"
]

FEATURES = [
    "ratio_child",
    "ratio_elderly",
    "ratio_young",
    "ratio_middle",
    "ratio_1person",
    "ratio_2person",
    "ratio_3plus",
    "카페수",
    "헬스장수",
    "학원수",
    "독서실수"
]

FEATURE_LABELS = {
    "ratio_child": "아동비율",
    "ratio_elderly": "고령비율",
    "ratio_young": "청년비율",
    "ratio_middle": "중년비율",
    "ratio_1person": "1인가구비율",
    "ratio_2person": "2인가구비율",
    "ratio_3plus": "3인이상가구비율",
    "카페수": "카페수",
    "헬스장수": "헬스장수",
    "학원수": "학원수",
    "독서실수": "독서실수"
}


# =========================
# 데이터 로드 함수
# =========================

@st.cache_data
def load_data():
    """
    GitHub 저장소 안의 data/output/apt_final_v3_biz.csv를 불러온다.
    인코딩 문제가 날 수 있어서 utf-8-sig → cp949 순서로 시도한다.
    """
    if not DATA_PATH.exists():
        st.error(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        st.stop()

    try:
        return pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(DATA_PATH, encoding="cp949")


df = load_data()

available_targets = [c for c in TARGETS if c in df.columns]
available_features = [c for c in FEATURES if c in df.columns]


# =========================
# 사이드바
# =========================

st.sidebar.title("메뉴")

page = st.sidebar.radio(
    "페이지 선택",
    [
        "1. 프로젝트 개요",
        "2. 데이터 현황",
        "3. 시설 보유율",
        "4. 지역별 비교",
        "5. 인구·세대 변수",
        "6. 가설 예비검증",
        "7. EDA 차트 모음",
        "8. 모델·SHAP 예정"
    ]
)


# =========================
# 1. 프로젝트 개요
# =========================

if page == "1. 프로젝트 개요":
    st.title("🏢 아파트 커뮤니티 시설 공급 패턴 분석")

    st.markdown(
        """
        서울시 신축 아파트 단지를 대상으로 커뮤니티 시설 공급 패턴을 분석하고,  
        인구 구조·세대 구성·주변 상권 데이터를 활용하여 시설 공급 여부를 설명하는 프로젝트입니다.
        """
    )

    st.info(
        "연구 전제: 시행사의 시설 공급 패턴을 수요의 간접 지표로 활용하되, "
        "결과는 실제 수요 예측이 아니라 '공급 패턴 예측'으로 해석합니다."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("분석 단지 수", f"{len(df):,}개")
    c2.metric("시설 타깃 수", f"{len(available_targets)}개")

    if "구" in df.columns:
        c3.metric("자치구 수", f"{df['구'].nunique()}개")
    else:
        c3.metric("자치구 수", "확인 필요")

    if "동" in df.columns:
        c4.metric("동 수", f"{df['동'].nunique()}개")
    else:
        c4.metric("동 수", "확인 필요")

    st.subheader("연구 질문")
    st.write(
        "서울시 신축 아파트에서 읍면동별 인구 구조와 주변 상권 밀도는 "
        "커뮤니티 시설 유형의 공급 여부를 예측하는 데 유의미한 변수인가?"
    )

    st.subheader("가설")
    st.markdown(
        """
        - Q1: 1인가구 비율이 높을수록 공유오피스·회의시설 공급률이 높은가?
        - Q2: 아동 비율이 높을수록 키즈시설 공급률이 높은가?
        - Q3: 고령 비율이 높을수록 시니어시설 공급률이 높은가?
        - Q4: 주변 카페·헬스장 밀도가 높을수록 단지 내 해당 시설이 감소하는가?
        """
    )


# =========================
# 2. 데이터 현황
# =========================

elif page == "2. 데이터 현황":
    st.title("📁 데이터 현황")

    st.subheader("최종 분석 데이터 미리보기")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("데이터 크기")
    st.write(f"행 개수: {df.shape[0]:,}개")
    st.write(f"열 개수: {df.shape[1]:,}개")

    st.subheader("컬럼 목록")
    st.write(df.columns.tolist())

    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="최종 데이터 CSV 다운로드",
        data=csv_data,
        file_name="apt_final_v3_biz.csv",
        mime="text/csv"
    )


# =========================
# 3. 시설 보유율
# =========================

elif page == "3. 시설 보유율":
    st.title("📊 시설별 보유율")

    if not available_targets:
        st.error("시설 타깃 컬럼을 찾을 수 없습니다.")
        st.stop()

    rate = df[available_targets].mean().sort_values(ascending=False) * 100
    count = df[available_targets].sum().loc[rate.index]

    summary = pd.DataFrame({
        "시설": rate.index,
        "보유율(%)": rate.round(1).values,
        "보유 단지 수": count.astype(int).values
    })

    st.subheader("시설별 보유율 표")
    st.dataframe(summary, use_container_width=True)

    st.subheader("시설별 보유율 그래프")
    st.bar_chart(rate)


# =========================
# 4. 지역별 비교
# =========================

elif page == "4. 지역별 비교":
    st.title("🗺️ 구별 시설 보급률 비교")

    if "구" not in df.columns:
        st.error("'구' 컬럼이 없어 지역별 비교를 할 수 없습니다.")
        st.stop()

    if not available_targets:
        st.error("시설 타깃 컬럼을 찾을 수 없습니다.")
        st.stop()

    target = st.selectbox("시설 선택", available_targets)

    count_col = "건물명" if "건물명" in df.columns else df.columns[0]

    gu_summary = (
        df.groupby("구")
          .agg(
              단지수=(count_col, "count"),
              보급률=(target, "mean")
          )
          .reset_index()
    )

    gu_summary["보급률(%)"] = (gu_summary["보급률"] * 100).round(1)
    gu_summary = gu_summary.sort_values("보급률(%)", ascending=False)

    st.subheader(f"자치구별 {target} 보급률")
    st.dataframe(
        gu_summary[["구", "단지수", "보급률(%)"]],
        use_container_width=True
    )

    st.subheader("그래프")
    chart_data = gu_summary.set_index("구")["보급률(%)"]
    st.bar_chart(chart_data)

    st.warning("단지 수가 적은 구는 보급률이 극단적으로 보일 수 있으므로 해석에 주의해야 합니다.")


# =========================
# 5. 인구·세대 변수
# =========================

elif page == "5. 인구·세대 변수":
    st.title("👥 인구·세대 변수 분포")

    if not available_features:
        st.error("분석 가능한 인구·세대·상권 변수 컬럼을 찾을 수 없습니다.")
        st.stop()

    feature = st.selectbox(
        "변수 선택",
        available_features,
        format_func=lambda x: FEATURE_LABELS.get(x, x)
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("평균", round(df[feature].mean(), 3))
    c2.metric("중앙값", round(df[feature].median(), 3))
    c3.metric("최솟값", round(df[feature].min(), 3))
    c4.metric("최댓값", round(df[feature].max(), 3))

    st.subheader(f"{FEATURE_LABELS.get(feature, feature)} 분포")
    st.bar_chart(df[feature])

    if "구" in df.columns:
        st.subheader("자치구별 평균")
        gu_feature = (
            df.groupby("구")[feature]
              .mean()
              .sort_values(ascending=False)
              .reset_index()
        )
        gu_feature.columns = ["구", "평균값"]
        st.dataframe(gu_feature, use_container_width=True)


# =========================
# 6. 가설 예비검증
# =========================

elif page == "6. 가설 예비검증":
    st.title("🔎 Q1~Q4 가설 예비검증")

    hypothesis = st.selectbox(
        "가설 선택",
        [
            "Q1: 1인가구비율 → 공유오피스_회의",
            "Q2: 아동비율 → 키즈시설",
            "Q3: 고령비율 → 시니어시설",
            "Q4: 상권 변수 → 단지 내 시설"
        ]
    )

    if hypothesis.startswith("Q1"):
        x_col = "ratio_1person"
        y_col = "공유오피스_회의"
        explanation = "1인가구비율이 높은 지역에서 공유오피스·회의시설 공급률이 높은지 확인합니다."

    elif hypothesis.startswith("Q2"):
        x_col = "ratio_child"
        y_col = "키즈시설"
        explanation = "아동비율이 높은 지역에서 키즈시설 공급률이 높은지 확인합니다."

    elif hypothesis.startswith("Q3"):
        x_col = "ratio_elderly"
        y_col = "시니어시설"
        explanation = "고령비율이 높은 지역에서 시니어시설 공급률이 높은지 확인합니다."

    else:
        possible_biz = [c for c in ["카페수", "헬스장수", "학원수", "독서실수"] if c in df.columns]
        if not possible_biz:
            st.error("상권 변수 컬럼을 찾을 수 없습니다.")
            st.stop()

        x_col = st.selectbox("상권 변수 선택", possible_biz)
        y_col = st.selectbox("시설 선택", available_targets)
        explanation = "주변 상권 밀도가 단지 내 유사 시설 공급과 어떤 관계가 있는지 확인합니다."

    st.info(explanation)

    if x_col not in df.columns or y_col not in df.columns:
        st.error(f"필요한 컬럼이 없습니다: {x_col}, {y_col}")
        st.stop()

    compare = (
        df.groupby(y_col)[x_col]
          .agg(["count", "mean", "median"])
          .reset_index()
    )

    compare[y_col] = compare[y_col].map({0: "미보유", 1: "보유"}).fillna(compare[y_col])

    st.subheader("시설 보유 여부별 변수 평균 비교")
    st.dataframe(compare, use_container_width=True)

    st.subheader("산점도")
    st.scatter_chart(df[[x_col, y_col]])


# =========================
# 7. EDA 차트 모음
# =========================

elif page == "7. EDA 차트 모음":
    st.title("🖼️ EDA 차트 모음")

    chart_files = {
        "시설별 보유율": CHART_DIR / "chart1_facility_rate.png",
        "인구·세대 변수 분포": CHART_DIR / "chart2_feature_dist.png",
        "상관관계 히트맵": CHART_DIR / "chart3_corr_heatmap.png",
        "시설 보유 여부별 비교": CHART_DIR / "chart4_facility_comparison.png",
        "구별 시설 보급률": CHART_DIR / "chart5_gu_heatmap.png",
        "가설 검증 산점도": CHART_DIR / "chart6_hypothesis_scatter.png",
    }

    selected = st.selectbox("차트 선택", list(chart_files.keys()))
    selected_path = chart_files[selected]

    if selected_path.exists():
        st.image(str(selected_path), use_container_width=True)
    else:
        st.error(f"차트 파일을 찾을 수 없습니다: {selected_path}")


# =========================
# 8. 모델·SHAP 예정
# =========================

elif page == "8. 모델·SHAP 예정":
    st.title("🤖 모델·SHAP 분석 예정")

    st.markdown(
        """
        최종 분석 단계에서 아래 내용을 추가할 예정입니다.

        1. 시설 유형별 이진 분류 모델
        2. Logistic Regression baseline
        3. Random Forest main model
        4. 5-Fold CV 기반 Macro F1 평가
        5. SHAP 변수 중요도 분석
        6. Q1~Q4 가설 다변량 재검증
        """
    )

    st.info("현재 앱은 중간보고서 단계의 EDA와 가설 예비검증을 중심으로 구성되어 있습니다.")
