from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# 기본 설정
# ============================================================

st.set_page_config(
    page_title="아파트 커뮤니티 시설 공급 패턴 분석",
    page_icon="🏢",
    layout="wide"
)

FINAL_DATA_PATH = Path("data/output/apt_final_1123.csv")
LEGACY_DATA_PATH = Path("data/output/apt_final_v3_biz.csv")
CHART_DIR = Path("notebooks/charts")

TARGETS = [
    "운동시설",
    "교육_학습시설",
    "키즈시설",
    "시니어시설",
    "공유오피스_회의",
    "라운지_휴게",
]

DEMO_FEATURES = [
    "ratio_child",
    "ratio_elderly",
    "ratio_young",
    "ratio_middle",
    "ratio_1person",
    "ratio_2person",
    "ratio_3plus",
]

BIZ_FEATURES = [
    "카페수",
    "헬스장수",
    "학원수",
    "기타학원수",
    "독서실수",
    "사우나수",
]

TARGET_LABELS = {
    "운동시설": "운동시설",
    "교육_학습시설": "교육·학습시설",
    "키즈시설": "키즈시설",
    "시니어시설": "시니어시설",
    "공유오피스_회의": "공유오피스·회의",
    "라운지_휴게": "라운지·휴게",
}

FEATURE_LABELS = {
    "ratio_child": "아동비율",
    "ratio_elderly": "고령비율",
    "ratio_young": "청년비율",
    "ratio_middle": "중년비율",
    "ratio_1person": "1인가구비율",
    "ratio_2person": "2인가구비율",
    "ratio_3plus": "3인이상비율",
}


# ============================================================
# 스타일
# ============================================================

def inject_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        .hero {
            padding: 2.2rem 2.4rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #0F172A 0%, #1D4ED8 58%, #38BDF8 100%);
            color: white;
            margin-bottom: 1.3rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.22);
        }
        .hero h1 {
            color: white;
            font-size: 2.35rem;
            line-height: 1.25;
            margin: 0 0 0.7rem 0;
        }
        .hero p {
            color: #E0F2FE;
            font-size: 1.02rem;
            line-height: 1.65;
            margin: 0;
        }
        .section-title {
            font-size: 1.22rem;
            font-weight: 800;
            color: #0F172A;
            margin-top: 1.6rem;
            margin-bottom: 0.65rem;
        }
        .soft-card {
            padding: 1.15rem 1.25rem;
            border-radius: 18px;
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }
        .badge {
            display: inline-block;
            padding: 0.28rem 0.58rem;
            border-radius: 999px;
            background: #DBEAFE;
            color: #1D4ED8;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.55rem;
        }
        .note-box {
            padding: 1rem 1.15rem;
            border-radius: 16px;
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            color: #1E3A8A;
            line-height: 1.6;
            margin-bottom: 1rem;
        }
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            padding: 1rem 1.15rem;
            border-radius: 18px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.055);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 데이터 로드
# ============================================================

@st.cache_data
def load_data():
    if FINAL_DATA_PATH.exists():
        data_path = FINAL_DATA_PATH
        data_version = "final_1123"
    elif LEGACY_DATA_PATH.exists():
        data_path = LEGACY_DATA_PATH
        data_version = "legacy_v3_biz"
    else:
        st.error(
            "데이터 파일을 찾을 수 없습니다. "
            "data/output/apt_final_1123.csv 또는 data/output/apt_final_v3_biz.csv를 업로드하세요."
        )
        st.stop()

    try:
        df = pd.read_csv(data_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding="cp949")

    return df, data_path, data_version


df, data_path, data_version = load_data()
inject_css()

available_targets = [col for col in TARGETS if col in df.columns]
available_demo_features = [col for col in DEMO_FEATURES if col in df.columns]
available_biz_features = [col for col in BIZ_FEATURES if col in df.columns]

if available_targets:
    facility_rate = df[available_targets].mean().sort_values(ascending=False) * 100
    top_facility = facility_rate.index[0]
    top_rate = facility_rate.iloc[0]
else:
    facility_rate = pd.Series(dtype=float)
    top_facility = "확인 필요"
    top_rate = 0


# ============================================================
# 사이드바
# ============================================================

st.sidebar.markdown("## 🏢 APT Community")
st.sidebar.markdown("서울시 신축 아파트 커뮤니티 시설 공급 패턴 분석")
st.sidebar.divider()

page = st.sidebar.radio(
    "페이지 이동",
    [
        "홈",
        "데이터 미리보기",
        "시설 보유율",
        "인구·상권 변수",
        "EDA·SHAP 차트",
        "최종 결론",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.caption(f"Data: {data_path}")
st.sidebar.caption("Data Mining Project | 2024720536 양세미")


# ============================================================
# 홈
# ============================================================

if page == "홈":
    st.markdown(
        """
        <div class="hero">
            <h1>서울시 신축 아파트<br>커뮤니티 시설 공급 패턴 분석</h1>
            <p>
                인구 구조·세대 구성·원본 상권정보 데이터를 활용하여
                아파트 커뮤니티 시설의 공급 패턴을 탐색하고 해석하는 대시보드입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="note-box">
            <b>연구 전제</b><br>
            본 연구의 타겟은 실제 이용 수요가 아니라 건축 인허가상 커뮤니티 시설 공급 여부입니다.
            따라서 결과는 실제 수요 예측이 아니라 <b>공급 패턴 예측</b>으로 해석합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 단지 수", f"{len(df):,}개")
    c2.metric("시설 타겟 수", f"{len(available_targets)}개")
    c3.metric("자치구 수", f"{df['구'].nunique()}개" if "구" in df.columns else "확인 필요")
    c4.metric("최고 보유 시설", TARGET_LABELS.get(top_facility, top_facility), f"{top_rate:.1f}%")

    st.markdown('<div class="section-title">최종 가설 결론</div>', unsafe_allow_html=True)

    conclusions = [
        ("Q1", "1인가구비율 → 공유오피스·회의", "기각", "단변량 상관이 거의 0이고 SHAP 순위도 낮음"),
        ("Q2", "아동비율 → 키즈시설", "지지", "아동비율은 키즈시설 예측의 핵심 변수"),
        ("Q3", "고령비율 → 시니어시설", "기각/재해석", "고령비율은 오히려 미보유 방향"),
        ("Q4", "외부 상권 → 단지 내 유사 시설", "부분 지지", "시설별 대체·보완 관계가 혼재"),
    ]

    for i in range(0, len(conclusions), 2):
        cols = st.columns(2)
        for col, item in zip(cols, conclusions[i:i + 2]):
            q, title, verdict, desc = item
            with col:
                st.markdown(
                    f"""
                    <div class="soft-card">
                        <div class="badge">{q} | {verdict}</div>
                        <h3>{title}</h3>
                        <p>{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if data_version != "final_1123":
        st.warning(
            "현재 앱은 legacy 데이터로 실행 중입니다. "
            "최종 기준으로 보려면 data/output/apt_final_1123.csv를 업로드하세요."
        )


# ============================================================
# 데이터 미리보기
# ============================================================

elif page == "데이터 미리보기":
    st.title("📁 데이터 미리보기")

    c1, c2, c3 = st.columns(3)
    c1.metric("행 개수", f"{df.shape[0]:,}개")
    c2.metric("열 개수", f"{df.shape[1]:,}개")
    c3.metric("데이터 버전", data_version)

    st.dataframe(df.head(50), use_container_width=True)

    with st.expander("컬럼 목록 보기"):
        st.write(df.columns.tolist())

    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="현재 데이터 CSV 다운로드",
        data=csv_data,
        file_name=data_path.name,
        mime="text/csv",
    )


# ============================================================
# 시설 보유율
# ============================================================

elif page == "시설 보유율":
    st.title("📊 시설별 보유율")

    if not available_targets:
        st.warning("시설 타겟 컬럼을 찾을 수 없습니다.")
        st.stop()

    rate = df[available_targets].mean().sort_values(ascending=False) * 100
    count = df[available_targets].sum().loc[rate.index]

    summary = pd.DataFrame({
        "시설": [TARGET_LABELS.get(x, x) for x in rate.index],
        "보유율(%)": rate.round(1).values,
        "보유 단지 수": count.astype(int).values,
        "미보유 단지 수": (len(df) - count).astype(int).values,
    })

    st.dataframe(summary, use_container_width=True)
    st.bar_chart(rate)

    if "구" in df.columns:
        st.markdown('<div class="section-title">자치구별 보급률</div>', unsafe_allow_html=True)
        target = st.selectbox("시설 선택", available_targets, format_func=lambda x: TARGET_LABELS.get(x, x))
        gu_summary = (
            df.groupby("구")
            .agg(단지수=("단지키" if "단지키" in df.columns else df.columns[0], "count"), 보급률=(target, "mean"))
            .reset_index()
        )
        gu_summary["보급률(%)"] = (gu_summary["보급률"] * 100).round(1)
        gu_summary = gu_summary.sort_values("보급률(%)", ascending=False)
        st.dataframe(gu_summary[["구", "단지수", "보급률(%)"]], use_container_width=True)
        st.bar_chart(gu_summary.set_index("구")["보급률(%)"])


# ============================================================
# 인구·상권 변수
# ============================================================

elif page == "인구·상권 변수":
    st.title("👥 인구·상권 변수")

    tab1, tab2 = st.tabs(["인구·세대 변수", "상권 변수"])

    with tab1:
        if not available_demo_features:
            st.warning("인구·세대 변수 컬럼을 찾을 수 없습니다.")
        else:
            feature = st.selectbox(
                "인구·세대 변수 선택",
                available_demo_features,
                format_func=lambda x: FEATURE_LABELS.get(x, x),
            )
            values = df[feature]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("평균", f"{values.mean() * 100:.2f}%")
            c2.metric("중앙값", f"{values.median() * 100:.2f}%")
            c3.metric("최솟값", f"{values.min() * 100:.2f}%")
            c4.metric("최댓값", f"{values.max() * 100:.2f}%")
            st.bar_chart(values)

    with tab2:
        if not available_biz_features:
            st.warning("상권 변수 컬럼을 찾을 수 없습니다.")
        else:
            feature = st.selectbox("상권 변수 선택", available_biz_features)
            values = df[feature]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("평균", f"{values.mean():.2f}")
            c2.metric("중앙값", f"{values.median():.2f}")
            c3.metric("최솟값", f"{values.min():.2f}")
            c4.metric("최댓값", f"{values.max():.2f}")
            st.bar_chart(values)


# ============================================================
# EDA·SHAP 차트
# ============================================================

elif page == "EDA·SHAP 차트":
    st.title("🖼️ EDA·SHAP 차트 모음")

    chart_files = {
        "그림 1. 시설별 보유율": "chart1_facility_rate.png",
        "그림 2. 인구·세대 변수 분포": "chart2_feature_dist.png",
        "그림 3. 피처 × 타겟 상관관계 히트맵": "chart3_corr_heatmap.png",
        "그림 4. 시설 보유 여부별 인구·세대 특성 차이": "chart4_facility_comparison.png",
        "그림 5. 자치구별 시설 보급률 히트맵": "chart5_gu_heatmap.png",
        "그림 6. Q1~Q4 핵심 가설 예비검증": "chart6_hypothesis_scatter.png",
        "그림 8. 시설별 SHAP 변수 중요도": "chart8_shap_importance.png",
        "그림 9. Q4 상권 방향 분석": "chart9_q4_direction.png",
        "그림 10. Q1~Q4 최종 결론표": "chart10_hypothesis_final_conclusion.png",
    }

    tab1, tab2 = st.tabs(["하나씩 보기", "전체 보기"])

    with tab1:
        selected = st.selectbox("차트 선택", list(chart_files.keys()))
        chart_path = CHART_DIR / chart_files[selected]
        st.subheader(selected)
        if chart_path.exists():
            st.image(str(chart_path), use_container_width=True)
        else:
            st.error(f"차트 파일을 찾을 수 없습니다: {chart_path}")

    with tab2:
        items = list(chart_files.items())
        for i in range(0, len(items), 2):
            cols = st.columns(2)
            for col, (title, filename) in zip(cols, items[i:i + 2]):
                with col:
                    chart_path = CHART_DIR / filename
                    st.markdown(f"### {title}")
                    if chart_path.exists():
                        st.image(str(chart_path), use_container_width=True)
                    else:
                        st.caption(f"파일 없음: {chart_path}")


# ============================================================
# 최종 결론
# ============================================================

elif page == "최종 결론":
    st.title("✅ 최종 결론")

    conclusion_rows = [
        {"가설": "Q1", "판정": "기각", "해석": "1인가구비율은 공유오피스·회의시설 공급과 단변량 상관이 거의 없고 SHAP 순위도 낮음"},
        {"가설": "Q2", "판정": "지지", "해석": "아동비율은 키즈시설 예측에서 가장 일관된 핵심 변수"},
        {"가설": "Q3", "판정": "기각/재해석", "해석": "고령비율은 시니어시설 공급과 음의 방향을 보여 지리적 불일치 가능성"},
        {"가설": "Q4", "판정": "부분 지지", "해석": "외부 상권의 대체효과는 시설별로 다르며 카페·학원은 보완/혼재 관계"},
    ]

    st.dataframe(pd.DataFrame(conclusion_rows), use_container_width=True)

    st.info(
        "최종 보고서에서는 기존 964개·1,044개 기준이 아니라 "
        "단지키 기준 1,123개 단지, 원본 상권정보 업종코드 기준 상권 변수, "
        "보유 클래스 F1 중심 평가로 통일합니다."
    )
