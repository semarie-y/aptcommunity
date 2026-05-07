from pathlib import Path

import streamlit as st
import pandas as pd


# =========================
# 기본 설정
# =========================

st.set_page_config(
    page_title="아파트 커뮤니티 시설 공급 패턴 분석",
    page_icon="🏢",
    layout="wide"
)


# =========================
# CSS 디자인
# =========================

def inject_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        .hero {
            padding: 2.4rem 2.6rem;
            border-radius: 26px;
            background: linear-gradient(135deg, #0F172A 0%, #1D4ED8 58%, #38BDF8 100%);
            color: white;
            margin-bottom: 1.4rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.22);
        }

        .hero .eyebrow {
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #BFDBFE;
            margin-bottom: 0.7rem;
        }

        .hero h1 {
            color: white;
            font-size: 2.45rem;
            line-height: 1.25;
            margin: 0 0 0.7rem 0;
        }

        .hero p {
            color: #E0F2FE;
            font-size: 1.05rem;
            line-height: 1.65;
            margin: 0;
        }

        .section-title {
            font-size: 1.22rem;
            font-weight: 800;
            color: #0F172A;
            margin-top: 1.7rem;
            margin-bottom: 0.7rem;
        }

        .soft-card {
            padding: 1.25rem 1.35rem;
            border-radius: 20px;
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }

        .soft-card h3 {
            color: #0F172A;
            font-size: 1.08rem;
            margin-top: 0;
            margin-bottom: 0.45rem;
        }

        .soft-card p {
            color: #475569;
            font-size: 0.94rem;
            line-height: 1.58;
            margin-bottom: 0;
        }

        .badge {
            display: inline-block;
            padding: 0.32rem 0.65rem;
            border-radius: 999px;
            background: #DBEAFE;
            color: #1D4ED8;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.65rem;
        }

        .note-box {
            padding: 1rem 1.15rem;
            border-radius: 18px;
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

        [data-testid="stMetricLabel"] {
            color: #64748B;
            font-weight: 700;
        }

        [data-testid="stMetricValue"] {
            color: #0F172A;
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# =========================
# 데이터 로드
# =========================

@st.cache_data
def load_data():
    data_path = Path("data/output/apt_final_v3_biz.csv")

    if not data_path.exists():
        st.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        st.stop()

    try:
        return pd.read_csv(data_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(data_path, encoding="cp949")


df = load_data()
inject_css()

CHART_DIR = Path("notebooks/charts")

# =========================
# 변수 설정
# =========================

TARGETS = [
    "운동시설",
    "교육_학습시설",
    "키즈시설",
    "시니어시설",
    "공유오피스_회의",
    "라운지_휴게"
]

available_targets = [col for col in TARGETS if col in df.columns]

if available_targets:
    facility_rate = df[available_targets].mean().sort_values(ascending=False) * 100
    top_facility = facility_rate.index[0]
    top_rate = facility_rate.iloc[0]
else:
    facility_rate = None
    top_facility = "확인 필요"
    top_rate = 0


# =========================
# 사이드바
# =========================

st.sidebar.markdown("## 🏢 APT Community")
st.sidebar.markdown("서울시 신축 아파트 커뮤니티 시설 공급 패턴 분석")
st.sidebar.divider()

page = st.sidebar.radio(
    "페이지 이동",
    ["홈", "데이터 미리보기", "시설 보유율", "EDA 차트 모음"],
    index=0
)

st.sidebar.divider()
st.sidebar.caption("Data Mining Project")
st.sidebar.caption("2024720536 양세미")


# =========================
# 홈
# =========================

if page == "홈":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Data Mining Project</div>
            <h1>서울시 신축 아파트<br>커뮤니티 시설 공급 패턴 분석</h1>
            <p>
                인구 구조·세대 구성·주변 상권 데이터를 활용하여  
                아파트 커뮤니티 시설의 공급 패턴을 탐색하는 Streamlit 대시보드입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="note-box">
            <b>연구 전제</b><br>
            시행사의 시설 공급 패턴을 수요의 간접 지표로 활용하되,
            결과는 실제 수요 예측이 아니라 <b>공급 패턴 예측</b>으로 해석합니다.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">핵심 지표</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("분석 단지 수", f"{len(df):,}개")
    c2.metric("시설 타깃 수", f"{len(available_targets)}개")

    if "구" in df.columns:
        c3.metric("자치구 수", f"{df['구'].nunique()}개")
    else:
        c3.metric("자치구 수", "확인 필요")

    if available_targets:
        c4.metric("최고 보유 시설", top_facility, f"{top_rate:.1f}%")
    else:
        c4.metric("최고 보유 시설", "확인 필요")

    st.markdown('<div class="section-title">연구 질문</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="soft-card">
            <h3>서울시 신축 아파트에서 인구 구조와 주변 상권은 커뮤니티 시설 공급과 관련이 있는가?</h3>
            <p>
                읍면동별 인구 구조, 세대 구성, 주변 상권 밀도가
                키즈시설·시니어시설·공유오피스·운동시설 등 커뮤니티 시설 공급 여부를
                설명할 수 있는지 탐색합니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">가설 구조</div>', unsafe_allow_html=True)

    hypotheses = [
        (
            "Q1",
            "1인가구비율 → 공유오피스",
            "1인가구 비율이 높은 지역에서 공유오피스·회의시설 공급률이 높은지 확인합니다."
        ),
        (
            "Q2",
            "아동비율 → 키즈시설",
            "아동 비율이 높은 지역에서 키즈시설 공급률이 높은지 확인합니다."
        ),
        (
            "Q3",
            "고령비율 → 시니어시설",
            "고령 비율이 높은 지역에서 시니어시설 공급률이 높은지 확인합니다."
        ),
        (
            "Q4",
            "상권 밀도 → 단지 내 시설",
            "주변 카페·헬스장 밀도가 높을수록 단지 내 유사 시설이 감소하는지 확인합니다."
        )
    ]

    for i in range(0, len(hypotheses), 2):
        cols = st.columns(2)
        for col, item in zip(cols, hypotheses[i:i+2]):
            q, title, desc = item
            with col:
                st.markdown(
                    f"""
                    <div class="soft-card">
                        <div class="badge">{q}</div>
                        <h3>{title}</h3>
                        <p>{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown('<div class="section-title">현재 앱에서 확인 가능한 내용</div>', unsafe_allow_html=True)

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            """
            <div class="soft-card">
                <h3>📁 데이터 현황</h3>
                <p>최종 분석 데이터의 행·열 개수, 컬럼 목록, 데이터 샘플을 확인합니다.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b:
        st.markdown(
            """
            <div class="soft-card">
                <h3>📊 시설 보유율</h3>
                <p>시설 유형별 보유율과 보유 단지 수를 표와 그래프로 확인합니다.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c:
        st.markdown(
            """
            <div class="soft-card">
                <h3>🔎 향후 확장</h3>
                <p>최종 단계에서 Random Forest, SHAP, 지도 시각화를 추가할 수 있습니다.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================
# 데이터 미리보기
# =========================

elif page == "데이터 미리보기":
    st.title("📁 데이터 미리보기")

    c1, c2 = st.columns(2)
    c1.metric("행 개수", f"{df.shape[0]:,}개")
    c2.metric("열 개수", f"{df.shape[1]:,}개")

    st.markdown('<div class="section-title">데이터 샘플</div>', unsafe_allow_html=True)
    st.dataframe(df.head(30), use_container_width=True)

    with st.expander("컬럼 목록 보기"):
        st.write(df.columns.tolist())


# =========================
# 시설 보유율
# =========================

elif page == "시설 보유율":
    st.title("📊 시설별 보유율")

    if not available_targets:
        st.warning("시설 타깃 컬럼을 찾을 수 없습니다.")
        st.stop()

    rate = df[available_targets].mean().sort_values(ascending=False) * 100
    count = df[available_targets].sum().loc[rate.index]

    summary = pd.DataFrame({
        "시설": rate.index,
        "보유율(%)": rate.round(1).values,
        "보유 단지 수": count.astype(int).values
    })

    st.markdown('<div class="section-title">시설별 보유율 요약</div>', unsafe_allow_html=True)
    st.dataframe(summary, use_container_width=True)

    st.markdown('<div class="section-title">시설별 보유율 그래프</div>', unsafe_allow_html=True)
    st.bar_chart(rate)

# =========================
# EDA 차트 모음
# =========================

elif page == "EDA 차트 모음":
    st.title("🖼️ EDA 시각화 6종")

    st.markdown(
        """
        중간보고서에서 사용한 탐색적 데이터 분석(EDA) 시각화 자료를 확인하는 페이지입니다.  
        시설별 보유율, 인구·세대 변수 분포, 상관관계, 시설 보유 여부별 비교, 구별 보급률,
        핵심 가설 산점도를 한 곳에서 볼 수 있습니다.
        """
    )

    chart_files = {
        "그림 1. 시설별 보유율": {
            "file": "chart1_facility_rate.png",
            "desc": "서울시 신축 아파트 단지의 커뮤니티 시설 유형별 보유율을 비교합니다."
        },
        "그림 2. 인구·세대 변수 분포": {
            "file": "chart2_feature_dist.png",
            "desc": "아동비율, 고령비율, 1인가구비율 등 주요 인구·세대 변수의 분포를 확인합니다."
        },
        "그림 3. 피처 × 타겟 상관관계 히트맵": {
            "file": "chart3_corr_heatmap.png",
            "desc": "인구·세대 변수와 시설 보유 여부 사이의 상관관계를 확인합니다."
        },
        "그림 4. 시설 보유 여부별 인구 특성 비교": {
            "file": "chart4_facility_comparison.png",
            "desc": "시설 보유 단지와 미보유 단지의 인구·세대 특성 차이를 비교합니다."
        },
        "그림 5. 구별 시설 보급률 히트맵": {
            "file": "chart5_gu_heatmap.png",
            "desc": "서울 자치구별 커뮤니티 시설 보급률 차이를 확인합니다."
        },
        "그림 6. Q1·Q2·Q3 가설 검증 산점도": {
            "file": "chart6_hypothesis_scatter.png",
            "desc": "1인가구비율, 아동비율, 고령비율과 주요 시설 보유 여부의 관계를 확인합니다."
        },
    }

    tab1, tab2 = st.tabs(["하나씩 보기", "전체 보기"])

    with tab1:
        selected_chart = st.selectbox(
            "확인할 EDA 차트 선택",
            list(chart_files.keys())
        )

        chart_info = chart_files[selected_chart]
        chart_path = CHART_DIR / chart_info["file"]

        st.markdown('<div class="section-title">' + selected_chart + '</div>', unsafe_allow_html=True)
        st.caption(chart_info["desc"])

        if chart_path.exists():
            st.image(str(chart_path), use_container_width=True)
        else:
            st.error(f"차트 파일을 찾을 수 없습니다: {chart_path}")

    with tab2:
        st.markdown('<div class="section-title">EDA 차트 전체 보기</div>', unsafe_allow_html=True)

        chart_items = list(chart_files.items())

        for i in range(0, len(chart_items), 2):
            cols = st.columns(2)

            for col, (title, chart_info) in zip(cols, chart_items[i:i+2]):
                chart_path = CHART_DIR / chart_info["file"]

                with col:
                    st.markdown(f"### {title}")
                    st.caption(chart_info["desc"])

                    if chart_path.exists():
                        st.image(str(chart_path), use_container_width=True)
                    else:
                        st.error(f"파일 없음: {chart_path}")
