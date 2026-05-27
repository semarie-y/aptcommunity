from pathlib import Path
import copy
import json

import numpy as np
import pandas as pd
import streamlit as st
import folium
from branca.colormap import LinearColormap
from streamlit_folium import st_folium


# ============================================================
# 기본 설정
# ============================================================

st.set_page_config(
    page_title="아파트 커뮤니티 시설 공급 패턴 분석",
    page_icon="🏢",
    layout="wide"
)

DATA_PATH = Path("data/output/apt_final_1123.csv")
CHART_DIR = Path("notebooks/charts")
MAP_DIR = Path("maps")
GEO_PATH = MAP_DIR / "seoul_gu_geo_enriched.json"
MAP_SUMMARY_PATH = MAP_DIR / "seoul_gu_map_summary.csv"

TARGETS = [
    "운동시설",
    "교육_학습시설",
    "키즈시설",
    "시니어시설",
    "공유오피스_회의",
    "라운지_휴게",
]

BIZ_FEATURES = [
    "카페수",
    "헬스장수",
    "학원수",
    "기타학원수",
    "독서실수",
    "사우나수",
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
# CSS
# ============================================================

def inject_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }

        .hero {
            padding: 2.4rem 2.5rem;
            border-radius: 28px;
            background: linear-gradient(135deg, #0F172A 0%, #1D4ED8 52%, #38BDF8 100%);
            color: white;
            margin-bottom: 1.4rem;
            box-shadow: 0 20px 48px rgba(15, 23, 42, 0.24);
        }

        .hero h1 {
            color: white;
            font-size: 2.45rem;
            line-height: 1.25;
            margin: 0 0 0.75rem 0;
        }

        .hero p {
            color: #E0F2FE;
            font-size: 1.06rem;
            line-height: 1.65;
            margin: 0;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 800;
            color: #0F172A;
            margin-top: 1.7rem;
            margin-bottom: 0.75rem;
        }

        .soft-card {
            padding: 1.18rem 1.28rem;
            border-radius: 20px;
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
            min-height: 135px;
        }

        .soft-card h3 {
            font-size: 1.03rem;
            margin: 0.2rem 0 0.45rem 0;
            color: #111827;
        }

        .soft-card p {
            color: #475569;
            line-height: 1.55;
            font-size: 0.93rem;
            margin: 0;
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

        .badge {
            display: inline-block;
            padding: 0.3rem 0.68rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.55rem;
        }

        .badge-green {
            background: #DCFCE7;
            color: #166534;
        }

        .badge-red {
            background: #FEE2E2;
            color: #991B1B;
        }

        .badge-yellow {
            background: #FEF3C7;
            color: #92400E;
        }

        .badge-blue {
            background: #DBEAFE;
            color: #1D4ED8;
        }

        .conclusion-card {
            padding: 1.2rem 1.25rem;
            border-radius: 18px;
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.055);
            margin-bottom: 1rem;
            min-height: 170px;
        }

        .conclusion-card h3 {
            font-size: 1.08rem;
            margin-top: 0.2rem;
            margin-bottom: 0.6rem;
        }

        .conclusion-card p {
            color: #475569;
            line-height: 1.55;
            font-size: 0.93rem;
            margin-bottom: 0.35rem;
        }

        .card-green {
            border-left: 8px solid #22C55E;
        }

        .card-red {
            border-left: 8px solid #EF4444;
        }

        .card-yellow {
            border-left: 8px solid #F59E0B;
        }

        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            padding: 1rem 1.15rem;
            border-radius: 18px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.055);
        }

        .small-muted {
            color: #64748B;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 데이터 로드
# ============================================================

@st.cache_data
def read_csv_safely(path: Path):
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"CSV 인코딩을 확인하세요: {path}")


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(
            "최종 데이터 파일을 찾을 수 없습니다. "
            "`data/output/apt_final_1123.csv`를 업로드한 뒤 다시 실행하세요."
        )
        st.stop()

    return read_csv_safely(DATA_PATH)


@st.cache_data
def load_geojson():
    if not GEO_PATH.exists():
        return None

    with open(GEO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_map_summary():
    if MAP_SUMMARY_PATH.exists():
        return read_csv_safely(MAP_SUMMARY_PATH)

    return None


df = load_data()
inject_css()

available_targets = [c for c in TARGETS if c in df.columns]
available_biz_features = [c for c in BIZ_FEATURES if c in df.columns]
available_demo_features = [c for c in DEMO_FEATURES if c in df.columns]


# ============================================================
# 지도용 데이터 생성
# ============================================================

def build_gu_summary_from_df(data: pd.DataFrame) -> pd.DataFrame:
    id_col = "단지키" if "단지키" in data.columns else "구"

    agg_dict = {
        "단지수": (id_col, "count"),
    }

    for target in available_targets:
        agg_dict[f"{target}_보급률"] = (target, "mean")

    for col in available_biz_features:
        agg_dict[f"{col}_평균"] = (col, "mean")

    gu = (
        data
        .groupby("구")
        .agg(**agg_dict)
        .reset_index()
    )

    for target in available_targets:
        gu[f"{target}_보급률_pct"] = (gu[f"{target}_보급률"] * 100).round(1)

    if "키즈시설_보급률_pct" in gu.columns and "교육_학습시설_보급률_pct" in gu.columns:
        gu["가족친화지수"] = (
            gu["키즈시설_보급률_pct"] + gu["교육_학습시설_보급률_pct"]
        ) / 2

    if "공유오피스_회의_보급률_pct" in gu.columns and "라운지_휴게_보급률_pct" in gu.columns:
        gu["선택형시설지수"] = (
            gu["공유오피스_회의_보급률_pct"] + gu["라운지_휴게_보급률_pct"]
        ) / 2

    if "운동시설_보급률_pct" in gu.columns:
        gu["운동복지지수"] = gu["운동시설_보급률_pct"]

    for col in gu.columns:
        if col.endswith("_평균") or col.endswith("지수"):
            gu[col] = gu[col].round(1)

    return gu


def get_map_summary() -> pd.DataFrame:
    uploaded_summary = load_map_summary()

    if uploaded_summary is not None:
        return uploaded_summary

    return build_gu_summary_from_df(df)


def enrich_geojson_with_summary(geo_data: dict, gu_summary: pd.DataFrame) -> dict:
    geo = copy.deepcopy(geo_data)
    summary_dict = gu_summary.set_index("구").to_dict(orient="index")

    for feature in geo["features"]:
        gu_name = feature["properties"].get("name")
        row = summary_dict.get(gu_name, {})

        for k, v in row.items():
            if pd.isna(v):
                feature["properties"][k] = None
            elif isinstance(v, (np.integer, np.floating, int, float)):
                feature["properties"][k] = float(v)
            else:
                feature["properties"][k] = v

    return geo


def get_palette(color_scheme: str):
    palettes = {
        "YlOrRd": ["#ffffcc", "#ffeda0", "#feb24c", "#f03b20", "#bd0026"],
        "YlGn": ["#ffffcc", "#c2e699", "#78c679", "#31a354", "#006837"],
        "PuBu": ["#fff7fb", "#d0d1e6", "#74a9cf", "#2b8cbe", "#045a8d"],
        "PuRd": ["#f7f4f9", "#d4b9da", "#df65b0", "#dd1c77", "#980043"],
        "OrRd": ["#fff7ec", "#fdd49e", "#fc8d59", "#d7301f", "#7f0000"],
        "YlGnBu": ["#ffffd9", "#c7e9b4", "#41b6c4", "#225ea8", "#081d58"],
    }
    return palettes.get(color_scheme, palettes["YlGnBu"])


def make_gu_choropleth(
    gu_summary: pd.DataFrame,
    geo_data: dict,
    value_col: str,
    title: str,
    unit: str = "%",
    color_scheme: str = "YlGnBu",
):
    values = gu_summary[value_col].dropna()

    if values.empty:
        st.warning(f"{value_col} 값이 없어 지도를 만들 수 없습니다.")
        return None

    vmin = float(values.min())
    vmax = float(values.max())

    if vmin == vmax:
        vmax = vmin + 1

    colormap = LinearColormap(
        colors=get_palette(color_scheme),
        vmin=vmin,
        vmax=vmax
    )
    colormap.caption = f"{title} ({unit})"

    enriched_geo = enrich_geojson_with_summary(geo_data, gu_summary)

    m = folium.Map(
        location=[37.5665, 126.9780],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    def style_function(feature):
        value = feature["properties"].get(value_col)

        if value is None:
            fill_color = "#CCCCCC"
        else:
            fill_color = colormap(value)

        return {
            "fillColor": fill_color,
            "color": "#555555",
            "weight": 1,
            "fillOpacity": 0.78,
        }

    def highlight_function(feature):
        return {
            "fillColor": "#FFD700",
            "color": "#000000",
            "weight": 2,
            "fillOpacity": 0.85,
        }

    tooltip_fields = ["name", "단지수", value_col]
    tooltip_aliases = ["자치구", "분석 단지 수", f"{title}"]

    for target in available_targets:
        col = f"{target}_보급률_pct"
        if col in gu_summary.columns:
            tooltip_fields.append(col)
            tooltip_aliases.append(f"{TARGET_LABELS.get(target, target)} 보급률(%)")

    for col in ["카페수_평균", "헬스장수_평균", "학원수_평균", "독서실수_평균"]:
        if col in gu_summary.columns:
            tooltip_fields.append(col)
            tooltip_aliases.append(col.replace("_평균", " 평균"))

    folium.GeoJson(
        enriched_geo,
        name=title,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=True,
            labels=True,
        )
    ).add_to(m)

    colormap.add_to(m)

    title_html = f"""
    <div style="
        position: fixed;
        top: 18px;
        left: 50px;
        z-index: 9999;
        background: white;
        padding: 12px 16px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        font-family: sans-serif;
    ">
        <div style="font-size: 17px; font-weight: 700;">{title}</div>
        <div style="font-size: 12px; color: #555;">
            서울시 신축 아파트 1,123개 단지 기준
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    return m


# ============================================================
# 결론 카드
# ============================================================

CONCLUSION_ROWS = [
    {
        "q": "Q1",
        "question": "1인가구비율이 높을수록 공유오피스·회의시설 공급률이 높은가?",
        "hypothesis": "1인가구비율↑ → 공유오피스·회의시설↑",
        "verdict": "기각",
        "emoji": "❌",
        "badge_class": "badge-red",
        "card_class": "card-red",
        "reason": "단변량 상관이 거의 0이고, SHAP 순위도 낮아 핵심 변수로 보기 어렵다.",
    },
    {
        "q": "Q2",
        "question": "아동비율이 높을수록 키즈시설 공급률이 높은가?",
        "hypothesis": "아동비율↑ → 키즈시설↑",
        "verdict": "지지",
        "emoji": "✅",
        "badge_class": "badge-green",
        "card_class": "card-green",
        "reason": "아동비율은 키즈시설 예측에서 SHAP 1위이며 단변량 방향도 일관된다.",
    },
    {
        "q": "Q3",
        "question": "고령비율이 높을수록 시니어시설 공급률이 높은가?",
        "hypothesis": "고령비율↑ → 시니어시설↑",
        "verdict": "기각/재해석",
        "emoji": "⚠️",
        "badge_class": "badge-red",
        "card_class": "card-red",
        "reason": "고령비율은 오히려 미보유 방향으로 작용해 신축 단지 공급지와 고령 인구 밀집지의 지리적 불일치를 시사한다.",
    },
    {
        "q": "Q4",
        "question": "외부 카페·헬스장·학원·독서실이 많을수록 단지 내 유사 시설은 감소하는가?",
        "hypothesis": "외부 상권↑ → 단지 내 유사 시설↓",
        "verdict": "부분 지지",
        "emoji": "🟨",
        "badge_class": "badge-yellow",
        "card_class": "card-yellow",
        "reason": "헬스장·독서실은 약한 대체 가능성이 있으나, 카페·학원은 보완/동반 관계도 보여 시설별로 혼재한다.",
    },
]


def render_conclusion_cards():
    for i in range(0, len(CONCLUSION_ROWS), 2):
        cols = st.columns(2)

        for col, row in zip(cols, CONCLUSION_ROWS[i:i + 2]):
            with col:
                st.markdown(
                    f"""
                    <div class="conclusion-card {row['card_class']}">
                        <span class="badge {row['badge_class']}">{row['emoji']} {row['q']} | {row['verdict']}</span>
                        <h3>{row['hypothesis']}</h3>
                        <p><b>질문:</b> {row['question']}</p>
                        <p><b>해석:</b> {row['reason']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# 사이드바
# ============================================================

st.sidebar.markdown("## 🏢 APT Community")
st.sidebar.markdown("서울시 신축 아파트 커뮤니티 시설 공급 패턴 분석")
st.sidebar.divider()

page = st.sidebar.radio(
    "페이지 이동",
    [
        "🏠 홈",
        "📁 데이터",
        "📊 시설 보유율",
        "🗺️ 지도 시각화",
        "🖼️ 차트 모음",
        "✅ 최종 결론",
    ],
    index=0
)

st.sidebar.divider()
st.sidebar.caption("최종 기준: 1,123개 단지")
st.sidebar.caption("Data Mining Project | 양세미")


# ============================================================
# 홈
# ============================================================

if page == "🏠 홈":
    st.markdown(
        """
        <div class="hero">
            <h1>🏢 서울시 신축 아파트<br>커뮤니티 시설 공급 패턴 분석</h1>
            <p>
                인구 구조 👥 · 세대 구성 🧩 · 원본 상권정보 🏪를 활용해
                아파트 커뮤니티 시설의 공급 패턴을 분석하고,
                Streamlit과 Folium 지도로 시각화한 대시보드입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="note-box">
            <b>연구 전제</b><br>
            본 연구는 실제 이용 수요가 아니라 <b>건축 인허가상 커뮤니티 시설 공급 여부</b>를 분석합니다.
            따라서 결과는 실제 수요 예측이 아니라 <b>공급 패턴 예측</b>으로 해석합니다.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏘️ 분석 단지", f"{len(df):,}개")
    c2.metric("🎯 시설 타겟", f"{len(available_targets)}개")
    c3.metric("🗺️ 자치구", f"{df['구'].nunique()}개" if "구" in df.columns else "확인 필요")

    if available_targets:
        top_facility = facility_rate.index[0]
        top_rate = facility_rate.iloc[0]
        c4.metric("🥇 최고 보유 시설", TARGET_LABELS.get(top_facility, top_facility), f"{top_rate:.1f}%")
    else:
        c4.metric("🥇 최고 보유 시설", "확인 필요")

    st.markdown('<div class="section-title">🔎 최종 가설 결론 한눈에 보기</div>', unsafe_allow_html=True)
    render_conclusion_cards()

    st.markdown('<div class="section-title">📌 앱에서 확인할 수 있는 내용</div>', unsafe_allow_html=True)

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            """
            <div class="soft-card">
                <span class="badge badge-blue">📊 EDA</span>
                <h3>시설 보유율·인구 변수·상관관계</h3>
                <p>최종 1,123개 단지 기준 시설별 보유율과 인구·세대 변수 분포를 확인합니다.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b:
        st.markdown(
            """
            <div class="soft-card">
                <span class="badge badge-blue">🗺️ Folium</span>
                <h3>서울 25구 지도 시각화</h3>
                <p>시설별 보급률, 가족친화 지수, Q4 상권 변수를 Choropleth 지도로 확인합니다.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c:
        st.markdown(
            """
            <div class="soft-card">
                <span class="badge badge-blue">🤖 SHAP</span>
                <h3>모델 해석과 가설 검증</h3>
                <p>SHAP 변수 중요도와 Q1~Q4 최종 결론을 시각적으로 정리합니다.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 데이터
# ============================================================

elif page == "📁 데이터":
    st.title("📁 데이터 현황")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("행 개수", f"{df.shape[0]:,}개")
    c2.metric("열 개수", f"{df.shape[1]:,}개")
    c3.metric("자치구 수", f"{df['구'].nunique()}개" if "구" in df.columns else "확인 필요")
    c4.metric("단지키 중복", f"{df['단지키'].duplicated().sum()}개" if "단지키" in df.columns else "확인 필요")

    st.markdown('<div class="section-title">최종 데이터 미리보기</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown('<div class="section-title">주요 변수</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("인구·세대 변수")
        st.write([FEATURE_LABELS.get(x, x) for x in available_demo_features])

    with col2:
        st.subheader("상권 변수")
        st.write(available_biz_features)

    with st.expander("전체 컬럼 보기"):
        st.write(df.columns.tolist())

    st.download_button(
        label="현재 데이터 CSV 다운로드",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="apt_final_1123.csv",
        mime="text/csv",
    )


# ============================================================
# 시설 보유율
# ============================================================

elif page == "📊 시설 보유율":
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

    st.markdown('<div class="section-title">자치구별 보급률 비교</div>', unsafe_allow_html=True)

    selected_target = st.selectbox(
        "시설 선택",
        available_targets,
        format_func=lambda x: TARGET_LABELS.get(x, x)
    )

    id_col = "단지키" if "단지키" in df.columns else df.columns[0]

    gu_summary = (
        df
        .groupby("구")
        .agg(
            단지수=(id_col, "count"),
            보급률=(selected_target, "mean")
        )
        .reset_index()
    )

    gu_summary["보급률(%)"] = (gu_summary["보급률"] * 100).round(1)
    gu_summary = gu_summary.sort_values("보급률(%)", ascending=False)

    st.dataframe(gu_summary[["구", "단지수", "보급률(%)"]], use_container_width=True)
    st.bar_chart(gu_summary.set_index("구")["보급률(%)"])


# ============================================================
# 지도 시각화
# ============================================================

elif page == "🗺️ 지도 시각화":
    st.title("🗺️ 서울 25구 Folium 지도 시각화")

    geo_data = load_geojson()

    if geo_data is None:
        st.error(
            "`maps/seoul_gu_geo_enriched.json` 파일을 찾을 수 없습니다. "
            "Folium 지도 생성 단계에서 만든 GeoJSON 파일을 GitHub의 `maps/` 폴더에 업로드하세요."
        )
        st.stop()

    gu_summary = get_map_summary()

    st.markdown(
        """
        자치구별 시설 보급률과 상권 변수를 Choropleth 지도로 확인합니다.  
        마우스를 자치구 위에 올리면 분석 단지 수, 시설별 보급률, 주요 상권 평균을 볼 수 있습니다.
        """
    )

    map_mode = st.selectbox(
        "지도 유형 선택",
        ["시설별 보급률", "종합지수", "Q4 상권 변수"]
    )

    color_scheme = "YlGnBu"

    if map_mode == "시설별 보급률":
        selected_target = st.selectbox(
            "시설 선택",
            available_targets,
            format_func=lambda x: TARGET_LABELS.get(x, x)
        )

        value_col = f"{selected_target}_보급률_pct"
        title = f"{TARGET_LABELS.get(selected_target, selected_target)} 보급률"
        unit = "%"

        color_map = {
            "운동시설": "PuBu",
            "교육_학습시설": "YlGn",
            "키즈시설": "OrRd",
            "시니어시설": "PuRd",
            "공유오피스_회의": "YlOrRd",
            "라운지_휴게": "YlGnBu",
        }
        color_scheme = color_map.get(selected_target, "YlGnBu")

    elif map_mode == "종합지수":
        index_options = {
            "가족친화 지수": "가족친화지수",
            "선택형·프리미엄 시설 지수": "선택형시설지수",
            "운동복지 지수": "운동복지지수",
        }

        selected_name = st.selectbox("지수 선택", list(index_options.keys()))
        value_col = index_options[selected_name]
        title = selected_name
        unit = "%"
        color_scheme = "OrRd" if value_col == "가족친화지수" else "PuBu"

    else:
        q4_options = {
            "카페수 평균": "카페수_평균",
            "헬스장수 평균": "헬스장수_평균",
            "학원수 평균": "학원수_평균",
            "독서실수 평균": "독서실수_평균",
        }

        selected_name = st.selectbox("상권 변수 선택", list(q4_options.keys()))
        value_col = q4_options[selected_name]
        title = selected_name
        unit = "개"
        color_scheme = "YlOrRd"

    if value_col not in gu_summary.columns:
        st.error(f"`{value_col}` 컬럼이 지도 요약 데이터에 없습니다.")
        st.stop()

    m = make_gu_choropleth(
        gu_summary=gu_summary,
        geo_data=geo_data,
        value_col=value_col,
        title=title,
        unit=unit,
        color_scheme=color_scheme
    )

    if m is not None:
        st_folium(m, width=1150, height=680)

    st.markdown('<div class="section-title">상위 자치구</div>', unsafe_allow_html=True)

    rank_table = (
        gu_summary[["구", "단지수", value_col]]
        .sort_values(value_col, ascending=False)
        .head(10)
        .rename(columns={value_col: title})
    )

    st.dataframe(rank_table, use_container_width=True)


# ============================================================
# 차트 모음
# ============================================================

elif page == "🖼️ 차트 모음":
    st.title("🖼️ EDA·SHAP 차트 모음")

    chart_files = {
        "그림 1. 시설별 보유율": {
            "file": "chart1_facility_rate.png",
            "desc": "시설 유형별 보유율과 보유 단지 수"
        },
        "그림 2. 인구·세대 변수 분포": {
            "file": "chart2_feature_dist.png",
            "desc": "아동비율, 고령비율, 1인가구비율 등 인구·세대 변수 분포"
        },
        "그림 3. 피처 × 타겟 상관관계 히트맵": {
            "file": "chart3_corr_heatmap.png",
            "desc": "최종 피처와 6개 시설 타겟 간 상관관계"
        },
        "그림 4. 시설 보유 여부별 인구·세대 특성 차이": {
            "file": "chart4_facility_comparison.png",
            "desc": "시설 보유 단지와 미보유 단지의 인구·세대 평균 차이"
        },
        "그림 5. 자치구별 시설 보급률 히트맵": {
            "file": "chart5_gu_heatmap.png",
            "desc": "서울 25개 자치구별 커뮤니티 시설 보급률"
        },
        "그림 6. Q1~Q4 핵심 가설 예비검증": {
            "file": "chart6_hypothesis_scatter.png",
            "desc": "Q1~Q4 단변량 및 분위별 보유율 검토"
        },
        "그림 8. 시설별 SHAP 변수 중요도": {
            "file": "chart8_shap_importance.png",
            "desc": "시설별 최종 모델의 SHAP Top 변수"
        },
        "그림 9. Q4 상권 방향 분석": {
            "file": "chart9_q4_direction.png",
            "desc": "외부 상권 변수와 단지 내 시설 보유율의 방향성"
        },
    }

    st.info("그림 10 최종 결론표 이미지는 제외하고, 최종 결론은 앱의 `✅ 최종 결론` 페이지에서 카드 형태로 제공합니다.")

    tab1, tab2 = st.tabs(["하나씩 보기", "전체 보기"])

    with tab1:
        selected_chart = st.selectbox("차트 선택", list(chart_files.keys()))

        chart_info = chart_files[selected_chart]
        chart_path = CHART_DIR / chart_info["file"]

        st.subheader(selected_chart)
        st.caption(chart_info["desc"])

        if chart_path.exists():
            st.image(str(chart_path), use_container_width=True)
        else:
            st.error(f"차트 파일을 찾을 수 없습니다: {chart_path}")

    with tab2:
        items = list(chart_files.items())

        for i in range(0, len(items), 2):
            cols = st.columns(2)

            for col, (title, chart_info) in zip(cols, items[i:i + 2]):
                with col:
                    st.markdown(f"### {title}")
                    st.caption(chart_info["desc"])

                    chart_path = CHART_DIR / chart_info["file"]

                    if chart_path.exists():
                        st.image(str(chart_path), use_container_width=True)
                    else:
                        st.caption(f"파일 없음: {chart_path}")


# ============================================================
# 최종 결론
# ============================================================

elif page == "✅ 최종 결론":
    st.title("✅ Q1~Q4 최종 결론")

    st.markdown(
        """
        아래 표는 최종 분석 결과를 바탕으로 정리한 Q1~Q4 가설 검증 결론입니다.  
        색상과 이모티콘으로 **지지 / 기각 / 부분 지지**를 구분했습니다.
        """
    )

    render_conclusion_cards()

    st.markdown('<div class="section-title">결론 요약표</div>', unsafe_allow_html=True)

    conclusion_df = pd.DataFrame([
        {
            "가설": row["q"],
            "질문": row["question"],
            "가설 내용": row["hypothesis"],
            "판정": f"{row['emoji']} {row['verdict']}",
            "핵심 해석": row["reason"],
        }
        for row in CONCLUSION_ROWS
    ])

    st.dataframe(conclusion_df, use_container_width=True)

    st.markdown('<div class="section-title">보고서용 핵심 문장</div>', unsafe_allow_html=True)

    st.markdown(
        """
        - **Q1 기각**: 1인가구비율은 공유오피스·회의시설 공급의 핵심 변수로 보기 어렵다.
        - **Q2 지지**: 아동비율은 키즈시설 공급을 설명하는 가장 일관된 변수로 확인되었다.
        - **Q3 기각/재해석**: 고령비율이 높을수록 시니어시설이 많아진다는 가설은 지지되지 않았다.
        - **Q4 부분 지지**: 외부 상권의 대체효과는 시설별로 다르며, 카페·학원은 보완/혼재 관계도 함께 나타났다.
        """
    )
