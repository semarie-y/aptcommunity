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
# 하드코딩 분석 결과 (최종보고서 기준)
# ============================================================

MODEL_PERFORMANCE = [
    {"시설": "키즈시설",        "보유율": 55.3, "Baseline보유F1": 0.712, "채택모델": "Random Forest",        "보유F1": 0.580, "MacroF1": 0.547, "BalAcc": 0.549, "판정": "⚠️ 재검토"},
    {"시설": "시니어시설",      "보유율": 45.4, "Baseline보유F1": 0.000, "채택모델": "Random Forest",        "보유F1": 0.534, "MacroF1": 0.565, "BalAcc": 0.566, "판정": "✅ 유의미"},
    {"시설": "운동시설",        "보유율": 33.6, "Baseline보유F1": 0.000, "채택모델": "Logistic Regression",  "보유F1": 0.423, "MacroF1": 0.526, "BalAcc": 0.535, "판정": "✅ 유의미"},
    {"시설": "교육·학습시설",   "보유율": 31.3, "Baseline보유F1": 0.000, "채택모델": "Logistic Regression",  "보유F1": 0.424, "MacroF1": 0.545, "BalAcc": 0.559, "판정": "✅ 유의미"},
    {"시설": "공유오피스·회의", "보유율": 17.5, "Baseline보유F1": 0.000, "채택모델": "Logistic Regression",  "보유F1": 0.295, "MacroF1": 0.467, "BalAcc": 0.542, "판정": "✅ 유의미"},
    {"시설": "라운지·휴게",     "보유율": 15.7, "Baseline보유F1": 0.000, "채택모델": "Random Forest",        "보유F1": 0.289, "MacroF1": 0.544, "BalAcc": 0.572, "판정": "✅ 유의미"},
]

SHAP_TOP3 = [
    {"시설": "운동시설",        "모델": "Logistic Regression", "보유F1": 0.423, "MacroF1": 0.526, "Top1": "카페수(log)",    "Top2": "서남권",       "Top3": "2인가구비율"},
    {"시설": "교육·학습시설",   "모델": "Logistic Regression", "보유F1": 0.424, "MacroF1": 0.545, "Top1": "아동비율",      "Top2": "학원수(log)",  "Top3": "헬스장수(log)"},
    {"시설": "키즈시설",        "모델": "Random Forest",       "보유F1": 0.580, "MacroF1": 0.547, "Top1": "아동비율",      "Top2": "2인가구비율",  "Top3": "사우나수(log)"},
    {"시설": "시니어시설",      "모델": "Random Forest",       "보유F1": 0.534, "MacroF1": 0.565, "Top1": "아동비율",      "Top2": "2인가구비율",  "Top3": "3인이상비율"},
    {"시설": "공유오피스·회의", "모델": "Logistic Regression", "보유F1": 0.295, "MacroF1": 0.467, "Top1": "카페수(log)",   "Top2": "사우나수(log)","Top3": "동북권"},
    {"시설": "라운지·휴게",     "모델": "Random Forest",       "보유F1": 0.289, "MacroF1": 0.544, "Top1": "3인이상비율",   "Top2": "1인가구비율",  "Top3": "아동비율"},
]

Q4_DIRECTION = [
    {"가설": "Q4-1", "관계": "카페수 → 공유오피스·회의", "보유-미보유 평균차": "+16.80개", "상관계수(log)": "+0.0483", "하위25%→상위25%": "14.1% → 19.8%", "해석": "🔵 보완/동반"},
    {"가설": "Q4-2", "관계": "헬스장수 → 운동시설",       "보유-미보유 평균차": "-1.10개",  "상관계수(log)": "-0.0122", "하위25%→상위25%": "36.8% → 34.7%", "해석": "🟡 약한 대체"},
    {"가설": "Q4-3", "관계": "학원수 → 교육·학습시설",   "보유-미보유 평균차": "+2.16개",  "상관계수(log)": "+0.0138", "하위25%→상위25%": "30.5% → 31.5%", "해석": "⚪ 약함/불명확"},
    {"가설": "Q4-4", "관계": "독서실수 → 교육·학습시설", "보유-미보유 평균차": "-0.53개",  "상관계수(log)": "-0.0139", "하위25%→상위25%": "32.3% → 31.2%", "해석": "⚪ 약함/불명확"},
]

# ============================================================
# CSS  ─  .streamlit/config.toml 다크 테마와 연동
# ============================================================

def inject_css():
    st.markdown(
        """
        <style>
        /* ── 레이아웃 ── */
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }

        /* ── 히어로 배너 ── */
        .hero {
            padding: 2.4rem 2.6rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #050B14 0%, #0F2358 45%, #1D4ED8 100%);
            color: #F8F4E3;
            margin-bottom: 1.4rem;
            box-shadow: 0 20px 48px rgba(0, 0, 0, 0.45);
            border: 1px solid rgba(56, 189, 248, 0.18);
        }
        .hero h1 {
            color: #F8F4E3;
            font-size: 2.3rem;
            line-height: 1.25;
            margin: 0 0 0.75rem 0;
        }
        .hero p {
            color: #BAE6FD;
            font-size: 1.05rem;
            line-height: 1.65;
            margin: 0;
        }

        /* ── 섹션 제목 ── */
        .section-title {
            font-size: 1.18rem;
            font-weight: 800;
            color: #F2C879;
            margin-top: 1.8rem;
            margin-bottom: 0.75rem;
            border-left: 4px solid #F2C879;
            padding-left: 0.6rem;
        }

        /* ── 일반 카드 ── */
        .soft-card {
            padding: 1.2rem 1.3rem;
            border-radius: 18px;
            background: #0E1B2E;
            border: 1px solid rgba(242, 200, 121, 0.15);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            margin-bottom: 1rem;
            min-height: 135px;
        }
        .soft-card h3 {
            font-size: 1.03rem;
            margin: 0.2rem 0 0.45rem 0;
            color: #F8F4E3;
        }
        .soft-card p {
            color: #94A3B8;
            line-height: 1.55;
            font-size: 0.92rem;
            margin: 0;
        }

        /* ── 노트 박스 ── */
        .note-box {
            padding: 1rem 1.2rem;
            border-radius: 16px;
            background: rgba(29, 78, 216, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.25);
            color: #93C5FD;
            line-height: 1.6;
            margin-bottom: 1rem;
            font-size: 0.95rem;
        }

        /* ── 경고/인포 박스 ── */
        .info-box {
            padding: 0.9rem 1.2rem;
            border-radius: 14px;
            background: rgba(234, 179, 8, 0.08);
            border: 1px solid rgba(242, 200, 121, 0.3);
            color: #FDE68A;
            line-height: 1.6;
            margin-bottom: 1rem;
            font-size: 0.93rem;
        }

        /* ── 뱃지 ── */
        .badge {
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .badge-green  { background: rgba(34,197,94,0.18);  color: #86EFAC; }
        .badge-red    { background: rgba(239,68,68,0.18);  color: #FCA5A5; }
        .badge-yellow { background: rgba(234,179,8,0.18);  color: #FDE68A; }
        .badge-blue   { background: rgba(56,189,248,0.18); color: #7DD3FC; }

        /* ── 결론 카드 ── */
        .conclusion-card {
            padding: 1.2rem 1.25rem;
            border-radius: 18px;
            background: #0E1B2E;
            border: 1px solid rgba(255, 255, 255, 0.07);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            margin-bottom: 1rem;
            min-height: 170px;
        }
        .conclusion-card h3 {
            font-size: 1.06rem;
            margin-top: 0.2rem;
            margin-bottom: 0.6rem;
            color: #F8F4E3;
        }
        .conclusion-card p {
            color: #94A3B8;
            line-height: 1.55;
            font-size: 0.92rem;
            margin-bottom: 0.3rem;
        }
        .card-green  { border-left: 6px solid #22C55E; }
        .card-red    { border-left: 6px solid #EF4444; }
        .card-yellow { border-left: 6px solid #F59E0B; }

        /* ── 모델 성능 카드 ── */
        .perf-card {
            padding: 1rem 1.2rem;
            border-radius: 16px;
            background: #0E1B2E;
            border: 1px solid rgba(56, 189, 248, 0.15);
            margin-bottom: 0.8rem;
        }
        .perf-card .facility-name {
            font-size: 1.05rem;
            font-weight: 700;
            color: #38BDF8;
            margin-bottom: 0.3rem;
        }
        .perf-bar-track {
            height: 8px;
            border-radius: 4px;
            background: rgba(255,255,255,0.08);
            margin: 0.4rem 0;
            overflow: hidden;
        }
        .perf-bar-fill {
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(90deg, #1D4ED8, #38BDF8);
        }

        /* ── 메트릭 오버라이드 ── */
        [data-testid="stMetric"] {
            background: #0E1B2E;
            border: 1px solid rgba(242, 200, 121, 0.15);
            padding: 1rem 1.15rem;
            border-radius: 16px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
        }

        .small-muted { color: #64748B; font-size: 0.88rem; }
        </style>
        """,
        unsafe_allow_html=True,
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

available_targets       = [c for c in TARGETS if c in df.columns]
available_biz_features  = [c for c in BIZ_FEATURES if c in df.columns]
available_demo_features = [c for c in DEMO_FEATURES if c in df.columns]


# ============================================================
# 지도용 데이터 생성
# ============================================================

def build_gu_summary_from_df(data: pd.DataFrame) -> pd.DataFrame:
    id_col = "단지키" if "단지키" in data.columns else "구"
    agg_dict = {"단지수": (id_col, "count")}
    for target in available_targets:
        agg_dict[f"{target}_보급률"] = (target, "mean")
    for col in available_biz_features:
        agg_dict[f"{col}_평균"] = (col, "mean")

    gu = data.groupby("구").agg(**agg_dict).reset_index()

    for target in available_targets:
        gu[f"{target}_보급률_pct"] = (gu[f"{target}_보급률"] * 100).round(1)

    if "키즈시설_보급률_pct" in gu.columns and "교육_학습시설_보급률_pct" in gu.columns:
        gu["가족친화지수"] = (gu["키즈시설_보급률_pct"] + gu["교육_학습시설_보급률_pct"]) / 2
    if "공유오피스_회의_보급률_pct" in gu.columns and "라운지_휴게_보급률_pct" in gu.columns:
        gu["선택형시설지수"] = (gu["공유오피스_회의_보급률_pct"] + gu["라운지_휴게_보급률_pct"]) / 2
    if "운동시설_보급률_pct" in gu.columns:
        gu["운동복지지수"] = gu["운동시설_보급률_pct"]

    for col in gu.columns:
        if col.endswith("_평균") or col.endswith("지수"):
            gu[col] = gu[col].round(1)
    return gu


def get_map_summary() -> pd.DataFrame:
    uploaded = load_map_summary()
    return uploaded if uploaded is not None else build_gu_summary_from_df(df)


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
        "YlOrRd":  ["#ffffcc", "#ffeda0", "#feb24c", "#f03b20", "#bd0026"],
        "YlGn":    ["#ffffcc", "#c2e699", "#78c679", "#31a354", "#006837"],
        "PuBu":    ["#fff7fb", "#d0d1e6", "#74a9cf", "#2b8cbe", "#045a8d"],
        "PuRd":    ["#f7f4f9", "#d4b9da", "#df65b0", "#dd1c77", "#980043"],
        "OrRd":    ["#fff7ec", "#fdd49e", "#fc8d59", "#d7301f", "#7f0000"],
        "YlGnBu":  ["#ffffd9", "#c7e9b4", "#41b6c4", "#225ea8", "#081d58"],
    }
    return palettes.get(color_scheme, palettes["YlGnBu"])


def make_gu_choropleth(gu_summary, geo_data, value_col, title, unit="%", color_scheme="YlGnBu"):
    values = gu_summary[value_col].dropna()
    if values.empty:
        st.warning(f"{value_col} 값이 없어 지도를 만들 수 없습니다.")
        return None

    vmin, vmax = float(values.min()), float(values.max())
    if vmin == vmax:
        vmax = vmin + 1

    colormap = LinearColormap(colors=get_palette(color_scheme), vmin=vmin, vmax=vmax)
    colormap.caption = f"{title} ({unit})"
    enriched_geo = enrich_geojson_with_summary(geo_data, gu_summary)

    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles="CartoDB positron")

    tooltip_fields  = ["name", "단지수", value_col]
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
        style_function=lambda f: {
            "fillColor": colormap(f["properties"].get(value_col)) if f["properties"].get(value_col) is not None else "#333",
            "color": "#555",
            "weight": 1,
            "fillOpacity": 0.78,
        },
        highlight_function=lambda f: {"fillColor": "#FFD700", "color": "#000", "weight": 2, "fillOpacity": 0.85},
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields, aliases=tooltip_aliases,
            localize=True, sticky=True, labels=True,
        ),
    ).add_to(m)
    colormap.add_to(m)

    title_html = f"""
    <div style="position:fixed;top:18px;left:50px;z-index:9999;background:#0E1B2E;
        padding:12px 16px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.45);
        font-family:sans-serif;color:#F8F4E3;border:1px solid rgba(56,189,248,.3);">
        <div style="font-size:17px;font-weight:700;">{title}</div>
        <div style="font-size:12px;color:#94A3B8;">서울시 신축 아파트 1,123개 단지 기준</div>
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))
    return m


# ============================================================
# 결론 카드 데이터
# ============================================================

CONCLUSION_ROWS = [
    {
        "q": "Q1", "question": "1인가구비율이 높을수록 공유오피스·회의시설 공급률이 높은가?",
        "hypothesis": "1인가구비율↑ → 공유오피스·회의시설↑",
        "verdict": "기각", "emoji": "❌", "badge_class": "badge-red", "card_class": "card-red",
        "reason": "단변량 상관 +0.004, 평균차 +0.11%p로 거의 무관. SHAP 순위도 12위에 불과해 핵심 변수로 보기 어렵다.",
    },
    {
        "q": "Q2", "question": "아동비율이 높을수록 키즈시설 공급률이 높은가?",
        "hypothesis": "아동비율↑ → 키즈시설↑",
        "verdict": "지지", "emoji": "✅", "badge_class": "badge-green", "card_class": "card-green",
        "reason": "단변량 상관 +0.144. SHAP에서도 키즈시설 예측 변수 1위로, 단변량·다변량 모두 일관된 신호.",
    },
    {
        "q": "Q3", "question": "고령비율이 높을수록 시니어시설 공급률이 높은가?",
        "hypothesis": "고령비율↑ → 시니어시설↑",
        "verdict": "기각/재해석", "emoji": "⚠️", "badge_class": "badge-red", "card_class": "card-red",
        "reason": "상관 -0.083, 보유 단지 고령비율이 오히려 낮음. 고령 인구 밀집 구도심과 신축 단지 공급지 간의 지리적 불일치로 해석.",
    },
    {
        "q": "Q4", "question": "외부 카페·헬스장·학원·독서실이 많을수록 단지 내 유사 시설은 감소하는가?",
        "hypothesis": "외부 상권↑ → 단지 내 유사 시설↓",
        "verdict": "부분 지지", "emoji": "🟨", "badge_class": "badge-yellow", "card_class": "card-yellow",
        "reason": "헬스장수↑→운동시설은 약한 대체(-0.012). 카페수↑→공유오피스는 오히려 보완/동반(+0.048). 시설별 방향 혼재.",
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
                        <p><b style="color:#CBD5E1;">질문:</b> {row['question']}</p>
                        <p><b style="color:#CBD5E1;">해석:</b> {row['reason']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
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
        "🤖 모델 · SHAP",
        "🖼️ 차트 모음",
        "✅ 최종 결론",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.caption("최종 기준: 1,123개 단지")
st.sidebar.caption("Data Mining Project | 양세미")


# ============================================================
# ① 홈
# ============================================================

if page == "🏠 홈":

    st.markdown(
        """
        <div class="hero">
            <h1>🏢 서울시 신축 아파트<br>커뮤니티 시설 공급 패턴 분석</h1>
            <p>
                인구 구조 👥 · 세대 구성 🧩 · 주변 상권 🏪 데이터를 결합해<br>
                서울시 신축 아파트 단지의 커뮤니티 시설 공급 패턴을 분류 모델과 SHAP으로 분석합니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="note-box">
            <b>📌 연구 전제</b><br>
            본 연구는 실제 이용 수요가 아닌 <b>건축 인허가상 커뮤니티 시설 공급 여부</b>를 분석합니다.
            시행사의 공급 결정을 수요의 간접 지표로 활용하되, 결과는 <b>공급 패턴 예측</b>으로 해석합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 주요 지표
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏘️ 분석 단지", f"{len(df):,}개")
    c2.metric("🎯 시설 타겟", f"{len(available_targets)}개")
    c3.metric("🗺️ 자치구", f"{df['구'].nunique()}개" if "구" in df.columns else "—")
    c4.metric("📊 모델 피처", "18개")

    # 결론 한눈에 보기
    st.markdown('<div class="section-title">🔎 최종 가설 결론 한눈에 보기</div>', unsafe_allow_html=True)
    render_conclusion_cards()

    # 주요 내용 카드
    st.markdown('<div class="section-title">📌 대시보드 구성</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)

    with a:
        st.markdown(
            """
            <div class="soft-card">
                <span class="badge badge-blue">📊 EDA</span>
                <h3>시설 보유율 · 인구·상권 변수 분포</h3>
                <p>1,123개 단지 기준 시설별 보유율과 인구·세대·상권 변수의 분포 및 상관관계를 확인합니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
            <div class="soft-card">
                <span class="badge badge-blue">🤖 모델 · SHAP</span>
                <h3>분류 모델 성능 · SHAP 변수 기여도</h3>
                <p>LR/RF 모델의 보유 클래스 F1, Macro F1, Balanced Accuracy와 시설별 SHAP Top 3 변수를 정리합니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            """
            <div class="soft-card">
                <span class="badge badge-blue">🗺️ Folium</span>
                <h3>서울 25구 시설 보급률 지도</h3>
                <p>시설별 보급률, 종합 지수, Q4 상권 변수를 Choropleth 지도로 자치구 단위로 탐색합니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ② 데이터
# ============================================================

elif page == "📁 데이터":
    st.title("📁 데이터 현황")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("행 개수", f"{df.shape[0]:,}개")
    c2.metric("열 개수", f"{df.shape[1]:,}개")
    c3.metric("자치구 수", f"{df['구'].nunique()}개" if "구" in df.columns else "—")
    c4.metric("단지키 중복", f"{df['단지키'].duplicated().sum()}개" if "단지키" in df.columns else "—")

    st.markdown('<div class="section-title">데이터 소스</div>', unsafe_allow_html=True)

    src_data = {
        "데이터 소스": ["세움터 건축HUB (복리분양시설)", "행안부 주민등록 인구통계", "행안부 세대원수별 세대수", "소상공인 상권정보 (원본)"],
        "수집 규모": ["5,441건 → 1,123단지", "329개 동", "329개 동", "534,978건"],
        "분석 용도": ["시설 보유 여부 (타겟)", "연령대별 인구 비율", "가구 구성 비율", "주변 상권 밀도 변수"],
    }
    st.dataframe(pd.DataFrame(src_data), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">최종 데이터 미리보기 (상위 50행)</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">인구·세대 피처 (7개)</div>', unsafe_allow_html=True)
        feat_df = pd.DataFrame(
            [{"변수명": k, "설명": v} for k, v in FEATURE_LABELS.items()
             if k in available_demo_features]
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
    with col2:
        st.markdown('<div class="section-title">상권 피처 (6개, log 변환)</div>', unsafe_allow_html=True)
        biz_df = pd.DataFrame({"변수명": available_biz_features})
        st.dataframe(biz_df, use_container_width=True, hide_index=True)

    with st.expander("전체 컬럼 목록"):
        st.write(df.columns.tolist())

    st.download_button(
        label="📥 현재 데이터 CSV 다운로드",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="apt_final_1123.csv",
        mime="text/csv",
    )


# ============================================================
# ③ 시설 보유율
# ============================================================

elif page == "📊 시설 보유율":
    st.title("📊 시설별 보유율")

    if not available_targets:
        st.warning("시설 타겟 컬럼을 찾을 수 없습니다.")
        st.stop()

    rate  = df[available_targets].mean().sort_values(ascending=False) * 100
    count = df[available_targets].sum().loc[rate.index]

    summary = pd.DataFrame({
        "시설":         [TARGET_LABELS.get(x, x) for x in rate.index],
        "보유율(%)":    rate.round(1).values,
        "보유 단지 수": count.astype(int).values,
        "미보유 단지 수": (len(df) - count).astype(int).values,
    })

    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">보유율 막대 차트</div>', unsafe_allow_html=True)
    st.bar_chart(rate.rename(index=TARGET_LABELS))

    st.markdown('<div class="section-title">자치구별 보급률 비교</div>', unsafe_allow_html=True)
    selected_target = st.selectbox(
        "시설 선택",
        available_targets,
        format_func=lambda x: TARGET_LABELS.get(x, x),
    )

    id_col = "단지키" if "단지키" in df.columns else df.columns[0]
    gu_summary = (
        df.groupby("구")
        .agg(단지수=(id_col, "count"), 보급률=(selected_target, "mean"))
        .reset_index()
    )
    gu_summary["보급률(%)"] = (gu_summary["보급률"] * 100).round(1)
    gu_summary = gu_summary.sort_values("보급률(%)", ascending=False)

    st.dataframe(gu_summary[["구", "단지수", "보급률(%)"]], use_container_width=True, hide_index=True)
    st.bar_chart(gu_summary.set_index("구")["보급률(%)"])


# ============================================================
# ④ 지도 시각화
# ============================================================

elif page == "🗺️ 지도 시각화":
    st.title("🗺️ 서울 25구 Folium 지도 시각화")

    geo_data = load_geojson()
    if geo_data is None:
        st.error(
            "`maps/seoul_gu_geo_enriched.json` 파일이 없습니다. "
            "GitHub의 `maps/` 폴더에 해당 GeoJSON 파일을 업로드하세요."
        )
        st.stop()

    gu_summary = get_map_summary()

    st.markdown(
        """
        <div class="note-box">
        자치구별 시설 보급률과 상권 변수를 Choropleth 지도로 확인합니다.
        마우스를 자치구 위에 올리면 분석 단지 수, 시설별 보급률, 주요 상권 평균을 볼 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    map_mode = st.selectbox("지도 유형 선택", ["시설별 보급률", "종합지수", "Q4 상권 변수"])
    color_scheme = "YlGnBu"

    if map_mode == "시설별 보급률":
        selected_target = st.selectbox("시설 선택", available_targets, format_func=lambda x: TARGET_LABELS.get(x, x))
        value_col = f"{selected_target}_보급률_pct"
        title = f"{TARGET_LABELS.get(selected_target, selected_target)} 보급률"
        unit = "%"
        color_map = {
            "운동시설": "PuBu", "교육_학습시설": "YlGn", "키즈시설": "OrRd",
            "시니어시설": "PuRd", "공유오피스_회의": "YlOrRd", "라운지_휴게": "YlGnBu",
        }
        color_scheme = color_map.get(selected_target, "YlGnBu")

    elif map_mode == "종합지수":
        index_options = {
            "가족친화 지수 (키즈+교육)": "가족친화지수",
            "선택형·프리미엄 시설 지수": "선택형시설지수",
            "운동복지 지수": "운동복지지수",
        }
        selected_name = st.selectbox("지수 선택", list(index_options.keys()))
        value_col = index_options[selected_name]
        title, unit = selected_name, "%"
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
        title, unit = selected_name, "개"
        color_scheme = "YlOrRd"

    if value_col not in gu_summary.columns:
        st.error(f"`{value_col}` 컬럼이 지도 요약 데이터에 없습니다.")
        st.stop()

    m = make_gu_choropleth(gu_summary, geo_data, value_col, title, unit, color_scheme)
    if m is not None:
        st_folium(m, width=1150, height=680)

    st.markdown('<div class="section-title">상위 자치구 순위</div>', unsafe_allow_html=True)
    rank_table = (
        gu_summary[["구", "단지수", value_col]]
        .sort_values(value_col, ascending=False)
        .head(10)
        .rename(columns={value_col: title})
    )
    st.dataframe(rank_table, use_container_width=True, hide_index=True)


# ============================================================
# ⑤ 모델 · SHAP
# ============================================================

elif page == "🤖 모델 · SHAP":
    st.title("🤖 분류 모델 성능 · SHAP 변수 기여도")

    # ── 모델 설계 요약 ──────────────────────────────────────
    st.markdown('<div class="section-title">모델 설계</div>', unsafe_allow_html=True)

    design_data = {
        "항목": ["타겟", "피처 수", "교차검증", "비교 모델", "클래스 불균형 대응", "주 평가 지표"],
        "내용": [
            "6개 시설 유형 각각 이진 분류 (보유=1, 미보유=0)",
            "18개: 인구·세대 7 + 상권 log 6 + 지역 통제 5",
            "5-Fold Stratified Cross Validation",
            "Dummy majority baseline / Logistic Regression / Random Forest",
            "class_weight=balanced",
            "보유 클래스 F1 (보조: Macro F1, Balanced Accuracy)",
        ],
    }
    st.dataframe(pd.DataFrame(design_data), use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="info-box">
        ⚠️ <b>키즈시설 해석 주의</b><br>
        키즈시설 보유율이 55.3%로 높아, 모든 단지를 '보유'로 예측하는 Majority Baseline의 보유F1이 0.712로 높게 나타납니다.
        키즈시설은 보유F1 단독이 아니라 <b>Macro F1 · Balanced Accuracy</b>를 함께 해석해야 합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 성능 테이블 ─────────────────────────────────────────
    st.markdown('<div class="section-title">최종 모델 성능 요약 (보고서 기준)</div>', unsafe_allow_html=True)

    perf_df = pd.DataFrame(MODEL_PERFORMANCE)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # ── 보유 클래스 F1 시각화 ─────────────────────────────
    st.markdown('<div class="section-title">보유 클래스 F1 비교 (채택 모델 기준)</div>', unsafe_allow_html=True)
    f1_series = pd.Series(
        {row["시설"]: row["보유F1"] for row in MODEL_PERFORMANCE}
    ).sort_values(ascending=False)
    st.bar_chart(f1_series)

    # ── SHAP Top 3 테이블 ──────────────────────────────────
    st.markdown('<div class="section-title">시설별 SHAP Top 3 변수</div>', unsafe_allow_html=True)
    shap_df = pd.DataFrame(SHAP_TOP3)[["시설", "모델", "보유F1", "MacroF1", "Top1", "Top2", "Top3"]]
    st.dataframe(shap_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="note-box">
        <b>SHAP 해석 방법</b><br>
        Logistic Regression은 LinearExplainer, Random Forest는 TreeExplainer를 사용하였습니다.
        SHAP 평균 절댓값은 변수 영향력의 크기를 나타내며, 방향성은 분위별 보유율 및 단변량 상관계수로 보완합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Q4 방향 분석 ──────────────────────────────────────
    st.markdown('<div class="section-title">Q4 방향 분석 — 외부 상권 × 단지 내 시설</div>', unsafe_allow_html=True)
    q4_df = pd.DataFrame(Q4_DIRECTION)
    st.dataframe(q4_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="note-box">
        Q4 가설의 '대체 효과'는 시설 유형별로 혼재합니다.
        헬스장수↑→운동시설은 약한 대체 가능성이 있으나,
        카페수↑→공유오피스는 오히려 보완/동반 관계로 나타나 단순 대체로 해석하기 어렵습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 차트 ──────────────────────────────────────────────
    chart8 = CHART_DIR / "chart8_shap_importance.png"
    chart9 = CHART_DIR / "chart9_q4_direction.png"

    tab1, tab2 = st.tabs(["SHAP 변수 중요도 차트", "Q4 방향 분석 차트"])
    with tab1:
        if chart8.exists():
            st.image(str(chart8), use_container_width=True)
        else:
            st.caption(f"파일 없음: {chart8}")
    with tab2:
        if chart9.exists():
            st.image(str(chart9), use_container_width=True)
        else:
            st.caption(f"파일 없음: {chart9}")


# ============================================================
# ⑥ 차트 모음
# ============================================================

elif page == "🖼️ 차트 모음":
    st.title("🖼️ EDA · 모델 차트 모음")

    chart_files = {
        "그림 1. 시설별 보유율": {
            "file": "chart1_facility_rate.png",
            "desc": "6개 시설 유형별 보유율과 보유 단지 수 (1,123개 단지 기준)",
        },
        "그림 2. 인구·세대 변수 분포": {
            "file": "chart2_feature_dist.png",
            "desc": "아동비율, 고령비율, 1인가구비율 등 7개 인구·세대 변수 분포 히스토그램",
        },
        "그림 3. 피처 × 타겟 상관관계 히트맵": {
            "file": "chart3_corr_heatmap.png",
            "desc": "18개 최종 피처와 6개 시설 타겟 간 상관관계",
        },
        "그림 4. 시설 보유 여부별 인구·세대 특성 차이": {
            "file": "chart4_facility_comparison.png",
            "desc": "시설 보유 단지 vs 미보유 단지의 인구·세대 평균 차이 막대그래프",
        },
        "그림 5. 자치구별 시설 보급률 히트맵": {
            "file": "chart5_gu_heatmap.png",
            "desc": "서울 25개 자치구별 커뮤니티 시설 보급률 히트맵",
        },
        "그림 6. Q1~Q4 핵심 가설 예비검증": {
            "file": "chart6_hypothesis_scatter.png",
            "desc": "Q1~Q3 단변량 및 분위별 보유율, Q4 상관계수 예비 검토",
        },
        "그림 7. 시설별 SHAP 변수 중요도": {
            "file": "chart8_shap_importance.png",
            "desc": "시설별 최종 채택 모델의 SHAP 평균 절댓값 Top 변수",
        },
        "그림 8. Q4 상권 방향 분석": {
            "file": "chart9_q4_direction.png",
            "desc": "외부 상권 변수(카페·헬스장·학원·독서실)와 단지 내 시설 보유율의 방향성",
        },
        "그림 9. Q1~Q4 최종 결론표": {
            "file": "chart10_hypothesis_final_conclusion.png",
            "desc": "Q1~Q4 단변량·SHAP·다변량 근거 종합 최종 판정표",
        },
    }

    st.markdown(
        """
        <div class="note-box">
        📌 최종 결론 (Q1~Q4 판정 카드)은 <b>✅ 최종 결론</b> 페이지에서 카드 형태로도 확인할 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["하나씩 보기", "전체 보기"])

    with tab1:
        selected_chart = st.selectbox("차트 선택", list(chart_files.keys()))
        ci = chart_files[selected_chart]
        chart_path = CHART_DIR / ci["file"]
        st.subheader(selected_chart)
        st.caption(ci["desc"])
        if chart_path.exists():
            st.image(str(chart_path), use_container_width=True)
        else:
            st.error(f"차트 파일이 없습니다: `{chart_path}`")

    with tab2:
        items = list(chart_files.items())
        for i in range(0, len(items), 2):
            cols = st.columns(2)
            for col, (title_str, ci) in zip(cols, items[i:i + 2]):
                with col:
                    st.markdown(f"**{title_str}**")
                    st.caption(ci["desc"])
                    chart_path = CHART_DIR / ci["file"]
                    if chart_path.exists():
                        st.image(str(chart_path), use_container_width=True)
                    else:
                        st.caption(f"파일 없음: {chart_path}")


# ============================================================
# ⑦ 최종 결론
# ============================================================

elif page == "✅ 최종 결론":
    st.title("✅ Q1~Q4 최종 결론")

    st.markdown(
        """
        <div class="note-box">
        아래 카드는 단변량 상관계수·SHAP 변수 중요도·분위별 보유율을 종합한 최종 판정 결과입니다.
        색상 — <b style="color:#86EFAC;">초록=지지</b> · <b style="color:#FCA5A5;">빨강=기각</b> · <b style="color:#FDE68A;">노랑=부분 지지</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_conclusion_cards()

    # 결론 요약표
    st.markdown('<div class="section-title">결론 요약표</div>', unsafe_allow_html=True)
    conclusion_df = pd.DataFrame([
        {
            "가설": row["q"],
            "질문": row["question"],
            "가설 내용": row["hypothesis"],
            "판정": f"{row['emoji']} {row['verdict']}",
            "핵심 근거": row["reason"],
        }
        for row in CONCLUSION_ROWS
    ])
    st.dataframe(conclusion_df, use_container_width=True, hide_index=True)

    # 핵심 문장
    st.markdown('<div class="section-title">보고서용 핵심 문장</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="soft-card">
            <ul style="color:#CBD5E1; line-height:2; margin:0; padding-left:1.2rem;">
                <li><b style="color:#FCA5A5;">Q1 기각</b>: 1인가구비율은 공유오피스·회의시설 공급의 핵심 변수로 보기 어렵다.
                (단변량 상관 +0.004, SHAP 12위)</li>
                <li><b style="color:#86EFAC;">Q2 지지</b>: 아동비율은 키즈시설 공급을 설명하는 가장 일관된 변수로,
                단변량·SHAP 모두 1위 수준의 신호를 보였다.</li>
                <li><b style="color:#FCA5A5;">Q3 기각/재해석</b>: 고령비율이 높을수록 시니어시설이 많아진다는 가설은
                지지되지 않았으며, 구도심 vs 신축 단지의 지리적 불일치로 해석된다.</li>
                <li><b style="color:#FDE68A;">Q4 부분 지지</b>: 외부 상권의 대체효과는 시설 유형별로 혼재한다.
                헬스장수↔운동시설은 약한 대체, 카페수↔공유오피스는 보완/동반 가능성이 함께 나타났다.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 연구 한계
    st.markdown('<div class="section-title">연구 한계 및 향후 과제</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="soft-card">
            <p style="color:#94A3B8; line-height:1.9; margin:0;">
            • 시설 보유 여부는 공급 데이터이며 실제 이용 수요를 직접 측정하지 않습니다.<br>
            • 인구·세대 비율 변수의 구조적 다중공선성으로 로지스틱 회귀 계수 해석이 제한됩니다.<br>
            • 상권 변수는 법정동 단위 카운트이므로 단지-업소 간 실제 거리 기반 접근성을 반영하지 않습니다.<br>
            • 향후: 반경 500m 공간 버퍼 상권 재집계, StratifiedGroupKFold 강건성 검증,
            단지 세대수·브랜드·준공연도 변수 추가를 계획합니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
