import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 한글 폰트 깨짐 방지
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="아파트 커뮤니티 시설 공급 패턴 분석",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 한글 폰트 설정 (OS별 처리)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else: # Linux (Streamlit Cloud 등)
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 커스텀 CSS (중간보고서 PDF의 정갈하고 분석적인 톤 반영)
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 1.5rem; font-weight: 600; color: #3B82F6; margin-top: 20px; }
    .info-text { font-size: 1.1rem; color: #4B5563; }
    .highlight { background-color: #DBEAFE; padding: 0.5rem; border-radius: 5px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 사이드바 내비게이션
# -----------------------------------------------------------------------------
st.sidebar.title("🏢 대시보드 메뉴")
menu = st.sidebar.radio(
    "이동할 페이지를 선택하세요:",
    ("1. 프로젝트 개요", "2. EDA 및 데이터 분포", "3. 모델링 및 SHAP 분석", "4. 최종 가설 검증 결론")
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**분석 개요**\n"
    "- 분석 대상: 서울시 신축 아파트 1,123개 단지\n"
    "- 모델 피처: 총 18개 (인구 7, 상권 6, 지역 5)\n"
    "- 예측 타겟: 커뮤니티 시설 6종\n"
    "- 작성자: 양세미"
)
st.sidebar.markdown("[GitHub Repository 바로가기](https://github.com/semarie-y/aptcommunity)")

# -----------------------------------------------------------------------------
# 3. 페이지별 로직 구현
# -----------------------------------------------------------------------------

# ==========================================
# 페이지 1: 프로젝트 개요
# ==========================================
if menu == "1. 프로젝트 개요":
    st.markdown('<div class="main-header">아파트 커뮤니티 시설 공급 패턴 분석</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-text">수요 간접 예측 - 서울시 공동주택 대상</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="sub-header">💡 연구 핵심 질문 (Q1~Q4)</div>', unsafe_allow_html=True)
        st.info(
            """
            **Q1.** 1인가구 비율이 높으면 → 공유오피스·회의시설이 많은가?\n
            **Q2.** 아동 비율이 높으면 → 키즈시설이 많은가?\n
            **Q3.** 고령 비율이 높으면 → 시니어시설이 많은가?\n
            **Q4.** 주변 상권(카페/헬스장 등) 밀도가 높으면 → 단지 내 해당 시설 비율은 적은가?
            """
        )
        
        st.markdown('<div class="sub-header">📌 데이터 수집 및 처리 요약</div>', unsafe_allow_html=True)
        st.write("""
        - **타겟 변수**: 세움터 건축HUB 복리분양시설 비정형 텍스트 분류 (키워드 사전 구축, 정확도 98%)
        - **피처 변수**: 
            - 행안부 인구통계 (연령대별 인구, 세대원수 비율)
            - 소상공인진흥공단 상권정보 (카페, 헬스장, 학원, 독서실 등)
        - **단지 집계**: 건물명 중복 방지를 위한 주소 기반 정규화 (최종 1,123개 단지)
        """)

    with col2:
        st.markdown('<div class="sub-header">🎯 연구 전제</div>', unsafe_allow_html=True)
        st.warning(
            "시행사의 시설 공급 패턴을 수요의 간접 지표로 활용하되, "
            "본 연구의 결과는 실제 이용 수요 예측이 아니라 **공급 패턴 예측**으로 해석합니다."
        )

# ==========================================
# 페이지 2: EDA 및 데이터 분포
# ==========================================
elif menu == "2. EDA 및 데이터 분포":
    st.markdown('<div class="main-header">탐색적 데이터 분석 (EDA)</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown('<div class="sub-header">📊 서울시 신축 아파트 커뮤니티 시설 유형별 보유율</div>', unsafe_allow_html=True)
    
    # 데이터셋 구성 (보고서 기반 하드코딩)
    facility_data = pd.DataFrame({
        '시설 유형': ['키즈시설', '시니어시설', '운동시설', '교육·학습시설', '공유오피스·회의', '라운지·휴게'],
        '보유율(%)': [55.3, 45.4, 33.6, 31.3, 17.5, 15.7],
        '보유 단지 수': [621, 510, 377, 351, 197, 176]
    })
    
    # 시각화
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='보유율(%)', y='시설 유형', data=facility_data, palette='Blues_r', ax=ax)
    
    # 막대 끝에 텍스트 추가
    for index, value in enumerate(facility_data['보유율(%)']):
        count = facility_data.loc[index, '보유 단지 수']
        ax.text(value + 0.5, index, f'{value}% ({count}개)', va='center', color='black')
        
    ax.set_xlim(0, 70)
    ax.set_xlabel('보유율 (%)')
    ax.set_ylabel('')
    sns.despine()
    st.pyplot(fig)
    
    st.markdown('<div class="sub-header">📈 주요 인구·세대 특성 분포 (예시)</div>', unsafe_allow_html=True)
    st.write("보고서 통계 요약표를 기반으로 한 기초 통계입니다.")
    
    # 기초 통계표
    demo_stats = pd.DataFrame({
        '변수': ['아동비율', '고령비율', '1인가구비율', '3인이상비율'],
        '평균': ['5.45%', '26.79%', '41.75%', '35.19%'],
        '표준편차': ['1.62%p', '4.87%p', '11.11%p', '9.97%p'],
        '최댓값': ['13.86%', '41.67%', '82.94%', '62.28%']
    })
    st.dataframe(demo_stats, use_container_width=True)

# ==========================================
# 페이지 3: 모델링 및 SHAP 분석
# ==========================================
elif menu == "3. 모델링 및 SHAP 분석":
    st.markdown('<div class="main-header">분류 모델 학습 및 변수 중요도(SHAP)</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown('<div class="sub-header">🤖 최종 모델 성능 요약</div>', unsafe_allow_html=True)
    st.write("클래스 불균형을 고려하여 **보유 클래스 F1(Positive-class F1)**을 주 지표로 사용했습니다.")
    
    model_results = pd.DataFrame({
        '시설 유형': ['운동시설', '교육·학습시설', '키즈시설', '시니어시설', '공유오피스·회의', '라운지·휴게'],
        '채택 모델': ['Logistic Regression', 'Logistic Regression', 'Random Forest', 'Random Forest', 'Logistic Regression', 'Random Forest'],
        'Baseline 보유F1': [0.000, 0.000, 0.712, 0.000, 0.000, 0.000],
        '보유 F1': [0.423, 0.424, 0.580, 0.534, 0.295, 0.289],
        'Macro F1': [0.526, 0.545, 0.547, 0.565, 0.467, 0.544]
    })
    st.dataframe(model_results, use_container_width=True)
    
    st.markdown('<div class="sub-header">🔥 시설별 SHAP Top 3 핵심 기여 변수</div>', unsafe_allow_html=True)
    
    shap_data = pd.DataFrame({
        '시설 유형': ['운동시설', '교육·학습시설', '키즈시설', '시니어시설', '공유오피스·회의', '라운지·휴게'],
        'Top 1': ['카페수(log)', '아동비율', '아동비율', '아동비율', '카페수(log)', '3인이상비율'],
        'Top 2': ['서남권', '학원수(log)', '2인가구비율', '2인가구비율', '사우나수(log)', '1인가구비율'],
        'Top 3': ['2인가구비율', '헬스장수(log)', '사우나수(log)', '3인이상비율', '동북권', '아동비율']
    })
    st.table(shap_data)
    st.info("💡 **다중공선성 대응**: 선형 모델 계수 불안정 문제를 보완하기 위해 SHAP 중요도와 분위별 보유율을 병행하여 해석했습니다.")

# ==========================================
# 페이지 4: 최종 가설 검증 결론
# ==========================================
elif menu == "4. 최종 가설 검증 결론":
    st.markdown('<div class="main-header">Q1~Q4 최종 가설 검증 결론</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Q1
    st.markdown("### ❌ Q1. 1인가구비율↑ → 공유오피스·회의시설↑ : **기각**")
    st.write("단변량 상관이 거의 없고(상관계수 +0.004), SHAP 순위도 낮아 핵심 변수로 보기 어렵습니다.")
    
    # Q2
    st.markdown("### ✅ Q2. 아동비율↑ → 키즈시설↑ : **지지**")
    st.write("아동비율은 단변량과 SHAP 분석 모두에서 키즈시설 공급을 설명하는 **가장 핵심적인 변수**로 확인되었습니다.")
    
    # Q3
    st.markdown("### ❌ Q3. 고령비율↑ → 시니어시설↑ : **기각 / 재해석**")
    st.write("고령비율이 높을수록 시니어시설이 많아진다는 가설은 지지되지 않았습니다. 이는 고령 인구 밀집지(구도심)와 신축 단지 패키지 공급 지역 간의 지리적 불일치로 해석됩니다.")
    
    # Q4
    st.markdown("### ⚠️ Q4. 외부 상권↑ → 단지 내 유사 시설↓ : **부분 지지**")
    st.write("""
    - **대체 가능성 (음의 관계):** 헬스장수는 운동시설과 단변량 음(-)의 관계가 있으나 기여도가 낮음.
    - **보완/동반 가능성 (양의 관계):** 카페수(공유오피스)와 학원수(교육·학습시설)는 오히려 보유 방향으로 작용하여 대체효과보다는 동반 관계에 가깝습니다.
    """)
    
    st.markdown("---")
    st.markdown('<div class="highlight">결론 및 시사점</div>', unsafe_allow_html=True)
    st.write("""
    공유오피스나 라운지처럼 보유율이 낮고 선택적인 시설은 단순 인구 변수만으로 설명하기 어렵습니다. 
    단지 규모, 분양가, 브랜드 등 아파트 자체의 특성 데이터가 결합될 때 더 높은 설명력을 가질 것으로 기대됩니다.
    """)
