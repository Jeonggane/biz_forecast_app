import os
if not os.path.exists("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    os.system("apt-get update && apt-get install -y fonts-nanum")
    os.system("fc-cache -f -v")
rc('font', family='NanumGothic')
import folium
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

# 한글 폰트 설정
plt.rcParams['font.family'] = ['NanumGothic', 'sans-serif']

# 페이지 설정
st.set_page_config(page_title="서울시 업종 전망 분석", layout="wide")

# 데이터 불러오기
url = "http://openapi.seoul.go.kr:8088/475a6c5976726b6439315741647171/xml/VwsmSignguNcmCnsmpW/1/1000/"
df = pd.read_xml(url) 
df.columns = df.columns.str.strip()

column_map = {
    'list_total_count': '총 데이터 건수',
    'CODE': '요청결과 코드',
    'MESSAGE': '요청결과 메시지',
    'STDR_YYQU_CD': '기준_년분기_코드',
    'SIGNGU_CD': '행정동_코드',
    'SIGNGU_CD_NM': '행정동_코드_명',
    'MT_AVRG_INCOME_AMT': '월_평균_소득_금액',
    'INCOME_SCTN_CD': '소득_구간_코드',
    'EXPNDTR_TOTAMT': '지출_총금액',
    'FDSTFFS_EXPNDTR_TOTAMT': '식료품_지출_총금액',
    'CLTHS_FTWR_EXPNDTR_TOTAMT': '의류_신발_지출_총금액',
    'LVSPL_EXPNDTR_TOTAMT': '생활용품_지출_총금액',
    'MCP_EXPNDTR_TOTAMT': '의료비_지출_총금액',
    'TRNSPORT_EXPNDTR_TOTAMT': '교통_지출_총금액',
    'EDC_EXPNDTR_TOTAMT': '교육_지출_총금액',
    'PLESR_EXPNDTR_TOTAMT': '유흥_지출_총금액',
    'LSR_CLTUR_EXPNDTR_TOTAMT': '여가_문화_지출_총금액',
    'ETC_EXPNDTR_TOTAMT': '기타_지출_총금액',
    'FD_EXPNDTR_TOTAMT': '음식_지출_총금액'
}
df.rename(columns=column_map, inplace=True)

# df = pd.read_csv(r"E:\python_chan\상권분석\서울시 상권분석서비스.csv", encoding='cp949')
# df.columns = df.columns.str.strip()

# 분기 컬럼 숫자 → 문자열 변환 및 정렬용 숫자 컬럼 생성
df['기준_년분기'] = df['기준_년분기_코드'].astype(str).str[:4] + '. ' + 'Q' + df['기준_년분기_코드'].astype(str).str[-1]
df = df.dropna(subset=['기준_년분기_코드'])  # 변환 전 필수
df['분기_정렬'] = df['기준_년분기_코드'].astype(int)


# 2024년 1분기(20241), 2024년 3분기(20243) 제외
df = df[~df['기준_년분기_코드'].isin([20241, 20243])]

target_columns = [
    '월_평균_소득_금액', '식료품_지출_총금액', '의류_신발_지출_총금액',
    '생활용품_지출_총금액', '의료비_지출_총금액', '교통_지출_총금액', '교육_지출_총금액',
    '유흥_지출_총금액', '여가_문화_지출_총금액', '음식_지출_총금액','지출_총금액', '기타_지출_총금액'
    ]

far_columns = [
    '식료품_지출_총금액', '의류_신발_지출_총금액',
    '생활용품_지출_총금액', '의료비_지출_총금액', '교통_지출_총금액', '교육_지출_총금액',
    '유흥_지출_총금액', '여가_문화_지출_총금액', '음식_지출_총금액'
    ]

# 분기 코드 문자열 변환 및 정렬용 컬럼
df['분기_정렬'] = df['기준_년분기_코드'].astype(int)

# 최근 4개 분기 필터링
recent_quarters = sorted(df['분기_정렬'].unique())[-4:]
df_recent = df[df['분기_정렬'].isin(recent_quarters)]

# 업종 선택 메뉴
업종_리스트 = [
    '식료품_지출_총금액', '의류_신발_지출_총금액', '생활용품_지출_총금액',
    '의료비_지출_총금액', '교통_지출_총금액', '교육_지출_총금액',
    '유흥_지출_총금액', '여가_문화_지출_총금액', '음식_지출_총금액'
    ]


# 사이드바 메뉴 생성
menu = st.sidebar.radio("메뉴 선택", [
    "자치구별 지출 비교",
    "자치구별 유망 업종 Top3 (3년)",
    "자치구 업종별 지출 비중 지도",
    "히트맵 비교"
])



if menu == "자치구별 지출 비교":
    # 메뉴 1: 자치구 다중 선택
    selected_gu_list = st.multiselect("자치구를 선택하세요", sorted(df['행정동_코드_명'].unique()), default=["강남구"])
    exclude_cols = ['기타_지출_총금액', '지출_총금액']
    forecast_cols = [col for col in far_columns if col not in exclude_cols]

    # 4개씩 행에 그래프 그리기
    for i in range(0, len(target_columns), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(target_columns):
                col = target_columns[i + j]
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    for gu in selected_gu_list:
                        gu_df = df[df['행정동_코드_명'] == gu].sort_values(by='분기_정렬')
                        ax.plot(gu_df['기준_년분기'], gu_df[col], marker='o', label=gu)
                    ax.set_title(f"{col} 지출 추이")
                    ax.set_xlabel("분기")
                    ax.set_ylabel("금액")
                    ax.tick_params(axis='x', rotation=45)
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)




elif menu == "자치구별 유망 업종 Top3 (3년)":

    selected_gu_list = st.multiselect("자치구를 선택하세요", sorted(df['행정동_코드_명'].unique()), default=["강남구"])
    exclude_cols = ['기타_지출_총금액', '지출_총금액']
    forecast_cols = [col for col in far_columns if col not in exclude_cols]

    # 유망 업종 예측 및 시각화
    st.subheader("\n📊 자치구별 유망 업종 예측 Top 3 (향후 3년)")
    for selected_gu in selected_gu_list:
        st.markdown(f"### 📍 {selected_gu}")
        forecast_columns = [col for col in far_columns if col not in ['기타_지출_총금액', '지출_총금액']]
        growth_data = []
        for col in forecast_columns:
            gu_df = df[df['행정동_코드_명'] == selected_gu].sort_values(by='분기_정렬')
            X = gu_df[['분기_정렬']]
            y = gu_df[col]

            # 선형 회귀 (1차: 기울기 기반)
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]
            future_periods = 5  # 향후 5분기 예측
            last_real = y.values[-1]

            # 기울기 기반 예상 상승률
            growth_percent = (slope * future_periods / last_real) * 100

            if growth_percent > 0:
                # 예측값도 기존처럼 포함해 시각화에 활용
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model_poly = LinearRegression()
                model_poly.fit(X_poly, y)
                future = pd.DataFrame({'분기_정렬': list(X['분기_정렬']) + [20244, 20251, 20252, 20253, 20254]})
                future_poly = poly.transform(future)
                y_pred = model_poly.predict(future_poly)
                growth_data.append((
                    col, growth_percent, y.copy(), y_pred.copy(),
                    list(gu_df['기준_년분기']) + ['2024. Q4', '2025. Q1', '2025. Q2', '2025. Q3', '2025. Q4']
                ))


        growth_data.sort(key=lambda x: x[1], reverse=True)
        top3_data = growth_data[:3]
        cols = st.columns(3)
        for idx, (col, growth_percent, y_vals, y_pred, full_quarters) in enumerate(top3_data):
            with cols[idx]:
                st.markdown(f"#### ✅ {col}")
                st.markdown(f"📈 기울기(성장 추세): **{growth_percent:.2f}**")
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(full_quarters[:len(y_vals)], y_vals, marker='o', label='실제')
                ax.plot(full_quarters, y_pred, linestyle='--', label='예측')
                ax.set_title(f"{col} 지촉 추이 및 전망")
                ax.tick_params(axis='x', rotation=45)
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)




elif menu == "자치구 업종별 지출 비중 지도":
    st.subheader("🗺️ 업종별 지출 비중 지도 시각화")
    selected_column = st.selectbox("업종을 선택하세요", 업종_리스트, index=8)
    gu_center_coords = {
        "강남구": [37.5172, 127.0473], "강북구": [37.6387, 127.0282], "강동구": [37.5302, 127.1237],
        "강서구": [37.5512, 126.8498], "관악구": [37.4803, 126.9527], "광진구": [37.5385, 127.0828],
        "구로구": [37.4953, 126.8877], "금천구": [37.4567, 126.8951], "노원구": [37.6541, 127.0567],
        "도봉구": [37.6687, 127.0466], "동대문구": [37.5743, 127.0395], "동작구": [37.5123, 126.9395],
        "중랑구": [37.6063, 127.0927], "서초구": [37.4836, 127.0326], "용산구": [37.5324, 126.9901],
        "마포구": [37.5638, 126.9084], "서대문구": [37.5792, 126.9368], "성동구": [37.5636, 127.0365],
        "성북구": [37.5901, 127.0165], "송파구": [37.5147, 127.1058], "양천구": [37.5172, 126.8663],
        "영등포구": [37.5263, 126.8959], "은평구": [37.6026, 126.9294], "종로구": [37.5809, 126.9828],
        "중구": [37.5642, 126.9976]
        }

    # 자치구별 최근 4분기 평균 값 계산
    grouped = df_recent.groupby('행정동_코드_명')[[selected_column, '지출_총금액']].mean().reset_index()
    grouped['비율'] = grouped[selected_column] / grouped['지출_총금액']
    grouped['비율'] = grouped['비율'].fillna(0)

    # 지도 생성
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
    for _, row in grouped.iterrows():
        gu = row['행정동_코드_명']
        ratio = row['비율']
        if gu in gu_center_coords:
            color = 'blue' if ratio > 0.15 else 'orange' if ratio > 0.10 else 'red'
            folium.CircleMarker(
                location=gu_center_coords[gu],
                radius=ratio * 100,
                popup=f"{gu}: {ratio:.2%}",
                color=color,
                fill=True,
                fill_opacity=0.6
            ).add_to(m)

    # 비율 순위 테이블 준비 (자치구명, 비율만)
    grouped_display = grouped.copy()

    # 숫자로 정렬
    grouped_display = grouped_display.sort_values(by='비율', ascending=False)

    # 그 후 포맷 적용
    grouped_display['비율'] = (grouped_display['비율'] * 100).astype(int).astype(str) + '%'
    grouped_display[selected_column] = (grouped_display[selected_column] / 10000).astype(int).astype(str) + "만원"
    grouped_display['지출_총금액'] = (grouped_display['지출_총금액'] / 10000).astype(int).astype(str) + "만원"

    # 필요한 컬럼만 보여주기
    grouped_display = grouped_display[['행정동_코드_명', '비율', selected_column, '지출_총금액']].reset_index(drop=True)

    # 지도 + 표를 나란히 배치
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("서울시 업종별 지출 비중 지도")
        st_folium(m, width=600, height=500)

    with right_col:
        st.subheader("자치구별 비율 순위")
        st.dataframe(grouped_display, use_container_width=True)



## 이건 잘 안되서 집에서 다시 해보기
# elif menu == "히트맵 비교":

#     st.header("📊 자치구별 업종별 지출 비중 히트맵")

#     # 자치구-업종별 평균 지출액 계산 후 정규화
#     pivot_df = df_recent.groupby('행정동_코드_명')[target_columns].mean()
#     pivot_normalized = pivot_df.div(pivot_df.sum(axis=0), axis=1)

#     # 히트맵 시각화
#     fig, ax = plt.subplots(figsize=(14, 10))
#     sns.heatmap(pivot_normalized, annot=True, fmt=".1%", cmap="YlGnBu", ax=ax)
#     ax.set_title("자치구별 업종별 지출 비중 (최근 4개 분기)")
#     ax.set_xlabel("업종")
#     ax.set_ylabel("자치구")
#     plt.xticks(rotation=45)
#     st.pyplot(fig)
