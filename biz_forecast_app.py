import folium
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from matplotlib import font_manager, rc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

# 한글 폰트 설정 제거 (또는 주석 처리)
# font_path = "malgun.ttf"
# font_name = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font_name)

# 대신 기본 폰트에 한글이 깨지지 않도록 matplotlib 설정
plt.rcParams['font.family'] = 'DejaVu Sans'  # 또는 나눔고딕 등이 설치된 경우 해당 이름


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
df['행정동_코드_명'] = df['행정동_코드_명'].str.strip()  # 자치구 이름 공백 제거

# df = pd.read_csv(r"E:\python_chan\상권분석\서울시 상권분석서비스.csv", encoding='cp949')
# df.columns = df.columns.str.strip()

# 분기 컬럼 숫자 → 문자열 변환 및 정렬용 숫자 컬럼 생성
df = df.dropna(subset=['기준_년분기_코드'])  # ① NaN 제거 먼저
df['기준_년분기_코드'] = df['기준_년분기_코드'].astype(int)  # ② 정수형으로 변환 (불필요 시 생략 가능)
df['기준_년분기'] = df['기준_년분기_코드'].astype(str).str[:4] + '. Q' + df['기준_년분기_코드'].astype(str).str[-1]
df['분기_정렬'] = df['기준_년분기_코드']


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

gu_center_coords = {
    "강남구": [37.5172, 127.0473], "강북구": [37.6387, 127.0282], "강동구": [37.5302, 127.1237],
    "강서구": [37.5512, 126.8498], "관악구": [37.4803, 126.9527], "광진구": [37.5385, 127.0828],
    "구로구": [37.4953, 126.8877], "금천구": [37.4567, 126.8951], "노원구": [37.6541, 127.0567],
    "도봉구": [37.6687, 127.0466], "동대문구": [37.5743, 127.0395], "동작구": [37.5123, 126.9395],
    "중랑구": [37.6063, 127.0927], "서초구": [37.4836, 127.0326], "용산구": [37.5324, 126.9901],
    "마포구": [37.5638, 126.9084], "서대문구": [37.5792, 126.9368], "성동구": [37.5636, 127.0365],
    "성북구": [37.5901, 127.0165], "송파구": [37.5147, 127.1058], "양천구": [37.5172, 126.8663],
    "영등포구": [37.5263, 126.8959], "은평구": [37.6026, 126.9294], "종로구": [37.5809, 126.9828],
    "중구": [37.5642, 126.9976]}


# 사이드바 메뉴 생성
menu = st.sidebar.radio("메뉴 선택", [
    "자치구별 지출 비교",
    "자치구 업종별 지출 비중 지도",
    "성장률 최고 업종 지도",
    "자치구, 업종 히트맵 비교",
    "최종 창업 추천 도우미"
])



if menu == "자치구별 지출 비교":
    # 메뉴 1: 자치구 다중 선택
    selected_gu_list = st.multiselect("자치구를 선택하세요", sorted(df['행정동_코드_명'].unique()), default=["강남구"])
    exclude_cols = ['기타_지출_총금액', '지출_총금액']
    forecast_cols = [col for col in far_columns if col not in exclude_cols]

    # 3개씩 행에 그래프 그리기
    for i in range(0, len(target_columns), 3):
        cols = st.columns(3)
        for j in range(3):
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




elif menu == "자치구 업종별 지출 비중 지도":
    st.subheader("🗺️ 업종별 지출 비중 지도 시각화")
    selected_column = st.selectbox("업종을 선택하세요", 업종_리스트, index=8)

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
            color = 'blue' if ratio > 0.3 else 'orange' if ratio > 0.2 else 'red'
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




elif menu == "성장률 최고 업종 지도":
    st.subheader("🚀 자치구별 성장률 최고 업종 지도")

    # 자치구 리스트 확보
    gu_list = df['행정동_코드_명'].unique()

    # 결과 저장 리스트
    top_growth_per_gu = []

    for gu in gu_list:
        gu_df = df[df['행정동_코드_명'] == gu].sort_values('분기_정렬')
        best_growth = -np.inf
        best_col = None
        for col in far_columns:
            if gu_df[col].isna().sum() > 0:
                continue
            X = gu_df[['분기_정렬']]
            y = gu_df[col]
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            growth_pct = (slope * 5 / y.values[-1]) * 100
            if growth_pct > best_growth:
                best_growth = growth_pct
                best_col = col
        if best_col:
            top_growth_per_gu.append({
                '자치구': gu,
                '업종': best_col,
                '성장률': best_growth
            })

    growth_df = pd.DataFrame(top_growth_per_gu)
    growth_df = growth_df[growth_df['자치구'].isin(gu_center_coords.keys())]

    # 지도 시각화
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
    for _, row in growth_df.iterrows():
        gu = row['자치구']
        coord = gu_center_coords[gu]
        업종 = row['업종']
        성장률 = row['성장률']
        color = 'blue' if 성장률 > 10 else 'orange' if 성장률 > 5 else 'red'
        folium.CircleMarker(
            location=coord,
            radius=6 + 성장률 / 10,
            popup=f"{gu} - {업종}\n성장률: {성장률:.2f}%",
            color=color,
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

    st_folium(m, width=700, height=500)

    # 테이블 표시
    st.dataframe(growth_df.sort_values(by='성장률', ascending=False).reset_index(drop=True))




elif menu == "자치구, 업종 히트맵 비교":
    st.header("📊 자치구별 업종별 지출 비중 히트맵")

    # 최근 분기 선택
    recent_n = st.slider("최근 분기 개수 선택", min_value=2, max_value=8, value=4)
    recent_quarters = sorted(df['분기_정렬'].unique())[-recent_n:]
    df_recent_n = df[df['분기_정렬'].isin(recent_quarters)]

    # ✅ 여기부터 추가
    df_recent_n['행정동_코드_명'] = df_recent_n['행정동_코드_명'].str.strip()
    df_recent_n = df_recent_n.drop_duplicates(subset=['행정동_코드_명', '분기_정렬'])

    # 자치구-업종별 평균 지출액 계산 후 정규화
    pivot_df = df_recent_n.groupby('행정동_코드_명')[far_columns].mean()
    pivot_normalized = pivot_df.div(pivot_df.sum(axis=0), axis=1)

    # 시각화
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot_normalized, annot=True, fmt=".1%", cmap="YlGnBu", ax=ax, annot_kws={"size": 7})
    ax.set_title(f"자치구별 업종별 지출 비중 (최근 {recent_n}개 분기)", fontsize=10)
    ax.set_xlabel("업종", fontsize=10)
    ax.set_ylabel("자치구", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    st.pyplot(fig)





elif menu == "최종 창업 추천 도우미":
    st.header("📌 최종 창업 추천 도우미")

    기준선택 = st.radio("먼저 선택할 항목을 고르세요", ["자치구 선택 → 유망 업종", "업종 선택 → 유망 자치구"])

    future_periods = 16
    last_code = df['분기_정렬'].max()
    year = int(str(last_code)[:4])
    quarter = int(str(last_code)[-1])

    future_codes = []
    for _ in range(future_periods):
        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1
        future_codes.append(int(f"{year}0{quarter}"))
    future_labels = [f"{str(code)[:4]}. Q{str(code)[-1]}" for code in future_codes]

    def normalize_score(value, max_value):
        return min(100, (value / max_value) * 100) if max_value else 0

    max_market_reference = df[far_columns].mean().max()
    max_competition_std = df[far_columns].std().max()

    import matplotlib.pyplot as plt
    import numpy as np

    def plot_radar(labels, growth, stability, market, competition):
        categories = ['성장률', '안정성', '시장규모', '경쟁도']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(2.5, 2.5), subplot_kw=dict(polar=True))
        for idx, label in enumerate(labels):
            values = [growth[idx] * 0.25, stability[idx] * 0.25, market[idx] * 0.25, competition[idx] * 0.25]
            values += values[:1]
            ax.plot(angles, values, label=label)
            ax.fill(angles, values, alpha=0.1)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 25)
        ax.set_title('업종별 종합 점수 비교 (정규화)', fontsize=10)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=7)
        st.pyplot(fig)


    if 기준선택 == "자치구 선택 → 유망 업종":
        selected_gu = st.selectbox("자치구를 선택하세요", sorted(df['행정동_코드_명'].unique()))
        st.markdown(f"### ✅ {selected_gu}에서 유망한 업종 Top 3")

        growth_data = []
        for col in far_columns:
            gu_df = df[df['행정동_코드_명'] == selected_gu].sort_values(by='분기_정렬')
            y = gu_df[col]
            if len(y) < 4 or y.isna().sum() > 0:
                continue

            window = 4
            y_ma = y.rolling(window=window, min_periods=1).mean()
            slope = np.polyfit(range(len(y_ma)), y_ma, 1)[0]
            y_pred = list(y_ma) + [y_ma.iloc[-1] + slope * i for i in range(1, future_periods + 1)]

            pred_start_avg = np.mean(y_pred[len(y):len(y)+4])
            pred_end_avg = np.mean(y_pred[-4:])
            growth_percent = ((pred_end_avg - pred_start_avg) / pred_start_avg) * 100

            volatility = np.std(y[-window:]) / np.mean(y[-window:]) if np.mean(y[-window:]) > 0 else np.inf
            stability_score = max(0, 1 - volatility) * 100

            market_raw = np.mean(y[-window:])
            market_score = normalize_score(market_raw, max_market_reference)

            competition_values = df.groupby('행정동_코드_명')[col].mean()
            competition_std = np.std(competition_values)
            competition_score = max(0, 100 - normalize_score(competition_std, max_competition_std))

            final_score = 0.25 * growth_percent + 0.25 * stability_score + 0.25 * market_score + 0.25 * competition_score

            growth_data.append((col, growth_percent, stability_score, market_score, competition_score, final_score,
                                y.copy(), y_pred.copy(), list(gu_df['기준_년분기']) + future_labels))

        growth_data.sort(key=lambda x: x[5], reverse=True)
        top3_data = [d for d in growth_data if d[1] > 0 and d[5] > 0][:3]

        radar_labels = []
        radar_growth = []
        radar_stability = []
        radar_market = []
        radar_competition = []

        cols = st.columns(3)
        for idx, (col_name, growth_percent, stability_score, market_score, competition_score, final_score, y_vals, y_pred, full_quarters) in enumerate(top3_data):
            full_quarters = full_quarters[:len(y_pred)]
            with cols[idx]:
                st.markdown(f"#### ✅ {col_name}")
                st.markdown(f"📈 예상 성장률(4년): **{growth_percent * 0.25:.2f}점**")
                st.markdown(f"🔒 안정성 점수: **{stability_score * 0.25:.1f}점**")
                st.markdown(f"💰 시장 규모 점수: **{market_score * 0.25:.1f}점**")
                st.markdown(f"⚔️ 경쟁도 점수: **{competition_score * 0.25:.1f}점**")
                st.markdown(f"⭐ 최종 추천 점수: **{final_score:.1f}점**")
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(full_quarters[:len(y_vals)], y_vals, marker='o', label='실제')
                ax.plot(full_quarters, y_pred, linestyle='--', label='예측')
                ax.set_title(f"{col_name} 지출 추이 및 전망", fontsize=11)
                ax.tick_params(axis='x', labelrotation=45, labelsize=8)
                ax.legend(fontsize=8)
                st.pyplot(fig)
                plt.close(fig)

                radar_labels.append(col_name)
                radar_growth.append(growth_percent)
                radar_stability.append(stability_score)
                radar_market.append(market_score)
                radar_competition.append(competition_score)

        if radar_labels:
            plot_radar(radar_labels, radar_growth, radar_stability, radar_market, radar_competition)

    elif 기준선택 == "업종 선택 → 유망 자치구":
        selected_field = st.selectbox("업종을 선택하세요", far_columns, index=far_columns.index('음식_지출_총금액'))
        st.markdown(f"### ✅ '{selected_field}' 업종에 유망한 자치구 Top 3")

        growth_data = []
        for gu in df['행정동_코드_명'].unique():
            gu_df = df[df['행정동_코드_명'] == gu].sort_values(by='분기_정렬')
            y = gu_df[selected_field]
            if len(y) < 4 or y.isna().sum() > 0:
                continue

            window = 4
            y_ma = y.rolling(window=window, min_periods=1).mean()
            slope = np.polyfit(range(len(y_ma)), y_ma, 1)[0]
            y_pred = list(y_ma) + [y_ma.iloc[-1] + slope * i for i in range(1, future_periods + 1)]

            pred_start_avg = np.mean(y_pred[len(y):len(y)+4])
            pred_end_avg = np.mean(y_pred[-4:])
            growth_percent = ((pred_end_avg - pred_start_avg) / pred_start_avg) * 100

            volatility = np.std(y[-window:]) / np.mean(y[-window:]) if np.mean(y[-window:]) > 0 else np.inf
            stability_score = max(0, 1 - volatility) * 100

            market_raw = np.mean(y[-window:])
            market_score = normalize_score(market_raw, max_market_reference)

            competition_values = df.groupby('행정동_코드_명')[selected_field].mean()
            competition_std = np.std(competition_values)
            competition_score = max(0, 100 - normalize_score(competition_std, max_competition_std))

            final_score = 0.25 * growth_percent + 0.25 * stability_score + 0.25 * market_score + 0.25 * competition_score

            growth_data.append((gu, growth_percent, stability_score, market_score, competition_score, final_score,
                                y.copy(), y_pred.copy(), list(gu_df['기준_년분기']) + future_labels))

        growth_data.sort(key=lambda x: x[5], reverse=True)
        top3_data = [d for d in growth_data if d[1] > 0 and d[5] > 0][:3]

        radar_labels = []
        radar_growth = []
        radar_stability = []
        radar_market = []
        radar_competition = []

        cols = st.columns(3)
        for idx, (gu_name, growth_percent, stability_score, market_score, competition_score, final_score, y_vals, y_pred, full_quarters) in enumerate(top3_data):
            full_quarters = full_quarters[:len(y_pred)]
            with cols[idx]:
                st.markdown(f"#### 📍 {gu_name}")
                st.markdown(f"📈 예상 성장률(4년): **{growth_percent * 0.25:.2f}점**")
                st.markdown(f"🔒 안정성 점수: **{stability_score * 0.25:.1f}점**")
                st.markdown(f"💰 시장 규모 점수: **{market_score * 0.25:.1f}점**")
                st.markdown(f"⚔️ 경쟁도 점수: **{competition_score * 0.25:.1f}점**")
                st.markdown(f"⭐ 최종 추천 점수: **{final_score:.1f}점**")
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(full_quarters[:len(y_vals)], y_vals, marker='o', label='실제')
                ax.plot(full_quarters, y_pred, linestyle='--', label='예측')
                ax.set_title(f"{gu_name}의 지출 추이 및 전망", fontsize=11)
                ax.tick_params(axis='x', labelrotation=45, labelsize=8)
                ax.legend(fontsize=8)
                st.pyplot(fig)
                plt.close(fig)

                radar_labels.append(gu_name)
                radar_growth.append(growth_percent)
                radar_stability.append(stability_score)
                radar_market.append(market_score)
                radar_competition.append(competition_score)

        if radar_labels:
            plot_radar(radar_labels, radar_growth, radar_stability, radar_market, radar_competition)
