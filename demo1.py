import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import folium_static

# 거리 계산 함수 정의
def calculate_distance(lat1, lon1, lat2, lon2):
    # 하버사인 공식을 사용하여 거리 계산
    R = 6371000  # 지구 반지름 (미터)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * \
        np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c  # 거리 (미터)

# 지리적 다양성을 고려한 위치 추천 함수 정의
def recommend_diverse_locations(last_lat, last_lon, data, model, min_distance=200):
    # 각 격자 중심점과의 거리 계산 및 모델 입력 데이터 생성
    data['Distance'] = data.apply(
        lambda row: calculate_distance(last_lat, last_lon, row['Latitude'], row['Longitude']), axis=1)
    data['predicted_probability'] = model.predict(
        data[['Distance', 'Crossroad', 'Slope', 'Traffic_Light']])

    # 발견 확률이 높은 순으로 데이터 정렬
    sorted_data = data.sort_values(by='predicted_probability', ascending=False)

    # 최소 거리 필터 적용하여 지점 추천
    recommended = []
    for _, row in sorted_data.iterrows():
        if all(calculate_distance(row['Latitude'], row['Longitude'], rec['Latitude'], rec['Longitude']) > min_distance
               for rec in recommended):
            recommended.append(row)
            if len(recommended) == 3:
                break

    return pd.DataFrame(recommended), min_distance

# 메인 함수
def main():
    st.title("실종자 발견 위치 예측 시스템")

    st.write("실종자의 마지막 위치 좌표와 최소 거리 기준을 입력하세요.")

    # 사용자 입력
    last_lat = st.number_input("위도 (Latitude)", format="%.6f")
    last_lon = st.number_input("경도 (Longitude)", format="%.6f")
    min_distance = st.number_input("추천 지점 간 최소 거리 (미터)", min_value=0, value=200, step=50)

    if st.button("예측 실행"):
        # 모델 및 데이터 로드
        model = joblib.load('location_probability_base_model.pkl')
        df = pd.read_csv('grid_center_points_with_probability_weighted.csv')

        # 위치 추천
        recommended_locations, used_min_distance = recommend_diverse_locations(
            last_lat, last_lon, df, model, min_distance=min_distance)

        if recommended_locations.empty:
            st.write("조건에 맞는 추천 지점을 찾을 수 없습니다.")
            return

        # 결과 표시
        st.subheader("추천 위치 좌표 및 확률")
        st.write(recommended_locations[['Latitude', 'Longitude', 'predicted_probability']])

        # 지도 생성
        m = folium.Map(location=[last_lat, last_lon], zoom_start=13)

        # 마지막 위치 마커 추가
        folium.Marker([last_lat, last_lon], tooltip='마지막 위치', icon=folium.Icon(color='red')).add_to(m)

        # 추천 위치 마커 추가
        for _, row in recommended_locations.iterrows():
            folium.Marker(
                [row['Latitude'], row['Longitude']],
                tooltip=f"예측 확률: {row['predicted_probability']:.4f}",
                icon=folium.Icon(color='blue')
            ).add_to(m)

        # 지도 표시
        st.subheader("추천 위치 지도")
        folium_static(m)

        # 추가 정보 출력
        st.write(f"\n최소 거리 사용: {used_min_distance} m")
        st.write(f"마지막 GPS 위치: Latitude = {last_lat}, Longitude = {last_lon}")

if __name__ == '__main__':
    main()
