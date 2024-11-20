import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import folium_static
from folium.features import CustomIcon
import base64
import os


# 거리 계산 함수 정의
def calculate_distance(lat1, lon1, lat2, lon2):
    """
    하버사인 공식을 사용하여 두 지점 간의 거리를 계산합니다.

    Parameters:
    - lat1, lon1: 첫 번째 지점의 위도와 경도
    - lat2, lon2: 두 번째 지점의 위도와 경도

    Returns:
    - 거리 (미터 단위)
    """
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
def recommend_diverse_locations(last_lat, last_lon, data, model, input_features, min_distance=200,
                                num_recommendations=3):
    """
    마지막 위치를 기준으로 모델을 사용하여 추천 지점을 선정합니다.

    Parameters:
    - last_lat, last_lon: 실종자의 마지막 위치 위도와 경도
    - data: 추천 후보 지점 데이터프레임
    - model: 예측 모델
    - input_features: 모델 예측에 사용되는 피처 리스트
    - min_distance: 추천 지점 간 최소 거리 (미터 단위)
    - num_recommendations: 추천 지점의 수

    Returns:
    - 추천 지점 데이터프레임과 사용된 최소 거리
    """
    # 각 격자 중심점과의 거리 계산 및 모델 입력 데이터 생성
    data['Distance'] = data.apply(
        lambda row: calculate_distance(last_lat, last_lon, row['Latitude'], row['Longitude']), axis=1)

    # 모델에 필요한 피처만 선택하여 예측 수행
    prediction_input = data[input_features]
    data['predicted_probability'] = model.predict(prediction_input)

    # 발견 확률이 높은 순으로 데이터 정렬
    sorted_data = data.sort_values(by='predicted_probability', ascending=False)

    # 최소 거리 필터 적용하여 지점 추천
    recommended = []
    for _, row in sorted_data.iterrows():
        if all(calculate_distance(row['Latitude'], row['Longitude'], rec['Latitude'], rec['Longitude']) > min_distance
               for rec in recommended):
            recommended.append(row)
            if len(recommended) == num_recommendations:
                break

    return pd.DataFrame(recommended), min_distance


# Base64로 이미지 인코딩하는 함수
def get_base64_image(image_path):
    """
    이미지 파일을 Base64로 인코딩합니다.

    Parameters:
    - image_path: 이미지 파일의 경로

    Returns:
    - Base64로 인코딩된 이미지 문자열
    """
    if not os.path.exists(image_path):
        st.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return None
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')


# 메인 함수
def main():
    # 사용자 입력을 사이드바로 이동
    st.sidebar.title("입력 값 설정")

    # 1. 모델 선택 옵션 추가
    model_selection = st.sidebar.selectbox(
        "모델 선택",
        ("일반 모델", "개인 모델")
    )

    # 2. 모델 파일 및 데이터셋 경로 설정
    if model_selection == "일반 모델":
        model_path = 'base_model_papaer.pkl'  # 일반 모델 파일 경로
        data_path = 'base_model_datset_weighted_paper.csv'  # 일반 모델 데이터셋 경로
        input_features = ['Distance', 'Crossroad', 'Slope', 'Traffic_light']  # 일반 모델 피처
    else:
        model_path = 'personal_model_paper.pkl'  # 개인 모델 파일 경로
        data_path = 'personal_model_dataset_weighted_paper.csv'  # 개인 모델 데이터셋 경로
        input_features = ['Distance', 'Crossroad', 'Slope', 'Traffic_light', 'Frequent_Place']  # 개인 모델 피처

    # 3. 나머지 입력 값 설정
    last_lat = st.sidebar.number_input("위도 (Latitude)", format="%.6f", value=0.0)
    last_lon = st.sidebar.number_input("경도 (Longitude)", format="%.6f", value=0.0)
    min_distance = st.sidebar.number_input("추천 지점 간 최소 거리 (미터)", min_value=0, value=200, step=50)
    circle_radius = st.sidebar.selectbox("추천 지점 주변 원의 반경 (미터)", options=[0, 100, 150, 200], index=1)
    map_tiles = st.sidebar.selectbox("지도 배경 선택", (
        "OpenStreetMap",
        "Stamen Terrain",
        "Stamen Toner",
        "Stamen Watercolor",
        "CartoDB positron",
        "CartoDB dark_matter"
    ))
    num_recommendations = st.sidebar.number_input("추천 지점 수", min_value=1, value=3, step=1)

    # 제목과 로고를 가로로 배치 using columns
    col1, col2 = st.columns([1, 5])  # 비율을 조정하여 공간 분배

    with col1:
        # 로고 이미지 삽입 (로컬 이미지 파일 사용)
        logo_path = "logo3.png"  # 로고 이미지 파일 경로
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
        else:
            st.error(f"로고 파일을 찾을 수 없습니다: '{logo_path}'")

    with col2:
        # 제목 폰트 사이즈 조정 및 두 줄로 표시
        st.markdown(
            """
            <h1 style='font-size: 35px; text-align: left;'>
                WENDY:<br>
                실종자 발견 위치 예측 시스템
            </h1>
            """,
            unsafe_allow_html=True
        )

    st.write("실종자의 마지막 위치 좌표와 최소 거리 기준을 입력하세요.")

    if st.sidebar.button("예측 실행"):
        # 모델 및 데이터 로드
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            st.error(f"모델 파일을 찾을 수 없습니다: '{model_path}'")
            return
        except Exception as e:
            st.error(f"모델 로드 중 오류가 발생했습니다: {e}")
            return

        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            st.error(f"데이터 파일을 찾을 수 없습니다: '{data_path}'")
            return
        except Exception as e:
            st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
            return

        # 위치 추천
        recommended_locations, used_min_distance = recommend_diverse_locations(
            last_lat, last_lon, df, model, input_features=input_features, min_distance=min_distance,
            num_recommendations=num_recommendations
        )

        if recommended_locations.empty:
            st.write("조건에 맞는 추천 지점을 찾을 수 없습니다.")
            return

        # 결과 표시 - 예쁜 표 형식으로 출력
        st.markdown("<hr>", unsafe_allow_html=True)  # 구분선 추가

        # 소제목과 아이콘을 가로로 배치 using columns
        sub_col1, sub_col2 = st.columns([1, 10])

        with sub_col1:
            # 소제목 옆에 아이콘 추가 (로컬 이미지 파일 사용)
            coord_icon_path = "result4.png"  # 소제목 아이콘 이미지 파일 경로
            if os.path.exists(coord_icon_path):
                st.image(coord_icon_path, width=35)
            else:
                st.error(f"소제목 아이콘 파일을 찾을 수 없습니다: '{coord_icon_path}'")

        with sub_col2:
            # 소제목 스타일링
            st.markdown(
                """
                <h2 style='font-size: 20px; color: black;'>
                    예상 위치 좌표 및 확률
                </h2>
                """,
                unsafe_allow_html=True
            )

        styled_df = recommended_locations[['Latitude', 'Longitude', 'predicted_probability']].copy()
        styled_df.columns = ['위도', '경도', '예측 확률']
        styled_df['예측 확률'] = styled_df['예측 확률'].map('{:.4f}'.format)

        # 인덱스를 리셋하고 드롭하여 인덱스 컬럼 제거
        styled_df.reset_index(drop=True, inplace=True)

        # 스타일 적용
        st.table(styled_df.style.set_properties(**{'text-align': 'center', 'color': 'black'}))

        # 소제목과 아이콘을 가로로 배치 using columns
        map_sub_col1, map_sub_col2 = st.columns([1, 10])

        with map_sub_col1:
            # 소제목 옆에 아이콘 추가 (로컬 이미지 파일 사용)
            map_icon_path = "map1.png"  # 소제목 아이콘 이미지 파일 경로
            if os.path.exists(map_icon_path):
                st.image(map_icon_path, width=35)
            else:
                st.error(f"지도 소제목 아이콘 파일을 찾을 수 없습니다: '{map_icon_path}'")

        with map_sub_col2:
            # 소제목 스타일링
            st.markdown(
                """
                <h2 style='font-size: 20px; color: black;'>
                    예상 위치 지도
                </h2>
                """,
                unsafe_allow_html=True
            )

        # 지도 생성
        m = folium.Map(location=[last_lat, last_lon], zoom_start=13, tiles=map_tiles)

        # 마지막 위치 마커 아이콘 설정
        last_icon_image = 'person4.png'  # 마지막 위치 아이콘 이미지 파일 경로
        last_icon_base64 = get_base64_image(last_icon_image)
        if last_icon_base64:
            last_custom_icon = CustomIcon(
                f"data:image/png;base64,{last_icon_base64}",
                icon_size=(30, 30)
            )
        else:
            # 기본 아이콘 사용
            last_custom_icon = folium.Icon(color='red', icon='user')

        # 마지막 위치 마커 추가
        folium.Marker(
            [last_lat, last_lon],
            tooltip='마지막 위치',
            icon=last_custom_icon
        ).add_to(m)

        # 추천 위치 마커 추가
        for _, row in recommended_locations.iterrows():
            # 커스텀 아이콘 사용
            icon_image = 'person5.png'  # 추천 위치 아이콘 이미지 파일 경로
            icon_base64 = get_base64_image(icon_image)
            if icon_base64:
                custom_icon = CustomIcon(
                    f"data:image/png;base64,{icon_base64}",
                    icon_size=(30, 30)
                )
            else:
                # 기본 아이콘 사용
                custom_icon = folium.Icon(color='blue', icon='flag')

            # 팝업 및 툴팁 설정
            popup = folium.Popup(f"<b>예측 확률:</b> {row['predicted_probability']:.4f}", max_width=300)
            tooltip = f"위치: ({row['Latitude']:.6f}, {row['Longitude']:.6f})"

            # 마커 추가
            folium.Marker(
                [row['Latitude'], row['Longitude']],
                tooltip=tooltip,
                popup=popup,
                icon=custom_icon  # 커스텀 아이콘 적용
            ).add_to(m)

            # 추천 지점 주변에 원 그리기 (테두리 없이 영역만 옅게 표시)
            if circle_radius > 0:
                folium.Circle(
                    location=[row['Latitude'], row['Longitude']],
                    radius=circle_radius,
                    color='blue',  # 테두리 색상 (무시됨)
                    fill=True,
                    fill_color='blue',  # 채우기 색상
                    fill_opacity=0.2,  # 채우기 투명도
                    weight=0  # 테두리 두께를 0으로 설정하여 테두리 없애기
                ).add_to(m)

        # 지도 표시
        folium_static(m)

        # 추가 정보 출력 (글자 색상 검정)
        st.markdown(
            f"""
            <p style='font-size: 18px; color: black;'>
                <b>최소 거리 사용:</b> {used_min_distance} m
            </p>
            <p style='font-size: 18px; color: black;'>
                <b>마지막 GPS 위치:</b> 위도 = {last_lat}, 경도 = {last_lon}
            </p>
            """,
            unsafe_allow_html=True
        )


# 실행
if __name__ == '__main__':
    main()
