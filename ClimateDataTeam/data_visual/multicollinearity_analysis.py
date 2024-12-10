import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# 데이터 불러오기 함수
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일 경로를 확인하세요: {file_path}")
    return pd.read_csv(file_path)

# 상관 행렬 시각화 함수 (숫자 표시 선택 가능)
def plot_correlation_matrix(data, title, annotate=True):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 10))  # 크기 확장
    sns.heatmap(
        correlation_matrix, 
        annot=annotate,         # 숫자 표시 여부 설정
        fmt=".2f" if annotate else "",  # 숫자 포맷 (표시할 경우)
        cmap='coolwarm', 
        linewidths=0.5, 
        cbar_kws={'shrink': 0.8}  # 색상 막대 크기 조정
    )
    plt.title(title, fontsize=16)  # 제목 크기 키움
    plt.xticks(fontsize=12)        # 축 라벨 크기 키움
    plt.yticks(fontsize=12)
    plt.tight_layout()            # 레이아웃 조정
    plt.show()

# VIF 계산 함수
def calculate_vif(df):
    vif = pd.DataFrame()
    vif["Feature"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif

# VIF 제거 함수
def remove_high_vif_features(data, threshold=10):
    vif_df = calculate_vif(data)
    print("초기 VIF 계산:")
    print(vif_df)
    
    high_vif_features = vif_df[vif_df["VIF"] > threshold]["Feature"]
    reduced_data = data.drop(columns=high_vif_features)
    
    reduced_vif_df = calculate_vif(reduced_data)
    print("\nVIF 제거 후 계산:")
    print(reduced_vif_df)
    return reduced_data

# 파일 경로 설정
air_pollutants_path = 'NewData/Weekly_Air_Pollutants.csv'
population_path = 'NewData/Weekly_Greenhouse_Gas_Population.csv'
power_usage_path = 'NewData/Weekly_Power_Usage.csv'
climate_data_path = 'ClimateDataTeam/climate_data/merged_weekly_avg_temp.csv'
weekly_co2_path = 'ClimateDataTeam/climate_data/weekly_co2.csv'

# x 데이터 불러오기 및 병합
air_pollutants = load_data(air_pollutants_path)
population = load_data(population_path)[['week_range', 'TotalPopulation']]
power_usage = load_data(power_usage_path)

merged_x_data = air_pollutants
merged_x_data = pd.merge(merged_x_data, population, on='week_range')
merged_x_data = pd.merge(merged_x_data, power_usage, on='week_range')

# y 데이터 불러오기 및 병합
climate_data = load_data(climate_data_path)
weekly_co2 = load_data(weekly_co2_path)

# 열 이름의 공백 제거
climate_data.columns = climate_data.columns.str.strip()
weekly_co2.columns = weekly_co2.columns.str.strip()

# 필요한 열 선택
weekly_co2 = weekly_co2[['datetime', 'molfrac']]

# 'datetime' 열을 datetime 객체로 변환
climate_data['datetime'] = pd.to_datetime(climate_data['datetime'])
weekly_co2['datetime'] = pd.to_datetime(weekly_co2['datetime'])

# 데이터 병합
merged_y_data = pd.merge(climate_data, weekly_co2, on='datetime', how='inner')

# 숫자형 데이터 추출
numeric_x_data = merged_x_data.select_dtypes(include=[np.number])
numeric_y_data = merged_y_data.select_dtypes(include=[np.number])

# X 데이터 상관 행렬 시각화
plot_correlation_matrix(numeric_x_data, "X data correlation matrix", annotate=True)

# Y 데이터 상관 행렬 시각화
plot_correlation_matrix(numeric_y_data, "Y data correlation matrix", annotate=True)

# X 데이터에서 VIF 제거
reduced_x_data = remove_high_vif_features(numeric_x_data)

# 최종 X 데이터프레임 구성
final_x_data = pd.concat([reduced_x_data, merged_x_data.drop(columns=numeric_x_data.columns)], axis=1)

# 결과 출력
print("최종 X 데이터프레임:")
print(final_x_data.head())


##################################### VIF 계산결과 #####################################

# 초기 VIF 계산:
#                     Feature          VIF
# 0                        CO     4.834031
# 1                       Nox     8.012287
# 2                       Sox    44.380880
# 3                       TSP   251.173196
# 4                     PM-10   241.663798
# 5                      VOCs    51.672701
# 6                       NH3     5.665047
# 7           TotalPopulation  3662.429334
# 8               Residential          inf
# 9                   General          inf
# 10              Educational          inf
# 11               Industrial          inf
# 12             Agricultural          inf
# 13             Streetlights          inf
# 14                    Other          inf
# 15  Total Electricity Usage          inf

# VIF 제거 후 계산:
#   Feature         VIF
# 0      CO  203.612520
# 1     Nox  135.507384
# 2     NH3   56.401421
# 최종 X 데이터프레임:
#              CO           Nox           NH3             week_range
# 0  8.227669e+08  1.242265e+09  2.225806e+08  2002-01-06~2002-01-12
# 1  8.224332e+08  1.244570e+09  2.228252e+08  2002-01-13~2002-01-19
# 2  8.220995e+08  1.246875e+09  2.230698e+08  2002-01-20~2002-01-26
# 3  8.217658e+08  1.249181e+09  2.233145e+08  2002-01-27~2002-02-02
# 4  8.214321e+08  1.251486e+09  2.235591e+08  2002-02-03~2002-02-09

# VIF 계산 결과, 'TSP', 'PM-10', 'VOCs' 등의 열이 높은 다중공선성을 보이므로 제거하였다.
# 일반적으로 VIF가 10 이상이면 다중공선성이 있다고 판단함. 5 이하를 이상적인 값으로 간주함.

# 결론: CO, Nox, NH3를 현재 상태로 모두 사용하는 것은 바람직하지 않음. 하지만 다음과 같은 방법으로 개선 가능함.
# CO 또는 Nox 중 하나만 선택하여 사용.
# PCA를 적용해 새로운 변수로 대체.
# 규제 모델을 사용해 다중공선성 문제를 완화.