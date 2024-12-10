import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import seaborn as sns
import os

# 데이터 불러오기 함수
def load_data(file_path):
    if not os.path.exists(file_path):f
    raise FileNotFoundError(f"파일 경로를 확인하세요: {file_path}")
    return pd.read_csv(file_path)

# VIF 계산 함수
def calculate_vif(df):
    vif = pd.DataFrame()
    vif["Feature"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif

# VIF 값이 높은 변수 제거 함수
def remove_high_vif_features(data, threshold=10):
    while True:
        vif_df = calculate_vif(data)
        max_vif = vif_df['VIF'].max()
        if max_vif > threshold:
            max_vif_feature = vif_df.loc[vif_df['VIF'] == max_vif, 'Feature'].values[0]
            data = data.drop(columns=[max_vif_feature])
            print(f"VIF가 {threshold} 초과인 '{max_vif_feature}' 변수를 제거하였습니다.")
        else:
            break
    return data

# 정상성 검정 함수
def test_stationarity(data, column_name):
    result = adfuller(data[column_name].dropna())
    print(f"ADF Test Statistic for {column_name}: {result[0]}")
    print(f"P-Value: {result[1]}")
    if result[1] <= 0.05:
        print(f"'{column_name}'은 정상성을 만족합니다.")
    else:
        print(f"'{column_name}'은 정상성을 만족하지 않습니다. 추가 변환 필요.")

# 시계열 전처리 함수
def preprocess_time_series(data, column_name, period=52):
    # 계절성 분해
    decomposed = seasonal_decompose(data[column_name], model='additive', period=period)
    data[f"{column_name}_detrended"] = data[column_name] - decomposed.trend
    data[f"{column_name}_deseasonalized"] = data[f"{column_name}_detrended"] - decomposed.seasonal
    return data

# 상관 행렬 시각화 함수
def plot_correlation_matrix(data, title, annotate=True):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=annotate,
        fmt=".2f" if annotate else "",
        cmap='coolwarm',
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

# 데이터 경로 설정
air_pollutants_path = 'NewData/Weekly_Air_Pollutants.csv'

# 데이터 로드
air_pollutants = load_data(air_pollutants_path)

# 'week_range' 열 처리: 시작일 추출
air_pollutants['start_date'] = air_pollutants['week_range'].str.split('~').str[0]
air_pollutants['start_date'] = pd.to_datetime(air_pollutants['start_date'], format='%Y-%m-%d')

# 시계열 데이터로 설정
air_pollutants.set_index('start_date', inplace=True)
air_pollutants.drop(columns=['week_range'], inplace=True)

# 모든 변수 선택
columns_to_analyze = air_pollutants.columns.tolist()
# 주요 변수 선택
# columns_to_analyze = ['CO', 'Nox', 'NH3']

# 추세 및 계절성 제거
for column in columns_to_analyze:
    air_pollutants = preprocess_time_series(air_pollutants, column, period=52)

# 정상성 테스트
for column in columns_to_analyze:
    test_stationarity(air_pollutants, f"{column}_deseasonalized")

# VIF 계산을 위한 데이터 구성
processed_columns = [f"{col}_deseasonalized" for col in columns_to_analyze]
vif_data = air_pollutants[processed_columns].dropna()

# VIF 계산 및 재계산
print("\n초기 VIF 계산 결과:")
vif_results = calculate_vif(vif_data)
print(vif_results)

# VIF 값이 높은 변수 제거 및 재계산
reduced_vif_data = remove_high_vif_features(vif_data)

print("\n최종 VIF 계산 결과:")
final_vif_results = calculate_vif(reduced_vif_data)
print(final_vif_results)

# 상관 행렬 시각화
plot_correlation_matrix(reduced_vif_data, "Processed X data correlation matrix", annotate=True)

# Granger 인과 검정 (예: 'Nox'가 'CO'에 영향을 미치는지 확인)
print("\nGranger Causality Test (maxlag=4):")
grangercausalitytests(air_pollutants[['CO_deseasonalized', 'Nox_deseasonalized']].dropna(), maxlag=4)