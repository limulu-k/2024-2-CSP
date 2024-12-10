import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def plot_time_series(merged_data, population_column):
    """
    시계열 시각화 함수
    :param merged_data: 병합된 데이터프레임
    :param population_column: y축에 사용할 인구 변수 (예: 'TotalPopulation')
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 첫 번째 y축에 선택된 인구 변수
    ax1.plot(merged_data['start_date'], merged_data[population_column], label=population_column, color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(population_column, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 두 번째 y축에 다른 변수들
    ax2 = ax1.twinx()
    ax2.plot(merged_data['start_date'], merged_data['tempmax'], label='Max Temperature', color='red')
    ax2.plot(merged_data['start_date'], merged_data['tempmin'], label='Min Temperature', color='green')
    ax2.plot(merged_data['start_date'], merged_data['temp'], label='Average Temperature', color='orange')
    ax2.plot(merged_data['start_date'], merged_data['dew'], label='Dew Point', color='purple')
    ax2.plot(merged_data['start_date'], merged_data['humidity'], label='Humidity', color='brown')
    ax2.plot(merged_data['start_date'], merged_data['precip'], label='Precipitation', color='pink')
    ax2.plot(merged_data['start_date'], merged_data['windspeed'], label='Wind Speed', color='cyan')
    ax2.plot(merged_data['start_date'], merged_data['sealevelpressure'], label='Sea Level Pressure', color='magenta')
    ax2.plot(merged_data['start_date'], merged_data['moonphase'], label='Moon Phase', color='yellow')
    ax2.plot(merged_data['start_date'], merged_data['molfrac'], label='CO2 Mole Fraction', color='black')

    ax2.set_ylabel('Climate Variables', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # 범례 설정
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    fig.tight_layout()  # 그래프가 겹치지 않도록 조정
    plt.title(f'{population_column} vs Various Climate Variables')
    plt.grid()
    plt.show()

# 데이터 불러오기
ghg_population = pd.read_csv('NewData/Weekly_Greenhouse_Gas_Population.csv')
avg_temp = pd.read_csv('ClimateDataTeam/climate_data/merged_weekly_avg_temp.csv')
weekly_co2 = pd.read_csv('ClimateDataTeam/climate_data/weekly_co2.csv')

# 데이터 전처리
avg_temp.columns = avg_temp.columns.str.replace(' ', '')
weekly_co2.columns = weekly_co2.columns.str.replace(' ', '')
weekly_co2 = weekly_co2[['datetime', 'molfrac']]

# week_range에서 시작일 추출하고 datetime으로 변환
ghg_population['start_date'] = ghg_population['week_range'].apply(lambda x: pd.to_datetime(x.split('~')[0].strip()))
ghg_population.drop(columns=['week_range'], inplace=True)

# datetime 열을 datetime 형식으로 변환하고 시간 정보 제거
ghg_population['start_date'] = pd.to_datetime(ghg_population['start_date']).dt.date
avg_temp['datetime'] = pd.to_datetime(avg_temp['datetime']).dt.date
weekly_co2['datetime'] = pd.to_datetime(weekly_co2['datetime']).dt.date

# 데이터 병합 (inner join 사용)
merged_data = pd.merge(ghg_population, avg_temp, left_on='start_date', right_on='datetime', how='inner')
merged_data = pd.merge(merged_data, weekly_co2, left_on='start_date', right_on='datetime', how='inner')

# 불필요한 열 제거
merged_data.drop(columns=['datetime_x', 'datetime_y', 'datetime'], errors='ignore', inplace=True)

# 수치형 열의 결측값 처리
numeric_columns = merged_data.select_dtypes(include=[np.number]).columns
merged_data[numeric_columns] = merged_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
merged_data[numeric_columns] = merged_data[numeric_columns].fillna(method='ffill').fillna(method='bfill')

# 날짜별로 정렬
merged_data.sort_values('start_date', inplace=True)

# 상관계수와 p-value를 저장할 데이터프레임 생성
correlation_matrix = pd.DataFrame(index=numeric_columns, columns=numeric_columns)
pvalue_matrix = pd.DataFrame(index=numeric_columns, columns=numeric_columns)

# 상관계수와 p-value 계산
for col1 in numeric_columns:
    for col2 in numeric_columns:
        corr_coeff, p_value = pearsonr(merged_data[col1], merged_data[col2])
        correlation_matrix.loc[col1, col2] = corr_coeff
        pvalue_matrix.loc[col1, col2] = p_value

# 상관계수 히트맵 시각화
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix.astype(float), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Coefficient Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# p-value 히트맵 시각화
plt.figure(figsize=(14, 10))
sns.heatmap(pvalue_matrix.astype(float), annot=True, cmap='viridis', fmt='.2e', linewidths=0.5, annot_kws={"size": 8})
plt.title('P-value Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# p-value 0.05 이하인 값들의 순서쌍 출력
significant_pairs = []

for col1 in numeric_columns:
    for col2 in numeric_columns:
        if pvalue_matrix.loc[col1, col2] <= 0.05 and col1 != col2:
            significant_pairs.append((col1, col2, pvalue_matrix.loc[col1, col2]))

print("Significant pairs with p-value <= 0.05:")
for pair in significant_pairs:
    print(f"{pair[0]} - {pair[1]}: p-value = {pair[2]:.4e}")



plot_time_series(merged_data, 'TotalPopulation')

# # 시계열 시각화 함수 호출 (모든 인구 변수에 대해)
# population_columns = [
#     'TotalPopulation', 'BusanPopulation', 'ChungcheongbukPopulation', 'ChungcheongnamPopulation',
#     'DaeguPopulation', 'DaejeonPopulation', 'GangwonPopulation', 'GwanguPopulation',
#     'GyeonggiPopulation', 'GyeongsangbukPopulation', 'GyeongsangnamPopulation', 'InchenPopulation',
#     'JejuPopulation', 'JeollanamPopulation', 'JeonbukPopulation', 'SeoulPopulation', 'UlsanPopulation'
# ]

# for population_column in population_columns:
#     plot_time_series(merged_data, population_column)



# 결과: molfrac이 가장 높은 상관관계를 보임. (molfrac - TotalPopulation: p-value = 0.0000e+00)