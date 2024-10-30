import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기 및 컬럼 공백 제거
temp_data = pd.read_csv("C:/Users/ehddb/2024-2-CSP/ClimateDataTeam/climateData/merged_weekly_avg_temp.csv")
co2_data = pd.read_csv("C:/Users/ehddb/2024-2-CSP/ClimateDataTeam/climateData/weekly_co2.csv")
co2_data.columns = co2_data.columns.str.strip()  # 공백 제거

# 일교차 계산
temp_data['diurnal_range'] = temp_data['tempmax'] - temp_data['tempmin']

# 데이터 병합
merged_data = pd.merge(temp_data[['datetime', 'diurnal_range']], co2_data[['datetime', 'molfrac']], on='datetime', how='inner')
merged_data['datetime'] = pd.to_datetime(merged_data['datetime'])

# 이동 평균 계산
merged_data['diurnal_range_ma'] = merged_data['diurnal_range'].rolling(window=10, center=True).mean()
merged_data['molfrac_ma'] = merged_data['molfrac'].rolling(window=10, center=True).mean()

# NaN 값 제거
merged_data.dropna(subset=['diurnal_range_ma'], inplace=True)

# 그래프 그리기
fig, ax1 = plt.subplots(figsize=(14, 7))

# 일교차 선 그래프
ax1.plot(merged_data['datetime'], merged_data['diurnal_range_ma'], color='b', label='Diurnal Temperature Range (F)', alpha=0.7)
ax1.set_ylabel('Diurnal Temperature Range (°F)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# CO2 농도 축
ax2 = ax1.twinx()
ax2.plot(merged_data['datetime'], merged_data['molfrac_ma'], color='g', label='CO2 Concentration (mol fraction)', alpha=0.7)
ax2.set_ylabel('CO2 Concentration (mol fraction)', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# 추세선 추가
z = np.polyfit(merged_data['datetime'].map(pd.Timestamp.toordinal), merged_data['diurnal_range_ma'], 1)
p = np.poly1d(z)

# 추세선 그래프
ax1.plot(merged_data['datetime'], p(merged_data['datetime'].map(pd.Timestamp.toordinal)), color='red', linestyle='--', linewidth=2, label='Trend Line (Diurnal Range)')

# 제목 및 범례 추가
plt.title('Weekly Diurnal Temperature Range and CO2 Concentration Over Time')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
fig.tight_layout()
plt.show()

