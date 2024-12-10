import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

# Load the datasets
air_pollutants_path = '../../NewData/datetime/Weekly_Air_Pollutants.csv'
climate_data_path = '../climate_data/merged_weekly_avg_temp.csv'

# 데이터 불러오기
air_pollutants_df = pd.read_csv(air_pollutants_path)
climate_data_df = pd.read_csv(climate_data_path)

# datetime 컬럼을 datetime 객체로 변환
air_pollutants_df['datetime'] = pd.to_datetime(air_pollutants_df['datetime'])
climate_data_df['datetime'] = pd.to_datetime(climate_data_df['datetime'])

# 데이터 병합
merged_df = pd.merge(air_pollutants_df, climate_data_df, on='datetime', how='inner')

# 제외할 컬럼 지정
exclude_cols = ['sealevelpressure', 'windspeed']

# 수치형 변수만 선택 (제외 컬럼 필터링)
air_pollutants_cols = air_pollutants_df.select_dtypes(include=[np.number]).columns
climate_data_cols = [col for col in climate_data_df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

# Granger 인과관계 분석 결과 저장
granger_results = []

# 기후 데이터 ↔ 오염 데이터 쌍만 분석
for var1 in air_pollutants_cols:
    for var2 in climate_data_cols:
        try:
            granger_test = grangercausalitytests(merged_df[[var1, var2]], maxlag=5, verbose=False)
            min_p_value = min(granger_test[lag][0]['ssr_ftest'][1] for lag in granger_test.keys())
            granger_results.append((var1, var2, min_p_value))
        except Exception as e:
            granger_results.append((var1, var2, None))  # 실패 시 None

# Granger causality results: filter and sort
granger_sorted = sorted(granger_results, key=lambda x: x[2] if x[2] is not None else float('inf'))
filtered_granger = [(var1, var2, p_value) for var1, var2, p_value in granger_sorted if p_value is not None and p_value < 0.02]

# Prepare data for plotting
plot_data = pd.DataFrame(filtered_granger, columns=['Variable 1', 'Variable 2', 'p-value'])
plot_data = plot_data.sort_values('p-value', ascending=True).reset_index(drop=True)

# Highlight significant results (p-value < 0.005)
plot_data['Significant'] = plot_data['p-value'] < 0.005

# Plot
plt.figure(figsize=(14, 7))
colors = plot_data['Significant'].map({True: 'red', False: 'blue'})  # Red for significant, blue otherwise
bars = plt.bar(
    x=range(len(plot_data)),
    height=plot_data['p-value'],
    color=colors
)

# Add horizontal line for threshold
plt.axhline(y=0.005, color='green', linestyle='--', label='Significance Threshold (p=0.005)')

# Add the p-value at the top of each bar
for i, row in plot_data.iterrows():
    plt.text(i, row['p-value'], f"{row['p-value']:.4f}", ha='center', va='bottom', fontsize=9)

# Set custom labels for each bar with adjusted font size and rotation
plt.xticks(
    range(len(plot_data)),
    labels=[f"{row['Variable 1']}\n→\n{row['Variable 2']}" for _, row in plot_data.iterrows()],
    rotation=45,  # Adjusted rotation for better readability
    fontsize=8  # Smaller font size
)

plt.ylabel('Granger Causality p-value')
plt.title('Granger Causality Analysis (Sorted by p-value)')

# Add legend
plt.legend()

# Display
plt.tight_layout()
plt.show()
