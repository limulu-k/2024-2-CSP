import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
greenhouse_gas_path = '../../NewData/datetime/Weekly_Greenhouse_Gas.csv'
climate_data_path = '../climate_data/merged_weekly_avg_temp.csv'

# 데이터 불러오기
greenhouse_gas_df = pd.read_csv(greenhouse_gas_path)
climate_data_df = pd.read_csv(climate_data_path)

# datetime 컬럼을 datetime 객체로 변환
greenhouse_gas_df['datetime'] = pd.to_datetime(greenhouse_gas_df['datetime'])
climate_data_df['datetime'] = pd.to_datetime(climate_data_df['datetime'])

# 데이터 병합
merged_df = pd.merge(climate_data_df, greenhouse_gas_df, on='datetime', how='inner')

# 기후 데이터에서 필요한 속성만 선택
selected_climate_cols = ['temp', 'tempmax', 'tempmin', 'dew', 'humidity']
climate_cols = [col for col in selected_climate_cols if col in climate_data_df.columns]

# 분석 데이터의 수치형 컬럼 추출
greenhouse_cols = greenhouse_gas_df.select_dtypes(include=[np.number]).columns

# Granger 인과관계 분석 결과 저장용
granger_results = []

for var2 in greenhouse_cols:  # 원인 변수
    for var1 in climate_cols:  # 결과 변수
        try:
            granger_test = grangercausalitytests(merged_df[[var2, var1]], maxlag=5, verbose=False)
            min_p_value = min(granger_test[lag][0]['ssr_ftest'][1] for lag in granger_test.keys())
            granger_results.append((var2, var1, min_p_value))
        except Exception:
            granger_results.append((var2, var1, None))

# 결과를 DataFrame으로 변환
granger_df = pd.DataFrame(granger_results, columns=['Cause', 'Effect', 'p-value'])

# p-value가 없는 경우를 NaN으로 표시
granger_df['p-value'] = granger_df['p-value'].fillna(1)  # NaN을 1로 대체하여 시각화 시 모든 값이 표시되도록

# 시각화
plt.figure(figsize=(14, 8))
heatmap_data = granger_df.pivot_table(values='p-value', index='Cause', columns='Effect')

# 사용자 정의 색상 맵: 0.005 이하 파란색, 그 이상 흰색
cmap = sns.color_palette([ (0, 0, 1),(1, 1, 1)])  # 흰색과 파란색

# 시각화
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap=cmap, vmin=0, vmax=0.005, mask=heatmap_data.isnull(),
            cbar=False, linecolor='black', linewidth=1, yticklabels=True, xticklabels=True)

plt.title('Granger Causality p-value between Greenhouse Gas and Selected Climate Data')
plt.show()
