import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 최종 병합된 데이터 가져오기
data2 = pd.read_csv('../../AI/data_preprocessing/merged_final_data.csv')

# 우선 데이터 속성값 그 자체로 분석을 진행합니다.

# # 상관 관계를 파악하기 위해서 Seaborn을 활용한 시각화
# sns.pairplot(data2, vars=["Polution","Enviroment_Polution","Biodiversity_Loss","Acid_Rain","Water_Pollution","Climate_Crisis",
#                           "Accelerated_Global_Warming","Ozone_Layer_Depletion","Hazardous_Substance_Leakage","Carbon_Dioxide",
#                           "Weekly_News_Count","News_Ratio","tempmax","tempmin","temp","dew","humidity","precip","windspeed",
#                           "sealevelpressure","moonphase"],
#              kind='scatter', dropna=True)

# 위의 결과는, 1개 정도의 결과에 주목해볼 수 있겠다.
# 해수면과 기온의 관계 -> 반비례 관계 : 과학적으로 옳은 관계입니다.

# 피어슨 상관계수를 이용하여 수치적으로 분석해보자.
# 상관계수 계산
date_column = data2.columns[1]
removed_data2 = data2.drop(columns=[date_column])
print(removed_data2.head())
correlation_matrix = removed_data2.corr()

# 상관계수 히트맵
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
# plt.show()

# 결론 : 기압과 온도 사이의 단순한 관계를 파악할 수 있는 정도였다.

# 이상 기후 빈도수에 대한 기준을 마련하여 뉴스 데이터 키워드 수와 비교를 할 수 있도록 하자.

# 1 단계 : 이상 기후의 기준 마련.
# 1) 폭염 및 한파 -> tempmax > 90 || tempmin < 10
# 2) 장마 -> humidity > 85
# 3) 강수량 -> prep > 1
# 4) 해수면기압 < 1005 || windspeed > 20: 폭풍
# 5) 극건조 -> 해수면 기압 > 1025 && humidyty < 50 또는 52
# 6) 열대야 -> tempmax > 90 || humidity > 85

# 이상 기후의 기준
def classify_abnormal_weather(row):
    abnormal_weather_count = 0

    # 속성별 기준
    if row['tempmax'] > 90:
        abnormal_weather_count += 1  # 폭염
    if row['tempmax'] < 10:
        abnormal_weather_count += 1  # 한파
    if row['precip'] > 1:
        abnormal_weather_count += 1  # 폭우
    # 속성 간 조합 기준
    if row['tempmax'] > 80 and row['humidity'] > 85:
        abnormal_weather_count += 1  # 열대야
    if row['sealevelpressure'] < 1005 and row['windspeed'] > 20:
        abnormal_weather_count += 1  # 폭풍
    if row['sealevelpressure'] > 1025 and row['humidity'] < 52:
        abnormal_weather_count += 1  # 가뭄

    return abnormal_weather_count

# 2단계 : 이상 기후의 빈도를 데이터셋에 추가
for index, row in data2.iterrows():
    abnormal_weather_count = classify_abnormal_weather(row)
    data2.loc[index, 'abnormal_weather_count'] = abnormal_weather_count

output_csv_path = '../../AI/data_preprocessing/analyze_abnormal.csv'
data2.to_csv(output_csv_path, index=False)

print("Modified data saved in", output_csv_path)

data3 = pd.read_csv('../../AI/data_preprocessing/analyze_abnormal.csv')
plt.figure(figsize=(12, 6))
plt.plot(data3.index, data3['abnormal_weather_count'], marker='o', linestyle='-', label='Abnormal Weather Count')

# 3단계 : 상관계수 분석을 사용해서 관계의 정도를 측정해보자.
date_column = data3.columns[1]
removed_data3 = data3.drop(columns=[date_column])
print(removed_data3.head())
correlation_matrix = removed_data3.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (after add abnormal weather count)")
# plt.show()

# 결론 : 뉴스 데이터와 이상 기후의 빈도수 사이의 선형적 관계를 발견하기에는 거의 불가능한 부분이다.

# 정규화가 안되어 있어서 문제가 발생했을 수도 있으니, 정규화를 시도합니다.

# 정규화 함수
def normalize_func(row):
    selected_columns = row.index[3:13]  # 선택된 열의 이름 추출
    selected_mean = row[selected_columns].mean()  # 평균 계산
    selected_std = row[selected_columns].std()  # 표준편차 계산

    # 선택된 열에 대해 정규화 수행
    for column in selected_columns:
        row[column] = (row[column] - selected_mean) / selected_std
    return row


# 데이터프레임에 정규화 적용
data3 = data3.apply(normalize_func, axis=1)

# 정규화된 데이터프레임 저장
output_path = '../../AI/data_preprocessing/normalized_analyze_abnormal.csv'
data3.to_csv(output_path, index=False)

print(f"Normalized data saved to {output_path}")

data4 = pd.read_csv('../../AI/data_preprocessing/normalized_analyze_abnormal.csv')
removed_data3 = data3.drop(columns=[date_column])
print(removed_data3.head())
correlation_matrix = removed_data3.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (after apply normalized news data count)")
plt.show()

# 결론 : 정규화를 하고 나서 상관계수를 구해보아도 뉴스 데이터와 기후 데이터 사이의 관계를 찾기란 쉽지 않았다.
# 정규화를 한 이후의 히트맵을 살펴보면 이산화탄소와 오염이라는 단어가 반비례관계이며 환경 오염과 지구 온난화 사이의 관계가
# 상당히 비례한 선형관계를 갖고 있음을 알 수 있었다.

# 지구 온난화와

# 그렇다면 가을의 길이를 분석하여보자.

sample_data = pd.read_csv('../climate_data/merged_weekly_avg_temp.csv')
print(sample_data.head())


data3['datetime'] = pd.to_datetime(data3['datetime'])

# Set datetime as the index
data3.set_index('datetime', inplace=True)

# Monthly and yearly data calculation
monthly_data = data3['abnormal_weather_count'].resample('M').sum()
yearly_data = data3['abnormal_weather_count'].resample('Y').sum()

# Create a figure with two subplots
plt.figure(figsize=(14, 8))

# First subplot: Monthly data (blue bar graph)
plt.subplot(2, 1, 1)
monthly_data.plot(kind='bar', color='skyblue', width=0.8)
plt.title('Monthly Abnormal Weather Count', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Abnormal Weather Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks([])  # Remove x-axis labels for clarity

# Second subplot: Yearly data (red line graph)
plt.subplot(2, 1, 2)
plt.plot(yearly_data.index.year, yearly_data, color='red', marker='o', linestyle='-', label='Yearly Abnormal Weather Count')
plt.title('Yearly Abnormal Weather Count', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Abnormal Weather Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

# Adjust layout and display
plt.tight_layout()
plt.show()
