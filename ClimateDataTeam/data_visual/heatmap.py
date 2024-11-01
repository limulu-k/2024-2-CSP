import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 파일 경로
file_path = "../climate_data/merged_weekly_avg_temp.csv"

# 데이터 로드
data = pd.read_csv(file_path)

# 데이터의 첫 5행 및 열 이름 출력
print(data.head())
print(data.columns)

# 'datetime' 열을 datetime 형식으로 변환
data['datetime'] = pd.to_datetime(data['datetime'])

# 2000년 1월 2일부터 2024년 1월 7일까지 데이터 필터링
filtered_data = data[(data['datetime'] >= '2000-01-02') & (data['datetime'] <= '2024-01-07')]

# 'datetime'을 인덱스로 설정
filtered_data.set_index('datetime', inplace=True)

# 주별 평균 온도 계산
weekly_avg_temp = filtered_data.resample('W')['temp'].mean()

# 주별 평균 온도를 2차원 배열로 변환 (년도와 주도로 인덱스 재구성)
weekly_avg_temp = weekly_avg_temp.reset_index()
weekly_avg_temp['year'] = weekly_avg_temp['datetime'].dt.year
weekly_avg_temp['week'] = weekly_avg_temp['datetime'].dt.isocalendar().week  # ISO 주 번호

# 중복을 처리하기 위해 그룹화하여 평균 계산
weekly_avg_temp = weekly_avg_temp.groupby(['year', 'week'], as_index=False)['temp'].mean()

# 피벗 테이블을 사용하여 데이터 재구성
heatmap_data = weekly_avg_temp.pivot(index='year', columns='week', values='temp')

# 히트맵 시각화
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=False, fmt='.1f', linewidths=.5)
plt.title('Weekly Average Temperature (2000-2024)')
plt.xlabel('Week Number')
plt.ylabel('Year')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

