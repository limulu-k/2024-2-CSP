import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# ---------------------------------------------
# 1. 폭염 빈도수 탐지 코드
# 폭염 탐지 기준: tempmax가 화씨 105도 이상인 경우
# ---------------------------------------------
file_path = '../climate_data/merged_weekly_avg_temp.csv'
df = pd.read_csv(file_path)
# print(df.head())

# 연간 평균 구하기
df["datetime"] = pd.to_datetime(df["datetime"])
yearly_avg_temp = df.groupby(df["datetime"].dt.year)["temp"].mean()

# 추세선 계산
coefficients = np.polyfit(yearly_avg_temp.index, yearly_avg_temp.values, 1)
trendline = np.poly1d(coefficients)

# 연간 평균에 대한 시각화
plt.figure(figsize=(10, 6))
plt.plot(yearly_avg_temp.index, yearly_avg_temp.values, marker='o', linestyle='-', color='b')
plt.plot(yearly_avg_temp.index, trendline(yearly_avg_temp.index), marker='o', linestyle='-', color='r')
plt.title('Average Temperature by Year')
plt.xlabel('Year')
plt.ylabel('Average Temperature')
plt.grid(True)
plt.show()

# 폭염 빈도수 분석
df["datetime"] = pd.to_datetime(df["datetime"])
heatwave_count = df[df["tempmax"] > 91.4].groupby(df["datetime"].dt.year).size()
# for year, count in heatwave_count.items():
#     print(f"{year}년: {count}건의 폭염 발생")

# 유의미한 결과를 도출하지 못한다..

# datetime 열을 datetime 형식으로 변환
df["datetime"] = pd.to_datetime(df["datetime"])
# tempmax와 tempmin의 차이가 10보다 큰 경우 필터링
large_temp_diff_count = df[(df["tempmax"] - df["tempmin"]) > 20].groupby(df["datetime"].dt.year).size()
# 결과 출력
for year, count in large_temp_diff_count.items():
    print(f"{year}년: {count}건의 tempmax와 tempmin 차이가 10보다 큰 날")




# ---------------------------------------------
# 2. 한파 빈도수 탐지 코드
# 한파 탐지 기준: tempmin이 전체 평균보다 5도 이상 낮은 경우
# ---------------------------------------------


# ---------------------------------------------
# 3. 홍수 빈도수 탐지 코드
# 홍수 탐지 기준: precip의 주 강수량이 전체 평균의 3배 이상인 경우
# ---------------------------------------------


