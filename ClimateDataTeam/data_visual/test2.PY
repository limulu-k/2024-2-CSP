# import pandas as pd
# import matplotlib.pyplot as plt

# # 데이터 파일 경로
# file_path = '../climate_data/merged_weekly_avg_temp.csv'
# df = pd.read_csv(file_path)

# # datetime 열을 datetime 형식으로 변환
# df['datetime'] = pd.to_datetime(df['datetime'])

# # 주별 데이터로 그룹화
# weekly_data = df.set_index('datetime').resample('W').mean()  # 평균 온도로 그룹화

# # 평균 온도를 섭씨로 변환
# #weekly_data['temp_celsius'] = (weekly_data['temp'] - 32) * 5 / 9
# weekly_data['temp_celsius'] = weekly_data['temp']

# # 월별로 서브플롯 생성
# months = weekly_data.index.month.unique()
# fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))
# axes = axes.flatten()  # 2D 배열을 1D로 변환

# # 각 월에 대한 그래프 생성
# for month in range(1, 13):  # 1월부터 12월까지 반복
#     month_data = weekly_data[weekly_data.index.month == month]
#     axes[month-1].plot(month_data.index, month_data['temp_celsius'], label=f'Month {month}', color='blue')
#     axes[month-1].set_title(f'Month {month}')
#     axes[month-1].set_xlabel('Date')
#     axes[month-1].set_ylabel('Temperature (°C)')
#     axes[month-1].grid()
#     axes[month-1].legend()

# # 빈 서브플롯 제거 (있는 경우)
# for i in range(len(months), len(axes)):
#     fig.delaxes(axes[i])

# plt.tight_layout()
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import linregress

# # 데이터 파일 경로
# file_path = '../climate_data/merged_weekly_avg_temp.csv'
# df = pd.read_csv(file_path)

# # datetime 열을 datetime 형식으로 변환
# df['datetime'] = pd.to_datetime(df['datetime'])

# # 주별 데이터로 그룹화
# weekly_data = df.set_index('datetime').resample('W').mean()  # 평균 온도로 그룹화

# # 평균 온도를 섭씨로 변환
# weekly_data['temp_celsius'] = (weekly_data['tempmin'] - 32) * 5 / 9

# # 월별로 서브플롯 생성
# months = weekly_data.index.month.unique()
# fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))
# axes = axes.flatten()  # 2D 배열을 1D로 변환

# # 각 월에 대한 그래프 생성 및 회귀선 추가
# for month in range(1, 13):  # 1월부터 12월까지 반복
#     month_data = weekly_data[weekly_data.index.month == month]
    
#     # 날짜를 일수로 변환하여 회귀 계산
#     x = (month_data.index - month_data.index[0]).days  # 기준 날짜로부터의 일수
#     y = month_data['temp_celsius']
    
#     # 회귀선 계산
#     slope, intercept, r_value, p_value, std_err = linregress(x, y)
#     trend_line = intercept + slope * x
    
#     # 온도 데이터와 회귀선 그리기
#     axes[month-1].plot(month_data.index, month_data['temp_celsius'], label=f'Month {month}', color='blue')
#     axes[month-1].plot(month_data.index, trend_line, color='red', linestyle='--', label='Trend Line')
    
#     # 플롯 꾸미기
#     axes[month-1].set_title(f'Month {month}')
#     axes[month-1].set_xlabel('Date')
#     axes[month-1].set_ylabel('Temperature (°C)')
#     axes[month-1].grid()
#     axes[month-1].legend()

# # 빈 서브플롯 제거 (있는 경우)
# for i in range(len(months), len(axes)):
#     fig.delaxes(axes[i])

# plt.tight_layout()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# # 데이터 파일 경로
# file_path = "../climate_data/merged_weekly_avg_temp.csv"
# df = pd.read_csv(file_path)

# # datetime 열을 datetime 형식으로 변환
# df['datetime'] = pd.to_datetime(df['datetime'])

# # 2000년 첫 주부터 현재까지 주별 강수량 합산
# df_2000 = df[df['datetime'] >= '2000-01-01']  # 2000년 1월 1일부터 필터링
# weekly_precip = df_2000.set_index('datetime').resample('W').sum()['dew']  # 주별 강수량 합산

# # 그래프 생성
# plt.figure(figsize=(12, 6))
# plt.plot(weekly_precip.index, weekly_precip.values, label='Weekly Precipitation', color='blue')
# plt.title('Weekly Precipitation from 2000 to Present')
# plt.xlabel('Date')
# plt.ylabel('Precipitation (mm)')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.dates import date2num, DateFormatter

# file_path = "../climate_data/merged_weekly_avg_temp.csv"
# df = pd.read_csv(file_path)

# df['datetime'] = pd.to_datetime(df['datetime'])

# weekly_data = df.set_index('datetime').resample('W').mean()

# weekly_data['temp_celsius'] = (weekly_data['temp'] - 32) * 5 / 9
# weekly_data['temp_celsius'] = weekly_data['temp']

# temp_diff = []

# for date in weekly_data.index:
#     next_year_date = date + pd.DateOffset(years=10)
#     one_week_range = weekly_data[(weekly_data.index >= next_year_date) & 
#                                  (weekly_data.index < next_year_date + pd.DateOffset(days=7))]
#     if not one_week_range.empty:
#         current_temp = weekly_data.loc[date, 'temp_celsius']
#         next_year_temp = one_week_range.iloc[0]['temp_celsius']
#         temp_diff.append({
#             'date': date,
#             'temp_diff': next_year_temp - current_temp
#         })

# temp_diff_df = pd.DataFrame(temp_diff)

# temp_diff_df['date_numeric'] = date2num(temp_diff_df['date'])

# sns.lmplot(x='date_numeric', y='temp_diff', data=temp_diff_df, aspect=2, height=7, line_kws={'color': 'red'})
# plt.title('Year-over-Year Weekly Temperature Difference with Regression Line (2000-2024)')
# plt.xlabel('Date')
# plt.ylabel('Temperature Difference (°C)')
# plt.axhline(0, color='gray', linestyle='--')
# plt.grid()
# plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
# plt.xticks(rotation=45)
# plt.show()



# import pandas as pd

# # 데이터 로드
# data = pd.read_csv('../climate_data/merged_weekly_avg_temp.csv')

# # 화씨에서 섭씨로 변환
# def fahrenheit_to_celsius(fahrenheit):
#     return (fahrenheit - 32) * 5 / 9

# # 평균 온도(temp) 열을 섭씨로 변환
# data['temp_celsius'] = data['tempmin'].apply(fahrenheit_to_celsius)

# # 'datetime' 열을 datetime 타입으로 변환
# data['datetime'] = pd.to_datetime(data['datetime'])

# # 'year' 열을 추가하여 연도별로 그룹화
# data['year'] = data['datetime'].dt.year

# # 평균 온도가 -10도 이하인 날짜만 필터링
# cold_days = data[data['temp_celsius'] <= -10]

# # 연도별로 -10도 이하인 날의 개수를 세기
# cold_days_per_year = cold_days.groupby('year').size()

# # 결과 출력
# print(cold_days_per_year)


# import pandas as pd
# import matplotlib.pyplot as plt

# # 데이터 로드
# data = pd.read_csv('../climate_data/merged_weekly_avg_temp.csv')

# # 화씨에서 섭씨로 변환
# def fahrenheit_to_celsius(fahrenheit):
#     return (fahrenheit - 32) * 5 / 9

# # 평균 온도(temp) 열을 섭씨로 변환
# data['temp_celsius'] = data['tempmax'].apply(fahrenheit_to_celsius)

# # 'datetime' 열을 datetime 타입으로 변환
# data['datetime'] = pd.to_datetime(data['datetime'])

# # 'year' 열을 추가하여 연도별로 그룹화
# data['year'] = data['datetime'].dt.year

# # 섭씨 값이 -5도 이하인 행만 필터링
# cold_days = data[data['temp_celsius'] >= 28]

# # 연도별로 -5도 이하인 날의 개수 세기
# cold_days_per_year = cold_days.groupby('year').size()

# # 막대그래프 시각화
# plt.figure(figsize=(10, 6))
# cold_days_per_year.plot(kind='bar', color='skyblue')

# # 그래프 제목 및 레이블 설정
# plt.title('Number of Days with Temperature <= -5°C per Year', fontsize=16)
# plt.xlabel('Year', fontsize=12)
# plt.ylabel('Number of Days', fontsize=12)

# # x축 레이블 회전
# plt.xticks(rotation=45)

# # 그래프 출력
# plt.tight_layout()
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# # 데이터 로드
# data = pd.read_csv('../climate_data/merged_weekly_avg_temp.csv')

# # 화씨에서 섭씨로 변환
# def fahrenheit_to_celsius(fahrenheit):
#     return (fahrenheit - 32) * 5 / 9

# # 평균 온도(tempmax)와 최저 온도(tempmin) 열을 섭씨로 변환
# data['tempmax_celsius'] = data['tempmax'].apply(fahrenheit_to_celsius)
# data['tempmin_celsius'] = data['tempmin'].apply(fahrenheit_to_celsius)

# # 'datetime' 열을 datetime 타입으로 변환
# data['datetime'] = pd.to_datetime(data['datetime'])

# # 'year' 열을 추가하여 연도별로 그룹화
# data['year'] = data['datetime'].dt.year

# # tempmax가 28도 이상이거나 tempmin이 -8도 이하인 경우 필터링
# extreme_days = data[(data['tempmax_celsius'] >= 30) | (data['tempmin_celsius'] <=-10)]

# # 연도별로 조건을 만족하는 날의 개수 세기
# extreme_days_per_year = extreme_days.groupby('year').size()

# # 막대그래프 시각화
# plt.figure(figsize=(10, 6))
# extreme_days_per_year.plot(kind='bar', color='skyblue')

# # 그래프 제목 및 레이블 설정
# plt.title('Number of Extreme Days per Year (Tempmax > 28°C or Tempmin < -11°C)', fontsize=16)
# plt.xlabel('Year', fontsize=12)
# plt.ylabel('Number of Extreme Days', fontsize=12)

# # x축 레이블 회전
# plt.xticks(rotation=45)

# # 그래프 출력
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# # 데이터 파일 경로
# file_path = '../../AI/data_preprocessing/merged_final_data.csv'
# data1 = pd.read_csv(file_path)

# # 'datetime' 열을 datetime 형식으로 변환
# data1['datetime'] = pd.to_datetime(data1['datetime'])

# # 날짜별 News_Ratio 시각화
# plt.figure(figsize=(12, 6))
# plt.plot(data1['datetime'], data1['News_Ratio'], color='blue', label='News Ratio')

# # 제목과 레이블 추가
# plt.title('News Ratio Over Time')
# plt.xlabel('Date')
# plt.ylabel('News Ratio')

# # 그래프 세부 조정
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()
# plt.legend()

# # 그래프 표시
# plt.show()


# import pandas as pd
# from scipy.stats import pearsonr

# # 두 번째 데이터셋 로드
# file_path2 = '../../AI/data_preprocessing/merged_final_data.csv'
# df2 = pd.read_csv(file_path2)

# # datetime 열을 datetime 형식으로 변환
# df2['datetime'] = pd.to_datetime(df2['datetime'])

# # tempmax와 tempmin의 차이 계산 (온도 범위)
# df2['temp_range'] = df2['tempmax'] - df2['tempmin']

# # 온도 범위와 강수량을 결합한 새로운 변수 'combined_metric' 생성
# df2['combined_metric'] = df2['temp_range'] * df2['precip']  # 두 변수를 곱하는 방식

# # 월별로 그룹화
# monthly_groups = df2.groupby(df2['datetime'].dt.month)

# # 각 월별로 상관 분석 진행
# for month, group in monthly_groups:
#     corr_combined_metric_news, _ = pearsonr(group['precip'], group['News_Ratio'])
#     print(f"Month {month} - 온도 범위와 강수량의 결합 변수와 뉴스 비율 간 상관 계수: {corr_combined_metric_news}")


# import pandas as pd
# from scipy.stats import pearsonr

# # 두 번째 데이터셋 로드
# file_path2 = '../../AI/data_preprocessing/merged_final_data.csv'
# df2 = pd.read_csv(file_path2)

# # datetime 열을 datetime 형식으로 변환
# df2['datetime'] = pd.to_datetime(df2['datetime'])
# df2['temp_range'] = df2['tempmax'] - df2['tempmin']

# # 월별로 그룹화
# monthly_groups = df2.groupby(df2['datetime'].dt.month)


# # 각 월별로 상관 분석 진행
# for month, group in monthly_groups:
#     if not group['precip'].isnull().all() and not group['News_Ratio'].isnull().all():  # 결측값 검증
#         corr_precip_news, _ = pearsonr(group['precip'], group['News_Ratio'])
#         print(f"Month {month} - 강수량과 뉴스 비율 간 상관 계수: {corr_precip_news:.4f}")
#     else:
#         print(f"Month {month} - 강수량 또는 뉴스 비율 데이터에 결측값만 존재하여 상관 분석 불가")

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr

# # 두 번째 데이터셋 로드
# file_path2 = '../../AI/data_preprocessing/merged_final_data.csv'
# df2 = pd.read_csv(file_path2)

# # datetime 열을 datetime 형식으로 변환
# df2['datetime'] = pd.to_datetime(df2['datetime'])
# df2['temp_range'] = df2['tempmax'] - df2['tempmin']

# # 월별로 그룹화
# monthly_groups = df2.groupby(df2['datetime'].dt.month)

# # 상관 계수를 저장할 리스트
# corr_results = []

# # 각 월별로 상관 분석 진행
# for month, group in monthly_groups:
#     if not group['precip'].isnull().all() and not group['News_Ratio'].isnull().all():  # 결측값 검증
#         corr_precip_news, _ = pearsonr(group['precip'], group['News_Ratio'])
#         corr_results.append((month, corr_precip_news))
#     else:
#         corr_results.append((month, None))  # 상관 분석 불가한 경우

# # 결과를 데이터프레임으로 변환
# corr_df = pd.DataFrame(corr_results, columns=['Month', 'Correlation'])

# # 시각화
# plt.figure(figsize=(14, 7))
# sns.barplot(x='Month', y='Correlation', data=corr_df, palette='coolwarm')

# # 제목과 레이블 설정
# plt.title('Month-wise Correlation between Precipitation and News Ratio', fontsize=16)
# plt.xlabel('Month', fontsize=14)
# plt.ylabel('Correlation Coefficient', fontsize=14)
# plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# # 결측값을 표시하기 위한 설정
# for index, row in corr_df.iterrows():
#     if pd.isnull(row['Correlation']):
#         plt.text(row['Month'], 0, 'No Data', ha='center', va='bottom', color='red')

# plt.show()



# import pandas as pd
# from scipy.stats import pearsonr

# # 두 번째 데이터셋 로드
# file_path2 = '../../AI/data_preprocessing/merged_final_data.csv'
# df2 = pd.read_csv(file_path2)

# # datetime 열을 datetime 형식으로 변환
# df2['datetime'] = pd.to_datetime(df2['datetime'])

# # 계절 정보 추가
# def get_season(month):
#     if month in [3, 4, 5]:
#         return 'Spring'  # 봄
#     elif month in [6, 7, 8]:
#         return 'Summer'  # 여름
#     elif month in [9, 10, 11]:
#         return 'Autumn'  # 가을
#     else:
#         return 'Winter'  # 겨울

# df2['season'] = df2['datetime'].dt.month.apply(get_season)
# df2['temp_range'] = df2['tempmax'] - df2['tempmin']

# # 계절별로 그룹화
# seasonal_groups = df2.groupby('season')

# # 각 계절별로 상관 분석 진행
# for season, group in seasonal_groups:
#     if not group['precip'].isnull().all() and not group['News_Ratio'].isnull().all():  # 결측값 검증
#         corr_precip_news, _ = pearsonr(group['precip'], group['News_Ratio'])
#         print(f"Season {season} - 강수량과 뉴스 비율 간 상관 계수: {corr_precip_news:.4f}")
#     else:
#         print(f"Season {season} - 강수량 또는 뉴스 비율 데이터에 결측값만 존재하여 상관 분석 불가")


# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from scipy.stats import pearsonr

# # 데이터 로드 및 준비
# file_path2 = '../../AI/data_preprocessing/merged_final_data.csv'
# df2 = pd.read_csv(file_path2)
# df2['datetime'] = pd.to_datetime(df2['datetime'])

# # 계절별로 그룹화
# seasons = {
#     'Spring': (3, 4, 5),
#     'Summer': (6, 7, 8),
#     'Autumn': (9, 10, 11),
#     'Winter': (12, 1, 2)
# }

# # 시각화를 위한 색상과 레이블 설정
# color_map = {
#     'Strong Positive': 'red',
#     'Moderate Positive': 'orange',
#     'Weak Positive': 'yellow',
#     'No Correlation': 'gray'
# }

# # 시각화 준비
# plt.figure(figsize=(10, 6))
# handles = []

# # 각 계절별로 상관계수 계산 및 시각화
# for season, months in seasons.items():
#     season_data = df2[df2['datetime'].dt.month.isin(months)]
#     corr_precip_news, _ = pearsonr(season_data['precip'], season_data['News_Ratio'])
    
#     if not (season_data['precip'].isnull().all() or season_data['News_Ratio'].isnull().all()):
#         if corr_precip_news > 0.5:
#             color = color_map['Strong Positive']
#             label = 'Strong Positive'
#         elif corr_precip_news > 0.3:
#             color = color_map['Moderate Positive']
#             label = 'Moderate Positive'
#         elif corr_precip_news > 0:
#             color = color_map['Weak Positive']
#             label = 'Weak Positive'
#         else:
#             color = color_map['No Correlation']
#             label = 'No Correlation'
        
#         plt.bar(season, corr_precip_news, color=color, label=label if season not in [h.get_label() for h in handles] else "")
    
# # 범례 추가
# handles = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
# plt.legend(handles=handles, title='Correlation Strength')
# plt.title('Seasonal Correlation Between Precipitation and News Ratio')
# plt.ylabel('Correlation Coefficient')
# plt.ylim(0, 1)  # y축 고정
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

# 데이터 로드 및 준비
file_path2 = '../../AI/data_preprocessing/merged_final_data.csv'
df2 = pd.read_csv(file_path2)
df2['datetime'] = pd.to_datetime(df2['datetime'])

# 월별로 그룹화
monthly_groups = df2.groupby(df2['datetime'].dt.month)

# 시각화를 위한 색상과 레이블 설정
color_map = {
    'Strong Positive': 'red',
    'Moderate Positive': 'orange',
    'Weak Positive': 'yellow',
    'No Correlation': 'gray'
}

# 시각화 준비
plt.figure(figsize=(14, 7))
handles = []

# 각 월별로 상관계수 계산 및 시각화
for month, group in monthly_groups:
    if not (group['precip'].isnull().all() or group['News_Ratio'].isnull().all()):
        corr_precip_news, _ = pearsonr(group['precip'], group['News_Ratio'])
        
        if corr_precip_news > 0.5:
            color = color_map['Strong Positive']
            label = 'Strong Positive'
        elif corr_precip_news > 0.3:
            color = color_map['Moderate Positive']
            label = 'Moderate Positive'
        elif corr_precip_news > 0:
            color = color_map['Weak Positive']
            label = 'Weak Positive'
        else:
            color = color_map['No Correlation']
            label = 'No Correlation'
        
        plt.bar(str(month), corr_precip_news, color=color, label=label if month not in [h.get_label() for h in handles] else "")
    
# 범례 추가
handles = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
plt.legend(handles=handles, title='Correlation Strength')
plt.title('Monthly Correlation Between Precipitation and News Ratio')
plt.ylabel('Correlation Coefficient')
plt.ylim(0, 1)  # y축 고정
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.show()
