import pandas as pd
import matplotlib.pyplot as plt
# ---------------------------------------------
# 1. 주 평균 온도 데이터 시각화
# ---------------------------------------------


def plot_temperature_trend(file_path, title="Weekly Average Temperature Trend"):
    """
    CSV 파일을 읽어와 주 평균 온도 추세를 시각화합니다.

    Args:
        file_path (str): 시각화할 CSV 파일 경로.
        title (str): 그래프 제목.
    """
    # CSV 파일 읽어오기
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])  # datetime 형식으로 변환

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(df['datetime'], df['temp'], marker='o', linestyle='-', color='b', markersize=3)
    plt.title(title)
    plt.xlabel("Date")  # x축 레이블은 설정하지만 숨깁니다.
    plt.ylabel("Temperature (°C)")

    # x축 레이블 숨기기
    plt.xticks([])

    # 그래프 보여주기
    plt.show()


# 주 평균 온도 데이터 시각화 실행
plot_temperature_trend("./climateData/merged_weekly_avg_temp.csv",
                       "Weekly Average Temperature (2000-2002)")



