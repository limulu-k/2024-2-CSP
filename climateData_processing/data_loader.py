import pandas as pd

# 데이터 속성값 선택 코드 #
# 원하는 속성을 selected_columns 에 넣고 실행하면 그에 상응하는 csv파일 생성.
# for i in range(12):
#     start_year = f"{2 * i:02d}"
#     end_year = f"{2 * (i + 1):02d}"
#
#     # CSV 파일 읽어오기
#     df = pd.read_csv(f"climateData/seoul 20{start_year}-01-01 to 20{end_year}-01-01.csv")
#
#
#     # 속성값 중 일부 선택 -> datetime, tempmax, tempmin, temp, dew, humidity, precip,
#     # windspeed, sealevelpressure, moonphase 정도를 수집한다. (총 10개의 속성)
#     selected_columns = ["datetime", "tempmax", "tempmin", "temp", "dew", "humidity", "precip", "windspeed", "sealevelpressure", "moonphase"]
#     new_df = df[selected_columns]
#
#     new_df.to_csv(f"CD_20{start_year}-20{end_year}.csv", index=False)
# 데이터 속성값 선택 코드 끝 #


