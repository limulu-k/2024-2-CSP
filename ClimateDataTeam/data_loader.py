import pandas as pd



# ---------------------------------------------
# 1. 특정 속성 선택 및 CSV 파일 생성
# ---------------------------------------------

def select_columns_and_save(start_year, end_year):
    """
    특정 기간의 CSV 파일에서 필요한 열만 선택하여 새로운 CSV 파일로 저장합니다.

    Args:
        start_year (str): 시작 연도 (두 자리 형식).
        end_year (str): 종료 연도 (두 자리 형식).
    """
    # CSV 파일 읽어오기
    df = pd.read_csv(f"./climateData/seoul 20{start_year}-01-01 to 20{end_year}-01-01.csv")

    # 선택할 속성 목록
    selected_columns = ["datetime", "tempmax", "tempmin", "temp", "dew", "humidity",
                        "precip", "windspeed", "sealevelpressure", "moonphase"]
    new_df = df[selected_columns]

    # 새로운 CSV 파일로 저장
    new_df.to_csv(f"./climateData/CD_20{start_year}-20{end_year}.csv", index=False)


# 12개의 기간별 파일에 대해 데이터 선택 및 저장 실행
# for i in range(12):
#     start_year = f"{2 * i:02d}"
#     end_year = f"{2 * (i + 1):02d}"
#     select_columns_and_save(start_year, end_year)


# ---------------------------------------------
# 2. 일주일 단위로 데이터 평균 계산 및 저장
# ---------------------------------------------

def weekly_average_and_save(start_year, end_year):
    """
    특정 기간의 CSV 파일에서 일주일 단위 평균을 계산하여 새로운 CSV 파일로 저장합니다.

    Args:
        start_year (str): 시작 연도 (두 자리 형식).
        end_year (str): 종료 연도 (두 자리 형식).
    """
    # CSV 파일 읽어오기
    df = pd.read_csv(f"./climateData/CD_20{start_year}-20{end_year}.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])  # datetime 형식으로 변환

    # 일주일 단위 평균 계산
    df.set_index("datetime", inplace=True)
    weekly_avg_df = df.resample('W').mean()

    # 새로운 CSV 파일로 저장
    weekly_avg_df.to_csv(f"./climateData/avg_selected_climate/weekly_avg_20{start_year}-20{end_year}.csv",
                         index_label="datetime")


# 12개의 기간별 파일에 대해 일주일 평균 계산 및 저장 실행
# for i in range(12):
#     start_year = f"{2 * i:02d}"
#     end_year = f"{2 * (i + 1):02d}"
#     weekly_average_and_save(start_year, end_year)


def load_and_select_columns(start_year, end_year):
    """
    특정 기간의 CSV 파일에서 'datetime'과 'temp' 열만 선택하여 반환합니다.

    Args:
        start_year (str): 시작 연도 (두 자리 형식).
        end_year (str): 종료 연도 (두 자리 형식).

    Returns:
        DataFrame: 선택한 열만 포함된 데이터프레임.
    """
    # CSV 파일 읽기
    df = pd.read_csv(f"./climateData/avg_selected_climate/weekly_avg_20{start_year}-20{end_year}.csv")

    # datetime 열을 datetime 형식으로 변환
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 필요한 열만 선택하여 반환
    return df[['datetime', 'tempmax', 'tempmin','temp', 'dew', 'humidity',
               'precip', 'windspeed', 'sealevelpressure', 'moonphase']]


def merge_temperature_data():
    """
    12개의 기간별 CSV 파일에서 'datetime'과 'temp' 열만 선택하여 병합하고, 하나의 CSV 파일로 저장합니다.

    Returns:
        DataFrame: 병합된 데이터프레임.
    """
    # 이어붙일 데이터프레임을 저장할 리스트
    df_list = []

    # 12개의 파일을 반복문으로 처리
    for i in range(12):
        start_year = f"{2 * i:02d}"
        end_year = f"{2 * (i + 1):02d}"
        # 각 파일에서 'datetime'과 'temp' 열을 선택하여 리스트에 추가
        df_list.append(load_and_select_columns(start_year, end_year))

    # 데이터프레임을 수평으로 이어붙이기
    merged_df = pd.concat(df_list, ignore_index=True)

    # 병합된 결과를 새로운 CSV 파일로 저장
    merged_df.to_csv("./climateData/merged_weekly_avg_temp.csv", index=False)

    return merged_df


def display_merged_data_preview(merged_df, num_rows=5):
    """
    병합된 데이터프레임의 상위 및 하위 데이터를 출력합니다.

    Args:
        merged_df (DataFrame): 병합된 데이터프레임.
        num_rows (int): 출력할 행의 개수. 기본값은 5.
    """
    print("Head of the merged data:")
    print(merged_df.head(num_rows))
    print("\nTail of the merged data:")
    print(merged_df.tail(num_rows))


# 병합 데이터 생성 및 저장
# merged_df = merge_temperature_data()

# 병합된 데이터 미리보기
# display_merged_data_preview(merged_df)
