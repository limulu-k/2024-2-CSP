# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics

# # 데이터 파일 경로
# file_path = "../../AI/data_preprocessing/merged_final_data.csv"
# data1 = pd.read_csv(file_path)
# data1['Label'] = ((data1['tempmax'] > (30) * 9/5  + 32) | (data1['tempmin'] < (-10) * 9/5  + 32)).astype(int) #30도 이상 -도 이하
# data1.to_csv(file_path, index=False)
# x_train, x_test, y_train, y_test = train_test_split(data1[['Polution','Enviroment_Polution','Biodiversity_Loss',
#                                                            'Acid_Rain','Water_Pollution','Climate_Crisis','Accelerated_Global_Warming',
#                                                            'Ozone_Layer_Depletion','Hazardous_Substance_Leakage',
#                                                            'Carbon_Dioxide']], data1['Label'],test_size=0.2,
#                                                             shuffle=True,stratify=data1['Label'])
# print("The dimension of the trainset is ", x_train.shape)
# print("The dimension of the testset is ", x_test.shape)
# scaler = StandardScaler()
# xtrain_scaled = scaler.fit_transform(x_train)
# xtest_scaled = scaler.transform(x_test)
# model1 = LogisticRegression(tol = 1e-06).fit(xtrain_scaled,y_train)
# print("___________________________________________________")
# print("regression coefficients are ", model1.coef_)
# print("intercept is ", model1.intercept_)
# print("model score is ",model1.score(xtrain_scaled,y_train))
# print("___________________________________________________")

# model1.predict_proba(xtest_scaled)
# y_pred = model1.predict(xtest_scaled)
# print("predicted labels are", y_pred)
# confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
# print(confusion_matrix)
# print(metrics.classification_report(y_test, y_pred, target_names= ['class1','class2']))


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics

# # 데이터 파일 경로
# file_path = "../../AI/data_preprocessing/merged_final_data.csv"
# data1 = pd.read_csv(file_path)

# # 'datetime' 열을 datetime 형식으로 변환
# data1['datetime'] = pd.to_datetime(data1['datetime'])

# # 주간 종료일 계산
# data1['end_date'] = data1['datetime'] + pd.Timedelta(days=6)

# # 1년 전 날짜 추가
# data1['previous_date'] = data1['datetime'] - pd.DateOffset(years=1)

# # 화씨를 섭씨로 변환
# data1['temp_celsius'] = (data1['temp'] - 32) * 5.0 / 9.0
# data1['tempmax_celsius'] = (data1['tempmax'] - 32) * 5.0 / 9.0
# data1['tempmin_celsius'] = (data1['tempmin'] - 32) * 5.0 / 9.0

# # 1년 전 날짜가 포함된 주간 데이터를 매칭
# matched_data = []
# for _, row in data1.iterrows():
#     previous_date = row['previous_date']
#     match = data1[(data1['datetime'] <= previous_date) & (data1['end_date'] >= previous_date)]
#     if not match.empty:
#         for _, prev_row in match.iterrows():
#             matched_data.append({
#                 'current_date': row['datetime'],
#                 'current_tempmax_celsius': row['tempmax_celsius'],
#                 'current_tempmin_celsius': row['tempmin_celsius'],
#                 'current_temp_celsius': row['temp_celsius'],
#                 'previous_tempmax_celsius': prev_row['tempmax_celsius'],
#                 'previous_tempmin_celsius': prev_row['tempmin_celsius'],
#                 'previous_temp_celsius': prev_row['temp_celsius'],
#                 'Polution': row['Polution'],
#                 'Enviroment_Polution': row['Enviroment_Polution'],
#                 'Biodiversity_Loss': row['Biodiversity_Loss'],
#                 'Acid_Rain': row['Acid_Rain'],
#                 'Water_Pollution': row['Water_Pollution'],
#                 'Climate_Crisis': row['Climate_Crisis'],
#                 'Accelerated_Global_Warming': row['Accelerated_Global_Warming'],
#                 'Ozone_Layer_Depletion': row['Ozone_Layer_Depletion'],
#                 'Hazardous_Substance_Leakage': row['Hazardous_Substance_Leakage'],
#                 'Carbon_Dioxide': row['Carbon_Dioxide'],
#                 'News_Ratio' : row['News_Ratio'],
#                 'Label': int(abs(row['temp_celsius'] - prev_row['temp_celsius']) > 10)  # 섭씨 기준 절댓값 10 초과 여부
#             })

# # 매칭된 데이터를 DataFrame으로 변환
# matched_df = pd.DataFrame(matched_data)

# label_counts = matched_df['Label'].value_counts()
# print("Label counts:\n", label_counts)

# # 결과 확인
# print("Matched data shape:", matched_df.shape)
# print(matched_df.head())

# # 모델링에 사용할 열과 타겟 설정
# x = matched_df[['Polution', 'Enviroment_Polution', 'Biodiversity_Loss', 
#                 'Acid_Rain', 'Water_Pollution', 'Climate_Crisis', 
#                 'Accelerated_Global_Warming', 'Ozone_Layer_Depletion', 
#                 'Hazardous_Substance_Leakage', 'Carbon_Dioxide','News_Ratio']]
# #x = matched_df[['News_Ratio']]
# y = matched_df['Label']

# # 데이터셋 분할
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)

# # 데이터 정규화
# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)

# # 로지스틱 회귀 모델 학습
# model = LogisticRegression(tol=1e-06)
# model.fit(x_train_scaled, y_train)

# # 학습 결과 출력
# print("Model coefficients:", model.coef_)
# print("Intercept:", model.intercept_)
# print("Training accuracy:", model.score(x_train_scaled, y_train))

# # 테스트 예측 및 평가
# y_pred = model.predict(x_test_scaled)
# conf_matrix = metrics.confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", conf_matrix)
# print(metrics.classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics

# # 데이터 파일 경로
# file_path = "../../AI/data_preprocessing/merged_final_data.csv"
# data1 = pd.read_csv(file_path)

# # 'datetime' 열을 datetime 형식으로 변환
# data1['datetime'] = pd.to_datetime(data1['datetime'])

# # 주간 종료일 계산
# data1['end_date'] = data1['datetime'] + pd.Timedelta(days=6)

# # 1년 전 날짜 추가
# data1['previous_date'] = data1['datetime'] - pd.DateOffset(years=1)

# # 화씨를 섭씨로 변환
# data1['temp_celsius'] = (data1['temp'] - 32) * 5.0 / 9.0
# data1['tempmax_celsius'] = (data1['tempmax'] - 32) * 5.0 / 9.0
# data1['tempmin_celsius'] = (data1['tempmin'] - 32) * 5.0 / 9.0

# # 매칭된 데이터를 생성할 때 'Label'을 tempmax - tempmin의 폭으로 설정
# matched_data = []
# for _, row in data1.iterrows():
#     previous_date = row['previous_date']
#     match = data1[(data1['datetime'] <= previous_date) & (data1['end_date'] >= previous_date)]
#     if not match.empty:
#         for _, prev_row in match.iterrows():
#             # 최고 온도와 최저 온도의 차이 계산
#             temp_range = row['tempmax_celsius'] - row['tempmin_celsius']
            
#             # Label 정의: 온도 차이가 특정 임계값 이상일 경우 1, 그렇지 않으면 0
#             label = int(temp_range > 10)  # 예: 차이가 10°C 이상일 때 1로 설정
            
#             matched_data.append({
#                 'current_date': row['datetime'],
#                 'current_temp_celsius': row['temp_celsius'],
#                 'previous_temp_celsius': prev_row['temp_celsius'],
#                 'Temp_Max': row['tempmax_celsius'],
#                 'Temp_Min': row['tempmin_celsius'],
#                 'Temp_Range': temp_range,  # 최고 온도와 최저 온도의 차이
#                 'Polution': row['Polution'],
#                 'Enviroment_Polution': row['Enviroment_Polution'],
#                 'Biodiversity_Loss': row['Biodiversity_Loss'],
#                 'Acid_Rain': row['Acid_Rain'],
#                 'Water_Pollution': row['Water_Pollution'],
#                 'Climate_Crisis': row['Climate_Crisis'],
#                 'Accelerated_Global_Warming': row['Accelerated_Global_Warming'],
#                 'Ozone_Layer_Depletion': row['Ozone_Layer_Depletion'],
#                 'Hazardous_Substance_Leakage': row['Hazardous_Substance_Leakage'],
#                 'Carbon_Dioxide': row['Carbon_Dioxide'],
#                 'News_Ratio': row['News_Ratio'],
#                 'Label': label
#             })

# # 매칭된 데이터를 DataFrame으로 변환
# matched_df = pd.DataFrame(matched_data)

# # Label 분포 확인
# label_counts = matched_df['Label'].value_counts()
# print("Label counts:\n", label_counts)

# # 결과 확인
# print("Matched data shape:", matched_df.shape)
# print(matched_df.head())

# # 모델링에 사용할 열과 타겟 설정
# x = matched_df[['Polution', 'Enviroment_Polution', 'Biodiversity_Loss', 
#                 'Acid_Rain', 'Water_Pollution', 'Climate_Crisis', 
#                 'Accelerated_Global_Warming', 'Ozone_Layer_Depletion', 
#                 'Hazardous_Substance_Leakage', 'Carbon_Dioxide', 'News_Ratio']]
# y = matched_df['Label']

# # 데이터셋 분할
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)

# # 데이터 정규화
# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)

# # 로지스틱 회귀 모델 학습
# model = LogisticRegression(tol=1e-06)
# model.fit(x_train_scaled, y_train)

# # 학습 결과 출력
# print("Model coefficients:", model.coef_)
# print("Intercept:", model.intercept_)
# print("Training accuracy:", model.score(x_train_scaled, y_train))

# # 테스트 예측 및 평가
# y_pred = model.predict(x_test_scaled)
# conf_matrix = metrics.confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", conf_matrix)
# print(metrics.classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# 데이터 파일 경로
file_path = "../../AI/data_preprocessing/merged_final_data.csv"
data1 = pd.read_csv(file_path)

# 'datetime' 열을 datetime 형식으로 변환
data1['datetime'] = pd.to_datetime(data1['datetime'])

# 주간 종료일 계산
data1['end_date'] = data1['datetime'] + pd.Timedelta(days=6)

# 1년 전 날짜 추가
data1['previous_date'] = data1['datetime'] - pd.DateOffset(years=1)

# 화씨를 섭씨로 변환
data1['temp_celsius'] = (data1['temp'] - 32) * 5.0 / 9.0
data1['tempmax_celsius'] = (data1['tempmax'] - 32) * 5.0 / 9.0
data1['tempmin_celsius'] = (data1['tempmin'] - 32) * 5.0 / 9.0

# 1년 전 날짜가 포함된 주간 데이터를 매칭
matched_data = []
for _, row in data1.iterrows():
    previous_date = row['previous_date']
    match = data1[(data1['datetime'] <= previous_date) & (data1['end_date'] >= previous_date)]
    if not match.empty:
        for _, prev_row in match.iterrows():
            # 이전 온도와 현재 온도
            prev_temp = prev_row['temp_celsius']
            curr_temp = row['temp_celsius']

            # 온도 차이와 상승률 계산
            temp_difference = abs(curr_temp - prev_temp)
            if prev_temp > 5:  # 기준 온도가 5°C 이상일 때만 상승률 계산
                temp_increase_rate = ((curr_temp - prev_temp) / abs(prev_temp)) * 100
            else:
                temp_increase_rate = 0  # 기준 온도가 낮으면 상승률 계산 제외

            # Label 정의 (예: 상승률 20% 이상 AND 절댓값 차이 2°C 이상)
            label = int((temp_difference > 2) and (abs(temp_increase_rate) > 20))

            matched_data.append({
                'current_date': row['datetime'],
                'current_temp_celsius': row['temp_celsius'],
                'previous_temp_celsius': prev_row['temp_celsius'],
                'Temp_Difference': temp_difference,  # 절댓값 차이
                'Temp_Increase_Rate': temp_increase_rate,  # 상승률
                'Polution': row['Polution'],
                'Enviroment_Polution': row['Enviroment_Polution'],
                'Biodiversity_Loss': row['Biodiversity_Loss'],
                'Acid_Rain': row['Acid_Rain'],
                'Water_Pollution': row['Water_Pollution'],
                'Climate_Crisis': row['Climate_Crisis'],
                'Accelerated_Global_Warming': row['Accelerated_Global_Warming'],
                'Ozone_Layer_Depletion': row['Ozone_Layer_Depletion'],
                'Hazardous_Substance_Leakage': row['Hazardous_Substance_Leakage'],
                'Carbon_Dioxide': row['Carbon_Dioxide'],
                'News_Ratio': row['News_Ratio'],
                'Label': label
            })

# 매칭된 데이터를 DataFrame으로 변환
matched_df = pd.DataFrame(matched_data)

# Label 분포 확인
label_counts = matched_df['Label'].value_counts()
print("Label counts:\n", label_counts)

# 결과 확인
print("Matched data shape:", matched_df.shape)
print(matched_df.head())

# 모델링에 사용할 열과 타겟 설정
x = matched_df[['Polution', 'Enviroment_Polution', 'Biodiversity_Loss', 
                'Acid_Rain', 'Water_Pollution', 'Climate_Crisis', 
                'Accelerated_Global_Warming', 'Ozone_Layer_Depletion', 
                'Hazardous_Substance_Leakage', 'Carbon_Dioxide', 'News_Ratio']]
y = matched_df['Label']

# 데이터셋 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)

# 데이터 정규화
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 로지스틱 회귀 모델 학습
model = LogisticRegression(tol=1e-06)
model.fit(x_train_scaled, y_train)

# 학습 결과 출력
print("Model coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Training accuracy:", model.score(x_train_scaled, y_train))

# 테스트 예측 및 평가
y_pred = model.predict(x_test_scaled)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print(metrics.classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
