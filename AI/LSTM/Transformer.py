import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 전처리
pollutants_path = '../../NewData/Weekly_Air_Pollutants.csv'
temperature_path = '../../ClimateDataTeam/climate_data/merged_weekly_avg_temp.csv'
target_path = pollutants_path

# 데이터 로드
pollutants_df = pd.read_csv(target_path)
temperature_df = pd.read_csv(temperature_path)

# datetime 변환 및 병합
pollutants_df['datetime'] = pd.to_datetime(pollutants_df['datetime'])
temperature_df['datetime'] = pd.to_datetime(temperature_df['datetime'])
merged_df = pd.merge(pollutants_df, temperature_df, on='datetime', how='inner')

# 날짜 정보 추가
merged_df['year'] = merged_df['datetime'].dt.year
merged_df['month'] = merged_df['datetime'].dt.month
merged_df['week'] = merged_df['datetime'].dt.isocalendar().week
merged_df = merged_df.drop(columns=['datetime'])  # datetime 제거

# 주기적 특징 추가 함수
def add_periodic_features(df, period_column, max_period):
    df[f"{period_column}_sin"] = np.sin(2 * np.pi * df[period_column] / max_period)
    df[f"{period_column}_cos"] = np.cos(2 * np.pi * df[period_column] / max_period)
    return df

# 주기적 특징 추가
merged_df = add_periodic_features(merged_df, "week", 52)
merged_df = add_periodic_features(merged_df, "month", 12)

# 출력 및 입력 변수 설정
pollutants_column = pollutants_df.columns.tolist()[1:]  # datetime 제거
climate_column = temperature_df.columns.tolist()[1:]
X = merged_df.drop(columns=pollutants_column[:])  # 입력 변수 (주기적 특징 포함)
y = merged_df[climate_column[:]]  # 출력 변수

# 2. 시계열 윈도우 생성 함수
def create_time_series_features(X, y, lag):
    X_features, y_labels = [], []
    for i in range(len(X) - lag):
        X_features.append(X.iloc[i:i+lag].values)
        y_labels.append(y.iloc[i+lag].values)
    return np.array(X_features), np.array(y_labels)

# 시계열 윈도우 생성
lag = 30
X_ts, y_ts = create_time_series_features(X, y, lag)

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_ts, y_ts, test_size=0.2, random_state=42)

# 3. Transformer 블록 정의
def transformer_block(x, num_heads, key_dim, ff_dim, dropout_rate):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = Add()([x, attn_output])
    out1 = LayerNormalization(epsilon=1e-6)(out1)

    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = Add()([out1, ffn_output])
    out2 = LayerNormalization(epsilon=1e-6)(out2)
    return out2

# 4. Transformer 모델 생성
def create_fully_transformer_model(input_shape, output_dim, num_heads=4, key_dim=32, ff_dim=128, dropout_rate=0.1):
    inputs = Input(shape=input_shape)

    # Transformer Blocks
    x = transformer_block(inputs, num_heads, key_dim, ff_dim, dropout_rate)
    x = transformer_block(x, num_heads, key_dim, ff_dim, dropout_rate)
    x = transformer_block(x, num_heads, key_dim, ff_dim, dropout_rate)
    x = transformer_block(x, num_heads, key_dim, ff_dim, dropout_rate)

    # Pooling and Dense Layers
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)

    # Output Layer
    outputs = Dense(output_dim)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=["mae"])
    return model

# 5. 부트스트랩 학습 함수
def bootstrap_training(X, y, create_model_fn, n_bootstrap=5, epochs=50, batch_size=32, validation_split=0.2):
    models = []
    results = []
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    for i in range(n_bootstrap):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_bootstrap, y_bootstrap = X[indices], y[indices]
        model = create_model_fn()
        history = model.fit(X_bootstrap, y_bootstrap, epochs=epochs, batch_size=batch_size, 
                            validation_split=validation_split, callbacks=[early_stopping], verbose=1)
        models.append(model)
        results.append(min(history.history['val_loss']))
    return models, results

# 6. 앙상블 예측 함수
def ensemble_predict(models, X_test):
    predictions = [model.predict(X_test) for model in models]
    return np.mean(predictions, axis=0)

# 7. 부트스트랩 학습 실행
n_bootstrap = 5
models, results = bootstrap_training(
    X_train, y_train, 
    lambda: create_fully_transformer_model(input_shape=(lag, X_train.shape[2]), output_dim=y_train.shape[1]),
    n_bootstrap=n_bootstrap
)

# 8. 예측 및 평가
y_pred = ensemble_predict(models, X_test)
mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')

print("Validation Loss from Bootstrap Samples:", results)
print("Mean Validation Loss:", np.mean(results))
print("MAE per output variable:", mae)
print("MSE per output variable:", mse)

# 9. 시각화
plt.figure(figsize=(14, 7))
plt.plot(range(len(y_test)), y_test[:, 0], label="Original Data", color="blue", linestyle="--")
plt.plot(range(len(y_pred)), y_pred[:, 0], label="Predicted Data", color="red", alpha=0.8)
plt.title("Comparison of Actual vs Predicted Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
