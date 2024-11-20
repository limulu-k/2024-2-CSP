import tensorflow as tf

# 하이퍼파라미터
lr = 0.001
epochs = 15
batch_sz = 100

# 입력 데이터와 출력 데이터 (예시 데이터)
# 실제 데이터로 대체하세요.
import numpy as np
X_data = np.random.rand(60000, 784).astype(np.float32)  # 60,000 샘플, 784 특성 (예: MNIST 데이터)
y_data = tf.keras.utils.to_categorical(np.random.randint(0, 10, size=(60000,)), num_classes=10)

# 데이터셋 분할 (훈련/검증 데이터)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 배치 데이터셋 생성
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(batch_sz)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_sz)

# 신경망 모델 구축
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),  # 입력층
    tf.keras.layers.Dense(256, activation='relu'),                      # 은닉층
    tf.keras.layers.Dense(10)                                          # 출력층 (10 클래스)
])

# 손실 함수와 옵티마이저 설정
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

# 모델 컴파일
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# 모델 학습
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# 모델 평가
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# 예측
predictions = model.predict(X_val[:5])  # 상위 5개 샘플 예측
print("Predictions:", tf.argmax(predictions, axis=1).numpy())
print("Actual:", tf.argmax(y_val[:5], axis=1).numpy())