import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# ==== 資料處理 ====
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# ==== 模型建構 ====
model = Sequential([
    Flatten(input_shape=(28, 28), name='flatten'),
    Dense(256, activation='relu', name='dense_1'),
    Dense(10, activation='softmax', name='dense_2')
])

# ==== 編譯與訓練 ====
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=64)

# ==== 儲存模型（可選）====
model.save('model.h5')  # 可忽略此檔

# ==== 儲存權重 ====
weights = model.get_weights()
np.savez('fashion_mnist.npz', *weights)

# ==== 儲存模型結構（只保留非 InputLayer 的層）====
layers = [layer for layer in model.get_config()['layers'] if layer['class_name'] != 'InputLayer']
with open('fashion_mnist.json', 'w') as f:
    json.dump(layers, f, indent=2)
