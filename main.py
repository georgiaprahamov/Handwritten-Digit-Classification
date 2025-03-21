import ssl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

ssl._create_default_https_context = ssl._create_unverified_context

# Зареждане на MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Нормализиране на входните данни
X_train, X_test = X_train / 255.0, X_test / 255.0

# Изграждане на MLP модела
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),   # Преобразуване на 28x28 вектор
    layers.Dense(128, activation="relu"),   # Плътен слой с 128 неврона
    layers.Dense(64, activation="relu"),    # Още един скрит слой
    layers.Dense(10, activation="softmax")  # Изходен слой с 10 класа (0-9)
])

# Компилиране на модела
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Обучение на модела
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

num_images = 5
indices = np.random.choice(len(X_test), num_images)

plt.figure(figsize=(10, 4))
for i, idx in enumerate(indices):
    img = X_test[idx]
    true_label = y_test[idx]
    predicted_label = np.argmax(model.predict(img.reshape(1, 28, 28)))

    plt.subplot(1, num_images, i + 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(f"Real: {true_label}\nPredicted: {predicted_label}")

plt.show()
