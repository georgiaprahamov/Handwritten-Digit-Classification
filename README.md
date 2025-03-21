# MNIST Handwritten Digit Classification

## Overview
This project implements a simple Multi-Layer Perceptron (MLP) model using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The model consists of fully connected layers and is trained to recognize digits (0-9) from 28x28 grayscale images.

## Features
- Loads the MNIST dataset
- Normalizes input data for better performance
- Implements a neural network with ReLU activations
- Uses the Adam optimizer for training
- Evaluates the model on test data
- Displays predictions alongside the actual images

## Requirements
Make sure you have the following dependencies installed before running the script:

```bash
pip install numpy tensorflow matplotlib
```

## Installation
1. Clone the repository or copy the script.
2. Install the required dependencies using `pip`.
3. Run the script using Python 3.

## Code Explanation
### 1. Import Dependencies
The script imports the required libraries:
- `ssl`: Handles SSL verification bypass for dataset downloading
- `numpy`: Used for numerical operations
- `tensorflow`: Provides deep learning functionalities
- `matplotlib.pyplot`: Used for visualization

### 2. Load and Preprocess Data
The MNIST dataset is loaded and normalized by scaling pixel values to the range `[0,1]`.

```python
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
```

### 3. Define the Model
The model is built using a sequential API:
- `Flatten` layer converts 28x28 images into a 1D vector.
- `Dense(128, activation='relu')` applies a fully connected layer with 128 neurons.
- `Dense(64, activation='relu')` applies another fully connected layer with 64 neurons.
- `Dense(10, activation='softmax')` provides output probabilities for 10 digit classes.

```python
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

### 4. Compile and Train the Model
The model is compiled with:
- `Adam` optimizer for efficient training.
- `Sparse Categorical Crossentropy` as the loss function.
- `Accuracy` as the evaluation metric.

The model is trained for 5 epochs with a batch size of 32:

```python
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
```

### 5. Make Predictions and Visualize Results
The script selects 5 random images from the test set, predicts their labels, and visualizes the results.

```python
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
```

## Running the Script
To run the script, simply execute:

```bash
python mnist_classifier.py
```

## Expected Output
- Training logs showing accuracy and loss per epoch.
- A visualization of sample test images with their predicted labels.

## License
This project is open-source and available for educational purposes.

