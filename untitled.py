# Importing required libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Visualize the first training image
plt.imshow(x_train[0], cmap='gray')
plt.title(f'Label: {y_train[0]}')
plt.show()

# Build a simple neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the input image to a 1D vector
    layers.Dense(128, activation='relu'),  # First fully connected layer with 128 neurons and ReLU activation
    layers.Dense(10, activation='softmax') # Output layer for 10 classes (digits 0-9) with softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc:.4f}')

# Make predictions on the test set
predictions = model.predict(x_test)

# Visualize the prediction for the first test image
plt.imshow(x_test[0], cmap='gray')
plt.title(f'Predicted: {predictions[0].argmax()}, Actual: {y_test[0]}')
plt.show()
