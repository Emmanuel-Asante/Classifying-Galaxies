# Import modules
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from utils import load_galaxy_data

import app

# Load data
input_data, labels = load_galaxy_data()

# Print out the dimension of input_data
print(input_data.shape)

# Print out the dimension of labels
print(labels.shape)

# Split data in training and validation data
x_train, x_valid, y_train, y_valid = train_test_split(input_data, labels, test_size=0.20, stratify=labels, shuffle=True, random_state=222)

# Preprocess data (pixel normalization)
data_generator = ImageDataGenerator(rescale=1./255)

# Create a training data iterator
training_iterator = data_generator.flow(x_train, y_train, batch_size=5)

# Create a validation data iterator
validation_iterator = data_generator.flow(x_valid, y_valid, batch_size=5)

# Create a Sequential model
model = tf.keras.Sequential()

# Add input layer to the model
model.add(tf.keras.Input(shape=(128, 128, 3)))

# Add output layer to the model
#model.add(tf.keras.layers.Dense(4, activation="softmax"))

# Add a Conv2D layer with 8 filters each size 3x3, and stride of 2
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu"))

# Add a max pooling layer
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Add a Conv2D layer with 8 filters each size 3x3, and stride of 2
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu"))

# Add a max pooling layer
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

# Add a Flatten layer
model.add(tf.keras.layers.Flatten())

# Add hidden Dense Layer with 16 hidden units
model.add(tf.keras.layers.Dense(16, activation="relu"))

# Add uoutput Dense Layer
model.add(tf.keras.layers.Dense(4, activation="softmax"))

# Print out the model's summary
print(model.summary())

# Compile the model
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
  loss=tf.keras.losses.CategoricalCrossentropy(),
  metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()]
)

# Train the model on training_iterator
model.fit(training_iterator, steps_per_epoch=len(x_train)/5, epochs=8, validation_data=validation_iterator, validation_steps=len(x_valid)/5)

# Import function for visualization
from visualize import visualize_activations

# Visualize images
visualize_activations(model,validation_iterator)