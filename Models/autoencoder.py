# %% Import libraries for isolation forest
# import os
import time

# Data processing
import numpy as np
import pandas as pd
from collections import Counter

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Model and performance
import tensorflow as tf
from tensorflow.keras import layers, losses
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report


# %% Start timer
totaltime = time.time()

# %% Train test split
data_normal = data_normal.astype('float32')
# Separate the features and labels
X = data_normal.drop(['y'], axis=1)
y = data_normal['y']

# Split the data into training and testing sets, stratifying on the label column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Concatenate the features and labels for the training and testing sets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)


# Check the number of records
print('The number of records in the training dataset is', X_train.shape[0])
print('The number of records in the test dataset is', X_test.shape[0])
print(f"The training dataset has {sorted(Counter(y_train).items())[0][1]} records for the majority class and {sorted(Counter(y_train).items())[1][1]} records for the minority class.")# %% Setting up the autoencoder layers

# %%
# Keep only the normal data for the training dataset
X_train_normal = X_train[y_train == 0]

# Input layer
input = tf.keras.layers.Input(shape=(24,))
# Encoder layers
encoder = tf.keras.Sequential([
  layers.Dense(12, activation='relu'),
  layers.Dense(6, activation='relu'),
  layers.Dense(3, activation='relu')])(input)
# Decoder layers
decoder = tf.keras.Sequential([
      layers.Dense(6, activation="relu"),
      layers.Dense(12, activation="relu"),
      layers.Dense(24, activation="sigmoid")])(encoder)
# Create the autoencoder
autoencoder = tf.keras.Model(inputs=input, outputs=decoder)

# %% Run the autoencoder
# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mae')
# Fit the autoencoder
history = autoencoder.fit(X_train_normal, X_train_normal, 
          epochs=20, 
          batch_size=64,
          validation_data=(X_test, X_test),
          shuffle=True)

# %%
# Store the loss and val_loss values in separate arrays
loss_values = history.history['loss']
val_loss_values = history.history['val_loss']

# Define the number of epochs
epochs = range(1, 21)

# Create the line plot
plt.plot(epochs, loss_values, label='Training loss')
plt.plot(epochs, val_loss_values, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# %%
# Predict anomalies/outliers in the training dataset
prediction = autoencoder.predict(X_test)
# Get the mean absolute error between actual and reconstruction/prediction
prediction_loss = tf.keras.losses.mae(prediction, X_test)
# Check the prediction loss threshold for 2% of outliers
loss_threshold = np.percentile(prediction_loss, 95)
print(f'The prediction loss threshold for 5% of outliers is {loss_threshold:.2f}')
# Visualize the threshold
sns.histplot(prediction_loss, bins=100, alpha=0.8)
plt.axvline(x=loss_threshold, color='orange')
# %%
# Check the model performance at 2% threshold
threshold_prediction = [0 if i < loss_threshold else 1 for i in prediction_loss]
# # Check the prediction performance
print(classification_report(y_test, threshold_prediction))



# %%
