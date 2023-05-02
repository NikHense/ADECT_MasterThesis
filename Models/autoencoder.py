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
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# %% Start timer
totaltime = time.time()

# %% Define data for autoencoder

data_auto = pd.DataFrame(data=input, columns=input.columns)
data_auto.insert(0, 'INDEX', total_payments.index)
data_auto = pd.merge(data_auto, data_normal[['INDEX', 'y_2']],
                     on='INDEX', how='left')
data_auto = data_auto.drop(['INDEX'], axis=1)
data_auto.info()

# %% Train test split
input_auto = data_auto.astype('float32')
# Separate the features and labels
X = input_auto.drop(['y_2'], axis=1)
y = input_auto['y_2']

# Scaling the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=input_auto.columns.drop(['y_2']))

# Split the data into train and test sets, stratifying on the label column
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Concatenate the features and labels for the training and testing sets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)


# Check the number of records
print('The number of records in the training dataset is', X_train.shape[0])
print('The number of records in the test dataset is', X_test.shape[0])
print(f'The training dataset has {sorted(Counter(y_train).items())[0][1]} '
      f'records for the majority class and '
      f'{sorted(Counter(y_train).items())[1][1]} '
      f'records for the minority class.')

print(f'The test data set has {sorted(Counter(y_test).items())[0][1]} '
      f'records for the majority class and '
      f'{sorted(Counter(y_test).items())[1][1]} '
      f'records for the minority class.')

# %% Setting up the autoencoder layers
# Keep only the normal data for the training dataset
X_train_normal = X_train[y == 0]

# Input layer
input_auto = tf.keras.layers.Input(shape=(20,))

# Encoder layers
encoder = tf.keras.Sequential([
  layers.Dense(12, activation='relu'),
  layers.Dense(6, activation='relu'),
  layers.Dense(3, activation='relu')])(input_auto)

# Decoder layers
decoder = tf.keras.Sequential([
      layers.Dense(6, activation="relu"),
      layers.Dense(12, activation="relu"),
      layers.Dense(20, activation="sigmoid")])(encoder)
# Create the autoencoder
autoencoder = tf.keras.Model(inputs=input_auto, outputs=decoder)

# %% Run the autoencoder
# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mae')
# Fit the autoencoder
history = autoencoder.fit(X_train_normal, X_train_normal,
                          # epochs=10000,
                          # epochs=500,
                          epochs=75,
                          batch_size=128,
                          validation_data=(X_test, X_test),
                          shuffle=True)

# %%
# Store the loss and val_loss values in separate arrays
loss_values = history.history['loss']
val_loss_values = history.history['val_loss']

# %% Define the number of epochs
epochs = range(1, 76)

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
# Check the prediction loss threshold for 0.1% of outliers
loss_threshold = np.percentile(prediction_loss, 99.9)
print(f'The prediction loss threshold for 0.1% of '
      f'outliers is {loss_threshold:.2f}')
# Visualize the threshold
sns.histplot(prediction_loss, bins=50, alpha=0.8, color='b')
plt.axvline(x=loss_threshold, color='orange')

# %%
# Check the model performance at 2% threshold
threshold_prediction = [0 if i < loss_threshold else 1
                        for i in prediction_loss]
# # Check the prediction performance
print(classification_report(y_test, threshold_prediction))

# %%
