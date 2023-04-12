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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# %% Select columns for isolation forest
data_Autoencoder = total_payments[['Payment_Number',
                                   'Gen_Jnl_Line_Number',
                                   'Line_Number', 'ID_Vendor_Entry',
                                   'Object_Number', 'Vendor_Number',
                                   'Country_Region_Code', 'Amount_Applied',
                                   'Amount_Initial', 'Discount_Applied',
                                   'Discount_Allowed', 'Discount_Rate',
                                   'Discount_Possible', 'Payment_Method_Code',
                                   'Customer_IBAN', 'Vendor_IBAN_BIC',
                                   'Vendor_Bank_Origin', 'Posting_Date',
                                   'Due_Date', 'Entry_Cancelled',
                                   'Blocked_Vendor', 'Review_Status',
                                   'Created_By', 'Source_System',
                                   'Year-Month', 'Mandant']]


# %% Convert 'Posting_Date' and 'Year-month' to integer
data_Autoencoder['Posting_Date'] = data_Autoencoder[
    'Posting_Date'].dt.strftime('%Y%m%d').astype(int)
data_Autoencoder['Year-Month'] = data_Autoencoder['Year-Month'].dt.strftime(
    '%Y%m').astype(int)
data_Autoencoder['Due_Date'] = data_Autoencoder['Due_Date'].dt.strftime(
    '%Y%m%d').astype(int)
# %% Convert the categories of to distinct integers using cat.codes
cat_cols = ['Payment_Number', 'Country_Region_Code', 'Payment_Method_Code',
            'Customer_IBAN', 'Vendor_Bank_Origin', 'Vendor_IBAN_BIC',
            'Created_By',  'Mandant']
for col in cat_cols:
    data_Autoencoder[col+'_encoded'] = data_Autoencoder[col].cat.codes


# %% Create a binary encoding column
# for 'Review_Status' and 'Source_System' having 0 and 1
bin_cols = ['Review_Status', 'Source_System']
for col in bin_cols:
    data_Autoencoder[col+'_encoded'] = data_Autoencoder[col].cat.codes

# %% Select columns for autoencoder input
# Select columns for autoencoder
auto_input = data_Autoencoder[['Payment_Number_encoded',
                           'Gen_Jnl_Line_Number',
                           'Line_Number', 'ID_Vendor_Entry',
                           'Object_Number', 'Vendor_Number',
                           'Country_Region_Code_encoded', 'Amount_Applied',
                           'Amount_Initial', 'Discount_Applied',
                           'Discount_Allowed', 'Discount_Rate',
                           'Discount_Possible',
                           'Payment_Method_Code_encoded',
                           'Customer_IBAN_encoded',
                           'Vendor_Bank_Origin_encoded',
                           'Vendor_IBAN_BIC_encoded', 'Posting_Date',
                           'Due_Date', 'Entry_Cancelled', 'Blocked_Vendor', 
                           'Review_Status_encoded', 'Created_By_encoded', 
                           'Source_System_encoded', 'Year-Month',
                            'Mandant_encoded']]
# Convert only integer columns to float
auto_input = auto_input.astype(float)
auto_input.info()
# %% Train test split
X_train, X_test = train_test_split(auto_input,
                                   test_size=0.2, random_state=42)

# Check the number of records
print('The number of records in the training dataset is', X_train.shape[0])
print('The number of records in the test dataset is', X_test.shape[0])
# %% Setting up the autoencoder layers
# Input layer
input = tf.keras.layers.Input(shape=(26,))
# Encoder layers
encoder = tf.keras.Sequential([
  layers.Dense(14, activation='relu'),
  layers.Dense(7, activation='relu'),
  layers.Dense(3, activation='relu')])(input)
# Decoder layers
decoder = tf.keras.Sequential([
      layers.Dense(7, activation="relu"),
      layers.Dense(14, activation="relu"),
      layers.Dense(26, activation="sigmoid")])(encoder)
# Create the autoencoder
autoencoder = tf.keras.Model(inputs=input, outputs=decoder)

# %% Run the autoencoder
# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mae')
# Fit the autoencoder
history = autoencoder.fit(X_train, X_train, 
          epochs=200, 
          batch_size=64,
          validation_data=(X_test, X_test),
          shuffle=True)


# %%
# Store the loss and val_loss values in separate arrays
loss_values = history.history['loss']
val_loss_values = history.history['val_loss']

# Define the number of epochs
epochs = range(1, 201)

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
