# %% Import libraries for data cleaning
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.stats as stats
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
# import datetime
# import json
# import requests
# import itertools
# from multiprocessing import Pool
from sqlalchemy import create_engine, text
# from sklearn.preprocessing import StandardScaler


# %% params sql connection
SERVER = 'T-SQLDWH-DEV'  # os.environ.get('SERVER')
DB = 'ML'
USERNAME = os.environ.get('USERNAME')
PASSWORD = os.environ.get('PASSWORD')

# %% sql connection
conn = ("Driver={ODBC Driver 18 for SQL Server};"
        "Server="+SERVER+";"
        "Database="+DB+";"
        "UID="+USERNAME+";"
        "PWD="+PASSWORD+";"
        "Encrypt=YES;"
        "TrustServerCertificate=YES")
engine = create_engine(
    f'mssql+pyodbc://?odbc_connect={conn}',
    fast_executemany=True)

# %%
SQL_TOTAL_PAYMENTS = 'SELECT * FROM ADECT.TOTAL_PAYMENTS'
total_payments = pd.DataFrame(engine.connect().execute(
    text(SQL_TOTAL_PAYMENTS)))

total_payments.insert(0, 'INDEX', total_payments.index)

total_payments.info()
# %% Define new data frame for data cleaning and preparation
input = total_payments.copy()

# %% Combining Vendor_IBAN and Vendor_BIC
input['Vendor_IBAN'] = input['Vendor_IBAN'].astype(str)
input['Vendor_BIC'] = input['Vendor_BIC'].astype(str)
input['Vendor_IBAN_BIC'] = input['Vendor_IBAN'] + input['Vendor_BIC']

# %% Change data type of specific columns

# Convert data type to datetime
date_cols = ['Posting_Date', 'Due_Date']
for col in date_cols:
    input[col] = pd.to_datetime(input[col])

# Change data type to integer type
int_cols = ['Vendor_Number', 'Posting_Date', 'Due_Date']
for col in int_cols:
    input[col] = input[col].astype(int)

# # Convert 'Posting_Date' & 'Due_Date to integer
# input['Posting_Date'] = input['Posting_Date'].astype(int)
# input['Due_Date'] = input['Due_Date'].astype(int)

# Change string data type to categorical values
cat_cols = ['Payment_Number', 'Object_Number',
            'Payment_Method_Code', 'Customer_IBAN',
            'Vendor_IBAN_BIC', 'Vendor_Bank_Origin', 'Country_Region_Code',
            'Created_By', 'Source_System', 'Mandant']
for col in cat_cols:
    input[col] = input[col].astype('category')

# get the unique categorical values and corresponding codes for column
    # Vendor_Bank_Origin = input['Vendor_Bank_Origin'].cat.categories
    # for i in range(len(Vendor_Bank_Origin)):
    # print(f"{Vendor_Bank_Origin[i]}: {i}")

# Convert data type to float
float_cols = ['Amount_Applied', 'Amount_Initial', 'Discount_Applied',
              'Discount_Rate']

for col in float_cols:
    input[col] = input[col].astype(float)

# Create a binary encoding column for 'Source_System' having 0 and 1
input['Source_System'] = input['Source_System'].replace({'BFSN': 0, 'RELN': 1})
# input.drop(columns=['Source_System'], inplace=True)

# %% Replace ' ' with NaN and print out count of empty cells per column
# Replace ' ' with NaN
input.replace(' ', np.nan, inplace=True)

# Count the number of empty cells per column
empty_cells = input.isna().sum()

# Print the result
print(empty_cells)

# %% Select columns for encoding
input = input[['Payment_Number',
               'Gen_Jnl_Line_Number',
               'Line_Number',
               'Object_Number',
               'Vendor_Number',
               'Country_Region_Code',
               'Amount_Applied',
               'Amount_Initial',
               'Discount_Applied',
               'Discount_Allowed',
               'Discount_Rate',
               'Payment_Method_Code',
               'Customer_IBAN',
               'Vendor_IBAN_BIC',
               'Vendor_Bank_Origin',
               'Posting_Date',
               'Due_Date',
               'Created_By',
               'Source_System',
               'Mandant']]

# %% Select columns for catgeorical encoding
# Change categorical data type to label encoding
cat_cols = ['Payment_Number', 'Object_Number',
            'Country_Region_Code', 'Payment_Method_Code',
            'Customer_IBAN', 'Vendor_IBAN_BIC',
            'Vendor_Bank_Origin', 'Created_By',
            'Mandant']

for col in cat_cols:
    labelencoder = LabelEncoder()
    input[col] = labelencoder.fit_transform(input[col])

print(input.info())

# %%
# for col in input.columns:
#     # create a histogram for the current column
#     plt.hist(input[col], bins=20)

#     # add labels and title to the plot
#     plt.xlabel(col)
#     plt.ylabel('Frequency')
#     plt.title(f'Distribution of {col}')
#     # display the plot
#     plt.show()

# %%
# for col in input_scaled.columns:
#     # create a histogram for the current column
#     plt.hist(input_scaled[col], bins=50)

#     # add labels and title to the plot
#     plt.xlabel(col)
#     plt.ylabel('Frequency')
#     plt.title(f'Distribution of {col}')

#     # display the plot
#     plt.show()
# %% One-hot encoding of categorical columns
# creating one hot encoder object
# onehotencoder = OneHotEncoder()

# # select the columns to be one-hot encoded
# columns_to_encode = ['Payment_Number', 'Object_Number',
#                      'Country_Region_Code', 'Payment_Method_Code',
#                      'Customer_IBAN', 'Vendor_IBAN_BIC',
#                      'Vendor_Bank_Origin',
#                      'Created_By', 'Mandant']

# # apply one-hot encoding to the selected columns
# for col in columns_to_encode:
#     data = input_cluster[[col]]
#     X = onehotencoder.fit_transform(data.values.reshape(-1,1)).toarray()
#     dfOneHot = pd.DataFrame(X, columns=[col+"_"+str(int(i))
#                                         for i in range(X.shape[1])])
#     input_cluster = pd.concat([input_cluster, dfOneHot], axis=1)
#     input_cluster.drop([col], axis=1, inplace=True)

# # print the resulting dataframe
# print(input_cluster.info())
