# %% Import libraries for data cleaning
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# import datetime
# import json
# import requests
# import itertools
# from multiprocessing import Pool
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler


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

# %% Print out the info of data frame
total_payments.info()

# %% Combining Vendor_IBAN and Vendor_BIC
total_payments['Vendor_IBAN'] = total_payments['Vendor_IBAN'].astype(str)
total_payments['Vendor_BIC'] = total_payments['Vendor_BIC'].astype(str)
total_payments['Vendor_IBAN_BIC'] = total_payments['Vendor_IBAN']
+ total_payments['Vendor_BIC']
# Insert a column with the value of 'Vendor_IBAN_BIC'
# to the position after 'Vendor_BIC'
# total_payments.insert(11, 'Vendor_IBAN_BIC',
#                       total_payments['Vendor_IBAN_BIC'])

# Combine Country_Region_Code and Vendor_Bank_Origin
# total_payments['Vendor_Bank_Origin'] = total_payments['Vendor_Bank_Origin'].astype(str)
# total_payments['Country_Region_Code'] = total_payments['Country_Region_Code'].astype(str)
# total_payments['Country_Codes'] = total_payments['Country_Region_Code']
# + total_payments['Vendor_Bank_Origin']

# %% Change data type of specific columns
# Change data type to integer type
int_cols = ['Vendor_Number']
for col in int_cols:
    total_payments[col] = total_payments[col].astype(int)

# Change string data type to categorical values
cat_cols = ['Payment_Number', 'Object_Number',
            'Payment_Method_Code', 'Customer_IBAN',
            'Vendor_IBAN_BIC', 'Vendor_Bank_Origin', 'Country_Region_Code',
            'Created_By', 'Source_System', 'Mandant']
for col in cat_cols:
    total_payments[col] = total_payments[col].astype('category')

# get the unique categorical values and corresponding codes for column
    # Vendor_Bank_Origin = total_payments['Vendor_Bank_Origin'].cat.categories
    # for i in range(len(Vendor_Bank_Origin)):
    # print(f"{Vendor_Bank_Origin[i]}: {i}")

# Convert data type to float
float_cols = ['Amount_Applied', 'Amount_Initial', 'Discount_Applied',
              'Discount_Rate']

for col in float_cols:
    total_payments[col] = total_payments[col].astype(float)

# Convert data type to datetime
date_cols = ['Posting_Date', 'Due_Date']
for col in date_cols:
    total_payments[col] = pd.to_datetime(total_payments[col])

# %% Replace ' ' with NaN and print out count of empty cells per column
# Replace ' ' with NaN
total_payments.replace(' ', np.nan, inplace=True)

# Count the number of empty cells per column
empty_cells = total_payments.isna().sum()

# Print the result
print(empty_cells)

# %% Create a binary encoding column
# for 'Review_Status' and 'Source_System' having 0 and 1
total_payments['Source_System_encoded'] = total_payments['Source_System'].replace({'BFSN': 0, 'RELN': 1})
total_payments.drop(columns=['Source_System'], inplace=True)


# %% Convert 'Posting_Date' & 'Due_Date to integer
total_payments['Posting_Date'] = total_payments['Posting_Date'].astype(int)
total_payments['Due_Date'] = total_payments['Due_Date'].astype(int)

# %% Select columns for one-hot encoding
input_cluster = total_payments[['Payment_Number',
                                'Gen_Jnl_Line_Number',
                                'Line_Number',
                                'Object_Number', 'Vendor_Number',
                                'Country_Region_Code', 'Amount_Applied',
                                'Amount_Initial', 'Discount_Applied',
                                'Discount_Allowed', 'Discount_Rate',
                                'Payment_Method_Code',
                                'Customer_IBAN', 'Vendor_IBAN_BIC',
                                'Vendor_Bank_Origin', 'Posting_Date',
                                'Due_Date', 'Created_By',
                                'Source_System_encoded', 'Mandant']]

input_tree = total_payments[['Payment_Number',
                             'Gen_Jnl_Line_Number',
                             'Line_Number',
                             'Object_Number', 'Vendor_Number',
                             'Country_Region_Code', 'Amount_Applied',
                             'Amount_Initial', 'Discount_Applied',
                             'Discount_Allowed', 'Discount_Rate',
                             'Payment_Method_Code',
                             'Customer_IBAN', 'Vendor_IBAN_BIC',
                             'Vendor_Bank_Origin', 'Posting_Date',
                             'Due_Date', 'Created_By',
                             'Source_System_encoded', 'Mandant']]

# %% Select columns for catgeorical encoding
# Change categorical data type to label encoding
cat_cols = ['Payment_Number', 'Object_Number',
            'Country_Region_Code', 'Payment_Method_Code',
            'Customer_IBAN', 'Vendor_IBAN_BIC',
            'Vendor_Bank_Origin', 'Created_By',
            'Mandant']

for col in cat_cols:
    labelencoder = LabelEncoder()
    input_tree[col] = labelencoder.fit_transform(input_tree[col])

print(input_tree.info())
# %% One-hot encoding of categorical columns
# creating one hot encoder object 
onehotencoder = OneHotEncoder()

# select the columns to be one-hot encoded
columns_to_encode = ['Payment_Number', 'Object_Number',
                     'Country_Region_Code', 'Payment_Method_Code',
                     'Customer_IBAN', 'Vendor_IBAN_BIC',
                     'Vendor_Bank_Origin', 
                     'Created_By', 'Mandant']

# apply one-hot encoding to the selected columns
for col in columns_to_encode:
    data = input_cluster[[col]]
    X = onehotencoder.fit_transform(data.values.reshape(-1,1)).toarray()
    dfOneHot = pd.DataFrame(X, columns=[col+"_"+str(int(i)) for i in range(X.shape[1])])
    input_cluster = pd.concat([input_cluster, dfOneHot], axis=1)
    input_cluster.drop([col], axis=1, inplace=True)

# print the resulting dataframe
print(input_cluster.info())

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

# %% Scaling the input data

# scaler = MinMaxScaler(feature_range=(-25, 25))
# input_scaled = scaler.fit_transform(input)
# input_scaled = pd.DataFrame(input_scaled, columns=input.columns)


scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_cluster)
input_scaled = pd.DataFrame(input_scaled, columns=input_cluster.columns)

input_scaled.mean()
input_scaled.var()
# %%
input_scaled.info()
# # %%
# for col in input_scaled.columns:
#     # create a histogram for the current column
#     plt.hist(input_scaled[col], bins=50)
    
#     # add labels and title to the plot
#     plt.xlabel(col)
#     plt.ylabel('Frequency')
#     plt.title(f'Distribution of {col}')
    
#     # display the plot
#     plt.show()
# %%
