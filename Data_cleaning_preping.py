# %% Import libraries for data cleaning
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
# import time
# import datetime
# import json
# import requests
# import itertools
# from multiprocessing import Pool
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler


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

# %% Change data type of specific columns
# Change data type to integer type
int_cols = ['Object_Number', 'Vendor_Number']
for col in int_cols:
    total_payments[col] = total_payments[col].astype(int)

# Change string data type to categorical values
cat_cols = ['Payment_Number', 'Country_Region_Code', 'Payment_Method_Code',
            'Customer_IBAN', 'Vendor_Bank_Origin', 'Vendor_IBAN_BIC',
            'Review_Status', 'Created_By', 'Source_System',  'Mandant']
for col in cat_cols:
    total_payments[col] = total_payments[col].astype('category')

# get the unique categorical values and corresponding codes for column
    # Vendor_Bank_Origin = total_payments['Vendor_Bank_Origin'].cat.categories
    # for i in range(len(Vendor_Bank_Origin)):
    # print(f"{Vendor_Bank_Origin[i]}: {i}")

# Convert data type to float
float_cols = ['Amount_Applied', 'Amount_Initial', 'Discount_Applied',
              'Discount_Rate', 'Discount_Possible']

for col in float_cols:
    total_payments[col] = total_payments[col].astype(float)

# Convert data type to datetime
date_cols = ['Posting_Date', 'Due_Date', 'Year-Month']
for col in date_cols:
    total_payments[col] = pd.to_datetime(total_payments[col])

# %% Replace ' ' with NaN and print out count of empty cells per column
# Replace ' ' with NaN
total_payments.replace(' ', np.nan, inplace=True)

# Count the number of empty cells per column
empty_cells = total_payments.isna().sum()

# Print the result
print(empty_cells)

# %% Print out the info of data frame
total_payments.info()

# %%