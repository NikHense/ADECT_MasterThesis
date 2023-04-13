# %% Import libraries for data cleaning
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import time
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

# %% Change data type of specific columns
# Change data type to integer type
int_cols = ['Vendor_Number']
for col in int_cols:
    total_payments[col] = total_payments[col].astype(int)

# Change string data type to categorical values
cat_cols = ['Payment_Number', 'Object_Number', 'Country_Region_Code',
            'Payment_Method_Code', 'Customer_IBAN',
            'Vendor_Bank_Origin', 'Vendor_IBAN_BIC',
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

# %% Select columns for algortims
total_payments = total_payments[['Payment_Number',
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
# include index column in data frame for isolation forest
# data_IsoForest = data_IsoForest.reset_index()

# %% Print out the info of data frame
total_payments.info()

# Show the decoded DataFrame
print(total_payments)

# %% Convert the categories of to distinct integers using cat.codes
cat_cols = ['Payment_Number', 'Object_Number',
            'Country_Region_Code', 'Payment_Method_Code',
            'Customer_IBAN', 'Vendor_Bank_Origin', 'Vendor_IBAN_BIC',
            'Created_By',  'Mandant']
for col in cat_cols:
    total_payments[col+'_encoded'] = total_payments[col].cat.codes


# %% Create a binary encoding column
# for 'Review_Status' and 'Source_System' having 0 and 1
bin_cols = ['Review_Status', 'Source_System']
for col in bin_cols:
    total_payments[col+'_encoded'] = total_payments[col].cat.codes

# %% Convert 'Posting_Date' and 'Year-month' to integer
total_payments['Posting_Date'] = total_payments[
    'Posting_Date'].dt.strftime('%Y%m%d').astype(int)
total_payments['Year-Month'] = total_payments['Year-Month'].dt.strftime(
    '%Y%m').astype(int)


# %% Select columns for isolation forest input
# Select columns for isolation forest
if_data = total_payments[['Payment_Number_encoded',
                           'Gen_Jnl_Line_Number',
                           'Line_Number', 'ID_Vendor_Entry',
                           'Object_Number_encoded', 'Vendor_Number',
                           'Country_Region_Code_encoded', 'Amount_Applied',
                           'Amount_Initial', 'Discount_Applied',
                           'Discount_Allowed', 'Discount_Rate',
                           'Discount_Possible',
                           'Payment_Method_Code_encoded',
                           'Customer_IBAN_encoded',
                           'Vendor_Bank_Origin_encoded',
                           'Vendor_IBAN_BIC_encoded', 'Posting_Date',
                           'Blocked_Vendor', 'Review_Status_encoded',
                           'Created_By_encoded', 'Source_System_encoded',
                           'Year-Month', 'Mandant_encoded']]

if_data.info()



# %% Select columns for kmeans input
kmeans_data = total_payments[['Payment_Number_encoded',
                              'Gen_Jnl_Line_Number',
                              'Line_Number', 'ID_Vendor_Entry',
                              'Object_Number_encoded', 'Vendor_Number',
                              'Country_Region_Code_encoded', 'Amount_Applied',
                              'Amount_Initial', 'Discount_Applied',
                              'Discount_Allowed', 'Discount_Rate',
                              'Discount_Possible',
                              'Payment_Method_Code_encoded',
                              'Customer_IBAN_encoded',
                              'Vendor_Bank_Origin_encoded',
                              'Vendor_IBAN_BIC_encoded', 'Posting_Date',
                              'Blocked_Vendor', 'Review_Status_encoded',
                              'Created_By_encoded', 'Source_System_encoded',
                              'Year-Month', 'Mandant_encoded']]

kmeans_data.info()

# %%
