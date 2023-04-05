# %% Import libraries for isolation forest
# import os
import time
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
# from sklearn.inspection import DecisionBoundaryDisplay
from numpy import bincount


# %% Select columns for isolation forest
total_payments_if = total_payments[['Payment_Number',
                                    'Gen_Jnl_Line_Number',
                                    'Line_Number', 'ID_Vendor_Entry',
                                    'Object_Number', 'Vendor_Number',
                                    'Country_Region_Code', 'Amount_Applied',
                                    'Amount_Initial', 'Discount_Applied',
                                    'Discount_Allowed', 'Discount_Rate',
                                    'Discount_Possible', 'Payment_Method_Code',
                                    'Customer_IBAN', 'Vendor_Bank_Origin',
                                    'Vendor_IBAN_BIC', 'Posting_Date',
                                    'Blocked_Vendor', 'Review_Status',
                                    'Created_By', 'Source_System',
                                    'Year-Month', 'Mandant']]
# include index column in data frame for isolation forest
# total_payments_if = total_payments_if.reset_index()

# %% Print out the info of data frame
total_payments_if.info()

# Show the decoded DataFrame
print(total_payments_if)

# %% Convert the categories of to distinct integers using cat.codes
cat_cols = ['Payment_Number', 'Country_Region_Code', 'Payment_Method_Code',
            'Customer_IBAN', 'Vendor_Bank_Origin', 'Vendor_IBAN_BIC',
            'Created_By',  'Mandant']
for col in cat_cols:
    total_payments_if[col+'_encoded'] = total_payments_if[col].cat.codes


# %% Create a binary encoding column
# for 'Review_Status' and 'Source_System' having 0 and 1
bin_cols = ['Review_Status', 'Source_System']
for col in bin_cols:
    total_payments_if[col+'_encoded'] = total_payments_if[col].cat.codes

# %% Convert 'Posting_Date' and 'Year-month' to integer
total_payments_if['Posting_Date'] = total_payments_if[
    'Posting_Date'].dt.strftime('%Y%m%d').astype(int)
total_payments_if['Year-Month'] = total_payments_if['Year-Month'].dt.strftime(
    '%Y%m').astype(int)


# %% Select columns for isolation forest input
# Select columns for isolation forest
if_input = total_payments_if[['Payment_Number_encoded',
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
                              'Blocked_Vendor', 'Review_Status_encoded',
                              'Created_By_encoded', 'Source_System_encoded',
                              'Year-Month', 'Mandant_encoded']]

if_input.info()
# %% Create the inital isolation forest function
# isof = IsolationForest(max_samples='auto', contamination='auto')

# isof.fit(if_input)

# isof.decision_function(if_input)

# y_pred = isof.predict(if_input)


# # Results of the isolation forest
# print(y_pred)

# n_outliers = bincount((y_pred == -1).astype(int))[1]
# n_inliers = bincount((y_pred == 1).astype(int))[1]

# print("Number of outliers: ", n_outliers)
# print("Number of inliers: ", n_inliers)
# %% Modify the isolation forest function
starttime = time.time()
isof = IsolationForest(n_estimators=1000, max_samples=10000,
                       contamination='auto', random_state=42, verbose=0)

isof.fit(if_input)

scores1 = isof.decision_function(if_input)

anomaly1 = isof.predict(if_input)

# print(y_pred)
#  Results of the isolation forest
n_outliers = bincount((anomaly1 == -1).astype(int))[1]
n_inliers = bincount((anomaly1 == 1).astype(int))[1]

print("Number of outliers: ", n_outliers)
print("Number of inliers: ", n_inliers)
print(f'Database read process took {time.time() - starttime} seconds')
# %% add the scores1 and anomaly1 values to the respective rows
# in the data frame to position 0
total_payments_if.insert(0, "scores1", scores1, True)
total_payments_if.insert(1, "anomaly1", anomaly1, True)


# %% Get a subsample of the data frame with same distribution of anomalies
# as the original data frame
subsample = total_payments_if[total_payments_if['anomaly1']
                              == -1].sample(n=10, random_state=42)
subsample = subsample.append(total_payments_if[
    total_payments_if['anomaly1'] == 1].sample(n=10, random_state=42))

# get a subsample1 where all blocked vendor are inlcuded
# (blocked vendor is not 0)
subsample1 = total_payments_if[total_payments_if['Blocked_Vendor']
                               != 0].sample(n=125, random_state=42)


# %%
