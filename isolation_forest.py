# %% Import libraries for isolation forest
# import os
import time
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
# from sklearn.inspection import DecisionBoundaryDisplay
from numpy import bincount
from multiprocessing import Pool

# %% Select columns for isolation forest
data_IsoForest = total_payments[['Payment_Number',
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
data_IsoForest.info()

# Show the decoded DataFrame
print(data_IsoForest)

# %% Convert the categories of to distinct integers using cat.codes
cat_cols = ['Payment_Number', 'Country_Region_Code', 'Payment_Method_Code',
            'Customer_IBAN', 'Vendor_Bank_Origin', 'Vendor_IBAN_BIC',
            'Created_By',  'Mandant']
for col in cat_cols:
    data_IsoForest[col+'_encoded'] = data_IsoForest[col].cat.codes


# %% Create a binary encoding column
# for 'Review_Status' and 'Source_System' having 0 and 1
bin_cols = ['Review_Status', 'Source_System']
for col in bin_cols:
    data_IsoForest[col+'_encoded'] = data_IsoForest[col].cat.codes

# %% Convert 'Posting_Date' and 'Year-month' to integer
data_IsoForest['Posting_Date'] = data_IsoForest[
    'Posting_Date'].dt.strftime('%Y%m%d').astype(int)
data_IsoForest['Year-Month'] = data_IsoForest['Year-Month'].dt.strftime(
    '%Y%m').astype(int)


# %% Select columns for isolation forest input
# Select columns for isolation forest
if_input = data_IsoForest[['Payment_Number_encoded',
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
                       contamination='auto', random_state=42,
                       verbose=0, n_jobs=-1)

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
data_IsoForest.insert(0, "scores1", scores1, True)
data_IsoForest.insert(1, "anomaly1", anomaly1, True)


# # %% Get a subsample of the data frame with same distribution of anomalies
# # as the original data frame
# subsample = data_IsoForest[data_IsoForest['anomaly1']
#                            == -1].sample(n=10, random_state=42)
# subsample = subsample.append(data_IsoForest[
#     data_IsoForest['anomaly1'] == 1].sample(n=10, random_state=42))

# # get a subsample1 where all blocked vendor are inlcuded
# # (blocked vendor is not 0)
# subsample1 = data_IsoForest[data_IsoForest['Blocked_Vendor']
#                             != 0].sample(n=125, random_state=42)

# %% Execute grid search for isolation forest
starttime = time.time()

# Define list of parameter values to test
n_estimators_list = list(range(250, 2001, 250))
max_samples_list = list(range(1000, 52001, 1000))

# Initialize empty lists to store results
results = []
params = []

# Define a function to fit the IsolationForest model and compute the results
def fit_isof(params):
    n_estimators, max_samples = params
    # starttime1 = time.time()
    isof = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                           contamination='auto', random_state=42,
                           verbose=0, n_jobs=1)
    isof.fit(if_input)
    # scores = isof.decision_function(if_input)
    anomaly = isof.predict(if_input)
    n_outliers = bincount((anomaly == -1).astype(int))[1]
    n_inliers = bincount((anomaly == 1).astype(int))[1]
    # elapsed_time1 = time.time() - starttime1
    # print(f"n_estimators: {n_estimators}, max_samples: {max_samples}; "
    #       f"Number of outliers: {n_outliers}, Number of inliers: {n_inliers}; "
    #       f"Elapsed time: {elapsed_time1:.2f} seconds")
    return (n_outliers, n_inliers, n_estimators, max_samples)


# Use multiprocessing to parallelize the loop over parameter combinations
with Pool() as p:
    params = [(n_estimators, max_samples) for n_estimators in n_estimators_list
              for max_samples in max_samples_list]
    results = p.map(fit_isof, params)

# # Print the results and parameters for each combination
# for n_outliers, n_inliers, n_estimators, max_samples in results:
#     print(f"Parameters: n_estimators={n_estimators},
#           max_samples={max_samples}; "
#           f"Number of outliers: {n_outliers},
#             Number of inliers: {n_inliers}")

# Create a dataframe from the results and params lists
iso_output = pd.DataFrame({'n_estimators': [n_estimators for _, _,
                                            n_estimators, _ in results],
                           'max_samples': [max_samples for _, _, _,
                                           max_samples in results],
                           'n_outliers': [n_outliers for n_outliers, _,  _,
                                          _ in results],
                           'n_inliers': [n_inliers for _, n_inliers, _,
                                         _ in results], })

# Create a new csv file form data frame
iso_output.to_csv('iso_output.csv', index=False)

# Print the time the process took
minutes = int((time.time() - starttime) / 60)
seconds = int((time.time() - starttime) % 60)
print(f'Isolation Forest grid search process took {minutes} minutes and {seconds} seconds')

# %% Evaluate grid search of isolation forest by plotting the results
# Plot the results of the grid search
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='n_estimators', y='n_outliers', hue='max_samples',
             data=iso_output, ax=ax)
ax.set_title('Number of outliers for different parameter combinations')
ax.set_xlabel('n_estimators')
ax.set_ylabel('Number of outliers')
plt.show()

# %% Plot the results of the grid search using matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
for max_samples in max_samples_list:
    ax.plot(iso_output[iso_output['max_samples'] == max_samples]['n_estimators'],
            iso_output[iso_output['max_samples'] == max_samples]['n_outliers'],
            label=f'max_samples={max_samples}')
ax.set_title('Number of outliers for different parameter combinations')
ax.set_xlabel('n_estimators')
ax.set_ylabel('Number of outliers')
ax.legend()
plt.show()
