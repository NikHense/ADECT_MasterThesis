# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------
# Generate data seen as "normal"
# ------------------------------------------------------------------------------
# %% Create a data frame
# Rename the labels column
combined_results = if_output.copy()
combined_results = combined_results.rename(columns={'labels_if': 'Anomaly_if'})

# Drop scores column
combined_results = combined_results.drop(['scores'], axis=1)

# %%
# Merge the 'labels_dbscan' column from dbscan_output combined_results based on index
combined_results = pd.merge(combined_results,
                       dbscan_output[['INDEX', 'Anomaly_dbscan']],
                       on='INDEX',
                       how='left')

combined_results = pd.merge(combined_results,
                       hdbscan_output[['INDEX', 'Anomaly_hdbscan']],
                       on='INDEX',
                       how='left')

combined_results = pd.merge(combined_results,
                       lof_output[['INDEX', 'Anomaly_lof']],
                       on='INDEX',
                       how='left')

# %%
# Align the labels columns next to each other
combined_results = combined_results[['INDEX', 'Anomaly_dbscan', 'Anomaly_hdbscan',
                           'Anomaly_if', 'Anomaly_lof', 'Payment_Number',
                           'Gen_Jnl_Line_Number', 'Line_Number',
                           'Object_Number', 'Vendor_Number',
                           'Country_Region_Code', 'Amount_Applied',
                           'Amount_Initial', 'Discount_Applied',
                           'Discount_Allowed', 'Discount_Rate',
                           'Payment_Method_Code',
                           'Customer_IBAN',
                           'Vendor_IBAN_BIC',
                           'Vendor_Bank_Origin', 'Posting_Date',
                           'Due_Date', 'Created_By',
                           'Source_System', 'Mandant']]

# # Count the number of true in Anomaly_dbscan & Anomaly_hdbscan & Anomaly_if
# # and create a new column with the sum
# combined_results['Anomaly_sum'] = combined_results[['Anomaly_dbscan',
#                                           'Anomaly_hdbscan',
#                                           'Anomaly_if',
#                                           'Anomaly_lof']].sum(axis=1)
# combined_results['Anomaly_sum'].value_counts()
# # %%
# # Create a data frame with a column stating if the data point is normal or not
# combined_results['y'] = np.where(combined_results['Anomaly_sum'] == 0, 0, 1)
# combined_results['y_1'] = np.where(combined_results['Anomaly_sum'] <= 1, 0, 1)
# combined_results['y_2'] = np.where(combined_results['Anomaly_sum'] <= 2, 0, 1)
# combined_results['y_3'] = np.where(combined_results['Anomaly_sum'] <= 3, 0, 1)
# print(combined_results['y'].value_counts())
# print(combined_results['y_1'].value_counts())
# print(combined_results['y_2'].value_counts())
# print(combined_results['y_3'].value_counts())

# # %% Plot the correlation between the anomaly columns
# # Create a correlation matrix
# corr = combined_results[['Anomaly_dbscan', 'Anomaly_hdbscan', 'Anomaly_if',
#                     'Anomaly_lof']].corr()
# corr.style.background_gradient(cmap='coolwarm')


# %% Merge the anomaly columns into the total_payments_academic dataframe
total_payments_academic.insert(0, 'INDEX', range(0, len(total_payments_academic)))
total_payments_academic = pd.merge(total_payments_academic,
                          combined_results[['INDEX', 'Anomaly_dbscan',
                                       'Anomaly_hdbscan', 'Anomaly_if',
                                       'Anomaly_lof',
                                       'Anomaly_sum',]],
                          on='INDEX', how='left')

# # %% Export the total_payments_academic dataframe to a csv file
# total_payments_academic.to_csv('total_payments_academic_results.csv',
#                                index=False)

# %%


# %%
