import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Compare HDBSCAN with DBSCAN
# -----------------------------------------------------------------------------
# %% Check whether hdbscan_noise points are in dbscan_noise based on index

# assuming both dataframes have columns called 'index'
merged = pd.merge(dbscan_noise, hdbscan_noise, on='INDEX', how='inner')
matching_obs = merged['INDEX'].tolist()

# Calculate number of noise points from hdbscan_noise that are in dbscan_noise
print(len(matching_obs),
      f'of {len(dbscan_noise)} noise points in total are similar')
print('Percentage: ',
      round(len(matching_obs)/len(dbscan_noise)*100, 2), '%')
 
# assuming both dataframes have columns called 'index'
merged_outer = pd.merge(dbscan_noise, hdbscan_noise, on='INDEX',
                        how='outer', indicator=True)

# get rows that are only in one of the dataframes
not_in_both = merged_outer.loc[merged_outer['_merge'].isin(['left_only',
                                                            'right_only'])]

# Calculate number of noise points from left_only and right_only
left_only = not_in_both.loc[not_in_both['_merge'].isin(['left_only'])]
right_only = not_in_both.loc[not_in_both['_merge'].isin(['right_only'])]
print('Only in dbscan_noise: ', len(left_only))
print('Only in hdbscan_noise: ', len(right_only))

# -----------------------------------------------------------------------------
# Compare DBSCAN with IF
# -----------------------------------------------------------------------------
# %%
# assuming both dataframes have columns called 'index'
merged = pd.merge(dbscan_noise, if_noise, on='INDEX', how='inner')
matching_obs = merged['INDEX'].tolist()

# Calculate number of noise points from dbscan_noise that are in if_noise
print(len(matching_obs),
      f'of {len(dbscan_noise)} noise points in total are similar')
print('Percentage: ',
      round(len(matching_obs)/len(dbscan_noise)*100, 2), '%')

# %% Check wheter dbscan_noise points are in if_noise based on index
# assuming both dataframes have columns called 'index'
merged_outer = pd.merge(dbscan_noise, if_noise, on='INDEX',
                        how='outer', indicator=True)

# get rows that are only in one of the dataframes
not_in_both = merged_outer.loc[merged_outer['_merge'].isin(
    ['left_only', 'right_only'])]

# Calculate number of noise points from left_only and right_only
left_only = not_in_both.loc[not_in_both['_merge'].isin(['left_only'])]
right_only = not_in_both.loc[not_in_both['_merge'].isin(['right_only'])]
print('Only in dbscan_noise: ', len(left_only))
print('Only in if_noise: ', len(right_only))

# -----------------------------------------------------------------------------
# Compare HDBSCAN with IF
# -----------------------------------------------------------------------------
# %%
# assuming both dataframes have columns called 'index'
merged = pd.merge(hdbscan_noise, if_noise, on='INDEX', how='inner')
matching_obs = merged['INDEX'].tolist()

# Calculate number of noise points from dbscan_noise that are in if_noise
print(len(matching_obs),
      f'of {len(if_noise)} noise points in total are similar')
print('Percentage: ',
      round(len(matching_obs)/len(if_noise)*100, 2), '%')

# %% Check wheter dbscan_noise points are in if_noise based on index
# assuming both dataframes have columns called 'index'
merged_outer = pd.merge(hdbscan_noise, if_noise, on='INDEX',
                        how='outer', indicator=True)

# get rows that are only in one of the dataframes
not_in_both = merged_outer.loc[merged_outer['_merge'].isin(
    ['left_only', 'right_only'])]

# Calculate number of noise points from left_only and right_only
left_only = not_in_both.loc[not_in_both['_merge'].isin(['left_only'])]
right_only = not_in_both.loc[not_in_both['_merge'].isin(['right_only'])]
print('Only in hdbscan_noise: ', len(left_only))
print('Only in if_noise: ', len(right_only))

# ------------------------------------------------------------------------------
# Generate data seen as "normal"
# ------------------------------------------------------------------------------
# %% Create a data frame with only the normal data points
# Rename the labels column
data_normal = if_output
data_normal = data_normal.rename(columns={'labels_if': 'Anomaly_if'})

# Drop scores column
data_normal = data_normal.drop(['scores'], axis=1)

# %%
# Merge the 'labels_dbscan' column from dbscan_output data_normal based on index
data_normal = pd.merge(data_normal, dbscan_output[['INDEX', 'Anomaly_dbscan']],
                       on='INDEX', how='left')

data_normal = pd.merge(data_normal, hdbscan_output[['INDEX', 'Anomaly_hdbscan']],
                       on='INDEX', how='left')

data_normal = pd.merge(data_normal, lof_output[['INDEX', 'Anomaly_lof']],
                       on='INDEX', how='left')

# %%
# Align the labels columns next to each other
data_normal = data_normal[['INDEX', 'Anomaly_dbscan', 'Anomaly_hdbscan',
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

# Count the number of true in Anomaly_dbscan & Anomaly_hdbscan & Anomaly_if
# and create a new column with the sum
data_normal['Anomaly_sum'] = data_normal[['Anomaly_dbscan',
                                          'Anomaly_hdbscan',
                                          'Anomaly_if',
                                          'Anomaly_lof']].sum(axis=1)
data_normal['Anomaly_sum'].value_counts()
# %%
# Create a data frame with a column stating if the data point is normal or not
data_normal['y'] = np.where(data_normal['Anomaly_sum'] == 0, 0, 1)
data_normal['y_1'] = np.where(data_normal['Anomaly_sum'] <= 1, 0, 1)
data_normal['y_2'] = np.where(data_normal['Anomaly_sum'] <= 2, 0, 1)
data_normal['y_3'] = np.where(data_normal['Anomaly_sum'] <= 3, 0, 1)
print(data_normal['y'].value_counts())
print(data_normal['y_1'].value_counts())
print(data_normal['y_2'].value_counts())
print(data_normal['y_3'].value_counts())

# %% Plot the correlation between the anomaly columns
# Create a correlation matrix
corr = data_normal[['Anomaly_dbscan', 'Anomaly_hdbscan', 'Anomaly_if',
                    'Anomaly_lof']].corr()
corr.style.background_gradient(cmap='coolwarm')


# %% Merge the anomaly columns into the total_payments dataframe
total_payments = pd.merge(total_payments,
                          data_normal[['INDEX', 'Anomaly_dbscan',
                                       'Anomaly_hdbscan', 'Anomaly_if',
                                       'Anomaly_lof', 'Anomaly_sum',]],
                          on='INDEX', how='left')

# %% Export the total_payments dataframe to a csv file
total_payments.to_csv('total_payments_results.csv', index=False)

# %%
# Create a scatter plot for the anomalies with color based on the 'Anomaly_sum' column
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(total_payments['Amount_Applied'], total_payments['Discount_Applied'],
                        c=data_normal['Anomaly_sum'], cmap='coolwarm')
legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Anomaly")
ax.add_artist(legend1)
plt.xlabel('Amount_Applied')
plt.ylabel('Discount_Applied')
plt.title('Anomaly detection')
plt.show()


# %%
