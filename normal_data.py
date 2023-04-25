#------------------------------------------------------------------------------
# Generate data seen as "normal"
#------------------------------------------------------------------------------
# %% Create a data frame with only the normal data points
# Rename the labels column
data_normal = if_output
data_normal = data_normal.rename(columns={'labels_if': 'Anomaly_if'})

# Drop scores column
data_normal = data_normal.drop(['scores'], axis=1)

# %%
#Merge the 'labels_dbscan' column from dbscan_output data_normal based on index
data_normal = pd.merge(data_normal, dbscan_output[['INDEX', 'Anomaly_dbscan']],
                        on='INDEX', how='left')

data_normal = pd.merge(data_normal, hdbscan_output[['INDEX', 'Anomaly_hdbscan']],
                        on='INDEX', how='left')

# %%
# Align the labels columns next to each other
data_normal = data_normal[['INDEX', 'Anomaly_dbscan', 'Anomaly_hdbscan',
                           'Anomaly_if', 'Payment_Number_encoded',
                           'Gen_Jnl_Line_Number', 'Line_Number',
                           'Object_Number_encoded', 'Vendor_Number',
                           'Country_Region_Code_encoded', 'Amount_Applied',
                           'Amount_Initial', 'Discount_Applied',
                           'Discount_Allowed','Discount_Rate',
                           'Payment_Method_Code_encoded',
                           'Customer_IBAN_encoded',
                           'Vendor_IBAN_BIC_encoded', 
                           'Vendor_Bank_Origin_encoded', 'Posting_Date',
                           'Created_By_encoded',
                           'Source_System_encoded', 'Mandant_encoded']]



# Count the number of true in Anomaly_dbscan & Anomaly_hdbscan & Anomaly_if
# and create a new column with the sum
data_normal['Anomaly_sum'] = data_normal[['Anomaly_dbscan',
                                          'Anomaly_hdbscan',
                                          'Anomaly_if']].sum(axis=1)
data_normal['Anomaly_sum'].value_counts()
# %%
# Create a data frame with a column stating if the data point is normal or not
data_normal['y'] = np.where(data_normal['Anomaly_sum'] == 0, 0, 1)
data_normal['y_1'] = np.where(data_normal['Anomaly_sum'] <= 1, 0, 1)
data_normal['y_2'] = np.where(data_normal['Anomaly_sum'] <= 2, 0, 1)
print(data_normal['y'].value_counts())
print(data_normal['y_1'].value_counts())
print(data_normal['y_2'].value_counts())

# %%
