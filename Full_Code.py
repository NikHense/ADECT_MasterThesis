# %% Import libraries
# import os
import time
import pandas as pd
import numpy as np
from numpy import bincount
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
# from sqlalchemy import create_engine, text
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import run_diagnostic
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
# from sklearn import metrics
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from kneed import KneeLocator
from multiprocessing import Pool
from statistics import mean
import hdbscan
# from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# from anonymizedf.anonymizedf import anonymize

# %%
# ---------------------------------------------------------------------------
# Import Data
# ---------------------------------------------------------------------------
# # %% Params SQL connection
# SERVER = 'P-SQLDWH'  # os.environ.get('SERVER')
# DB = 'ML'
# USERNAME = os.environ.get('USERNAME')
# PASSWORD = os.environ.get('PASSWORD')

# # %% SQL Connection
# # Be aware that this connection is only possible with the
# # respective Usernames and Passwords
# conn = ("Driver={ODBC Driver 18 for SQL Server};"
#         "Server="+SERVER+";"
#         "Database="+DB+";"
#         "UID="+USERNAME+";"
#         "PWD="+PASSWORD+";"
#         "Encrypt=YES;"
#         "TrustServerCertificate=YES")
# engine = create_engine(
#     f'mssql+pyodbc://?odbc_connect={conn}',
#     fast_executemany=True)

# SQL_TOTAL_PAYMENTS = 'SELECT * FROM ADECT.TOTAL_PAYMENTS'
# total_payments = pd.DataFrame(engine.connect().execute(
#                               text(SQL_TOTAL_PAYMENTS)))
# # %% Print out a LaTex table with the column information
# # # Create a dictionary with the column information
# # column_info = {
# #     'Column Number': list(range(len(total_payments.columns))),
# #     'Feature Name': total_payments.columns,
# #     'Data Type': total_payments.dtypes.values.astype(str)
# #     }


# # # Create a new DataFrame with the column information
# # column_df = pd.DataFrame(column_info)

# # # Generate the LaTeX table
# # latex_table = column_df.to_latex(index=False)

# # # Replace 'object' with 'textual' in the LaTeX table
# # latex_table = latex_table.replace('object', 'textual')

# # # Replace 'int64' with 'numerical' in the LaTeX table
# # latex_table = latex_table.replace('int64', 'numerical')

# # # Replace 'datetime64[ns]' with 'Date' in the LaTeX table
# # latex_table = latex_table.replace('datetime64[ns]', 'date')

# # # Print the LaTeX table
# # print(latex_table)

# # %% Import Fraudulent Invoices
# # Define the data types for each feature
# dtypes_fraud = {
#     'Object_Number': str
# }

# # Import Fraudulent Invoices csv file
# fraud_invoices = pd.read_csv('Fraud_Invoices.csv',
#                              dtype=dtypes_fraud,
#                              na_values='NA',
#                              sep=';')

# # Replace the ',' with '.' in the following columns
# fraud_invoices['Amount_Applied'] = fraud_invoices['Amount_Applied'].str.replace(',', '.')
# fraud_invoices['Amount_Initial'] = fraud_invoices['Amount_Initial'].str.replace(',', '.')
# fraud_invoices['Discount_Applied'] = fraud_invoices['Discount_Applied'].str.replace(',', '.')
# fraud_invoices['Discount_Rate'] = fraud_invoices['Discount_Rate'].str.replace(',', '.')


# # Transform data types to float
# fraud_invoices['Amount_Applied'] = fraud_invoices['Amount_Applied'].astype(
#     float)
# fraud_invoices['Amount_Initial'] = fraud_invoices['Amount_Initial'].astype(
#     float)
# fraud_invoices['Discount_Applied'] = fraud_invoices['Discount_Applied'].astype(
#     float)
# fraud_invoices['Discount_Rate'] = fraud_invoices['Discount_Rate'].astype(float)

# # Replace ' ' with NaN
# fraud_invoices.replace(' ', np.nan, inplace=True)

# fraud_invoices.info()


# # %% Add fraud_invoices to total_payments
# # Add fraud_invoices to total_payments
# total_payments = total_payments.append(fraud_invoices, ignore_index=True)

                          
# # %% Anonymize the data
# # Prepare the data to be anonymized
# an = anonymize(total_payments)

# # Select what data you want to anonymize and your preferred style
# an.fake_categories('Created_By')
# an.fake_ids('Vendor_IBAN')
# an.fake_ids('Customer_IBAN')

# # # Save the final dataset as a csv file (test purposes)
# # total_payments.to_csv('total_payments_fakes.csv',
# #                                sep=";", index=False)

# # %% Split up fraud_invoices from total_payments
# # Create two independent data sets
# fraud_invoices = total_payments[total_payments['Fraud'] == 1]
# total_payments = total_payments[total_payments['Fraud'] != 1]

# # Delete all columns that contain only NaN values
# fraud_invoices = fraud_invoices.dropna(axis=1, how='all')

# # Delete 'Vendor_IBAN'  and 'Created_By' columns
# fraud_invoices = fraud_invoices.drop(['Vendor_IBAN', 'Created_By',
#                                       'Posting_Description_1',
#                                       'Document_Number_external',
#                                       'Document_Number_internal',
#                                       'Customer_IBAN'], axis=1)
# total_payments = total_payments.drop(['Vendor_IBAN', 'Created_By',
#                                       'Customer_IBAN'], axis=1)

# # Rename 'Fake_Vendor_IBAN' to 'Vendor_IBAN'
# fraud_invoices = fraud_invoices.rename(columns={'Fake_Vendor_IBAN': 'Vendor_IBAN'})
# total_payments = total_payments.rename(columns={'Fake_Vendor_IBAN': 'Vendor_IBAN'})

# # Rename 'Fake_Created_By' to 'Created_By'
# fraud_invoices = fraud_invoices.rename(columns={'Fake_Created_By': 'Created_By'})
# total_payments = total_payments.rename(columns={'Fake_Created_By': 'Created_By'})

# # Rename 'Fake_Customer_IBAN' to 'Customer_IBAN'
# fraud_invoices = fraud_invoices.rename(columns={'Fake_Customer_IBAN': 'Customer_IBAN'})
# total_payments = total_payments.rename(columns={'Fake_Customer_IBAN': 'Customer_IBAN'})

# # %% Due to confidentiality reasons, the data is not available as it is
# # and has to be cut down
# # Select specific columns, resulting in final dataset for analysis
# total_payments_academic = total_payments[['Payment_Number',
#                                           'Gen_Jnl_Line_Number',
#                                           'Line_Number',
#                                           'Object_Number',
#                                           'Vendor_Number',
#                                           'Country_Region_Code',
#                                           'Amount_Applied',
#                                           'Amount_Initial',
#                                           'Discount_Applied',
#                                           'Discount_Allowed',
#                                           'Discount_Rate',
#                                           'Payment_Method_Code',
#                                           'Customer_IBAN',
#                                           'Vendor_IBAN',
#                                           'Vendor_BIC',
#                                           'Vendor_Bank_Origin',
#                                           'Posting_Date',
#                                           'Due_Date',
#                                           'Created_By',
#                                           'Source_System',
#                                           'Mandant',
#                                           'Review_Status'
#                                           ]]

# # %% 
# # Save the final datasets as a csv file
# total_payments_academic.to_csv('total_payments_academic.csv',
#                                sep=";", index=False)
# fraud_invoices.to_csv('Fraud_Invoices_final.csv',
#                       sep=";", index=False)

# ---------------------------------------------------------------------------
# Load final dataset ready for study
# ---------------------------------------------------------------------------
# %% Load the final dataset
# Define the data types for each feature as they were in the original dataset

# When improrting a csv file, VS Code automatically converts the
# data types which is not wanted
dtypes_totpaym = {
    'Payment_Number': str,
    'Object_Number': str,
    'Vendor_Number': str,
    'Country_Region_Code': str,
    'Payment_Method_Code': str,
    'Customer_IBAN': str,
    'Vendor_IBAN': str,
    'Vendor_BIC': str,
    'Vendor_Bank_Origin': str,
    'Created_By': str,
    'Source_System': str,
    'Mandant': str,
    'Review_Status': str,
    'Gen_Jnl_Line_Number': int,
    'Line_Number': int,
    'Discount_Allowed': int,
    'Amount_Applied': float,
    'Amount_Initial': float,
    'Discount_Applied': float,
    'Discount_Rate': float,
}

# Specify the columns to be parsed as datetime
date_columns = ['Posting_Date', 'Due_Date']

# Import the final dataset with the correct data types
total_payments_academic = pd.read_csv('Input_Data/total_payments_academic.csv',
                                      dtype=dtypes_totpaym,
                                      parse_dates=date_columns,
                                      na_values='NA',
                                      sep=';')
# ----------------------------------------------------------------------------------

# %% Import Fraudulent Invoices
# Define the data types for each feature
dtypes_fraud = {
    'Object_Number': str
}

# Import Fraudulent Invoices csv file
fraud_invoices = pd.read_csv('Input_Data/Fraud_Invoices_final.csv',
                             dtype=dtypes_fraud,
                             na_values='NA',
                             sep=';')

# # Replace the ',' with '.' in the following columns
# fraud_invoices['Amount_Applied'] = fraud_invoices['Amount_Applied'].str.replace(',', '.')
# fraud_invoices['Amount_Initial'] = fraud_invoices['Amount_Initial'].str.replace(',', '.')
# fraud_invoices['Discount_Applied'] = fraud_invoices['Discount_Applied'].str.replace(',', '.')
# fraud_invoices['Discount_Rate'] = fraud_invoices['Discount_Rate'].str.replace(',', '.')

# Transform data types to float
fraud_invoices['Amount_Applied'] = fraud_invoices['Amount_Applied'].astype(
    float)
fraud_invoices['Amount_Initial'] = fraud_invoices['Amount_Initial'].astype(
    float)
fraud_invoices['Discount_Applied'] = fraud_invoices['Discount_Applied'].astype(
    float)
fraud_invoices['Discount_Rate'] = fraud_invoices['Discount_Rate'].astype(float)

# Replace ' ' with NaN
fraud_invoices.replace(' ', np.nan, inplace=True)

fraud_invoices.info()

# -----------------------------------------------------------------------------
# Generate synthetic data based on fraud_invoices
# -----------------------------------------------------------------------------
# %% Prepare Metadata for generating synthetic data

metadata = SingleTableMetadata()

metadata.detect_from_dataframe(data=fraud_invoices)
metadata.validate()

print(metadata.to_dict())
print(metadata.primary_key)
# ---------------------------------------------------------------------------

# %% Generate synthetic data (Gaussian Copula Synthesizer) 71% Quality

# Step 1: Load Synthesizer
synthesizer = GaussianCopulaSynthesizer.load(filepath='Synthesizer_Models/Gaussian_Copula_Synthesizer.pkl')

# Step 2: Generate synthetic data (0.5% of the original data)
synthetic_data = synthesizer.sample(num_rows=int(len(total_payments_academic) *
                                                 0.005),
                                    batch_size=100)

# Step 3: Evaluate the quality of the synthetic data
quality_report = evaluate_quality(real_data=fraud_invoices,
                                  synthetic_data=synthetic_data,
                                  metadata=metadata)

# Visualize the quality report
# quality_report.get_visualization(property_name='Column Shapes')
# quality_report.get_visualization(property_name='Column Pair Trends')

# Run a diagnostic on the synthetic data
diagnostic_report = run_diagnostic(real_data=fraud_invoices,
                                   synthetic_data=synthetic_data,
                                   metadata=metadata)

# Change values in 'Fraud' of sybthetic_data to 2
synthetic_data['Fraud'] = 2
# ---------------------------------------------------------------------------

# %% Generate synthetic data (CTGAN Synthesizer) 83% Quality

# Step 1: Load the synthesizer
synthesizer1 = CTGANSynthesizer.load(filepath='Synthesizer_Models/CTGAN_Synthesizer.pkl')

# Step 2: Generate synthetic data (0.5% of the original data)
synthetic_data1 = synthesizer1.sample(num_rows=int(
                                      len(total_payments_academic) * 0.005),
                                      batch_size=100)

# Step 3: Evaluate the quality of the synthetic data
quality_report = evaluate_quality(real_data=fraud_invoices,
                                  synthetic_data=synthetic_data1,
                                  metadata=metadata)

# Visualize the quality report
# quality_report.get_visualization(property_name='Column Shapes')
# quality_report.get_visualization(property_name='Column Pair Trends')

# Run a diagnostic on the synthetic data
diagnostic_report = run_diagnostic(real_data=fraud_invoices,
                                   synthetic_data=synthetic_data1,
                                   metadata=metadata)
# Change values in 'Fraud' of sybthetic_data1 to 2
synthetic_data1['Fraud'] = 2

# ---------------------------------------------------------------------------

# %% Generate synthetic data (TVAE Synthesizer) 87% Quality

# Step 1: Load the synthesizer
synthesizer2 = TVAESynthesizer.load(filepath='Synthesizer_Models/TVAE_Synthesizer.pkl')

# Step 2: Generate synthetic data (0.5% of the original data)
synthetic_data2 = synthesizer2.sample(num_rows=int(
                                      len(total_payments_academic) * 0.005),
                                      batch_size=100)

# Step 3: Evaluate the quality of the synthetic data
quality_report = evaluate_quality(real_data=fraud_invoices,
                                  synthetic_data=synthetic_data2,
                                  metadata=metadata)

# Visualize the quality report
# quality_report.get_visualization(property_name='Column Shapes')
# quality_report.get_visualization(property_name='Column Pair Trends')

# Run a diagnostic on the synthetic data
diagnostic_report = run_diagnostic(real_data=fraud_invoices,
                                   synthetic_data=synthetic_data2,
                                   metadata=metadata)

# Change values in 'Fraud' of sybthetic_data2 to 2
synthetic_data2['Fraud'] = 2


# ---------------------------------------------------------------------------

# %% Generate synthetic data (CopulaGAN Synthesizer) 85% Quality

# Step 1: Load the synthesizer
synthesizer3 = CopulaGANSynthesizer.load(filepath='Synthesizer_Models/CopulaGAN_Synthesizer.pkl')

# Step 2: Generate synthetic data (0.5% of the original data)
synthetic_data3 = synthesizer3.sample(num_rows=int(
                                      len(total_payments_academic)*0.005),
                                      batch_size=100)

# Step 3: Evaluate the quality of the synthetic data
quality_report = evaluate_quality(real_data=fraud_invoices,
                                  synthetic_data=synthetic_data3,
                                  metadata=metadata)

# Visualize the quality report
# quality_report.get_visualization(property_name='Column Shapes')
# quality_report.get_visualization(property_name='Column Pair Trends')

# Run a diagnostic on the synthetic data
diagnostic_report = run_diagnostic(real_data=fraud_invoices,
                                   synthetic_data=synthetic_data3,
                                   metadata=metadata)

# Change values in 'Fraud' of sybthetic_data3 to 2
synthetic_data3['Fraud'] = 2
# ---------------------------------------------------------------------------

# %% Include the synthetic_data to the fraud_invoices
fraud_invoices1 = pd.concat([fraud_invoices, synthetic_data],
                            ignore_index=True)
fraud_invoices2 = pd.concat([fraud_invoices, synthetic_data1],
                            ignore_index=True)
fraud_invoices3 = pd.concat([fraud_invoices, synthetic_data2],
                            ignore_index=True)  #  highest Overall Quality score
fraud_invoices4 = pd.concat([fraud_invoices, synthetic_data3],
                            ignore_index=True)

# ---------------------------------------------------------------------------
# Data Preprocessing & Feature Engineering
# ---------------------------------------------------------------------------
# %% Add entries of fraudulent invoices to total_payments
total_payments_academic = pd.concat([total_payments_academic,
                                     fraud_invoices3], ignore_index=True)

# Transforn nan of 'Fraud' column to 0 and 'Review_Status' to 'synthetic'
total_payments_academic['Fraud'] = total_payments_academic['Fraud'].fillna(0)
total_payments_academic['Review_Status'] = total_payments_academic['Review_Status'].fillna('synthetic')

total_payments_academic.info()

# %% Define new data frame for data cleaning and preparation
input = total_payments_academic.copy()

# %% Combining Vendor_IBAN and Vendor_BIC
input['Vendor_IBAN'] = input['Vendor_IBAN'].astype(str)
input['Vendor_BIC'] = input['Vendor_BIC'].astype(str)
input['Vendor_IBAN_BIC'] = input['Vendor_IBAN'] + input['Vendor_BIC']

# %% Change data type of specific columns

# Convert data type to datetime
date_cols = ['Posting_Date', 'Due_Date']
for col in date_cols:
    input[col] = pd.to_datetime(input[col])

# Specify the observation period ending on 10.05.2023
input = input[(input['Posting_Date'] <= '2023-05-10')]

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

# %% Select columns for encoding and sequential input into models
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

# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

# # %% Create the inital isolation forest function for test reasons
# starttime = time.time()
# isof = IsolationForest(n_estimators=1000, max_samples=10000,
#                        contamination='auto', random_state=42,
#                        verbose=0, n_jobs=-1)

# isof.fit(input)

# scores1 = isof.decision_function(input)

# anomaly1 = isof.predict(input)

# # print(y_pred)
# #  Results of the isolation forest
# n_outliers = bincount((anomaly1 == -1).astype(int))[1]
# n_inliers = bincount((anomaly1 == 1).astype(int))[1]

# print("Number of outliers: ", n_outliers)
# print("Number of inliers: ", n_inliers)
# print(f'Database read process took {time.time() - starttime} seconds')
# # %% add the scores1 and anomaly1 values to the respective rows
# # in the data frame to position 0
# input.insert(0, "scores1", scores1, True)
# input.insert(1, "anomaly1", anomaly1, True)

# %% Execute grid search for isolation forest (!!! 12 min running time !!!)
starttime = time.time()

# Define list of parameter values to test
n_estimators_list = list(range(100, 1002, 100))
max_samples_list = list(range(1000, len(input), 1000))

# Initialize empty lists to store results
results = []
params = []


# Define a function to fit the IsolationForest model and compute the results
def fit_isof(params):
    n_estimators, max_samples = params
    # starttime1 = time.time()
    isof = IsolationForest(n_estimators=n_estimators,
                           max_samples=max_samples,
                           contamination='auto',
                           random_state=42,
                           verbose=1, n_jobs=1)
    isof.fit(input)
    # scores = isof.decision_function(input)
    anomaly = isof.predict(input)
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

# # Create a new csv file form data frame
# iso_output.to_csv('iso_output.csv', index=False)

# Print the time the process took
minutes = int((time.time() - starttime) / 60)
seconds = int((time.time() - starttime) % 60)
print(f'Isolation Forest grid search process took '
      f'{minutes} minutes and {seconds} seconds')

# %% Evaluate grid search of isolation forest by plotting the results
# Plot the results of the grid search in 2D
sns.set_style('whitegrid')
sns.relplot(x='max_samples', y='n_outliers', hue='n_estimators',
            legend='full', data=iso_output)
#plt.title('Number of outliers for different parameter combinations')
plt.ylabel('Number of anomalies')
plt.xlabel('Number of max_samples')
# plt.xlim(0, 40000)
# plt.ylim(1000, 4000)
plt.show()

# # Plot the results of the grid search in 3D
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(iso_output['n_estimators'], iso_output['max_samples'],
#            iso_output['n_outliers'], c=iso_output['n_outliers'],
#            cmap='viridis', linewidth=0.5)
# ax.set_xlabel('n_estimators')
# ax.set_ylabel('max_samples')
# ax.set_zlabel('n_outliers')
# ax.set_title('Number of outliers for different parameter combinations')
# plt.show()

# %% Plot average number of outliers per max_samples
iso_output.groupby('max_samples')['n_outliers'].mean().plot()
#plt.title('Average number of anomalies per max_samples')
plt.ylabel('Number of anomalies')
plt.xlabel('Number of max_samples')
plt.show()

# %% Calculate the maximum curvature point of the average number of outliers
# Print the list of average number of outliers per max_samples
avg_outlier = iso_output.groupby('max_samples')['n_outliers'].mean()
avg_outlier = np.array(avg_outlier)

# Create array that goes from 1000 to the length of avg_outlier in 1000 steps
x = max_samples_list

# %% Calculate the maximum curvature point of the average number of outliers
kneedle = KneeLocator(x, avg_outlier,
                      interp_method='polynomial',
                      #  smoothen the lines, otherwise bumpy
                      curve='convex', direction='decreasing')

print(round(kneedle.elbow, 0))
print(round(kneedle.elbow_y, 3))

plt.style.use('ggplot')
kneedle.plot_knee_normalized()
# plt.xlim(0, 0.05)
# plt.ylim(0.95, 1)
plt.show()

# Plot the k-distance graph with the knee point (zoomed in)
# plt.figure(figsize=(10, 10))
# plt.plot(x, avg_outlier, 'ro-', linewidth=2)
# plt.axvline(x=kneedle.elbow, color='b', linestyle='--')
# plt.axhline(y=kneedle.elbow_y, color='b', linestyle='--')
# plt.text(kneedle.elbow + 500, kneedle.elbow_y + 200,
#          f'max_samples ({round(kneedle.elbow, 0)})',
#          fontsize=13)
# plt.title('Average number of outliers per max_samples')
# plt.xlabel('Number of max_samples')
# plt.ylabel('Average number of outliers')
# plt.xlim(kneedle.elbow - 10000, kneedle.elbow + 10000)
# plt.ylim(kneedle.elbow_y - 500, kneedle.elbow_y + 1000)
# plt.show()

plt.style.use('ggplot')
kneedle.plot_knee()
plt.xlabel('Number of max_samples')
plt.ylabel('Number of outliers')
plt.title('')
plt.text(kneedle.elbow + 1500, kneedle.elbow_y + 200,
         f'max_samples ({round(kneedle.elbow, 0)})',
         fontsize=11)
plt.show()


# %% Create the isolation forest with tuned parameter
starttime = time.time()

avg_n_estimator = mean(n_estimators_list)

isof = IsolationForest(n_estimators=avg_n_estimator,
                       max_samples=int(kneedle.elbow),
                       contamination='auto',
                       random_state=42,
                       verbose=0,
                       n_jobs=-1)

isof.fit(input)

if_scores = isof.decision_function(input)

if_anomaly = isof.predict(input)

#  Results of the isolation forest
n_outliers = bincount((if_anomaly == -1).astype(int))[1]
n_inliers = bincount((if_anomaly == 1).astype(int))[1]

print("Number of outliers: ", n_outliers)
print("Number of inliers: ", n_inliers)
print(f'Database read process took {time.time() - starttime} seconds')

# -----------------------------------------------------------------------------
# Generate Output of IF
# -----------------------------------------------------------------------------
# %% Create a data frame with the results of the isolation forest
# Convert the if_output array to a pandas DataFrame
if_output = input.copy()

# Add the labels column to the if_output at position 0
if_output.insert(0, 'INDEX', total_payments_academic.index)
if_output.insert(1, 'labels_if', if_anomaly, True)
if_output.insert(2, "scores", if_scores, True)

# Transform the labels column to a boolean column (1 = False, -1 = True)
if_output['labels_if'] = if_output['labels_if'].apply(lambda x: True if x == -1
                                                      else False)

# Filter out the data frame with only noise points & clean
if_noise = if_output[if_output['labels_if']]

# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------
# %% Scaling the input data

scaler = StandardScaler()
input_scaled = scaler.fit_transform(input)
input_scaled = pd.DataFrame(input_scaled, columns=input.columns)

input_scaled.mean()
input_scaled.var()

# %% Test out the PCA and optimal number of principal components

# Perform a PCA on input_scaled
pca = PCA(n_components=len(input_scaled.columns))
input_pca = pca.fit_transform(input_scaled)

# # # Method 1: Scree plot
# # # Plot the eigenvalues of the principal components
# # plt.plot(range(1, pca.n_components_ + 1),
# #          pca.explained_variance_ratio_, 'ro-', linewidth=2)
# # plt.title('Scree Plot')
# # plt.xlabel('Principal Component')
# # plt.ylabel('Eigenvalue')
# # plt.show()

# Method 2: Cumulative proportion of variance explained
# Calculate the cum. proportion of variance explained
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot the cum. proportion of variance explained, with the 99% threshold
num_components = range(1, len(input_scaled.columns) + 1)
plt.style.use('ggplot')
plt.plot(num_components, cumulative_variance_ratio, 'ro-', linewidth=2)
plt.axhline(y=0.99, color='b', linestyle='--')
plt.title('Cumulative Proportion of Variance Explained')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Proportion of Variance Explained')

# Find the index of the element in y_values that is closest to 0.99
threshold_idx = (np.abs(cumulative_variance_ratio - 0.991)).argmin()

# Get the x-coordinate of the threshold (-1 becasue of rounding effect)
threshold_x = num_components[threshold_idx]-1

# Add a vertical line at the threshold x-coordinate
plt.axvline(x=threshold_x, color='b', linestyle='--')
plt.title('')
plt.show()

# retrieve the number of components that explain 99% of the variance
n_components = np.argmax(cumulative_variance_ratio >= 0.99)

print(f'{n_components} principal components explain 99% of the variance')

# if n_components > 15 than 15 otherwise
# n_components = np.argmax(cumulative_variance_ratio >= 0.99)
if n_components > 15:
    n_components = 15
else:
    n_components = np.argmax(cumulative_variance_ratio >= 0.99)


# %% Calculate the K-Distance Graph

# Perform a PCA on input_scaled
pca = PCA(n_components=n_components)
input_pca = pca.fit_transform(input_scaled)

# Calculate the distance between each point and its kth nearest neighbor
neigh = NearestNeighbors(n_neighbors=n_components)
nbrs = neigh.fit(input_pca)
distances, indices = nbrs.kneighbors(input_pca)

# Plot the k-distance graph
# Sort the distances of each point to its kth nearest neighbor in descending order
distances = np.sort(distances, axis=0)
distances = distances[:,-1]
distances = distances[::-1]

plt.figure(figsize=(10, 10))
plt.plot(distances, linewidth=2)
plt.title('K-Distance Graph')
plt.xlabel('Data Point sorted by distance')
plt.ylabel('Epsilon (distance to kth nearest neighbor)')
plt.show()

# # Zoom in on the elbow of the graph
# plt.figure(figsize=(20, 10))
# plt.plot(distances, 'ro-', linewidth=2)
# plt.title('K-Distance Graph')
# plt.xlabel('Data Point sorted by distance')
# plt.ylabel('Epsilon (distance to kth nearest neighbor)')
# plt.xlim(45000, 53000)
# plt.ylim(0, 25)
# plt.show()

# Inlcude index to the number of observation to array
distances = np.column_stack((np.arange(0, len(distances)), distances))

# %% Calculate the maximum curvature point of the k-distance graph
kneedle = KneeLocator(distances[:, 0], distances[:, 1],
                      S=5,
                      #   interp_method='polynomial',
                      curve='convex', direction='decreasing')

print(round(kneedle.knee, 0))
print(round(kneedle.elbow_y, 3))

# Normalized data, normalized knee, and normalized distance curve.
plt.style.use('ggplot')
kneedle.plot_knee_normalized()
# plt.xlim(0, 0.1)
# plt.ylim(0.9, 1)
plt.show()

plt.style.use('ggplot')
kneedle.plot_knee()
plt.text(kneedle.elbow + 2000, kneedle.elbow_y + 1,
         f'epsilon ({round(kneedle.elbow_y, 3)})',
         fontsize=11)
plt.xlabel('Data points sorted by distance')
plt.ylabel('Epsilon (distance to kth nearest neighbor)')
plt.title('')
plt.show()

# %% Run DBSCAN
starttime = time.time()

# Define optimal epsilon value
eps = round(kneedle.elbow_y, 3)

# Define the optimal min_samples value (acc. Sander's 1998)
min_samples = (2*n_components-1)

# Define the optimal min_samples value from the max Calinski-Harabasz score
# min_samples = results_chScore.iloc[results_chScore
#                                    ['CH_score'].idxmax()]['min_samples']

# Apply DBSCAN to cluster the data
y_pred = DBSCAN(eps=eps,
                min_samples=int(min_samples),
                n_jobs=-1)

y_pred.fit(input_pca)

core_samples_mask = np.zeros_like(y_pred.labels_, dtype=bool)
core_samples_mask[y_pred.core_sample_indices_] = True
labels_dbscan = y_pred.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_ = list(labels_dbscan).count(-1)

print('eps: %0.3f' % eps)
print('min_samples: %d' % min_samples)
# # print the Calinski-Harabasz score
# print("Calinski-Harabasz Score: %0.4f"
#       % metrics.calinski_harabasz_score(input_pca, labels_dbscan))
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

print(f'DBSCAN process took {time.time() - starttime} seconds')

# -----------------------------------------------------------------------------
# Generate Output of DBSCAN
# -----------------------------------------------------------------------------
# %%
# Invert the scaling applied by StandardScaler
dbscan_output = scaler.inverse_transform(input_scaled)

# # Invert the PCA transformation
# input_pca_inverted = pca.inverse_transform(input_pca)

# Convert the dbscan_output array to a pandas DataFrame
dbscan_output = pd.DataFrame(dbscan_output, columns=input_scaled.columns)
dbscan_output['Line_Number'] = dbscan_output['Line_Number'].round(0)

# Add the labels column to the dbscan_output at position 0
dbscan_output.insert(0, 'INDEX', total_payments_academic.index)
dbscan_output.insert(1, 'labels_dbscan', labels_dbscan)
dbscan_output.insert(2, 'Anomaly_dbscan', dbscan_output['labels_dbscan'] == -1)

# Filter out the a data frame with only noise points
dbscan_noise = dbscan_output[dbscan_output['Anomaly_dbscan']]

# -----------------------------------------------------------------------------
# HDBSCAN
# -----------------------------------------------------------------------------

# %% Run HDBSCAN algorithm
# Compute the clustering using HDBSCAN
hdbscan = hdbscan.HDBSCAN(metric='euclidean',
                          min_cluster_size=10,
                          allow_single_cluster=True)
hdbscan.fit(input_pca)

# Number of clusters in labels, ignoring noise if present.
labels_hdbscan = hdbscan.labels_
n_clusters_hdbscan = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan
                                                 else 0)
n_noise_hdbscan = list(labels_hdbscan).count(-1)

print('Estimated number of clusters: %d' % n_clusters_hdbscan)
print('Estimated number of noise points: %d' % n_noise_hdbscan)

# %% Apply the outlier detection of hdbscan
hdbscan.outlier_scores_

# %% Plot the outlier scores and zoom in on the tail
sns.distplot(hdbscan.outlier_scores_[np.isfinite(hdbscan.outlier_scores_)],
             rug=True)

plt.show()

# Defined the threshold for the outlier scores at 98% quantile
threshold = pd.Series(hdbscan.outlier_scores_).quantile(0.98)
outliers_hdbscan = np.where(hdbscan.outlier_scores_ > threshold)[0]

print(f'Number of outliers: {len(outliers_hdbscan)}')

# -----------------------------------------------------------------------------
# Generate Output of HDBSCAN
# -----------------------------------------------------------------------------
# %%

# Invert the scaling applied by StandardScaler
hdbscan_output = scaler.inverse_transform(input_scaled)

# Convert the dbscan_output array to a pandas DataFrame & clean
hdbscan_output = pd.DataFrame(hdbscan_output, columns=input.columns)
hdbscan_output['Line_Number'] = hdbscan_output['Line_Number'].round(0)

# Add the labels column to the dbscan_output at position 0
hdbscan_output.insert(0, 'INDEX', total_payments_academic.index)
hdbscan_output.insert(1, 'labels_hdbscan', labels_hdbscan)
hdbscan_output.insert(2, 'Noise_hdbscan',
                      hdbscan_output['labels_hdbscan'] == -1)
hdbscan_output.insert(3, 'Anomaly_hdbscan',
                      hdbscan.outlier_scores_ > threshold)

# Filter out the a data frame with only noise points & clean
hdbscan_noise = hdbscan_output[hdbscan_output['Anomaly_hdbscan']]

# -----------------------------------------------------------------------------
# LOF
# ----------------------------------------------------------------------------- 
# %% Run LOF
lof = LocalOutlierFactor(n_neighbors=20,
                         contamination='auto',
                         n_jobs=-1)
labels_lof = lof.fit_predict(input_pca)
n_outlier = np.count_nonzero(labels_lof == -1)

print("For n =", 20,
      "n_outlier :", n_outlier)

# -----------------------------------------------------------------------------
# Generate Output of LOF
# -----------------------------------------------------------------------------
# %% 
# Invert the scaling applied by StandardScaler
lof_output = scaler.inverse_transform(input_scaled)

# Invert the PCA transformation
# input_pca_inverted = pca.inverse_transform(input_pca)

# Convert the dbscan_output array to a pandas DataFrame
lof_output = pd.DataFrame(lof_output, columns=input_scaled.columns)

# Add the labels column to the dbscan_output at position 0
lof_output.insert(0, 'INDEX', total_payments_academic.index)
lof_output.insert(1, 'labels_lof', labels_lof)
lof_output.insert(2, 'Anomaly_lof', lof_output['labels_lof'] == -1)

# Filter out the a data frame with only noise points
lof_noise = lof_output[lof_output['Anomaly_lof']]

# -----------------------------------------------------------------------------
# Combine the results of the four algorithms
# -----------------------------------------------------------------------------
# %% Create a data frame
# Rename the labels column
combined_results = if_output.copy()
combined_results = combined_results.rename(columns={'labels_if': 'Anomaly_if'})

# Drop scores column
combined_results = combined_results.drop(['scores'], axis=1)

# %% Merge the results of the algorithms
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

# %% Reorder the columns
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

# %% Merge the anomaly columns into the total_payments_academic dataframe
total_payments_academic.insert(0, 'INDEX', range(0, len(total_payments_academic)))
total_payments_academic = pd.merge(total_payments_academic,
                          combined_results[['INDEX', 'Anomaly_dbscan',
                                       'Anomaly_hdbscan', 'Anomaly_if',
                                       'Anomaly_lof']],
                          on='INDEX', how='left')

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
# %% Merge algorithm results with the ones from IF

# REGULATION FOR ANOMALIES
# If 'Anomaly_if' = 1 and 'Algorythm II' = 1, then it is an anomaly
# If 'Anomaly_if' = 0 and 'Algorythm II' = 0, then it is not an anomaly
# If 'Anomaly_if' = 0 and 'Algorythm II' = 1, then it is an anomaly
# If 'Anomaly_if' = 1 and 'Algorythm II' = 0, then it is an anomaly

# Create new feature 'Combi_IF_AlgoII' with the above regulation
total_payments_academic['Combi_IF_DBSCAN'] = total_payments_academic['Anomaly_if'] + total_payments_academic['Anomaly_dbscan']
total_payments_academic['Combi_IF_HDBSCAN'] = total_payments_academic['Anomaly_if'] + total_payments_academic['Anomaly_hdbscan']
total_payments_academic['Combi_IF_LOF'] = total_payments_academic['Anomaly_if'] + total_payments_academic['Anomaly_lof']

# %% 
# Define true labels and predicted labels
true_labels = total_payments_academic['Fraud'] >= 1
if_labels = total_payments_academic['Anomaly_if']
dbscan_labels = total_payments_academic['Anomaly_dbscan']
hdbscan_labels = total_payments_academic['Anomaly_hdbscan']
lof_labels = total_payments_academic['Anomaly_lof']
combi_if_dbscan_labels = total_payments_academic['Combi_IF_DBSCAN']
combi_if_hdbscan_labels = total_payments_academic['Combi_IF_HDBSCAN']
combi_if_lof_labels = total_payments_academic['Combi_IF_LOF']

# Create a dataframe with the true labels and the predicted labels
df = pd.DataFrame({'True Label': true_labels,
                   'Isolation Forest': if_labels,
                   'DBSCAN': dbscan_labels,
                   'HDBSCAN': hdbscan_labels,
                   'LOF': lof_labels,
                   'Combi_IF_DBSCAN': combi_if_dbscan_labels,
                   'Combi_IF_HDBSCAN': combi_if_hdbscan_labels,
                   'Combi_IF_LOF': combi_if_lof_labels
                   })

# Get the unique class labels
labels = sorted(df['True Label'].unique())


# Loop through each model and generate its confusion matrix
for model_name in ['Isolation Forest', 'DBSCAN',
                   'HDBSCAN', 'LOF', 'Combi_IF_DBSCAN',
                   'Combi_IF_HDBSCAN', 'Combi_IF_LOF']:
    model_predictions = df[model_name]
    cm = confusion_matrix(df['True Label'], model_predictions, labels=labels)
    print(f"Confusion matrix for {model_name}:\n{cm}\n")

    # Calculate additional evaluation metrics
    accuracy = accuracy_score(df['True Label'], model_predictions)
    precision = precision_score(df['True Label'], model_predictions)
    recall = recall_score(df['True Label'], model_predictions)
    f1 = f1_score(df['True Label'], model_predictions)

    print(f"Accuracy for {model_name}: {accuracy}")
    print(f"Precision for {model_name}: {precision}")
    print(f"Recall for {model_name}: {recall}")
    print(f"F1-Score for {model_name}: {f1}\n")

    # Calculate the False Positive Rate (FPR)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    print(f"FPR for {model_name}: {fpr}\n")

    # Indicate the number of rightfully classified observations
    print(f"Number of rightfully classified observations for {model_name}: {tp}")

    # Plot the confusion matrix as a heatmap
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Greens', fmt='g')

    # Set the plot title and labels
    ax.set_title(f"Confusion matrix for {model_name}")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    # Show the plot
    plt.show()

# -----------------------------------------------------------------------------
# Summary Statistics
# -----------------------------------------------------------------------------
# %% Summary Statistics of numeric features
# Select the desired attributes/columns
selected_attributes = ['Amount_Applied', 'Amount_Initial', 'Discount_Applied', 'Discount_Rate']

# Get summary statistics for the selected attributes
summaryStats_num = input.loc[:, selected_attributes].describe(include='all')
summaryStats_num = summaryStats_num.round(2)

# Create a latex table for summary statistics
print(summaryStats_num.to_latex())

# %% Summary Statistics of categorical features
# Select the desired attributes/columns
selected_attributes = ['Payment_Number', 'Object_Number',
                        'Country_Region_Code', 'Discount_Allowed',
                        'Payment_Method_Code', 'Customer_IBAN',
                        'Vendor_IBAN_BIC', 'Vendor_Bank_Origin',
                        'Created_By', 'Source_System', 'Mandant']

# Get summary statistics for the selected attributes
summaryStats_cat = input.loc[:, selected_attributes].describe(include='all').transpose()


# Create a latex table for summary statistics
print(summaryStats_cat.to_latex())

# -----------------------------------------------------------------------------
# Box Plot
# -----------------------------------------------------------------------------
# %% Create box plot of Amount_Applied for each source system

# Extract 'Amount_Applied' and 'Source_system' columns
amount_applied = input['Amount_Applied']
source_system = input['Source_System']

# Create a dictionary to store the 'Amount_Applied' values for each 'Source_system'
data = {}
for system in source_system.unique():
    data[system] = amount_applied[source_system == system]

# Create a list to store the 'Amount_Applied' values for each 'Source_system'
values = []
for system, amounts in data.items():
    values.append(amounts)
    
# Create a box plot
plt.boxplot(values, labels=['BFSN', 'RELN'], sym='rx',
            patch_artist=True, vert=False, notch=True)

# Add labels and title
plt.ylabel('Source System')
plt.xlabel('Amount Applied')
# plt.title('Box Plot of Amount Applied by Source System')

# Show the plot
plt.show()

# -----------------------------------------------------------------------------
# Histogram
# -----------------------------------------------------------------------------
# %% Create a plot of the number of transactions by posting date
# Convert 'Posting_Date' column to datetime if it's not already in that format
total_payments_academic['Posting_Date'] = pd.to_datetime(total_payments_academic['Posting_Date'])

# Filter transactions that have a 1 in the 'Anomaly_if' feature
anomaly_transactions = total_payments_academic[total_payments_academic['Fraud']  != 2]

# Group transactions by posting date and count the number of transactions in each group
transactions_by_date = anomaly_transactions['Posting_Date'].value_counts().sort_index()

# Create a bar plot to visualize the distribution of transactions by posting date
plt.figure(figsize=(12, 6))
bars = plt.bar(transactions_by_date.index, transactions_by_date.values, label='Transactions')

# Customize the bar color
for bar in bars:
    bar.set_color('red')
    bar.set_alpha(0.5)

plt.xlabel('Posting_Date')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
# plt.legend()

# Set the background color to white
plt.gca().set_facecolor('white')

# Add horizontal gray grid lines
plt.grid(color='gray', linestyle='--', linewidth=0.5)

# Add a black frame around the plot
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_color('black')

plt.xlabel('Posting_Date')
plt.ylabel('Number of Transactions')
# plt.title('Number of Transactions by Posting_Date')
plt.xticks(rotation=45)
plt.legend()

# Get the date and count of the top 1 day with the largest transactions
top1_date = transactions_by_date.nlargest(1).index[0]
top1_count = transactions_by_date.nlargest(1).values[0]

# Highlight the top 1 day with a circle marker
plt.plot(top1_date, top1_count, marker='o', markersize=2, color='red')

# Add the number of transactions for the top 1 day as text next to the record
plt.text(top1_date, top1_count, str(top1_count), ha='left', va='bottom', color='red')

plt.show()

# Get me the dates and respective counts of the top 10 days with the most transactions
print(transactions_by_date.nlargest(5))

# calculate the average number of transactions per day
print(transactions_by_date.mean())

# -----------------------------------------------------------------------------
# Histogram 2
# -----------------------------------------------------------------------------
# %% Create a plot of the number of transactions by posting date
# Convert 'Posting_Date' column to datetime if it's not already in that format
total_payments_academic['Posting_Date'] = pd.to_datetime(total_payments_academic['Posting_Date'])


anomaly_transactions = total_payments_academic[total_payments_academic['Fraud']  != 2]

# Group filtered transactions by posting date and count the number of transactions in each group
transactions_by_date = anomaly_transactions['Posting_Date'].value_counts().sort_index()


# Create a bar plot to visualize the distribution of transactions by posting date
fig, ax = plt.subplots(figsize=(12, 6))
bars =  ax.bar(transactions_by_date.index, transactions_by_date.values)
ax.set_xlabel('Posting_Date')
ax.set_ylabel('Number of Transactions')
# ax.set_title('Distribution of Transactions by Posting Date')

# Set the x-axis tick frequency to show each 7-day period
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

# Set the x-axis tick labels starting from '2023-01-04' in a 7-day interval
start_date = pd.Timestamp('2023-01-04')
end_date = transactions_by_date.index[-1]  # Set the end date as the last date in the dataset
date_range = pd.date_range(start_date, end_date, freq='7D')
ax.set_xticks(date_range)
ax.set_xticklabels(date_range.strftime('%Y-%m-%d'))

# Set the x-axis limits using datetime values
plt.xlim(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-05-12'))

# Rotate the x-axis tick labels for better readability
plt.xticks(rotation=45, ha='right')

# Customize the bar color
for bar in bars:
    bar.set_color('red')
    bar.set_alpha(0.8)

plt.xlabel('Posting_Date')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
# plt.legend()

# Set the background color to white
plt.gca().set_facecolor('white')

# Add horizontal gray grid lines
plt.grid(color='gray', linestyle='--', linewidth=0.5)

# Add a black frame around the plot
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_color('black')

plt.show()

# -----------------------------------------------------------------------------
# Histogram 3
# -----------------------------------------------------------------------------
# %% Create a plot of the number of transactions by posting date
# Convert 'Posting_Date' column to datetime if it's not already in that format
total_payments_academic['Posting_Date'] = pd.to_datetime(total_payments_academic['Posting_Date'])

# Filter transactions that have a 1 in the 'Anomaly_if' feature
anomaly_transactions = total_payments_academic[total_payments_academic['Anomaly_if'] == 1]
anomaly_transactions = anomaly_transactions[anomaly_transactions['Fraud'] != 2]

# Group filtered transactions by posting date and count the number of transactions in each group
transactions_by_date = anomaly_transactions['Posting_Date'].value_counts().sort_index()

# Create a bar plot to visualize the distribution of transactions by posting date
plt.figure(figsize=(12, 6))
bars = plt.bar(transactions_by_date.index, transactions_by_date.values, label='Transactions')

# Customize the bar color
for bar in bars:
    bar.set_color('red')
    bar.set_alpha(0.8)

plt.xlabel('Posting_Date')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
# plt.legend()

# Set the background color to white
plt.gca().set_facecolor('white')

# Add horizontal gray grid lines
plt.grid(color='gray', linestyle='--', linewidth=0.5)

# Add a black frame around the plot
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_color('black')

plt.show()


# Get me the dates and respective counts of the top 10 days with the most transactions
print(transactions_by_date.nlargest(5))

# calculate the average number of transactions per day
print(transactions_by_date.mean())

# -----------------------------------------------------------------------------
# Top 5 dates with the most transactions
# -----------------------------------------------------------------------------
# %% Display the Top 5 transactions recorded with Anomaly_if = 1
# Get me the top 5 transactions with the highest anomaly score
print(transactions_by_date.nlargest(5))

# Get me number of distict Payment_Numbers for each of the top 5 dates
print(anomaly_transactions[anomaly_transactions['Posting_Date'] == '2023-05-10']['Payment_Number'].nunique())
print(anomaly_transactions[anomaly_transactions['Posting_Date'] == '2020-10-14']['Payment_Number'].nunique())
print(anomaly_transactions[anomaly_transactions['Posting_Date'] == '2022-03-02']['Payment_Number'].nunique())
print(anomaly_transactions[anomaly_transactions['Posting_Date'] == '2023-02-22']['Payment_Number'].nunique())
print(anomaly_transactions[anomaly_transactions['Posting_Date'] == '2023-03-22']['Payment_Number'].nunique())

print(anomaly_transactions[anomaly_transactions['Posting_Date'] == '2023-05-10']['Vendor_Number'].nunique())
print(anomaly_transactions[anomaly_transactions['Posting_Date'] == '2020-10-14']['Vendor_Number'].nunique())
print(anomaly_transactions[anomaly_transactions['Posting_Date'] == '2022-03-02']['Vendor_Number'].nunique())
print(anomaly_transactions[anomaly_transactions['Posting_Date'] == '2023-02-22']['Vendor_Number'].nunique())
print(anomaly_transactions[anomaly_transactions['Posting_Date'] == '2023-03-22']['Vendor_Number'].nunique())

# Store the top 5 dates in variables for inspection
top1 = anomaly_transactions[anomaly_transactions['Posting_Date'] == '2023-05-10']
top2 = anomaly_transactions[anomaly_transactions['Posting_Date'] == '2020-10-14']
top3 = anomaly_transactions[anomaly_transactions['Posting_Date'] == '2022-03-02']
top4 = anomaly_transactions[anomaly_transactions['Posting_Date'] == '2023-02-22']
top5 = anomaly_transactions[anomaly_transactions['Posting_Date'] == '2023-03-22']

# -----------------------------------------------------------------------------
# Scatter plot of all transactions with respect to the amount applied
# -----------------------------------------------------------------------------
# %% Scatter plot of all transactions with respect to the amount applied
# Filter the data for True values
true_data = total_payments_academic[total_payments_academic['Anomaly_if']]

# Filter the data for False values
false_data = total_payments_academic[~total_payments_academic['Anomaly_if']]

# Create the scatter plot with False values in the background
plt.scatter(x=false_data['Amount_Applied'],
            y=false_data['INDEX'],
            c='lightgrey',
            marker='x',
            label='Normal')

# Create the scatter plot with True values
plt.scatter(x=true_data['Amount_Applied'],
            y=true_data['INDEX'],
            c='red',
            marker='x',
            label='Anomalies')

# Set the x and y axis labels
# plt.xlim(0, 2100000)

# Set the background color to white
plt.gca().set_facecolor('white')

# Add horizontal gray grid lines
plt.grid(color='gray', linestyle='--', linewidth=0.5)

# Add a black frame around the plot
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_color('black')

# Set the x-axis and y-axis labels
plt.xlabel('Amount_Applied')
plt.ylabel('INDEX')

# Set the legend position to bottom right
plt.legend(loc='upper left')

# Set the figure size to stretch the plot horizontally
fig = plt.gcf()
fig.set_size_inches(15, 9)

# Show the plot
plt.show()
