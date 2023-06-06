# %% Import libraries
# import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
# from sqlalchemy import create_engine, text
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import run_diagnostic
from sklearn.preprocessing import LabelEncoder
# from anonymizedf.anonymizedf import anonymize

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
total_payments_academic = pd.read_csv('total_payments_academic.csv',
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
fraud_invoices = pd.read_csv('Fraud_Invoices_final.csv',
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
                            ignore_index=True) #  highest Overall Quality score
fraud_invoices4 = pd.concat([fraud_invoices, synthetic_data3],
                            ignore_index=True)

# ---------------------------------------------------------------------------
# Data Preprocessing & Feature Engineering
# ---------------------------------------------------------------------------
# %% Add entries of fraudulent invoices to total_payments
total_payments_academic = pd.concat([total_payments_academic, fraud_invoices3], ignore_index=True)

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

# %%
# Convert 'Posting_Date' column to datetime if it's not already in that format
total_payments_academic['Posting_Date'] = pd.to_datetime(total_payments_academic['Posting_Date'])

# Filter transactions that have a 1 in the 'Anomaly_if' feature
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
# %% 
