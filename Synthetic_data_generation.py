# %% Import libraries
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import run_diagnostic

# %% Params SQL connection
SERVER = 'P-SQLDWH'  # os.environ.get('SERVER')
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

# %% Import total_payments
SQL_TOTAL_PAYMENTS = 'SELECT * FROM ADECT.TOTAL_PAYMENTS'
total_payments = pd.DataFrame(engine.connect().execute(
                              text(SQL_TOTAL_PAYMENTS)))

# %% Import Fraudulent Invoices
# # Define the data types for each feature
dtypes = {
    'Object_Number': str
}

# Import Fraudulent Invoices csv file
fraud_invoices = pd.read_csv('Fraud_Invoices.csv', dtype=dtypes,
                             na_values='NA', sep=';')


# Delete row with index 20 until 23
fraud_invoices = fraud_invoices.drop([20, 21, 22, 23])

# Replace the , in 'Ammount_Applied', 'Ammount_Initial' and 'Discount_Applied' with
fraud_invoices['Amount_Applied'] = fraud_invoices['Amount_Applied'].str.replace(',', '.')
fraud_invoices['Amount_Initial'] = fraud_invoices['Amount_Initial'].str.replace(',', '.')
fraud_invoices['Discount_Applied'] = fraud_invoices['Discount_Applied'].str.replace(',', '.')
fraud_invoices['Discount_Rate'] = fraud_invoices['Discount_Rate'].str.replace(',', '.')


# Transform data type of column 'Amount_Applied', 'Amount_Initial' and 'Discount_Applied' to float
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

# %% Generate synthetic data (Gaussian Copula Synthesizer)

# Step 1: Load Synthesizer
synthesizer = GaussianCopulaSynthesizer.load(filepath='Synthesizer_Models/Gaussian_Copula_Synthesizer.pkl')

# Step 2: Generate synthetic data (0.5% of the original data)
synthetic_data = synthesizer.sample(num_rows=int(len(total_payments)*0.005),
                                    batch_size=100)

# Step 3: Evaluate the quality of the synthetic data
quality_report = evaluate_quality(real_data=fraud_invoices,
                                  synthetic_data=synthetic_data,
                                  metadata=metadata)

# Run a diagnostic on the synthetic data
diagnostic_report = run_diagnostic(real_data=fraud_invoices,
                                   synthetic_data=synthetic_data,
                                   metadata=metadata)

# Change values in 'Fraud' of sybthetic_data to 2
synthetic_data['Fraud'] = 2
# ---------------------------------------------------------------------------

# %% Generate synthetic data (CTGAN Synthesizer)

# Step 1: Load the synthesizer
synthesizer1 = CTGANSynthesizer.load(filepath='Synthesizer_Models/CTGAN_Synthesizer.pkl')

# Step 2: Generate synthetic data (0.5% of the original data)
synthetic_data1 = synthesizer1.sample(num_rows=int(len(total_payments)*0.005),
                                      batch_size=100)

# Step 3: Evaluate the quality of the synthetic data
quality_report = evaluate_quality(real_data=fraud_invoices,
                                  synthetic_data=synthetic_data1,
                                  metadata=metadata)

# Run a diagnostic on the synthetic data
diagnostic_report = run_diagnostic(real_data=fraud_invoices,
                                   synthetic_data=synthetic_data1,
                                   metadata=metadata)
# Change values in 'Fraud' of sybthetic_data1 to 2
synthetic_data1['Fraud'] = 2
# ---------------------------------------------------------------------------

# %% Generate synthetic data (TVAE Synthesizer)

# Step 1: Load the synthesizer
synthesizer2 = TVAESynthesizer.load(filepath='Synthesizer_Models/TVAE_Synthesizer.pkl')

# Step 2: Generate synthetic data (0.5% of the original data)
synthetic_data2 = synthesizer2.sample(num_rows=int(len(total_payments)*0.005),  
                                      batch_size=100)

# Step 3: Evaluate the quality of the synthetic data
quality_report = evaluate_quality(real_data=fraud_invoices,
                                  synthetic_data=synthetic_data2,
                                  metadata=metadata)

# Run a diagnostic on the synthetic data
diagnostic_report = run_diagnostic(real_data=fraud_invoices,
                                   synthetic_data=synthetic_data2,
                                   metadata=metadata)

# Change values in 'Fraud' of sybthetic_data2 to 2
synthetic_data2['Fraud'] = 2
# ---------------------------------------------------------------------------

# %% Generate synthetic data (CopulaGAN Synthesizer)

# Step 1: Load the synthesizer
synthesizer3 = CopulaGANSynthesizer.load(filepath='Synthesizer_Models/CopulaGAN_Synthesizer.pkl')

# Step 2: Generate synthetic data
synthetic_data3 = synthesizer3.sample(num_rows=int(
                                      len(total_payments)*0.005), # 0.5% of the original data
                                      batch_size=100)

# Step 3: Evaluate the quality of the synthetic data
quality_report = evaluate_quality(real_data=fraud_invoices,
                                  synthetic_data=synthetic_data3,
                                  metadata=metadata)

# Run a diagnostic on the synthetic data
diagnostic_report = run_diagnostic(real_data=fraud_invoices,
                                   synthetic_data=synthetic_data3,
                                   metadata=metadata)

# Change values in 'Fraud' of sybthetic_data3 to 2
synthetic_data3['Fraud'] = 2
# ---------------------------------------------------------------------------

# %% Include the synthetic_data to the fraud_invoices
fraud_invoices = pd.concat([fraud_invoices, synthetic_data],
                           ignore_index=True)
fraud_invoices1 = pd.concat([fraud_invoices, synthetic_data1],
                            ignore_index=True)
fraud_invoices2 = pd.concat([fraud_invoices, synthetic_data2],
                            ignore_index=True)
fraud_invoices3 = pd.concat([fraud_invoices, synthetic_data3],
                            ignore_index=True)

# %%
