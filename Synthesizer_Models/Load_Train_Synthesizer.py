# %% Import libraries
import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import run_diagnostic

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

#-------------------------------------------------------------------------------------------
# Generate synthetic data based on fraud_invoices
#-------------------------------------------------------------------------------------------

# %% Prepare Metadata for generating synthetic data

metadata = SingleTableMetadata()

metadata.detect_from_dataframe(data=fraud_invoices)
metadata.validate()

print(metadata.to_dict())
print(metadata.primary_key)

# %% Generate synthetic data (Gaussian Copula Synthesizer)

# Step 1: Create the synthesizer
synthesizer = GaussianCopulaSynthesizer(metadata,
                                        # enforce_min_max_values=False,
                                        enforce_rounding=True)

# Step 2: Train the synthesizer
synthesizer.fit(fraud_invoices)

# Step 3: Generate synthetic data (0.5% of the original data)
synthetic_data = synthesizer.sample(num_rows=int(118904*0.005),
                                    batch_size=100)

# %% Evaluate synthetic data
# Evaluate the quality of the synthetic data
quality_report = evaluate_quality(real_data=fraud_invoices,
                                  synthetic_data=synthetic_data,
                                  metadata=metadata)

# Run a diagnostic on the synthetic data
diagnostic_report = run_diagnostic(real_data=fraud_invoices,
                                   synthetic_data=synthetic_data,
                                   metadata=metadata)

# Change values in 'Fraud' of sybthetic_data to 2
synthetic_data['Fraud'] = 2
# %% Save  & Load Gaussian Copula Synthesizer

# Save the synthesizer
synthesizer.save(filepath='Gaussian_Copula_Synthesizer.pkl')

# Load the synthesizer
synthesizer = GaussianCopulaSynthesizer.load(filepath='Gaussian_Copula_Synthesizer.pkl')

# %% Generate synthetic data (CTGAN Synthesizer)

# Step 1: Create the synthesizer
synthesizer1 = CTGANSynthesizer(metadata,
                                # enforce_min_max_values=False,
                                enforce_rounding=True,
                                epochs=50000,
                                cuda=True,
                                verbose=True)

# Step 2: Train the synthesizer
synthesizer1.fit(fraud_invoices)

# Step 3: Generate synthetic data (0.5% of the original data)
synthetic_data1 = synthesizer1.sample(num_rows=int((118904*0.005)), 
                                      batch_size=100)

# Change values in 'Fraud' of sybthetic_data1 to 2
synthetic_data1['Fraud'] = 2

# Step 4: Evaluate the quality of the synthetic data
# Evaluate the quality of the synthetic data
quality_report = evaluate_quality(real_data=fraud_invoices,
                                  synthetic_data=synthetic_data1,
                                  metadata=metadata)

# Run a diagnostic on the synthetic data
diagnostic_report = run_diagnostic(real_data=fraud_invoices,
                                   synthetic_data=synthetic_data1,
                                   metadata=metadata)

# %% Save  & Load CTGAN Synthesizer

# Save the synthesizer
synthesizer1.save(filepath='CTGAN_Synthesizer.pkl')

# Load the synthesizer
synthesizer1 = CTGANSynthesizer.load(filepath='CTGAN_Synthesizer.pkl')

# %% Generate synthetic data (TVAE Synthesizer)

# Step 1: Create the synthesizer
synthesizer2 = TVAESynthesizer(metadata,
                               # enforce_min_max_values=False,
                               enforce_rounding=True,
                               epochs=50000,
                               cuda=True)

# Step 2: Train the synthesizer
synthesizer2.fit(fraud_invoices)

# Step 3: Generate synthetic data (0.5% of the original data)
synthetic_data2 = synthesizer2.sample(num_rows=int((118904*0.005)),  
                                      batch_size=100)

# Change values in 'Fraud' of sybthetic_data2 to 2
synthetic_data2['Fraud'] = 2

# Step 4: Evaluate the quality of the synthetic data
quality_report = evaluate_quality(real_data=fraud_invoices,
                                  synthetic_data=synthetic_data2,
                                  metadata=metadata)

# Run a diagnostic on the synthetic data
diagnostic_report = run_diagnostic(real_data=fraud_invoices,
                                   synthetic_data=synthetic_data2,
                                   metadata=metadata)

# %% Save  & Load TVAE Synthesizer

# Save the synthesizer
synthesizer2.save(filepath='TVAE_Synthesizer.pkl')

# Load the synthesizer
synthesizer2 = TVAESynthesizer.load(filepath='TVAE_Synthesizer.pkl')

# %% Generate synthetic data (CopulaGAN Synthesizer)

# Step 1: Create the synthesizer
synthesizer3 = CopulaGANSynthesizer(metadata,
                                    # enforce_min_max_values=False,
                                    enforce_rounding=True,
                                    epochs=50000,
                                    cuda=True,
                                    verbose=True)

# Step 2: Train the synthesizer
synthesizer3.fit(fraud_invoices)

# Step 3: Generate synthetic data
synthetic_data3 = synthesizer3.sample(num_rows=int((118904*0.005)), #  0.5% of the original data
                                      batch_size=100)

# Change values in 'Fraud' of sybthetic_data3 to 2
synthetic_data3['Fraud'] = 2

# Step 4: Evaluate the quality of the synthetic data
quality_report = evaluate_quality(real_data=fraud_invoices,
                                  synthetic_data=synthetic_data3,
                                  metadata=metadata)

# Run a diagnostic on the synthetic data
diagnostic_report = run_diagnostic(real_data=fraud_invoices,
                                   synthetic_data=synthetic_data3,
                                   metadata=metadata)


# %% Save  & Load CopulaGAN Synthesizer

# Save the synthesizer
synthesizer3.save(filepath='CopulaGAN_Synthesizer.pkl')

# Load the synthesizer
synthesizer3 = CopulaGANSynthesizer.load(filepath='CopulaGAN_Synthesizer.pkl')


