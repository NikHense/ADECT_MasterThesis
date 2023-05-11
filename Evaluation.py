# %% Load packages
from sklearn.metrics import confusion_matrix
import pandas as pd
# import numpy as np
# from tensorflow import keras

# Define true labels and predicted labels
true_labes = total_payments['Fraud']
if_labels = total_payments['Anomaly_if']
dbscan_labels = total_payments['Anomaly_dbscan']
hdbscan_labels = total_payments['Anomaly_hdbscan']
lof_labels = total_payments['Anomaly_lof']

# Create a dataframe with the true labels and the predicted labels
df = pd.DataFrame({'True Label': true_labes,
                   'Isolation Forest': if_labels,
                   'DBSCAN': dbscan_labels,
                   'HDBSCAN': hdbscan_labels,
                   'LOF': lof_labels})

# Get the unique class labels
labels = sorted(df['True Label'].unique())


# Loop through each model and generate its confusion matrix
for model_name in ['Isolation Forest', 'DBSCAN', 'HDBSCAN', 'LOF']:
    model_predictions = df[model_name]
    cm = confusion_matrix(df['True Label'], model_predictions, labels=labels)
    print(f"Confusion matrix for {model_name}:\n{cm}\n")

    # Plot the confusion matrix as a heatmap
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Greens', fmt='g')

    # Set the plot title and labels
    ax.set_title(f"Confusion matrix for {model_name}")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    # Show the plot
    plt.show()


# %% Evaluate number ob anomaly scores when Review_Status = 'Proposal'
# Get me all observations that have Review_Status = 'Proposal' and Anomaly_sum >= 2
print(total_payments[(total_payments['Review_Status'] == 'Proposal') &
                (total_payments['Anomaly_sum'] >= 2)].shape[0])

# Count them
print(total_payments[(total_payments['Review_Status'] == 'Proposal') &
                (total_payments['Anomaly_sum'] >= 3)].shape[0])

print(total_payments[(total_payments['Review_Status'] == 'Proposal') &
                (total_payments['Anomaly_sum'] >= 4)].shape[0])

thisweek = total_payments[(total_payments['Review_Status'] == 'Proposal') &
                (total_payments['Anomaly_sum'] >= 2)]

# Count the number of observations that have Proposal in Review_Status
print(total_payments[total_payments['Review_Status'] == 'Proposal'].shape[0])

#-------------------------------------------------------------------------------------------
# Generate synthetic data based on fraud_invoices
#-------------------------------------------------------------------------------------------

# %% Prepare Metadata for generating synthetic data
from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()

metadata.detect_from_dataframe(data=fraud_invoices)
metadata.validate()

print(metadata.to_dict())
print(metadata.primary_key)

# %% Generate synthetic data
from sdv.single_table import GaussianCopulaSynthesizer

# Step 1: Create the synthesizer
synthesizer = GaussianCopulaSynthesizer(metadata)

# Step 2: Train the synthesizer
synthesizer.fit(fraud_invoices)

# Step 3: Generate synthetic data
synthetic_data = synthesizer.sample(num_rows=int((len(input)*0.05)), #  0.5% of the original data
                                    batch_size=100) 
# %% Evaluate synthetic data

from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import run_diagnostic

# Evaluate the quality of the synthetic data
quality_report = evaluate_quality(
    real_data=fraud_invoices,
    synthetic_data=synthetic_data,
    metadata=metadata)

# Run a diagnostic on the synthetic data
diagnostic_report = run_diagnostic(
    real_data=fraud_invoices,
    synthetic_data=synthetic_data,
    metadata=metadata)

# %% Generate synthetic data (CTGAN Synthesizer)
starttime = time.time()
from sdv.single_table import CTGANSynthesizer

synthesizer1 = CTGANSynthesizer(metadata,
                                enforce_min_max_values=False,
                                enforce_rounding=True,
                                epochs=50000,
                                cuda=True,
                                verbose=True)


synthesizer1.fit(fraud_invoices)

synthetic_data1 = synthesizer1.sample(num_rows=int((len(input)*0.01)), #  0.5% of the original data
                                      batch_size=100)

endtime = time.time()
print(f"Runtime of the program is {endtime - starttime}")
# %%
# Evaluate the quality of the synthetic data
quality_report = evaluate_quality(
    real_data=fraud_invoices,
    synthetic_data=synthetic_data1,
    metadata=metadata)

# Run a diagnostic on the synthetic data
diagnostic_report = run_diagnostic(
    real_data=fraud_invoices,
    synthetic_data=synthetic_data1,
    metadata=metadata)

# %% Plot the loss curves of the CTGAN synthesizer
# Format the output into a table named loss_values
# epochs_output = str(output_CTGAN).split('\n')
# raw_values = [line.split(',') for line in epochs_output]

# loss_values = pd.DataFrame(raw_values)[:-1]
# loss_values.columns = ['Epoch', 'Generator Loss', 'Discriminator Loss']
# loss_values['Epoch'] = loss_values['Epoch'].str.extract('(\d+)').astype(int)
# loss_values['Generator Loss'] = loss_values['Generator Loss'].str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
# loss_values['Discriminator Loss'] = loss_values['Discriminator Loss'].str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
# %%
# Change values in 'Fraud' of sybthetic_data1 to 2
synthetic_data1['Fraud'] = 2



# %%
