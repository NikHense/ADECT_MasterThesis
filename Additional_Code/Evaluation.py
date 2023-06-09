# %% Load packages
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import numpy as np
# from tensorflow import keras


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


# # %% Evaluate number ob anomaly scores when Review_Status = 'Proposal'
# # Get me all observations that have Review_Status = 'Proposal' and Anomaly_if = 1
# print(total_payments_academic[(total_payments_academic['Review_Status'] == 'Proposal') &
#                 (total_payments_academic['Anomaly_if'] == 1)].shape[0])

# print(total_payments_academic[(total_payments_academic['Review_Status'] == 'Proposal') &
#                 (total_payments_academic['Anomaly_dbscan'] ==1)].shape[0])

# print(total_payments_academic[(total_payments_academic['Review_Status'] == 'Proposal') &
#                 (total_payments_academic['Anomaly_hdbscan'] ==1)].shape[0])

# print(total_payments_academic[(total_payments_academic['Review_Status'] == 'Proposal') &
#                 (total_payments_academic['Anomaly_lof'] ==1)].shape[0])

# print(total_payments_academic[(total_payments_academic['Review_Status'] == 'Proposal') &
#                 (total_payments_academic['Combi_IF_DBSCAN'] ==1)].shape[0])

# print(total_payments_academic[(total_payments_academic['Review_Status'] == 'Proposal') &
#                 (total_payments_academic['Combi_IF_HDBSCAN'] ==1)].shape[0])

# print(total_payments_academic[(total_payments_academic['Review_Status'] == 'Proposal') &
#                 (total_payments_academic['Combi_IF_LOF'] ==1)].shape[0])

# # Count the number of observations that have Proposal in Review_Status
# print(total_payments_academic[total_payments_academic['Review_Status'] == 'Proposal'].shape[0])


# %%
