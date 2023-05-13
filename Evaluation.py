# %% Load packages
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import numpy as np
# from tensorflow import keras

# Define true labels and predicted labels

true_labels = total_payments['Fraud'] >= 1
if_labels = total_payments['Anomaly_if']
dbscan_labels = total_payments['Anomaly_dbscan']
hdbscan_labels = total_payments['Anomaly_hdbscan']
lof_labels = total_payments['Anomaly_lof']

# Create a dataframe with the true labels and the predicted labels
df = pd.DataFrame({'True Label': true_labels,
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
    
    # Calculate additional evaluation metrics
    accuracy = accuracy_score(df['True Label'], model_predictions)
    precision = precision_score(df['True Label'], model_predictions)
    recall = recall_score(df['True Label'], model_predictions)
    f1 = f1_score(df['True Label'], model_predictions)
    
    print(f"Accuracy for {model_name}: {accuracy}")
    print(f"Precision for {model_name}: {precision}")
    print(f"Recall for {model_name}: {recall}")
    print(f"F1-Score for {model_name}: {f1}\n")
    
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





# %%
