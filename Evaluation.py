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


# %% Evaluate number ob anomaly scores when Review_Status = 'Proposal'
# Get me all observations that have Review_Status = 'Proposal' and Anomaly_if = 1
print(total_payments_academic[(total_payments_academic['Review_Status'] == 'Proposal') &
                (total_payments_academic['Anomaly_if'] == 1)].shape[0])

print(total_payments_academic[(total_payments_academic['Review_Status'] == 'Proposal') &
                (total_payments_academic['Anomaly_dbscan'] ==1)].shape[0])

print(total_payments_academic[(total_payments_academic['Review_Status'] == 'Proposal') &
                (total_payments_academic['Anomaly_hdbscan'] ==1)].shape[0])

print(total_payments_academic[(total_payments_academic['Review_Status'] == 'Proposal') &
                (total_payments_academic['Anomaly_lof'] ==1)].shape[0])

print(total_payments_academic[(total_payments_academic['Review_Status'] == 'Proposal') &
                (total_payments_academic['Combi_IF_DBSCAN'] ==1)].shape[0])

print(total_payments_academic[(total_payments_academic['Review_Status'] == 'Proposal') &
                (total_payments_academic['Combi_IF_HDBSCAN'] ==1)].shape[0])

print(total_payments_academic[(total_payments_academic['Review_Status'] == 'Proposal') &
                (total_payments_academic['Combi_IF_LOF'] ==1)].shape[0])

# Count the number of observations that have Proposal in Review_Status
print(total_payments_academic[total_payments_academic['Review_Status'] == 'Proposal'].shape[0])

# %%
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

# %%


# # Define true labels and predicted labels
# true_labels = total_payments_academic['Fraud'] >= 1
# if_labels = total_payments_academic['Anomaly_if']

# # Create a dataframe with the true labels and the predicted labels
# df = pd.DataFrame({'True Label': true_labels,
#                    'Isolation Forest': if_labels,
#                    'Amount_Applied': total_payments_academic['Amount_Applied'],
#                    'INDEX': total_payments_academic['INDEX']})

# # Get the TP and FN data
# tp_fn_data = df[(df['True Label'] == True) & (df['Isolation Forest'] == True)]

# # Create a scatter plot
# color_dict = {True: 'red', False: 'green'}
# plt.scatter(x=df.loc[~df.index.isin(tp_fn_data.index), 'Amount_Applied'],
#             y=df.loc[~df.index.isin(tp_fn_data.index), 'INDEX'],
#             color='grey',
#             label='Normal')
# plt.scatter(x=tp_fn_data.loc[tp_fn_data['True Label'] == True, 'Amount_Applied'],
#             y=tp_fn_data.loc[tp_fn_data['True Label'] == True, 'INDEX'],
#             color='green',
#             label='True Positive')
# plt.scatter(x=tp_fn_data.loc[tp_fn_data['True Label'] == False, 'Amount_Applied'],
#             y=tp_fn_data.loc[tp_fn_data['True Label'] == False, 'INDEX'],
#             color='red',
#             label='False Negative')

# # Set plot labels and title
# plt.xlabel('Amount_Applied')
# plt.ylabel('INDEX')
# plt.title('Anomaly Detection Results')

# # Add legend
# plt.legend()

# # Show the plot
# plt.show()



# %%

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
# plt.xlim(-250000, 250000)

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

# Show the plot
plt.show()

# %%
