# %% Import libraries for k-means clustering algorithm
import matplotlib.pyplot as plt
import time
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np
import pandas as pd
from multiprocessing import Pool


# %% Run an inital DBSCAN to test whether it works
# Standardize the features using StandardScaler
# starttime = time.time()
# scaler = StandardScaler()
# kmeans_data_scaled = scaler.fit_transform(kmeans_data)

# # Perform a PCA on kmeans_data
# pca = PCA(n_components=15)
# kmeans_data_pca = pca.fit_transform(kmeans_data_scaled)

# # Apply DBSCAN to cluster the data
# y_pred = DBSCAN(eps=0.9, min_samples=5,
#                 n_jobs=-1).fit(kmeans_data_pca)

# core_samples_mask = np.zeros_like(y_pred.labels_, dtype=bool)
# core_samples_mask[y_pred.core_sample_indices_] = True
# labels = y_pred.labels_

# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)

# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(kmeans_data_scaled, labels))

# # Invert the scaling applied by StandardScaler
# kmeans_output = scaler.inverse_transform(kmeans_data_scaled)

# # # Invert the PCA transformation
# # kmeans_data_pca_inverted = pca.inverse_transform(kmeans_data_pca)

# # Convert the kmeans_output array to a pandas DataFrame
# kmeans_output = pd.DataFrame(kmeans_output, columns=kmeans_data.columns)

# # Add the labels column to the kmeans_output
# kmeans_output['labels'] = labels
# kmeans_output['noise'] = kmeans_output['labels'] == -1

# print(f'DBSCAN process took {time.time() - starttime} seconds')


# %% Define list of parameter values to test

eps_list_u = list(np.arange(0.5, 7.1, 0.5)) # initial parameters
min_samples_list_u = list(range(25, 301, 25)) # initial parameters

# ---------------------------------------------------------------------
# lower range of eps

# eps_list_l = list(np.arange(1.4, 3.1, 0.1)) # closing in on best silhouette score
# min_samples_list_l = list(range(60, 201, 10)) # closing in on best silhouette score

# eps_list_l = list(np.arange(1.8, 2.2, 0.01)) # closing in on best silhouette score
# min_samples_list_l = list(range(160, 211, 2)) # closing in on best silhouette score

eps_list_l = list(np.arange(1.90, 1.92, 0.01)) # closing in on best silhouette score
min_samples_list_l = list(range(193, 204, 1)) # closing in on best silhouette score

# ---------------------------------------------------------------------------------
# upper range of eps

# eps_list_u = list(np.arange(4, 10.1, 0.5)) # closing in on best silhouette score
# min_samples_list_u = list(range(10, 100, 10)) # closing in on best silhouette score

# eps_list_u = list(np.arange(4, 5, 0.1)) # closing in on best silhouette score
# min_samples_list_u = list(range(50, 80, 5)) # closing in on best silhouette score

# eps_list_u = list(np.arange(3.8, 4.2, 0.01))  # closing in on best silhouette score
# min_samples_list_u = list(range(20, 30, 1))  # closing in on best silhouette score

# %% Run DBSCAN, testing the different paramter combinations (lower range of eps)
starttime = time.time()
scaler = StandardScaler()
kmeans_data_scaled = scaler.fit_transform(kmeans_data)


# Perform a PCA on kmeans_data
pca = PCA(n_components=15)
kmeans_data_pca = pca.fit_transform(kmeans_data_scaled)
# Initialize empty lists to store results
scores_db = []
params_db = []

# Define a function to compute DBSCAN for a given parameter combination
def dbscan_cluster(params):
    eps, min_samples = params
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
    labels = dbscan.fit_predict(kmeans_data_pca)
    n_clusters_ = len(np.unique(labels[labels != -1]))
    if n_clusters_ < 2:
        score = -1
    # set score to a negative value if there is only one cluster
    else:
        score = metrics.silhouette_score(kmeans_data_pca, labels)
    n_noise_ = list(labels).count(-1)
    print(f'eps={eps:.6f}, min_samples={min_samples:4d}, n_clusters={n_clusters_:3d}, n_noise={n_noise_:4d}, score={score:.3f}')
    return score


# Define a list of parameter combinations to test
param_list = [(eps, min_samples) for eps in eps_list_l
              for min_samples in min_samples_list_l]

# Create a Pool object
with Pool() as pool:
    # Compute DBSCAN for all parameter combinations in parallel
    scores_db = pool.map(dbscan_cluster, param_list)
    pool.close()
    pool.join()

# Find the parameter combination with the highest score
best_idx = np.argmax(scores_db)
best_params = param_list[best_idx]
best_score = scores_db[best_idx]
print(f'Best parameter combination: eps={best_params[0]:.6f}, min_samples={best_params[1]:4d}, score={best_score:.3f}')

# Save the best parameters for later use
eps_lower = float(f"{best_params[0]:.3f}")
min_samples_lower = int(f"{best_params[1]:d}")

# Print the time the process took
minutes = int((time.time() - starttime) / 60)
seconds = int((time.time() - starttime) % 60)
print(f'DBSCAN grid search process took {minutes} minutes and {seconds} seconds')


# %% Get the list with the Top 30 scores and their parameters
top_30_l = sorted(zip(scores_db, param_list), reverse=True)[:30]
top_30_l

# Plot Top_30
sns.set_style('whitegrid')
plt.figure(figsize=(10, 10))
plt.scatter([x[1][0] for x in top_30_l], [x[1][1] for x in top_30_l],
            c=[x[0] for x in top_30_l], cmap='viridis')
plt.colorbar()
plt.xlabel('eps_lower')
plt.ylabel('min_samples_lower')
plt.title('Top 30 DBSCAN scores (lower grid)')
plt.show()

# %% Run DBSCAN, testing the different paramter combinations (upper range of eps)
starttime = time.time()
scaler = StandardScaler()
kmeans_data_scaled = scaler.fit_transform(kmeans_data)


# Perform a PCA on kmeans_data
pca = PCA(n_components=15)
kmeans_data_pca = pca.fit_transform(kmeans_data_scaled)

# Initialize empty lists to store results
scores_db = []
params_db = []

# Define a function to compute DBSCAN for a given parameter combination
def dbscan_cluster(params):
    eps, min_samples = params
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
    labels = dbscan.fit_predict(kmeans_data_pca)
    n_clusters_ = len(np.unique(labels[labels != -1]))
    if n_clusters_ < 2:
        score = -1
    # set score to a negative value if there is only one cluster
    else:
        score = metrics.silhouette_score(kmeans_data_pca, labels)
    n_noise_ = list(labels).count(-1)
    print(f'eps={eps:.6f}, min_samples={min_samples:4d}, n_clusters={n_clusters_:3d}, n_noise={n_noise_:4d}, score={score:.3f}')
    return score


# Define a list of parameter combinations to test
param_list = [(eps, min_samples) for eps in eps_list_u 
              for min_samples in min_samples_list_u]

# Create a Pool object
with Pool() as pool:
    # Compute DBSCAN for all parameter combinations in parallel
    scores_db = pool.map(dbscan_cluster, param_list)
    pool.close()
    pool.join()

# Find the parameter combination with the highest score
best_idx = np.argmax(scores_db)
best_params = param_list[best_idx]
best_score = scores_db[best_idx]
print(f'Best parameter combination: eps={best_params[0]:.6f}, min_samples={best_params[1]:4d}, score={best_score:.3f}')

# Save the best parameters for later use
eps_upper = float(f"{best_params[0]:.3f}")
min_samples_upper = int(f"{best_params[1]:d}")

# Print the time the process took
minutes = int((time.time() - starttime) / 60)
seconds = int((time.time() - starttime) % 60)
print(f'DBSCAN (upper grid) search process took {minutes} minutes and {seconds} seconds')

# %% Get the list with the scores, their parameters, n_clusters and n_noise
top_30_u = sorted(zip(scores_db, param_list), reverse=True)[:30]
top_30_u

# Plot Top_30
sns.set_style('whitegrid')
plt.figure(figsize=(10, 10))
plt.scatter([x[1][0] for x in top_30_u], [x[1][1] for x in top_30_u],
            c=[x[0] for x in top_30_u], cmap='viridis')
plt.colorbar()
plt.xlabel('eps_upper')
plt.ylabel('min_samples_upper')
plt.title('Top 30 DBSCAN scores (upper grid)')
plt.show()


# %% 
scores_all = sorted(zip(scores_db, param_list), reverse=True)

# %% Run best parameter (lower range of eps)
starttime = time.time()

scaler = StandardScaler()
kmeans_data_scaled = scaler.fit_transform(kmeans_data)

# Perform a PCA on kmeans_data
pca = PCA(n_components=15)
kmeans_data_pca = pca.fit_transform(kmeans_data_scaled)

# Apply DBSCAN to cluster the data
y_pred = DBSCAN(eps=eps_lower, min_samples=min_samples_lower,
                n_jobs=-1).fit(kmeans_data_pca)

core_samples_mask = np.zeros_like(y_pred.labels_, dtype=bool)
core_samples_mask[y_pred.core_sample_indices_] = True
labels = y_pred.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(kmeans_data_scaled, labels))

# Invert the scaling applied by StandardScaler
kmeans_output = scaler.inverse_transform(kmeans_data_scaled)

# # Invert the PCA transformation
# kmeans_data_pca_inverted = pca.inverse_transform(kmeans_data_pca)

# Convert the kmeans_output array to a pandas DataFrame
kmeans_output = pd.DataFrame(kmeans_output, columns=kmeans_data.columns)

# Add the labels column to the kmeans_output
kmeans_output['labels'] = labels
kmeans_output['noise'] = kmeans_output['labels'] == -1

print(f'DBSCAN (lower grid) process took {time.time() - starttime} seconds')

# %% Run best parameter (upper range of eps)
starttime = time.time()

scaler = StandardScaler()
kmeans_data_scaled = scaler.fit_transform(kmeans_data)

# Perform a PCA on kmeans_data
pca = PCA(n_components=15)
kmeans_data_pca = pca.fit_transform(kmeans_data_scaled)

# Apply DBSCAN to cluster the data
y_pred = DBSCAN(eps=eps_upper, min_samples=min_samples_upper,
                n_jobs=-1).fit(kmeans_data_pca)

core_samples_mask = np.zeros_like(y_pred.labels_, dtype=bool)
core_samples_mask[y_pred.core_sample_indices_] = True
labels = y_pred.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(kmeans_data_scaled, labels))

# Invert the scaling applied by StandardScaler
kmeans_output = scaler.inverse_transform(kmeans_data_scaled)

# # Invert the PCA transformation
# kmeans_data_pca_inverted = pca.inverse_transform(kmeans_data_pca)

# Convert the kmeans_output array to a pandas DataFrame
kmeans_output = pd.DataFrame(kmeans_output, columns=kmeans_data.columns)

# Add the labels column to the kmeans_output
kmeans_output['labels'] = labels
kmeans_output['noise'] = kmeans_output['labels'] == -1

print(f'DBSCAN (upper grid) process took {time.time() - starttime} seconds')

# %% 
