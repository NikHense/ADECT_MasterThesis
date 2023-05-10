# %% Import libraries for DBSCAN algorithm
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn import metrics
from multiprocessing import Pool
from kneed import KneeLocator
import hdbscan

# -----------------------------------------------------------------------------
# Parameter selection for DBSCAN & HDBSCAN
# -----------------------------------------------------------------------------
# %% Start timer
totaltime = time.time()

# %% Scaling the input data

# scaler = MinMaxScaler(feature_range=(-25, 25))
# input_scaled = scaler.fit_transform(input)
# input_scaled = pd.DataFrame(input_scaled, columns=input.columns)


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

# Plot the cum. proportion of variance explained, with the 95% threshold
num_components = range(1, len(input_scaled.columns) + 1)
plt.plot(num_components, cumulative_variance_ratio, 'ro-', linewidth=2)
plt.axhline(y=0.95, color='b', linestyle='--')
plt.title('Cumulative Proportion of Variance Explained')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Proportion of Variance Explained')
# Find the index of the element in y_values that is closest to 0.95
threshold_idx = (np.abs(cumulative_variance_ratio - 0.95)).argmin()

# Get the x-coordinate of the threshold
threshold_x = num_components[threshold_idx]

# Add a vertical line at the threshold x-coordinate
plt.axvline(x=threshold_x, color='b', linestyle='--')
plt.show()

# retrieve the number of components that explain 95% of the variance
n_components = np.argmax(cumulative_variance_ratio >= 0.95)

print(f'{n_components} principal components explain 95% of the variance')

# if n_components > 15 than 15 otherwise
# n_components = np.argmax(cumulative_variance_ratio >= 0.99)
if n_components > 15:
    n_components = 15
else:
    n_components = np.argmax(cumulative_variance_ratio >= 0.95)


# %% Calculate the K-Distance Graph

# Perform a PCA on input_scaled
pca = PCA(n_components=n_components)
input_pca = pca.fit_transform(input_scaled)

# Calculate the distance between each point and its kth nearest neighbor
neigh = NearestNeighbors(n_neighbors=n_components)
nbrs = neigh.fit(input_pca)
distances, indices = nbrs.kneighbors(input_pca)

# Plot the k-distance graph
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.figure(figsize=(20, 10))
plt.plot(distances, 'ro-', linewidth=2)
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
                      # interp_method='polynomial',
                      curve='convex', direction='increasing')

print(round(kneedle.knee, 0))
print(round(kneedle.elbow_y, 3))

# Normalized data, normalized knee, and normalized distance curve.
plt.style.use('ggplot')
kneedle.plot_knee_normalized()
# plt.xlim(0, 0.1)
# plt.ylim(0.9, 1)
plt.show()

kneedle.plot_knee()

# plt.axhline(y=kneedle.elbow_y, color='r', linestyle='--')
# plt.text(kneedle.elbow - 10000, kneedle.elbow_y + 0.1,
#          f'elbow point ({round(kneedle.elbow, 0)}, '
#          f'{round(kneedle.elbow_y, 3)})', fontsize=10)
# plt.xlim(kneedle.elbow - 5000, kneedle.elbow + 5000)
# plt.ylim(kneedle.elbow_y - 0.5, kneedle.elbow_y + 1.5)
# plt.show()

# %%
# Plot the k-distance graph with the knee point (zoomed in)
# plt.figure(figsize=(10, 10))
# plt.plot(distances[:, 0], distances[:, 1], 'ro-', linewidth=2)
# plt.axvline(x=kneedle.elbow, color='b', linestyle='--')
# plt.axhline(y=kneedle.elbow_y, color='b', linestyle='--')
# plt.text(kneedle.elbow + 100, kneedle.elbow_y + 0.2,
#          f'elbow point ({round(kneedle.elbow, 0)}, '
#          f'{round(kneedle.elbow_y, 3)})', fontsize=12)
# plt.title('K-Distance Graph')
# plt.xlabel('Data Point sorted by distance')
# plt.ylabel('Epsilon (distance to kth nearest neighbor)')
# plt.xlim(kneedle.elbow - 500, kneedle.elbow + 500)
# plt.ylim(kneedle.elbow_y - 0.5, kneedle.elbow_y + 1.5)
# plt.show()

# %% Define list of parameter values to test

# Define optimal epsilon value
eps = round(kneedle.elbow_y, 3)

# Define the optimal min_samples value (acc. Sander's 1998)
min_samples_list = list(range(n_components, (2*n_components-1)+400, 1))
# min_samples_list = list(range(n_components, 3000, 1))


# ---------------------------------------------------------------------
# lower range of eps

# closing in on best silhouette score
# eps_list_l = list(np.arange(1.4, 3.1, 0.1))
# min_samples_list_l = list(range(60, 201, 10))

# eps_list_l = list(np.arange(1.8, 2.2, 0.01))
# min_samples_list_l = list(range(160, 211, 2))

# eps_list_l = list(np.arange(1.90, 1.92, 0.01))
# min_samples_list_l = list(range(193, 204, 1))

# ---------------------------------------------------------------------------------
# upper range of eps

# closing in on best silhouette score
# eps_list_u = list(np.arange(4, 10.1, 0.5))
# min_samples_list_u = list(range(10, 100, 10))

# eps_list_u = list(np.arange(4, 5, 0.1))
# min_samples_list_u = list(range(50, 80, 5))

# eps_list_u = list(np.arange(3.8, 4.2, 0.01))
# min_samples_list_u = list(range(20, 30, 1))

# -----------------------------------------------------------------------------
# DBSCAN (Silhouette Score)
# -----------------------------------------------------------------------------
# # %% Run DBSCAN, testing the different paramter combinations (inital run)
# starttime = time.time()

# # Initialize empty lists to store results
# results_silScore = []


# # Define a function to compute DBSCAN for a given parameter combination
# def dbscan_cluster(min_samples):
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
#     labels_dbscan = dbscan.fit_predict(input_pca)
#     n_clusters_ = len(np.unique(labels_dbscan[labels_dbscan != -1]))
#     if n_clusters_ < 1:
#         Sil_score = -1
#     # set score to a negative value if there is only one cluster
#     else:
#         Sil_score = metrics.silhouette_score(input_pca, labels_dbscan)
#     n_noise_ = list(labels_dbscan).count(-1)
#     print(f'eps={eps:.6f}, min_samples={min_samples:4d}, '
#           f'n_clusters={n_clusters_:3d}, n_noise={n_noise_:4d}, '
#           f'score={Sil_score:.3f}')
#     return (eps, min_samples, n_clusters_, n_noise_, Sil_score)


# # Create a Pool object
# with Pool() as pool:
#     # Compute DBSCAN for all parameter combinations in parallel
#     results_silScore = pool.map(dbscan_cluster, min_samples_list)
#     pool.close()
#     pool.join()

# # Store the results in a pandas DataFrame
# results_silScore = pd.DataFrame(results_silScore,
#                                 columns=['eps', 'min_samples',
#                                          'n_clusters', 'n_noise',
#                                          'Sil_score'])

# # Find the parameter combination with the highest score
# best_row = results_silScore.iloc[results_silScore['Sil_score'].idxmax()]
# best_params = (best_row['eps'], best_row['min_samples'],
#                best_row['n_clusters'], best_row['n_noise'])
# best_score = best_row['Sil_score']
# # print(f'Best parameter combination: eps={best_params[0]:.6f}, '
# #       f'min_samples={best_params[1]}, n_clusters={best_params[2]}, '
# #       f'n_noise={best_params[3]}, score={best_score:.3f}')

# # Print the time the process took
# minutes = int((time.time() - starttime) / 60)
# seconds = int((time.time() - starttime) % 60)
# print(f'DBSCAN grid search process took {minutes} minutes and '
#       f'{seconds} seconds')

# # %% Plot the results of the grid search
# # # Plot the results
# # sns.set()
# # sns.set_style("whitegrid")
# # sns.set_context("paper")
# # plt.figure(figsize=(10, 10))
# # sns.scatterplot(x='eps', y='min_samples', hue='score',
#                   data=results_silScore)
# # plt.title('DBSCAN parameter grid search')
# # plt.xlabel('eps')
# # plt.ylabel('min_samples')
# # plt.show()

# # Plot increase in score with increasing min_samples
# plt.figure(figsize=(10, 10))
# sns.lineplot(x='min_samples', y='Sil_score', data=results_silScore)
# plt.title('DBSCAN parameter grid search')
# plt.xlabel('min_samples')
# plt.ylabel('Sil_score')
# plt.show()

# # %% Run best parameter DBSCAN
# starttime = time.time()
# # Define the optimal min_samples value (acc. Sander's 1998)
# min_samples = (2*n_components-1)

# # Apply DBSCAN to cluster the data
# y_pred = DBSCAN(eps=eps, min_samples=min_samples,
#                 n_jobs=-1).fit(input_pca)

# core_samples_mask = np.zeros_like(y_pred.labels_, dtype=bool)
# core_samples_mask[y_pred.core_sample_indices_] = True
# labels_dbscan = y_pred.labels_

# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
# n_noise_ = list(labels_dbscan).count(-1)

# print('eps: %0.3f' % eps)
# print('min_samples: %d' % min_samples)
# # print the silhouette score
# print("Silhouette Coefficient: %0.4f"
#       % metrics.silhouette_score(input_pca, labels_dbscan))
# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)

# print(f'DBSCAN process took {time.time() - starttime} seconds')

# -----------------------------------------------------------------------------
# DBSCAN (Calinski-Harbasz Score)
# -----------------------------------------------------------------------------
# %% Run DBSCAN, testing Calinski-Harabasz score instead of Silhouette score
starttime = time.time()

# Initialize empty lists to store results
results_chScore = []


# Define a function to compute DBSCAN for a given parameter combination
def dbscan_cluster(min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
    labels_dbscan = dbscan.fit_predict(input_pca)
    n_clusters_ = len(np.unique(labels_dbscan[labels_dbscan != -1]))
    if n_clusters_ < 1:
        CH_score = -1
    # set score to a negative value if there is only one cluster
    else:
        CH_score = metrics.calinski_harabasz_score(input_pca, labels_dbscan)
    n_noise_ = list(labels_dbscan).count(-1)
    print(f'eps={eps:.6f}, min_samples={min_samples:4d}, '
          f'n_clusters={n_clusters_:3d}, n_noise={n_noise_:4d}, '
          f'CH_score={CH_score:.3f}')
    return (eps, min_samples, n_clusters_, n_noise_, CH_score)


# Create a Pool object
with Pool() as pool:
    # Compute DBSCAN for all parameter combinations in parallel
    results_chScore = pool.map(dbscan_cluster, min_samples_list)
    pool.close()
    pool.join()

# Store the results in a pandas DataFrame
results_chScore = pd.DataFrame(results_chScore,
                               columns=['eps', 'min_samples', 'n_clusters',
                                        'n_noise', 'CH_score'])

results_chScore['CH_score'] = results_chScore['CH_score'].astype(float)

# Find the parameter combination with the highest score
best_row = results_chScore.iloc[results_chScore['CH_score'].idxmax()]
best_params = (best_row['eps'], best_row['min_samples'],
               best_row['n_clusters'], best_row['n_noise'])
best_score = best_row['CH_score']
# print(f'Best parameter combination: eps={best_params[0]:.6f}, '
#       f'min_samples={best_params[1]}, n_clusters={best_params[2]}, '
#       f'n_noise={best_params[3]}, CH_score={best_score:.3f}')

# Print the time the process took
minutes = int((time.time() - starttime) / 60)
seconds = int((time.time() - starttime) % 60)
print(f'DBSCAN grid search process took {minutes} minutes and '
      f'{seconds} seconds')

# %% Plot the results of the grid search (Calinski-Harabasz scores)
# Plot increase in score with increasing min_samples
plt.figure(figsize=(10, 10))
sns.lineplot(x='min_samples', y='CH_score', data=results_chScore)
plt.title('DBSCAN parameter grid search')
plt.xlabel('min_samples')
plt.ylabel('CH_score')
plt.show()

# %% Run best parameter DBSCAN
starttime = time.time()
# Define the optimal min_samples value (acc. Sander's 1998)
# min_samples = (2*n_components-1)
# Define the optimal min_samples value from the max Calinski-Harabasz score
min_samples = results_chScore.iloc[results_chScore
                                   ['CH_score'].idxmax()]['min_samples']
min_samples = int(min_samples)


# Apply DBSCAN to cluster the data
y_pred = DBSCAN(eps=eps, min_samples=min_samples,
                n_jobs=-1).fit(input_pca)

core_samples_mask = np.zeros_like(y_pred.labels_, dtype=bool)
core_samples_mask[y_pred.core_sample_indices_] = True
labels_dbscan = y_pred.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_ = list(labels_dbscan).count(-1)

print('eps: %0.3f' % eps)
print('min_samples: %d' % min_samples)
# print the silhouette score
print("Calinski-Harabasz Score: %0.4f"
      % metrics.calinski_harabasz_score(input_pca, labels_dbscan))
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

print(f'DBSCAN process took {time.time() - starttime} seconds')

# %% End and print the total time of DBSCAN process
minutes = int((time.time() - totaltime) / 60)
seconds = int((time.time() - totaltime) % 60)
print(f' Total DBSCAN process took {minutes} minutes and '
      f'{seconds} seconds')


# -----------------------------------------------------------------------------
# HDBSCAN
# -----------------------------------------------------------------------------
# %% Start timer
totaltime = time.time()

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

# Defined the threshold for the outlier scores at 95% quantile
threshold = pd.Series(hdbscan.outlier_scores_).quantile(0.95)
outliers_hdbscan = np.where(hdbscan.outlier_scores_ > threshold)[0]

print(f'Number of outliers: {len(outliers_hdbscan)}')


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
dbscan_output.insert(0, 'INDEX', total_payments.index)
dbscan_output.insert(1, 'labels_dbscan', labels_dbscan)
dbscan_output.insert(2, 'Anomaly_dbscan', dbscan_output['labels_dbscan'] == -1)

# Filter out the a data frame with only noise points
dbscan_noise = dbscan_output[dbscan_output['Anomaly_dbscan']]

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
hdbscan_output.insert(0, 'INDEX', total_payments.index)
hdbscan_output.insert(1, 'labels_hdbscan', labels_hdbscan)
hdbscan_output.insert(2, 'Noise_hdbscan',
                      hdbscan_output['labels_hdbscan'] == -1)
hdbscan_output.insert(3, 'Anomaly_hdbscan',
                      hdbscan.outlier_scores_ > threshold)

# Filter out the a data frame with only noise points & clean
hdbscan_noise = hdbscan_output[hdbscan_output['Anomaly_hdbscan']]

# %%
