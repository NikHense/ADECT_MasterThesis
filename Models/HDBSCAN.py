# %% Import libraries for HDBSCAN algorithm
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.neighbors import NearestNeighbors
import hdbscan
# from sklearn import metrics
# from multiprocessing import Pool
# from kneed import KneeLocator

# %% Start timer
totaltime = time.time()

# %% Test out the PCA and optimal number of principal components

# Scale the data using StandardScaler
scaler = StandardScaler()
kmeans_data_scaled = scaler.fit_transform(kmeans_data)

# Perform a PCA on kmeans_data
pca = PCA(n_components=len(kmeans_data.columns))
kmeans_data_pca = pca.fit_transform(kmeans_data_scaled)

# # Method 1: Scree plot
# # Plot the eigenvalues of the principal components
# plt.plot(range(1, pca.n_components_ + 1),
#          pca.explained_variance_ratio_, 'ro-', linewidth=2)
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Eigenvalue')
# plt.show()

# Method 2: Cumulative proportion of variance explained
# Calculate the cum. proportion of variance explained
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot the cum. proportion of variance explained, with the 95% threshold
plt.plot(range(1, pca.n_components_ + 1),
         cumulative_variance_ratio, 'ro-', linewidth=2)
plt.axhline(y=0.95, color='b', linestyle='--')
plt.title('Cumulative Proportion of Variance Explained')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.show()

# retrieve the number of components that explain 90% of the variance
n_components = np.argmax(cumulative_variance_ratio >= 0.95)
print(f'{n_components} principal components explain 95% of the variance')

# %% Run HDBSCAN algorithm
starttime = time.time()
# Compute the clustering using HDBSCAN
hdbscan = hdbscan.HDBSCAN(metric='euclidean', 
                          min_cluster_size=10,
                          allow_single_cluster=True)
hdbscan.fit(kmeans_data_pca)

# Number of clusters in labels, ignoring noise if present.
labels_hdbscan = hdbscan.labels_
n_clusters_hdbscan = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan
                                                 else 0)
n_noise_hdbscan = list(labels_hdbscan).count(-1)

print('Estimated number of clusters: %d' % n_clusters_hdbscan)
print('Estimated number of noise points: %d' % n_noise_hdbscan)

print(f'HDBSCAN process took {time.time() - starttime} seconds')

# %%
# Invert the scaling applied by StandardScaler
hdbscan_output = scaler.inverse_transform(kmeans_data_scaled)

# Convert the dbscan_output array to a pandas DataFrame
hdbscan_output = pd.DataFrame(hdbscan_output, columns=kmeans_data.columns)

# Add the labels column to the dbscan_output at position 0
hdbscan_output.insert(0, 'INDEX', kmeans_data.index)
hdbscan_output.insert(1, 'labels', labels_hdbscan)
hdbscan_output.insert(2, 'noise', hdbscan_output['labels'] == -1)

hdbscan_noise = hdbscan_output[hdbscan_output['noise'] == True]

# %%Check wheter dbscan_noise points are in dbscan_noise based on index

# assuming both dataframes have columns called 'index'
merged = pd.merge(hdbscan_noise, dbscan_noise, on='INDEX', how='inner')
matching_obs = merged['INDEX'].tolist()

# Calculate number of noise points from dbscan_noise that are in if_noise
print(len(matching_obs),
      f'of {len(hdbscan_noise)} noise points in total are similar')
print('Percentage: ',
      round(len(matching_obs)/len(hdbscan_noise)*100, 2), '%')
 

# assuming both dataframes have columns called 'index'
merged_outer = pd.merge(hdbscan_noise, dbscan_noise, on='INDEX', how='outer', indicator=True)

# get rows that are only in one of the dataframes
not_in_both = merged_outer.loc[merged_outer['_merge'].isin(['left_only', 'right_only'])]

# Calculate number of noise points from left_only and right_only
left_only = not_in_both.loc[not_in_both['_merge'].isin(['left_only'])]
right_only = not_in_both.loc[not_in_both['_merge'].isin(['right_only'])]
print('Only in hdbscan_noise: ', len(left_only))
print('Only in dbscan_noise: ', len(right_only))

# %%
