# %% Import libraries for HDBSCAN algorithm
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.neighbors import NearestNeighbors
import hdbscan
# from sklearn import metrics
# from multiprocessing import Pool
# from kneed import KneeLocator

# %% Start timer
totaltime = time.time()

# %% Run HDBSCAN algorithm
# Compute the clustering using HDBSCAN
hdbscan = hdbscan.HDBSCAN(metric='euclidean',
                          min_cluster_size=100,
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
sns.distplot(hdbscan.outlier_scores_[np.isfinite(hdbscan.outlier_scores_)], rug=True)

plt.show()

# DEfined the threshold for the outlier scores at 95% quantile
threshold = pd.Series(hdbscan.outlier_scores_).quantile(0.90)
outliers = np.where(hdbscan.outlier_scores_ > threshold)[0]

print(f'Number of outliers: {len(outliers)}')


# %%
# Invert the scaling applied by StandardScaler
hdbscan_output = scaler.inverse_transform(input_scaled)

# Convert the dbscan_output array to a pandas DataFrame
hdbscan_output = pd.DataFrame(hdbscan_output, columns=input.columns)

# Add the labels column to the dbscan_output at position 0
hdbscan_output.insert(0, 'INDEX', input.index)
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
