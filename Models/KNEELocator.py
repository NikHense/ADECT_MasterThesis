#  %% Import libraries for DBSCAN algorithm
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

# %% Test out the PCA and optimal number of principal components

# Scale the data using StandardScaler
# scaler = StandardScaler()
# input_scaled = scaler.fit_transform(input)

# Perform a PCA on input
pca = PCA(n_components=len(input.columns))
input_pca = pca.fit_transform(input_scaled)

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
plt.axhline(y=0.99, color='b', linestyle='--')
plt.title('Cumulative Proportion of Variance Explained')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.show()

# retrieve the number of components that explain 99% of the variance
n_components = np.argmax(cumulative_variance_ratio >= 0.99)
print(f'{n_components} principal components explain 99% of the variance')

# if n_components > 15 than 15 otherwise n_components = np.argmax(cumulative_variance_ratio >= 0.99)
if n_components > 15:
      n_components = 15
else:
      n_components = np.argmax(cumulative_variance_ratio >= 0.99)               

# %% Calculate the K-Distance Graph

# Perform a PCA on input
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

sensitivity = list(range(1, 10, 1))

# %% Calculate the maximum curvature point of the k-distance graph

# hold knee points for each sensitivity
knees = []
norm_knees = []
for s in sensitivity:
    kl = KneeLocator(distances[:, 0], distances[:, 1], curve='convex', direction='increasing', S=s)
    knees.append(kl.knee)
    norm_knees.append(kl.norm_knee)

print(f'Knees: {knees}')

print(f'Normalized Knees: {[nk.round(2) for nk in norm_knees]}')

plt.style.use('ggplot');
plt.figure(figsize=(10, 10));
plt.plot(kl.x_normalized, kl.y_normalized);
plt.plot(kl.x_difference, kl.y_difference);

for k, s in zip(norm_knees, sensitivity):
    plt.vlines(k, 0, 1, linestyles='--', label=f'S = {s}');
plt.xlim(0, 0.1)
plt.ylim(0.8, 1)
plt.legend()
plt.show()


#------------------------------------------------------------------------------
# %%
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
kneedle = KneeLocator(distances[:, 0], distances[:, 1], S=7,
                      #interp_method='polynomial',
                      curve='convex', direction='increasing')

print(round(kneedle.knee, 0))
print(round(kneedle.elbow_y, 3))

# Normalized data, normalized knee, and normalized distance curve.
plt.style.use('ggplot')
kneedle.plot_knee_normalized()
# plt.xlim(0, 0.01)
# plt.ylim(0.97, 1)
plt.show()

kneedle.plot_knee()
# %%
