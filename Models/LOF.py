# %%
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import multiprocessing as mp
from multiprocessing import Pool
import seaborn as sns

#-----------------------------------------------------------------------------
# Parameter selection for K-Means
#-----------------------------------------------------------------------------
# %% Start timer
totaltime = time.time()

# %% Execute grid search for K-Means
starttime = time.time()

scaler = StandardScaler()
input_lof_scaled = scaler.fit_transform(input)
input_lof_scaled = pd.DataFrame(input_lof_scaled, columns=input.columns)

input_lof_scaled.mean()
input_lof_scaled.var()

# %% Test out the PCA and optimal number of principal components

# Perform a PCA on input_scaled
pca = PCA(n_components=len(input_lof_scaled.columns))
input_pca = pca.fit_transform(input_lof_scaled)

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
num_components = range(1, len(input_lof_scaled.columns) + 1)
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

# %% Perform a final PCA on input_ lof_scaled with the optimal number of components
pca = PCA(n_components=n_components)
input_pca = pca.fit_transform(input_lof_scaled)

# %%
# %%
# # do a grid search for the best parameters
# for n in range(1, 100):
#     lof = LocalOutlierFactor(n_neighbors=n, n_jobs=-1)
#     lof.fit(input_pca)
#     outlier_scores = lof.negative_outlier_factor_
#     CH_avg = metrics.calinski_harabasz_score(input_pca, outlier_scores)
#     print("For n =", n, "The average calinski_harabasz_score is :", CH_avg)

# %% Run LOF, testing Calinski-Harabasz score
starttime = time.time()

# Initialize empty lists to store results
results_LOF_CH = []

# Define a range of parameter values
n = list(range(1, 151, 1))


# Define a function to compute LOF for a given parameter combination
def LOF_CH(n):
    lof = LocalOutlierFactor(n_neighbors=n, contamination='auto',
                             n_jobs=1)
    labels_lof = lof.fit_predict(input_pca)
    # outlier_scores = lof.negative_outlier_factor_
    CH_score = metrics.calinski_harabasz_score(input_pca,labels_lof)
    n_outlier = np.count_nonzero(labels_lof == -1)
    print("For n =", n, "CH_score :", CH_score, "n_outlier :", n_outlier)
    return n, CH_score, n_outlier


# Create a Pool object
with Pool() as pool:
    # Compute DBSCAN for all parameter combinations in parallel
    results_LOF_CH = pool.map(LOF_CH, n)
    pool.close()
    pool.join()

# Print the time the process took
minutes = int((time.time() - starttime) / 60)
seconds = int((time.time() - starttime) % 60)
print(f'LOF grid search process took {minutes} minutes and '
      f'{seconds} seconds')
# %% Store the results in a pandas DataFrame
results_LOF_CH = pd.DataFrame(results_LOF_CH,
                              columns=['n', 'CH_score', 'n_outlier'])

results_LOF_CH['CH_score'] = results_LOF_CH['CH_score'].astype('float64')

# Find the parameter combination with the highest score
best_LOF_CH = results_LOF_CH.loc[results_LOF_CH['CH_score'].idxmax()]
best_n = best_LOF_CH['n'].astype('int64')


# %%
# def LOF_CH(n, input_pca):
#     lof = LocalOutlierFactor(n_neighbors=n, contamination='auto', n_jobs=1)
#     labels_lof = lof.fit_predict(input_pca)
#     outlier_scores = lof.negative_outlier_factor_
#     CH_score = metrics.calinski_harabasz_score(input_pca, outlier_scores)
#     n_outlier = np.count_nonzero(labels_lof == -1)
#     print(f"For n = {n}, CH_score: {CH_score:.4f}, n_outlier: {n_outlier}")
#     return n, CH_score, n_outlier

# if __name__ == '__main__':
#     starttime = time.time()

#     # Define input_pca here

#     n = list(range(1, 4000, 50))

#     with mp.Pool() as pool:
#         results = [pool.apply_async(LOF_CH, (i, input_pca)) for i in n]
#         output = [p.get() for p in results]

#     output = np.array(output)

#     results_LOF_CH = pd.DataFrame(data=output, columns=['n', 'CH_score', 'n_outlier'])

#     # Print the time the process took
#     minutes = int((time.time() - starttime) / 60)
#     seconds = int((time.time() - starttime) % 60)
#     print(f"LOF grid search process took {minutes} minutes and {seconds} seconds")


# %% Plot the silhouette scores
plt.plot(results_LOF_CH['n'], results_LOF_CH['CH_score'], 'ro-', linewidth=2)
plt.title('CH_Score over n')
plt.xlabel('Number of Neighbors')
plt.ylabel('Calinski-Harabasz Score')
plt.show()

# plot number of outliers vs. n (change style of plot)
plt.plot(results_LOF_CH['n'], results_LOF_CH['n_outlier'], 'ro-', linewidth=2)
plt.title('Number of Outliers over n')
plt.xlabel('Number of Neighbors')
plt.ylabel('Number of Outliers')
# plt.ylim(5000,7000)
plt.show()

# %%

lof_best = LocalOutlierFactor(n_neighbors=best_n, contamination='auto',
                             n_jobs=-1)
labels_lof = lof_best.fit_predict(input_pca)
CH_score = round(metrics.calinski_harabasz_score(input_pca,labels_lof), 4)
n_outlier = np.count_nonzero(labels_lof == -1)

print("For n =", best_n, "CH_score :", CH_score, "n_outlier :", n_outlier)


# %%
# Invert the scaling applied by StandardScaler
lof_output = scaler.inverse_transform(input_lof_scaled)

# # Invert the PCA transformation
# input_pca_inverted = pca.inverse_transform(input_pca)

# Convert the dbscan_output array to a pandas DataFrame
lof_output = pd.DataFrame(lof_output, columns=input_lof_scaled.columns)

# Add the labels column to the dbscan_output at position 0
lof_output.insert(0, 'INDEX', input.index)
lof_output.insert(1, 'labels_lof', labels_lof)
lof_output.insert(2, 'Anomaly_lof', lof_output['labels_lof'] == -1)

# Filter out the a data frame with only noise points
lof_noise = lof_output[lof_output['Anomaly_lof']]


# %%
