# %%
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from kneed import KneeLocator
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from multiprocessing import Pool
# import seaborn as sns


# %% Start timer
totaltime = time.time()

# %% Execute grid search for K-Means
starttime = time.time()

# scaler = StandardScaler()
# input_lof_scaled = scaler.fit_transform(input)
# input_lof_scaled = pd.DataFrame(input_lof_scaled, columns=input.columns)

# # input_lof_scaled.mean()
# # input_lof_scaled.var()

# # %% Test out the PCA and optimal number of principal components

# # Perform a PCA on input_scaled
# pca = PCA(n_components=len(input_lof_scaled.columns))
# input_pca = pca.fit_transform(input_lof_scaled)

# # # # Method 1: Scree plot
# # # # Plot the eigenvalues of the principal components
# # # plt.plot(range(1, pca.n_components_ + 1),
# # #          pca.explained_variance_ratio_, 'ro-', linewidth=2)
# # # plt.title('Scree Plot')
# # # plt.xlabel('Principal Component')
# # # plt.ylabel('Eigenvalue')
# # # plt.show()

# # Method 2: Cumulative proportion of variance explained
# # Calculate the cum. proportion of variance explained
# cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# # Plot the cum. proportion of variance explained, with the 99% threshold
# num_components = range(1, len(input_lof_scaled.columns) + 1)
# plt.plot(num_components, cumulative_variance_ratio, 'ro-', linewidth=2)
# plt.axhline(y=0.99, color='b', linestyle='--')
# plt.title('Cumulative Proportion of Variance Explained')
# plt.xlabel('Number of Principal Components')
# plt.ylabel('Cumulative Proportion of Variance Explained')
# # Find the index of the element in y_values that is closest to 0.99
# threshold_idx = (np.abs(cumulative_variance_ratio - 0.99)).argmin()

# # Get the x-coordinate of the threshold
# threshold_x = num_components[threshold_idx]

# # Add a vertical line at the threshold x-coordinate
# plt.axvline(x=threshold_x, color='b', linestyle='--')
# plt.show()

# # retrieve the number of components that explain 99% of the variance
# n_components = np.argmax(cumulative_variance_ratio >= 0.99)

# print(f'{n_components} principal components explain 99% of the variance')

# # if n_components > 15 than 15 otherwise
# # n_components = np.argmax(cumulative_variance_ratio >= 0.99)
# if n_components > 15:
#     n_components = 15
# else:
#     n_components = np.argmax(cumulative_variance_ratio >= 0.99)

# %% Perform a final PCA on input_lof_scaled w/ best num. of components
# pca = PCA(n_components=n_components)
# input_pca = pca.fit_transform(input_lof_scaled)

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
n = list(range(1, 51, 1))


# Define a function to compute LOF for a given parameter combination
def LOF_CH(n):
    lof = LocalOutlierFactor(n_neighbors=n,
                             contamination='auto',
                             n_jobs=1)
    labels_lof = lof.fit_predict(input_pca)
    # outlier_scores = lof.negative_outlier_factor_
    CH_score = metrics.calinski_harabasz_score(input_pca, labels_lof)
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

# # Find the parameter combination with the highest score
# best_LOF_CH = results_LOF_CH.loc[results_LOF_CH['CH_score'].idxmax()]
# best_n = best_LOF_CH['n'].astype('int64')

# %% Plot the Calinski-Harabasz scores
sns.lineplot(x=results_LOF_CH['n'], y=results_LOF_CH['CH_score'], linewidth=2)
plt.title('CH_Score over n')
plt.xlabel('Number of Neighbors')
plt.ylabel('Calinski-Harabasz Score')
plt.show()

# plot number of outliers vs. n (change style of plot)
plt.plot(results_LOF_CH['n'], results_LOF_CH['n_outlier'], linewidth=2)
plt.title('Number of Outliers over n')
plt.xlabel('Number of Neighbors')
plt.ylabel('Number of Outliers')
# plt.ylim(5000,7000)
plt.show()

# %% Calculate the maximum curvature point of the number of outliers
# num_outliers = results_LOF_CH['n_outlier']
# num_outliers = np.array(num_outliers)

# x_lof = results_LOF_CH['n']
# x_lof = np.array(x_lof)

# kneedle = KneeLocator(x_lof, num_outliers,
#                       S=1,
#                       #  interp_method='polynomial',
#                       curve='convex', direction='decreasing')

# print(round(kneedle.elbow, 0))
# print(round(kneedle.elbow_y, 3))

# plt.style.use('ggplot')
# kneedle.plot_knee_normalized()
# # plt.xlim(0, 0.05)
# # plt.ylim(0.95, 1)
# plt.show()

# kneedle.plot_knee()

# Get the best n with highest CH_score
best_n = results_LOF_CH.loc[results_LOF_CH['CH_score'].idxmax()]['n']
# best_n = kneedle.elbow

# %% Run LOF
lof = LocalOutlierFactor(n_neighbors=20,
                         contamination='auto',
                         n_jobs=-1)
labels_lof = lof.fit_predict(input_pca)
# CH_score = round(metrics.calinski_harabasz_score(input_pca, labels_lof), 4)
n_outlier = np.count_nonzero(labels_lof == -1)

print("For n =", 20,
    #   "CH_score :", CH_score,
      "n_outlier :", n_outlier)

# %% 
# Invert the scaling applied by StandardScaler
lof_output = scaler.inverse_transform(input_scaled)

# Invert the PCA transformation
# input_pca_inverted = pca.inverse_transform(input_pca)

# Convert the dbscan_output array to a pandas DataFrame
lof_output = pd.DataFrame(lof_output, columns=input_scaled.columns)

# Add the labels column to the dbscan_output at position 0
lof_output.insert(0, 'INDEX', total_payments_academic.index)
lof_output.insert(1, 'labels_lof', labels_lof)
lof_output.insert(2, 'Anomaly_lof', lof_output['labels_lof'] == -1)

# Filter out the a data frame with only noise points
lof_noise = lof_output[lof_output['Anomaly_lof']]

# %%
