# %% Import libraries for isolation forest
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# %% Create the isolation forest function
isof = IsolationForest(max_samples='auto', contamination= 'auto')

isof.fit(total_payments_if)

# %%
