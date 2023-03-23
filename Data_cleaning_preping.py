# %% Import libraries for data cleaning
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz
import time
import datetime
import json
import requests
import itertools
from multiprocessing import Pool

# %% Load csv-file into dataframe
df = pd.read_csv('total_payments.csv')
# %% Print out the info of data frame
df.info()

# %% Change specific features to a categorical data type
cat_cols = ['Country_Region_Code','Payment_Method_Code', 'Customer_IBAN','Review_Status', 'Created_By', 'Source_System',  'Mandant']

for col in cat_cols:
    df[col] = df[col].astype('category')

# get the unique categorical values and corresponding codes for the '???' column
    unique_mandant = df['Mandant'].cat.categories
    for i in range(len(unique_mandant)):
        print(f"{unique_mandant[i]}: {i}")


# %% Comparing and retrieving similar 'City' names
starttime = time.time()

#transform the entries of the 'City' column in the df data frame into a list of strings
df['City'] = df['City'].astype(str)
cities = df['City'].tolist()

# define a function to check if two values are similar
def is_similar(a, b):
    return fuzz.partial_ratio(a, b) in range(90,101)  # adjust similarity threshold as needed

# define a separate function to perform the comparison
def get_similarities(city1, cities1):
    similarities = []
    for city2 in cities1:
        if is_similar(city1, city2):
            similarities.append(city2)
    return similarities

# use a hash table to remove duplicates
cities = list(set(cities))

# use multiprocessing to parallelize the comparisons
pool = Pool()
similar_values = {}
threshold = 100
for i, city1 in enumerate(cities):
    similarities = pool.apply_async(get_similarities, (city1, cities[i+1:]))
    similar_cities = similarities.get()
    if len(similar_cities) > 0:
        similar_values[city1] = similar_cities

# print the similar values
for city, similar_cities in similar_values.items():
    print(f"{city}: {'; '.join(similar_cities)}")
    
print(f'Similarity Detection process took {time.time() - starttime:.1f} seconds')
# %%
