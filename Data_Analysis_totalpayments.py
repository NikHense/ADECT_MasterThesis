# %%
import os
# import time
import pandas as pd
import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
from sqlalchemy import create_engine, text
import sweetviz as sv
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import norm
# from sklearn.preprocessing import StandardScaler
# from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# %% params sql connection
SERVER = 'T-SQLDWH-DEV'  # os.environ.get('SERVER')
DB = 'ML'
USERNAME = os.environ.get('USERNAME')
PASSWORD = os.environ.get('PASSWORD')
# %% sql connection
conn = ("Driver={ODBC Driver 18 for SQL Server};"
        "Server="+SERVER+";"
        "Database="+DB+";"
        "UID="+USERNAME+";"
        "PWD="+PASSWORD+";"
        "Encrypt=YES;"
        "TrustServerCertificate=YES")
engine = create_engine(
    f'mssql+pyodbc://?odbc_connect={conn}',
    fast_executemany=True)

# %%
SQL_TOTAL_PAYMENTS = 'SELECT * FROM ADECT.TOTAL_PAYMENTS'
total_payments = pd.DataFrame(engine.connect().execute(
    text(SQL_TOTAL_PAYMENTS)))
# Save data from in csv file 
total_payments.to_csv('total_payments.csv', index=False)

# %% Print out the info of data frame
total_payments.info()
# %%
total_payments.describe()
# %%
total_payments.head()
# %%
# sns.distplot(total_payments['Amount_Applied'])
# %%
# Pairplot of total payments
sns.pairplot(total_payments)

# %%analyze single df, 
# or "compare" two df, "compare_intra" two subsets of df
sv.analyze(total_payments).show_html()