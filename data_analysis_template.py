''' Quick Data Exploration '''
# %%
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sqlalchemy import create_engine, text
import sweetviz as sv
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import norm
# from sklearn.preprocessing import StandardScaler
# from scipy import stats
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline
# %%
# df = pd.read_csv('file.csv')
# %%
# %% params sql connection
SERVER = 'T-SQLDWH-DEV' #os.environ.get('SERVER')
DB = 'HUB'
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
# %% sql query
SQL_QUERY1 = 'SELECT * FROM PAPO.RELN_RE_Payment_Entry'
SQL_QUERY2 = 'SELECT * FROM PAPO.RELN_RE_Payment_Line'
# %% read
#starttime = time.time()
df1= pd.DataFrame(engine.connect().execute(text(SQL_QUERY1)))
df2 = pd.DataFrame(engine.connect().execute(text(SQL_QUERY2)))
#print(f'Database read process took {time.time() - starttime} seconds')
# %%
#df1.info()
df2.info()
# %%
df.describe()  # .apply(lambda s: s.apply(lambda x: format(x, 'f')))
# %%
df1.head()
df2.head()
# %%
sns.distplot(df['Quantity'])
# %%
sns.pairplot(df)
# %%
df_piv = df.pivot_table(index='DayLongName',
                        columns='Quantity',
                        values='Hours_to_dispatch')
# %%
# sns.heatmap(df_piv, annot=True)
# %%
cmap = LinearSegmentedColormap.from_list('gr', ["g", "y", "r"], N=256)
ax = plt.axes()
ax.set_title('Ø Hours_to_dispatch')
sns.heatmap(df_piv, annot=True, cmap=cmap, ax=ax)
plt.show()
# %%
# df.plot.scatter(x='Quantity', y='Amount_Net_LCY')  # opt: xlim, ylim
# %%
df.groupby('Quantity').agg(['max', 'min', 'count', 'median', 'mean'])
# %%
# analyze single df, other: "compare" two df, "compare_intra" two subsets of df
sv.analyze(df1).show_html()
# %%
# penguins = sns.load_dataset("penguins")
# sns.pairplot(penguins)
# %%
# sns.pairplot(penguins, hue="species")
# %%
# sns.pairplot(penguins, hue="species", diag_kind="hist")
# %%
# sv.analyze(penguins).show_html()
# %%
