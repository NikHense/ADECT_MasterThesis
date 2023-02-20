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
SERVER = os.environ.get('SERVER')
DB = 'DMA'
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
SQL_QUERY = '''
SELECT
 sl.Sales_Header_Code
,sl.Quantity
,sl.Amount_Net_LCY
,sl.Discount_Amount_Net_LCY
,sl.Hours_to_dispatch
,CONCAT(d.DayOfWeekNumber,' ',d.DayLongName) AS DayLongName
FROM DMA.ONLN.FACT_Sales_Line sl
INNER JOIN DMA.ONLN.DIM_Date d ON sl.FK_Date_Order = d.PK_Date
WHERE Quantity > 0
AND FK_Date_Order > 20230101
'''
# %% read
starttime = time.time()
df = pd.DataFrame(engine.connect().execute(text(SQL_QUERY)))
print(f'Database read process took {time.time() - starttime} seconds')
# %%
df.info()
# %%
df.describe()  # .apply(lambda s: s.apply(lambda x: format(x, 'f')))
# %%
df.head()
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
sv.analyze(df).show_html()
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
