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
SERVER = 'T-SQLDWH-DEV'  #os.environ.get('SERVER')
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
# %% sql query
SQL_QUERY1 = 'SELECT * FROM PAPO.BFSN_Outletcity_Metzingen_GmbH_Change_Log_Entry'
SQL_QUERY2 = 'SELECT * FROM PAPO.BFSN_Outletcity_Metzingen_GmbH_Payment_Proposal'
SQL_QUERY3 = 'SELECT * FROM PAPO.BFSN_Outletcity_Metzingen_GmbH_Payment_Proposal_Head'
SQL_QUERY4 = 'SELECT * FROM PAPO.BFSN_Outletcity_Metzingen_GmbH_Payment_Proposal_Line'
SQL_QUERY5 = 'SELECT * FROM PAPO.BFSN_Outletcity_Metzingen_GmbH_Vendor'
SQL_QUERY6 = 'SELECT * FROM PAPO.BFSN_Outletcity_Metzingen_GmbH_Vendor_Bank_Account'
SQL_QUERY7 = 'SELECT * FROM PAPO.BFSN_RELN_Change_Log_Entry'
SQL_QUERY8 = 'SELECT * FROM PAPO.BFSN_RELN_Payment'
SQL_QUERY9 = 'SELECT * FROM PAPO.BFSN_RELN_Vendor'
SQL_QUERY10 = 'SELECT * FROM PAPO.RELN_Change_Log_Entry'
SQL_QUERY11 = 'SELECT * FROM PAPO.RELN_RE_Payment_Entry'
SQL_QUERY12 = 'SELECT * FROM PAPO.RELN_RE_Payment_Line'
SQL_QUERY13 = 'SELECT * FROM PAPO.RELN_RE_Payment_Proposal'
SQL_QUERY14 = 'SELECT * FROM PAPO.RELN_RE_Payment_Proposal_Entry'
SQL_QUERY15 = 'SELECT * FROM PAPO.RELN_RE_Payment_Proposal_Line'
SQL_QUERY16 = 'SELECT * FROM PAPO.RELN_RE_Payment_Transaction'
SQL_QUERY17 = 'SELECT * FROM PAPO.RELN_RE_Vendor_Bank_Account'
SQL_QUERY18 = 'SELECT * FROM PAPO.RELN_Vendor'
SQL_QUERY19 = 'SELECT * FROM PAPO.RELN_Vendor_Ledger_Entry'
# %% read
starttime = time.time()
BFSN_Change_Log_Entry = pd.DataFrame(engine.connect().execute(text(SQL_QUERY1)))
BFSN_Paym_Prop = pd.DataFrame(engine.connect().execute(text(SQL_QUERY2)))
BFSN_Paym_Prop_Head = pd.DataFrame(engine.connect().execute(text(SQL_QUERY3)))
BFSN_Paym_Prop_Line = pd.DataFrame(engine.connect().execute(text(SQL_QUERY4)))
BFSN_Vendor = pd.DataFrame(engine.connect().execute(text(SQL_QUERY5)))
BFSN_Vendor_Bank = pd.DataFrame(engine.connect().execute(text(SQL_QUERY6)))
BFSN_RELN_Change_Log_Entry = pd.DataFrame(engine.connect().execute(text(SQL_QUERY7)))
BFSN_RELN_Paym = pd.DataFrame(engine.connect().execute(text(SQL_QUERY8)))
BFSN_RELN_Vendor = pd.DataFrame(engine.connect().execute(text(SQL_QUERY9)))
RELN_Change_Log_Entry = pd.DataFrame(engine.connect().execute(text(SQL_QUERY10)))
RELN_Paym_Entry = pd.DataFrame(engine.connect().execute(text(SQL_QUERY11)))
RELN_Paym_Line = pd.DataFrame(engine.connect().execute(text(SQL_QUERY12)))
RELN_Paym_Prop = pd.DataFrame(engine.connect().execute(text(SQL_QUERY13)))
RELN_Paym_Prop_Entry = pd.DataFrame(engine.connect().execute(text(SQL_QUERY14)))
RELN_Paym_Prop_Line = pd.DataFrame(engine.connect().execute(text(SQL_QUERY15)))
RELN_Paym_Transaction = pd.DataFrame(engine.connect().execute(text(SQL_QUERY16)))
RELN_Vendor_Bank = pd.DataFrame(engine.connect().execute(text(SQL_QUERY17)))
RELN_Vendor = pd.DataFrame(engine.connect().execute(text(SQL_QUERY18)))
RELN_Vendor_LedgerEntry = pd.DataFrame(engine.connect().execute(text(SQL_QUERY19)))
print(f'Database read process took {time.time() - starttime} seconds')
# %% Print out the info of data frame
RELN_Paym_Prop_Entry.info()
# %%
RELN_Vendor_Bank.describe()  # .apply(lambda s: s.apply(lambda x: format(x, 'f')))
# %%
RELN_Vendor_Bank.head()
df2.head()
# %%
sns.distplot(df1['Quantity'])
# %%
sns.pairplot(df1)
# %%
df_piv = df1.pivot_table(index='DayLongName',
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
df1.groupby('Quantity').agg(['max', 'min', 'count', 'median', 'mean'])
# %%
# analyze single df, other: "compare" two df, "compare_intra" two subsets of df
sv.analyze(RELN_Paym_Prop_Line).show_html()
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
