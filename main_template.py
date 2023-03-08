"""Project Name & Short Description"""
# %%
import os
import time
import datetime
import requests
import json
import pandas as pd
from sqlalchemy import create_engine, text
# %% params sql connection
PROJECT_NAME = 'Test Project'
SERVER = 'T-SQLDWH-DEV' #os.environ.get('SERVER')
DB = 'HUB'
USERNAME = os.environ.get('USERNAME')
PASSWORD = os.environ.get('PASSWORD')
SLACK_WEBHOOK = os.environ.get('SLACK_WEBHOOK')
# %% slack alert start
msg_start = f'{datetime.datetime.fromtimestamp(time.time())}: ' \
            f'{PROJECT_NAME} started.'
# %% slack alert start: POST
response = requests.post(
    SLACK_WEBHOOK,
    headers={'Content-Type': 'application/json'},
    data=json.dumps({"text": msg_start})
    )
if response.status_code != 200:
    raise ValueError(
        'Request to slack returned an error %s, the response is:\n%s'
        % (response.status_code, response.text)
    )
# %%
starttime_main = time.time()
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
# %% sql query (select data base)
SQL_QUERY = 'SELECT * FROM PAPO.RELN_RE_Payment_Entry'
# %% read
starttime = time.time()
df = pd.DataFrame(engine.connect().execute(text(SQL_QUERY)))
print(f'Database read process took {time.time() - starttime:.1f} seconds')

# %%
# print(f'Main process took {time.time() - starttime_main:.1f} seconds overall')
# # %% slack alert finish
# msg_finish = f'{datetime.datetime.fromtimestamp(time.time())}: ' \
#              f'{PROJECT_NAME} finished in ' \
#              f'{time.time() - starttime_main:.1f} seconds'
# # %% slack alert finish: POST
# response = requests.post(
#     SLACK_WEBHOOK,
#     headers={'Content-Type': 'application/json'},
#     data=json.dumps({"text": msg_finish})
#     )
# if response.status_code != 200:
#     raise ValueError(
#         'Request to slack returned an error %s, the response is:\n%s'
#         % (response.status_code, response.text)
#     )
# %%'

