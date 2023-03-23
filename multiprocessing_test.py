# https://urban-institute.medium.com/using-multiprocessing-to-make-python-code-faster-23ea5ef996ba
# Process sends code to a processor as soon as the process is started. Pool sends a code to each available processor and doesn’t send any more until a processor has finished computing the first section of code. Pool does this so that processes don’t have to compete for computing resources, but this makes it slower than Process in cases where each process is lengthy.
# %%
import time
import multiprocessing 
import pandas as pd
# %%
def basic_func(x):
    if x == 0:
        return 'zero'
    elif x%2 == 0:
        return 'even'
    else:
        return 'odd'
# %%
def multiprocessing_func(x):
    y = x*x
    #time.sleep(2)
    print('{} squared results in a/an {} number'.format(x, basic_func(y)))
# %% multi process: better with few lengthy processes
if __name__ == '__main__':
    starttime = time.time()
    processes = []
    for i in range(0,1000):
        p = multiprocessing.Process(target=multiprocessing_func, args=(i,))
        processes.append(p)
        p.start()
    for process in processes:
        process.join() 
    print('That took {} seconds'.format(time.time() - starttime))
# %% pool: better with many short processes
# to test: comment out time.sleep(2) in multiprocessing_func
if __name__ == '__main__':
    
    starttime = time.time()
    pool = multiprocessing.Pool()
    pool.map(multiprocessing_func, range(0,1000))
    pool.close()
    print('That took {} seconds'.format(time.time() - starttime))
# %% test write func
r = range(0,4)
df_write = pd.DataFrame()
df_write['colA'] = r
print(df_write)
# %%
def write_func(x):
    df_write.loc[x,'colA'] = x*x
    print(df_write.loc[x,'colA'])
    return df_write.loc[x,'colA']
# %%
if __name__ == '__main__':
    starttime = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.map(write_func, r)
    print(results)
    print('That took {} seconds'.format(time.time() - starttime))
# %%
