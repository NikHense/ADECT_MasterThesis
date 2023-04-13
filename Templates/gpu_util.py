# %%
##########################
### GPU test performance
##########################
# %%
from numba import jit
import numpy as np
from timeit import default_timer as timer
# %% To run on CPU
def func(a):
    for i in range(100000000):
        a[i]+= 1
# %% To run on GPU
@jit
def func2(x):
    return x+1
if __name__=="__main__":
    n = 100000000
    a = np.ones(n, dtype = np.float64)
    start = timer()
    func(a)
    print("without GPU:", timer()-start)
    start = timer()
    func2(a)
    #numba.cuda.profile_stop()
    print("with GPU:", timer()-start)
# %%
#######################################
### CUDA test
#######################################
# %%
from distutils.command.install_egg_info import to_filename
import GPUtil
import torch
import os
import tensorflow as tf
import cuda
# %%
GPUtil.getAvailable()
# %%
GPUs = GPUtil.getGPUs()
# %%
GPUs
# %%
GPUavailability = GPUtil.getAvailability(
        GPUs, maxLoad = 0.5, maxMemory = 0.5
        , includeNan=True, excludeID=[], excludeUUID=[])
# %%
GPUavailability
############### pytorch
# %%
torch.zeros(1).cuda()
# %%
use_cuda = torch.cuda.is_available()
# %%
use_cuda
# %%
if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
# %%
device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ",device)
# &&
# %%
DEVICE_ID = GPUs[0]
# %%
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
# %%
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# %%
tf.config.list_physical_devices()
# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# %%
device = '/gpu:0'
print('Device ID (unmasked): ' + str(DEVICE_ID))
print('Device ID (masked): ' + str(0))
# %%
with tf.compat.v1.Session() as sess:
    # Select the device
    with tf.device(device):
        print(device)
        # Declare two numbers and add them together in TensorFlow
        a = tf.constant(42)
        b = tf.constant(5)
        result = sess.run(a/b)
        print('a+b=' + str(result))
# %%