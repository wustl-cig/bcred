from DataFidelities.RandomClass import RandomClass
from Regularizers.robjects_tf import *
from iterAlgs import *
from util import *

import scipy.io as sio
import numpy as np
import os


####################################################
####              HYPER-PARAMETERS               ###
####################################################

# indicate the GPU index if available. If not, just leave it
gpu_ind = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind;

# set the random seed, please do not comment this line
np.random.seed(128)

## optimal tau values for residual DnCNN with Random Matrix with 30 dB input SNR noise
DnCNN_taus_30dB = [0.3947665428470804, 0.33916115451527434, 0.31724239698349554, 0.39744749839135796,
                    0.2556451911121183, 0.4109487505623313, 0.6721322329761701, 0.322682764758653,
                    0.386136952197389, 0.3410113490614658]

## optimal tau values for residual DnCNN with Random Matrix with 40 dB input SNR noise
DnCNN_taus_40dB = [0.20051761636796708, 0.23342255207802717, 0.25608225198595563, 0.20984552245851243,
                    0.1944676571465673, 0.355732263964717, 0.28946240847511034, 0.28102085922777414,
                    0.3154775141742967, 0.2559106590509435]

# you can change the save path here
save_root = 'results/Demo_DnCNNstar_Random/'

# allocating folders
abs_save_path = os.path.abspath(save_root)
if os.path.exists(save_root):
    print("Removing '{:}'".format(abs_save_path))
    shutil.rmtree(abs_save_path, ignore_errors=True)
# make new path
print("Allocating '{:}'".format(abs_save_path))
os.makedirs(abs_save_path)

####################################################
####              DATA PREPARATION               ###
####################################################

# here we load all 10 test images
data_name = 'Knee_10'
data = sio.loadmat('data/{}.mat'.format(data_name), squeeze_me=True)
imgs = np.squeeze(data['img'])

# prepare for the data info
sigSize = np.array(imgs[...,0].shape)
num_blocks = 16
block_size = 40

downsample_rate = 0.71 # 0.71*0.71 = 0.5
noiseLevel = 40    # two noise levels {30,40} are available for the simulation.

# generate random measurement matrix used for all 10 images.
print()
print('Generating random measurement matrix . . .')
A, A_patches = RandomClass.genMeas(sigSize, num_blocks, block_size, downsample_rate=downsample_rate)
print('. . . Done')
print()

# number of iterations
iters = 100

####################################################
####            NETWORK INITIALIZATION           ###
####################################################

#-- Network Hyperparameters --#
input_channels = 1
truth_channels = 1

#-- Network Setup --#
# select the DnCNNstar model
# - Please use 'residual_DnCNNstar_LC=2/DnCNN_layer=7_sigma=5' to generate the optimal results for DnCNNstar.
model_name = 'DnCNN_layer=7_sigma=5'
model_path = 'models/residual_DnCNNstar_LC=2/{}/model.cpkt'.format(model_name)

####################################################
####                LOOP IMAGES                  ###
####################################################

numImgs = imgs.shape[2]
bcred_output = {}
red_output = {}
bcred_dist = np.zeros(iters)
red_dist  = np.zeros(iters)
bcred_snr = np.zeros(iters)
red_snr   = np.zeros(iters)

# select which image you want to reconstruct. By default we use the sixth image.
startIndex = 5
endIndex = 6

for i in range(startIndex,endIndex):

    # extract truth
    x = imgs[...,i]
    xtrue = x
    sigSize = np.array(x.shape)

    # measure
    y = RandomClass.fmult(x, A)

    # add white gaussian noise
    y,_ = util.addwgn(y, noiseLevel)

    ####################################################
    ####                    DnCNN                    ###
    ####################################################

    tau = DnCNN_taus_40dB[i]

    #-- Reconstruction --# 
    dObj = RandomClass(y, sigSize, A, A_patches)
    rObj = DnCNNClass(sigSize, tau, model_path, img_channels=input_channels, truth_channels=truth_channels)
    # rObj = TVClass(sigSize, 0.1, 0.001, maxiter=20)   # Qualitative analysis, parameters not optimized

    print()
    print('#######################')
    print('#### BCRED (epoch) ####')
    print('#######################')
    print()
    
    # - To try out direct DnCNN, set useNoise to False.
    # - To denoise with full denoiser, set pad to None.
    # - To denoise with block-wise denoiser, set pad to some scalar (5 by default).
    # We set the step-size to be 1/(L+2*tau)
    bcred_recon, bcred_out = bcredEst(dObj, rObj,
                            num_patch=num_blocks, patch_size=block_size, pad=5, numIter=iters, step=1/(2+2*tau),
                            useNoise=True, verbose=True, xtrue=xtrue)
    bcred_out['recon'] = bcred_recon

    print()
    print('###################')
    print('####    RED    ####')
    print('###################')
    print()
    
    # - To try out direct DnCNN, set useNoise to False.
    # - To save intermediate results, set is_save to True.
    red_recon, red_out = redEst(dObj, rObj,
                        numIter=iters, step=1/(6+2*tau), accelerate=False, mode='RED', useNoise=True, 
                        verbose=True, xtrue=xtrue)  # set useNoise to False if you want to try out direct DnCNN
    red_out['recon'] = red_recon

    # save out info
    bcred_output['img_{}'.format(i)] = bcred_out
    red_output['img_{}'.format(i)] = red_out

    sio.savemat(save_root + 'bcred_out.mat', bcred_output)
    sio.savemat(save_root + 'red_out.mat', red_output)

    bcred_dist = bcred_dist + np.array(bcred_out['dist'])
    red_dist = red_dist + np.array(red_out['dist'])

    bcred_snr = bcred_snr + np.array(bcred_out['snr'])
    red_snr = red_snr + np.array(red_out['snr'])

####################################################
####            PlOTTING CONVERGENCE             ###
####################################################

import matplotlib.pyplot as plt

num = endIndex -  startIndex

# compute the averaged distance to fixed points
avgDistBcred = np.squeeze(bcred_dist / num)
avgDistRed = np.squeeze(red_dist / num)
avgSnrBcred = np.squeeze(bcred_snr / num)
avgSnrRed = np.squeeze(red_snr / num)
xRange  = np.linspace(0,iters,iters)

fig, (ax1, ax2) = plt.subplots(1, 2)
# Convergence Plot
ax1.semilogy(xRange, avgDistBcred, label='BC-RED (epoch)')
ax1.semilogy(xRange, avgDistRed, label='RED')
ax1.set_xlim(0,iters)
ax1.set_ylim(1e-7,1)
ax1.set_xlabel('iteration')
ax1.set_ylabel('accuracy')
ax1.set_title('Convergence plot for BC-RED and RED')

plt.legend()

# SNR Plot
ax2.plot(xRange, avgSnrBcred, label='BC-RED (epoch)')
ax2.plot(xRange, avgSnrRed, label='RED')
ax2.set_xlim(0,iters)
ax2.set_ylim(0,30)
ax2.set_xlabel('iteration')
ax2.set_ylabel('SNR (dB)')
ax2.set_title('SNR plot for BC-RED and RED')

plt.legend()
plt.show()
