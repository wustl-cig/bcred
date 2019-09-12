'''
Class for quadratic-norm on subsampled 2D Fourier measurements
Yu Sun, CIG, WUSTL, 2019
Based on MATLAB code by U. S. Kamilov, CIG, WUSTL, 2017
'''

import numpy as np
import skimage
from util import extract_nonoverlap_patches, putback_nonoverlap_patches
from DataFidelities.DataClass import DataClass


###################################################
###                   CT Class                  ###
###################################################

class RadonClass(DataClass):

    def __init__(self, y, sigSize, theta=None, num_blocks=16, block_size=32):
        self.y = y
        self.sigSize = sigSize
        self.theta = theta   # here theta is a list of degrees
        self.num_blocks = num_blocks
        self.block_size = block_size
    
    def size(self):
        sigSize = self.sigSize
        return sigSize

    def eval(self, x):
        z = self.fmult(x, self.A)
        d = np.linalg.norm(z.flatten('F') - self.y.flatten('F')) ** 2
        return d
    
    def res(self, x):
        z = self.fmult(x, self.theta)
        return z-self.y

    def grad(self, x):
        res = self.res(x)
        g = self.ftran(res, self.theta)
        g = g.real
        d = np.linalg.norm(res.flatten('F')) ** 2
        return g, d
    
    def gradRes(self, res):
        return self.ftran(res, self.theta)

    def gradBloc(self, res, block_idx):  # need to be fixed
        g = self.ftran(res, self.theta)
        g_block = self.ftranBloc(res, block_idx, self.num_blocks, self.block_size, self.theta)
        return g_block

    def fmultPatch(self, patch, block_idx):
        return self.fmultBloc(patch, block_idx, self.num_blocks, self.block_size, self.theta)

    def draw(self, x):
        # plt.imshow(np.real(x),cmap='gray')
        pass


    @staticmethod
    def fmult(x, theta):
        z = skimage.transform.radon(x, theta=theta, circle=False)
        return z
    
    @staticmethod
    def ftran(z, theta): # output_size, theta):
        x = skimage.transform.iradon(z, theta=theta, filter=None, circle=False)
        return x

    @staticmethod
    def ftranBloc(z, block_idx, num_blocks, block_size, theta):
        x = skimage.transform.iradon(z, theta=theta, filter=None, circle=False)
        x_list = extract_nonoverlap_patches(x, num_blocks, block_size)
        return x_list[block_idx,...]

    @staticmethod
    def fmultBloc(x_block, block_idx, num_blocks, block_size, theta):
        dummy = np.zeros([num_blocks,block_size,block_size])
        dummy[block_idx,:,:] = x_block
        dummy_image = putback_nonoverlap_patches(dummy)
        z = skimage.transform.radon(dummy_image, theta=theta, circle=False)
        return z


