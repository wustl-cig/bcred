'''
Class for quadratic-norm on subsampled 2D Fourier measurements
Jianxing Liao, CIG, WUSTL, 2018
Based on MATLAB code by U. S. Kamilov, CIG, WUSTL, 2017
'''

import numpy as np
import math
import sys
import decimal
from util import extract_nonoverlap_patches, putback_nonoverlap_patches
from DataFidelities.DataClass import DataClass

###################################################
###                  MRI Class                  ###
###################################################

class FourierClass(DataClass):
    def __init__(self, y, mask, num_blocks=16, block_size=32):
        self.y = y
        self.mask = mask
        self.sigSize = mask.shape
        self.num_blocks = num_blocks
        self.block_size = block_size
    
    def size(self):
        sigSize = self.sigSize
        return sigSize

    def eval(self,x):
        z = self.fmult(x, self.mask)
        d = 0.5 * np.power(np.linalg.norm(self.y.flatten('F')-z.flatten('F')),2)
        return d

    def res(self, x):
        z = self.fmult(x, self.mask)
        return z-self.y

    def grad(self, x):
        res = self.res(x)
        g = self.ftran(res, self.mask)
        g = g.real
        d = np.linalg.norm(res.flatten('F')) ** 2
        return g,d

    def gradRes(self, res):
        return self.ftran(res, self.mask).real

    def gradBloc(self, res, block_idx):  # need to be fixed
        g = self.ftran(res, self.mask)
        g_block = self.ftranBloc(res, block_idx, self.num_blocks, self.block_size, self.mask)
        return g_block.real

    def fmultPatch(self, patch, block_idx):
        return self.fmultBloc(patch, block_idx, self.num_blocks, self.block_size, self.mask)

    def draw(self,x):
        # plt.imshow(np.real(x),cmap='gray')
        pass
    
    @staticmethod
    def genMask(imgSize, numLines):
        if imgSize[0] % 2 != 0 or imgSize[1] % 2 != 0:
            sys.stderr.write('image must be even sized! ')
            sys.exit(1)
        center = np.ceil(imgSize/2)+1
        freqMax = math.ceil(np.sqrt(np.sum(np.power((imgSize/2),2),axis=0)))
        ang = np.linspace(0, math.pi, num=numLines+1)
        mask = np.zeros(imgSize, dtype=bool)
        
        for indLine in range(0,numLines):
            ix = np.zeros(2*freqMax + 1)
            iy = np.zeros(2*freqMax + 1)
            ind = np.zeros(2*freqMax + 1, dtype=bool)
            for i in range(2*freqMax + 1):
                ix[i] = decimal.Decimal(center[1] + (-freqMax+i)*math.cos(ang[indLine])).quantize(0,rounding=decimal.ROUND_HALF_UP)
            for i in range(2*freqMax + 1):
                iy[i] = decimal.Decimal(center[0] + (-freqMax+i)*math.sin(ang[indLine])).quantize(0,rounding=decimal.ROUND_HALF_UP)
                 
            for k in range(2*freqMax + 1):
                if (ix[k] >= 1) & (ix[k] <= imgSize[1]) & (iy[k] >= 1) & (iy[k] <= imgSize[0]):
                    ind[k] = True
                else:
                    ind[k] = False
                
            ix = ix[ind]
            iy = iy[ind]
            ix = ix.astype(np.int64)
            iy = iy.astype(np.int64)
            
            for i in range(len(ix)):
                mask[iy[i]-1][ix[i]-1] = True
        
        return mask
     
    @staticmethod
    def fmult(x, mask):
        numPix = x.size
        z = np.multiply(mask,np.fft.fftshift(np.fft.fft2(x))) / math.sqrt(numPix)
        return z
    
    @staticmethod
    def ftran(z, mask):
        numPix = z.size
        x = np.multiply(np.fft.ifft2(np.fft.ifftshift(np.multiply(mask, z))), np.sqrt(numPix))
        return x

    @staticmethod
    def fmultBloc(x_block, block_idx, num_blocks, block_size, mask):
        dummy = np.zeros([num_blocks,block_size,block_size])
        dummy[block_idx,:,:] = x_block
        dummy_image = putback_nonoverlap_patches(dummy)
        z = np.multiply(mask,np.fft.fftshift(np.fft.fft2(dummy_image))) / math.sqrt(dummy_image.size)
        return z
        
    @staticmethod
    def ftranBloc(z, block_idx, num_blocks, block_size, mask):
        x = np.multiply(np.fft.ifft2(np.fft.ifftshift(np.multiply(mask, z))), np.sqrt(z.size))
        x_list = extract_nonoverlap_patches(x, num_blocks, block_size)
        return x_list[block_idx,...]


