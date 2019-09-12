# Class for quadratic-norm on subsampled 2D Radon measurements
# Moran Xu, Yu Sun, CIG, WUSTL, 2019

import numpy as np
import skimage
from DataFidelities.DataClass import DataClass


##### Helper #####

def sinogram_linear_interpolation(sinogram, shifted_sinogram_interp, column_num, position_start):
    row_start = int(np.round(position_start))
    row_end = row_start + sinogram.shape[0]

    shift_sinogram = np.zeros(sinogram.shape[0])

    length = sinogram.shape[0] - 1
    if round(position_start) - position_start >= 0:
        step = round(position_start) - position_start
        shift_sinogram[:length] = (sinogram[1:, column_num] - sinogram[0:length, column_num]) * step + sinogram[
                                                                                                       0:length,
                                                                                                       column_num]
        shift_sinogram[length] = sinogram[length, column_num]
    else:
        step = 1 - position_start + round(position_start)
        shift_sinogram[0] = sinogram[0, column_num]
        shift_sinogram[1:] = (sinogram[1:, column_num] - sinogram[0:length, column_num]) * step + sinogram[0:length,
                                                                                                  column_num]
    shifted_sinogram_interp[row_start: row_end, column_num] = shift_sinogram[:]
    return shifted_sinogram_interp


def sinogram_linear_inverse_interpolation(shifted_sinogram_interp, sinogram, column_num, position_start):
    row_start = int(np.round(position_start))
    row_end = row_start + sinogram.shape[0]
    ishift_sinogram = np.zeros(shifted_sinogram_interp.shape[0])

    #    length = shifted_sinogram_interp.shape[0] - 1
    if round(position_start) - position_start >= 0:
        step = 1 - round(position_start) + position_start
        ishift_sinogram[row_start] = shifted_sinogram_interp[row_start, column_num]
        ishift_sinogram[row_start + 1:row_end] = (shifted_sinogram_interp[row_start + 1:row_end, column_num] - \
                                                  shifted_sinogram_interp[row_start:row_end - 1, column_num]) * step + \
                                                 shifted_sinogram_interp[row_start:row_end - 1, column_num]
    else:
        step = -round(position_start) + position_start
        ishift_sinogram[row_start:row_end - 1] = (shifted_sinogram_interp[row_start + 1:row_end, column_num] - \
                                                  shifted_sinogram_interp[row_start:row_end - 1, column_num]) * step + \
                                                 shifted_sinogram_interp[row_start:row_end - 1, column_num]
        ishift_sinogram[row_end] = shifted_sinogram_interp[row_end, column_num]
    sinogram[:, column_num] = ishift_sinogram[row_start: row_end]
    return ishift_sinogram[row_start: row_end]


###################################################
###                   CT Class                  ###
###################################################

class RadonEffClass(DataClass):

    def __init__(self, y, sigSize, mask, theta=None, num_blocks=16, block_size=40):
        self.y = y
        self.sigSize = sigSize
        self.mask = mask
        self.theta = theta  # here theta is a list of degrees
        self.num_blocks = num_blocks
        self.block_size = block_size

    def size(self):
        return self.sigSize

    def eval(self, x):
        pass

    def res(self, x):
        z = self.fmult(x, self.theta)
        return z - self.y

    def grad(self, x):
        res = self.res(x)
        g = self.ftran(res, self.theta)
        g = g.real
        d = np.linalg.norm(res.flatten('F')) ** 2
        return g, d

    def gradRes(self, res):
        return self.ftran(res, self.theta)

    def gradBloc(self, res, block_idx):  # need to be fixed
        g_block = self.ftranBloc(res, block_idx, self.mask, self.sigSize, self.num_blocks, self.block_size, self.theta)
        return g_block

    def fmultPatch(self, patch, block_idx):
        return self.fmultBloc(patch, block_idx, self.sigSize, self.num_blocks, self.block_size, self.theta)

    def draw(self, x):
        pass

    @staticmethod
    def fmult(x, theta):
        z = skimage.transform.radon(x, theta=theta, circle=False)
        return z

    @staticmethod
    def ftran(z, theta):  # output_size, theta):
        x = skimage.transform.iradon(z, theta=theta, filter=None, circle=False)
        return x

    @staticmethod
    def fmultBloc(x_block, block_idx, sigSize, num_blocks, block_size, theta):
        dummy = np.zeros([num_blocks, block_size, block_size])
        dummy[block_idx, :, :] = x_block
        # num of blocks every row
        coordinate_x = block_idx // int(num_blocks ** 0.5)
        coordinate_y = block_idx % int(num_blocks ** 0.5)

        block_coordinate = [coordinate_x, coordinate_y]

        sinogram = skimage.transform.radon(x_block, theta=theta, circle=False)
        block_size = x_block.shape[0]

        block_movement_v, block_movement_h = (np.array(block_coordinate) + 1) * block_size - block_size // 2  # [10 10]
        H, W = sigSize  # [160 160]
        assert H == W, "Image should be squared."

        T_length = int(np.sqrt(2) * H) + 1
        shifted_sinogram_interp = np.zeros([T_length, sinogram.shape[1]])

        # distance from center
        block_movement_h_center = H / 2 - block_movement_h  # 70
        block_movement_v_center = H / 2 - block_movement_v  # 70

        delta_v = -block_movement_h_center
        delta_h = block_movement_v_center

        for column_num in range(sinogram.shape[1]):
            # put the sinogram into the same column, but shited rows
            th = (np.pi / 180.0) * theta[column_num]

            position_start = float(T_length) / 2 - float(sinogram.shape[0] / 2) + \
                             delta_v * np.cos(th) + delta_h * np.sin(th)
            shifted_sinogram_interp = sinogram_linear_interpolation(sinogram, shifted_sinogram_interp, column_num,
                                                                    position_start)
        z = shifted_sinogram_interp
        return z

    @staticmethod
    def ftranBloc(z, block_idx, mask, sigSize, num_blocks, block_size, theta):  # dummy variables
        shifted_res_patch = np.multiply(z, mask[block_idx])

        ######inverse interpolation to get original patch size######
        sinogram = np.zeros((int(np.sqrt(2) * block_size) + 1, len(theta)))
        res_patch = np.zeros((num_blocks, int(np.sqrt(2) * block_size) + 1, len(theta)))
        shifted_sinogram_interp = shifted_res_patch
        # num of blocks every row
        coordinate_x = block_idx // int(num_blocks ** 0.5)
        coordinate_y = block_idx % int(num_blocks ** 0.5)

        block_coordinate = [coordinate_x, coordinate_y]
        block_movement_v, block_movement_h = (np.array(block_coordinate) + 1) * block_size - block_size // 2

        H, W = sigSize  # [160 160]
        assert H == W, "Image should be squared."
        T_length = int(np.sqrt(2) * H) + 1
        # distance from center
        block_movement_h_center = H / 2 - block_movement_h
        block_movement_v_center = H / 2 - block_movement_v

        delta_v = -block_movement_h_center
        delta_h = block_movement_v_center

        for column_num in range(sinogram.shape[1]):
            # put the sinogram into the same column, but shited rows
            th = (np.pi / 180.0) * theta[column_num]
            position_start = float(T_length) / 2 - float(sinogram.shape[0] / 2) + \
                             delta_v * np.cos(th) + delta_h * np.sin(th)
            # linear interpolation
            sinogram[:, column_num] = sinogram_linear_inverse_interpolation(shifted_sinogram_interp, sinogram,
                                                                            column_num, position_start)
        res_patch[block_idx] = sinogram
        g_data = skimage.transform.iradon(res_patch[block_idx], theta, filter=None, circle=False)
        return g_data
