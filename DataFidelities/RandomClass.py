import numpy as np
from DataFidelities.DataClass import DataClass


class RandomClass(DataClass):

    def __init__(self, y, sigSize, A, A_patches):
        self.y = y
        self.A = A
        self.A_patches = A_patches
        self.sigSize = sigSize

    def eval(self, x):
        z = self.fmult(x, self.A)
        d = np.linalg.norm(z.flatten('F') - self.y.flatten('F')) ** 2
        return d

    def res(self, x):
        z = self.fmult(x, self.A)
        return z - self.y

    def gradBloc(self, res, block_idx):
        g = self.ftran(res, self.A_patches[block_idx, :, :])
        return g

    def gradRes(self, res):
        return self.ftran(res, self.A)

    def grad(self, x):
        res = self.res(x)
        d = np.linalg.norm(res.flatten('F')) ** 2
        return self.ftran(res, self.A), d

    def fmultPatch(self, patch, block_idx):
        result = self.fmult(patch, self.A_patches[block_idx, :, :])
        return result

    def draw(self, x):
        # plt.imshow(np.real(x),cmap='gray')
        pass

    @staticmethod
    def genMeas(sigSize, num_blocks, block_size, downsample_rate=1):
        # downsample rate is the downsample rate of nx and ny
        # size of image should be the power of 2
        # num of blocks better to be power of 2
        nx = sigSize[0]  # size of width
        ny = sigSize[1]  # size of height
        dx = int(downsample_rate * nx)  # downsample size of width
        dy = int(downsample_rate * ny)  # downsample size of height
        A = np.random.randn(dx * dy, nx * ny) / np.sqrt(dx * dy)

        # generate patch-wise A, first horizontal, then vertical
        A_patches = []
        for j in range(int(np.sqrt(num_blocks))):  # loop over rows
            joffset = j * block_size
            for i in range(int(np.sqrt(num_blocks))):  # loop over columns
                ioffset = joffset + i * block_size * ny
                sub_matrix = []
                for z in range(block_size):
                    start_ind = ioffset + z * ny
                    end_ind = start_ind + block_size
                    sub_matrix.append(A[:, start_ind:end_ind])
                patch = np.concatenate(sub_matrix, axis=1)
                A_patches.append(patch)
        return A, np.array(A_patches)

    @staticmethod
    def fmult(x, A):
        meas_size_sq = A.shape[0]
        meas_size = int(np.sqrt(meas_size_sq))
        z = np.dot(A, x.flatten('F'))
        return z.reshape(meas_size, meas_size, order='F')

    @staticmethod
    def ftran(z, A):
        block_size_sq = A.shape[1]
        block_size = int(np.sqrt(block_size_sq))
        x = np.dot(A.T, z.flatten('F'))
        return x.reshape(block_size, block_size, order='F')
