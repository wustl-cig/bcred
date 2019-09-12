'''
Yu Sun, CIG, WUSTL, 2019
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import scipy.io as sio
import scipy.misc as smisc
from scipy.optimize import fminbound


def to_rgb(img):
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img


def to_double(img):
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    return img


def save_mat(img, path):
    sio.savemat(path, {'img': img})


def save_img(img, path):
    img = to_rgb(img)
    smisc.imsave(path, img.round().astype(np.uint8))


def addwgn(x, inputSnr):
    noiseNorm = np.linalg.norm(x.flatten('F')) * 10 ** (-inputSnr / 20)
    xBool = np.isreal(x)
    real = True
    for e in np.nditer(xBool):
        if e == False:
            real = False
    if (real == True):
        noise = np.random.randn(np.shape(x)[0], np.shape(x)[1])
    else:
        noise = np.random.randn(np.shape(x)[0], np.shape(x)[1]) + 1j * np.random.randn(np.shape(x)[0], np.shape(x)[1])

    noise = noise / np.linalg.norm(noise.flatten('F')) * noiseNorm
    y = x + noise
    return y, noise


def extract_nonoverlap_patches(x, num_blocks, block_size):
    patches = np.zeros([num_blocks, block_size, block_size])
    nx, ny = x.shape
    count = 0
    for i in range(0, nx - block_size + 1, block_size):
        for j in range(0, ny - block_size + 1, block_size):
            patches[count, :] = x[i:i + block_size, j:j + block_size]
            count = count + 1
    return patches


def putback_nonoverlap_patches(patches):
    num_blocks, block_size, _ = patches.shape
    nx = ny = int(np.sqrt(num_blocks) * block_size)
    x = np.zeros([nx, ny])
    count = 0
    for i in range(0, nx - block_size + 1, block_size):
        for j in range(0, ny - block_size + 1, block_size):
            x[i:i + block_size, j:j + block_size] = patches[count]
            count = count + 1
    return x


def extract_padding_patches(x_input, patch_index, extend_p=5, num_blocks=16, block_size=40, pad_mode='reflect'):
    assert len(x_input.shape) != 0, "Input is empty."
    if extend_p is None: extend_p = 0
    if len(x_input.shape) == 3:
        num_blocks, block_size, _ = x_input.shape
        nx = ny = int(np.sqrt(num_blocks) * block_size)
        x = np.zeros([nx, ny])
        count = 0
        for i in range(0, nx - block_size + 1, block_size):
            for j in range(0, ny - block_size + 1, block_size):
                x[i:i + block_size, j:j + block_size] = x_input[count]
                count = count + 1
        x_input = x
    elif len(x_input.shape) == 2:
        pass

    x_input_padded = np.pad(x_input, ((extend_p,), (extend_p,)), pad_mode)
    x_shape0, x_shape1 = (x_input_padded.shape[0], x_input_padded.shape[1])
    patch_size = block_size + 2 * extend_p
    h_idx_list = list(range(0, x_shape0 - patch_size, block_size)) + [x_shape0 - patch_size]
    w_idx_list = list(range(0, x_shape1 - patch_size, block_size)) + [x_shape1 - patch_size]
    extended_patches = np.zeros([num_blocks, patch_size, patch_size])
    count = 0
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            # print(h_idx, w_idx)
            extended_patches[count, ...] = x_input_padded[h_idx:h_idx + patch_size, w_idx:w_idx + patch_size]
            count = count + 1
    return extended_patches[patch_index, ...].squeeze()


def powerIter(A, imgSize, iters=100, tol=1e-6, verbose=False):
    # compute singular value for A'*A
    # A should be a function (lambda:x)
    x = np.random.randn(imgSize[0], imgSize[1])
    x = x / np.linalg.norm(x.flatten('F'))
    lam = 1
    for i in range(iters):
        # apply Ax
        xnext = A(x)
        # xnext' * x / norm(x)^2
        lamNext = np.dot(xnext.flatten('F'), x.flatten('F')) / np.linalg.norm(x.flatten('F')) ** 2
        # only take the real part
        lamNext = lamNext.real
        # normalize xnext
        xnext = xnext / np.linalg.norm(xnext.flatten('F'))
        # compute relative difference
        relDiff = np.abs(lamNext - lam) / np.abs(lam)
        x = xnext
        lam = lamNext
        # verbose
        if verbose:
            print('[{}/{}] lam = {}, relative Diff = {:0.4f}'.format(i, iter, lam, relDiff))
        # stopping criterion
        if relDiff < tol:
            break
    return lam
