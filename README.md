# [Block Coordinate Regularization by Denoising](https://arxiv.org/abs/1905.05113)

We consider the problem of estimating a vector from its noisy measurements using a prior specified only through a denoising function. Recent work on plug- and-play priors (PnP) and regularization-by-denoising (RED) has shown the state-of-the-art performance of estimators under such priors in a range of imaging tasks. In this work, we develop a new block coordinate RED algorithm that decomposes a large-scale estimation problem into a sequence of updates over a small subset of the unknown variables. We theoretically analyze the convergence of the algorithm and discuss its relationship to the traditional proximal optimization. Our analysis complements and extends recent theoretical results for RED-based estimation methods. We numerically validate our method using several denoiser priors, including those based on convolutional neural network (CNN) denoisers.

## How to run the code

### Prerequisites

python 3.6  
tensorflow 1.12 or lower  
scipy 1.2.1 or lower  
numpy v1.17 or lower  
matplotlib v3.1.0

It is better to use Conda for installation of all dependecies.

### Run the Demo
We provide three scripts 
```
demo_DnCNNstar_Random.py
demo_DnCNNstar_Radon.py
demo_DnCNNstar_Fourier.py
```
to demonstrate the performance of BC-RED with Random matrix, Radon matrix, and Fourier matrix. For example, after installing all prerequisites, you can run the BC-RED for Radon matrix by typing

```
$ python Demo_DnCNNstar_Radon.py
```

To try with different settings, please open the script and follow the instruction inside.

### Citation
Feel free to cite our work as
```
@conference{Sun.etal2019b,
Author = {Sun, Y. and Liu, J. and Kamilov, U. S.},
Month = May,
Booktitle = {Proc. Ann. Conf. Neural Information Processing Systems ({N}eur{IPS})},
Title = {Block Coordinate Regularization by Denoising},
Year = {2019}}
```
