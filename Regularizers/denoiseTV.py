'''
Usage of TV regularizer and apply to only 2D data.
Jianxing Liao, CIG, WUSTL, 2018
Based on MATLAB code by U. S. Kamilov, CIG, WUSTL, 2017
'''

import numpy as np
import math
import scipy.ndimage

def denoiseTV(y,lambd,pin,maxiter=100,L=8,tol=1e-5,optim='fgp',verbose=False,img=np.array([]),bounds=[-math.inf, math.inf],bc='reflexive'):
    P = pin
    
    if L == 0:
        L = Lipschitz(y)/1.25
    
    count = 0
    flag = False
    
    if verbose:
        print('******************************************\n')
        print('**     Denoising with TV Regularizer    **\n')
        print('******************************************\n')
        print('#iter     relative-dif   \t fun_val         Duality Gap    \t   ISNR\n')
        print('====================================================================\n')
    
    if optim == 'fgp':
        t = 1
        F = np.array(P)
        for i in range(1, maxiter+1):
            K = y - lambd * AdjTVOp2D(F,bc)     
            Pnew = F + (1/(L*lambd)) * TVOp2D(project(K,bounds),bc)
            Pnew = projectL2(Pnew)
      
            re = np.linalg.norm(Pnew.flatten('F')-P.flatten('F'))/np.linalg.norm(Pnew.flatten('F'))
            if re < tol:
                count=count+1
            else:
                count=0
      
            tnew = (1+np.sqrt(1+4*np.power(t,2)))/2
            F = np.array(Pnew+(t-1)/tnew*(Pnew-P))
            P = np.array(Pnew)
            t = tnew
            
            if verbose:
                if img.size == 0:
                    k = y-lambd*AdjTVOp2D(P,bc)
                    x = project(k,bounds)
                    fun_val,_=cost(y,x,lambd,bc)
                    dual_fun_val=dualcost(y,k,bounds)
                    dual_gap=(fun_val-dual_fun_val)
                    print("{} \t {} \t {} \t {}".format(i,re,fun_val,dual_gap))
                    
                else:
                    k = y-lambd*AdjTVOp2D(P,bc)
                    x = project(k,bounds)
                    fun_val = cost(y,x,lambd,bc)
                    dual_fun_val = dualcost(y,k,bounds)
                    dual_gap = (fun_val-dual_fun_val)
                    ISNR = 20*np.log10(np.linalg.norm(y-img,ord='fro')/np.linalg.norm(x-img,ord='fro'))
                    print("%3d \t " % (i) + "%10.5f \t " % (re) + "%10.5f \t " % (fun_val) + "%2.8f \t " % (dual_gap) + "%2.8f\n" % (ISNR))
            
            if count >= 5:
                flag = True
                itera = i
                break
    
    elif optim == 'gp':
        for i in range(1, maxiter+1):
            
            K = y-lambd*AdjTVOp2D(P,bc)
            Pnew = P+(1/(L*lambd))*TVOp2D(project(K,bounds),bc)
            Pnew = projectL2(Pnew);
      
            re = np.linalg.norm(Pnew[:]-P[:])/np.linalg.norm(Pnew[:])
            if re < tol:
                count=count+1
            else:
                count=0
      
            P = Pnew
      
            if verbose:
                if img.size == 0:
                    k = y-lambd*AdjTVOp2D(P,bc)
                    x = project(k,bounds)
                    fun_val,_ = cost(y,x,lambd,bc)
                    dual_fun_val = dualcost(y,k,bounds)
                    dual_gap = (fun_val-dual_fun_val)
                    print("{} \t {} \t {} \t {}".format(i,re,fun_val,dual_gap))
                
                else:
                    k = y-lambd*AdjTVOp2D(P,bc)
                    x = project(k,bounds)
                    fun_val = cost(y,x,lambd,bc)
                    dual_fun_val = dualcost(y,k,bounds)
                    dual_gap = (fun_val-dual_fun_val)
                    ISNR = 20*np.log10(np.linalg.norm(y-img,ord='fro')/np.linalg.norm(x-img,ord='fro'))
                    print("%3d \t " % (i) + "%10.5f \t " % (re) + "%10.5f \t " % (fun_val) + "%2.8f \t " % (dual_gap) + "%2.8f\n" % (ISNR))
            
            if count >= 5:
                flag = True
                itera = i
                break
    
    if not flag:
        itera = maxiter
    x = project(y-lambd*AdjTVOp2D(P,bc),bounds)
    return x,P,itera,L


def TVOp2D(f,bc):
    filter1 = np.array([[0],[-1],[1]])
    filter2 = np.array([[0,-1,1]])
    [r,c]=f.shape
    Df=np.zeros([r,c,2])
    Df[:,:,0] = scipy.ndimage.filters.correlate(f,filter1,mode='wrap')
    Df[:,:,1] = scipy.ndimage.filters.correlate(f,filter2,mode='wrap')
    return Df


def AdjTVOp2D(P,bc):
    filter1 = np.array([[1],[-1],[0]])
    filter2 = np.array([[1,-1,0]])
    P1 = P[:,:,0]
    P1 = scipy.ndimage.filters.correlate(P1,filter1,mode='wrap')
    P2 = P[:,:,1]
    P2 = scipy.ndimage.filters.correlate(P2,filter2,mode='wrap')
    g = P1+P2
    return g


def projectL2(B):
    PB = np.divide(B, np.tile(np.maximum(1,np.sqrt(np.sum(np.power(B, 2),2)))[...,None],(1,1,2)))
    return PB


def project(x,bounds):
    lb = bounds[0]
    ub = bounds[1]
    if lb == -math.inf and ub == math.inf:
        Px = np.array(x)
    elif lb == -math.inf and np.isfinite(ub):
        x[x>ub] = ub
        Px = np.array(x)
    elif ub == math.inf and np.isfinite(lb):
        x[x<lb] = lb
        Px = np.array(x)
    else:
        x[x<lb] = lb
        x[x>ub] = ub
        Px = np.array(x)
    return Px


def cost(y,f,lambd,bc):
    filter1 = np.array([[0],[-1],[1]])
    filter2 = np.array([[0,-1,1]])
    fx = scipy.ndimage.filters.correlate(f,filter1,mode='wrap')
    fy = scipy.ndimage.filters.correlate(f,filter2,mode='wrap')
    TVf = np.sqrt(np.power(fx,2)+np.power(fy,2))
    TVnorm = np.sum(TVf[:])
    Q = 0.5*np.linalg.norm(y-f,ord='fro')**2 + lambd*TVnorm
    return Q,TVnorm
   
    
def dualcost(y,f,bounds):
    r = f - project(f,bounds)
    Q = 0.5*(np.sum(np.power(r[:],2))+np.sum(np.power(y[:],2))-np.sum(np.power(f[:],2)))
    return Q


def Lipschitz(y):
    [r,c] = y.shape
    hx = np.zeros([3,3])
    hx[:,1] = np.array([1, -1, 0])
    hy = hx.transpose()
    hx[r-1,c-1] = 0
    hy[r-1,c-1] = 0
    hx = np.roll(hx, -1, axis=1)
    hx = np.roll(hx, -1, axis=0)
    hy = np.roll(hy, -1, axis=1)
    hy = np.roll(hy, -1, axis=0)

    Op_eig = np.absolute(np.power(np.fft.fft2(hx),2)) + np.absolute(np.power(np.fft.fft2(hy),2))
    L = np.max(Op_eig[:])
    return L