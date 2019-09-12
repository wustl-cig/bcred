# library
import os
import shutil
import numpy as np
import time
# scripts
import util


######## Iterative Methods #######

def redEst(dObj, rObj, 
            numIter=100, step=1, accelerate=False, mode='RED', useNoise=True, 
            verbose=False, is_save=False, save_path='red_intermediate_results', xtrue=None, xinit=None):
    """
    Regularization by Denoising (RED)
    
    ### INPUT:
    dObj       ~ data fidelity term, measurement/forward model
    rObj       ~ regularizer term
    numIter    ~ total number of iterations
    accelerate ~ use APGM or PGM
    mode       ~ RED update or PROX update
    useNoise.  ~ true if CNN predict noise; false if CNN predict clean image
    step       ~ step-size
    verbose    ~ if true print info of each iteration
    is_save    ~ if true save the reconstruction of each iteration
    save_path  ~ the save path for is_save
    xtrue      ~ the ground truth of the image, for tracking purpose
    xinit      ~ initialization of x (zero otherwise)

    ### OUTPUT:
    x     ~ reconstruction of the algorithm
    outs  ~ detailed information including cost, snr, step-size and time of each iteration

    """
    
    ########### HELPER FUNCTION ###########

    evaluateSnr = lambda xtrue, x: 20*np.log10(np.linalg.norm(xtrue.flatten('F'))/np.linalg.norm(xtrue.flatten('F')-x.flatten('F')))

    ########### INITIALIZATION ###########
    
    # initialize save foler
    if is_save:
        abs_save_path = os.path.abspath(save_path)
        if os.path.exists(save_path):
            print("Removing '{:}'".format(abs_save_path))
            shutil.rmtree(abs_save_path, ignore_errors=True)
        # make new path
        print("Allocating '{:}'".format(abs_save_path))
        os.makedirs(abs_save_path)

    #initialize info data
    if xtrue is not None:
        xtrueSet = True
        snr = []
    else:
        xtrueSet = False

    loss = []
    dist = []
    timer = []
    
    # initialize variables
    if xinit is not None:
        pass
    else:    
        xinit = np.zeros(dObj.sigSize, dtype=np.float32)
    x = xinit
    s = x            # gradient update
    t = 1.           # controls acceleration
    p,pfull = rObj.init(1, dObj.sigSize[0])  # dual variable for TV
    p = p[0]

    ########### BC-RED (EPOCH) ############

    for indIter in range(numIter):
        timeStart = time.time()
        # get gradient
        g, _ = dObj.grad(s)
        if mode == 'RED':
            g_robj, p = rObj.red(s, step, p, useNoise=useNoise, extend_p=None)
            xnext = s - step*(g + g_robj)
        elif mode == 'PROX':
            xnext, p = rObj.prox(np.clip(s-step*g,0,np.inf), step, p)   # clip to [0, inf]
        elif mode == 'GRAD':
            xnext = s-step*g
        else:
            print("No such mode option")
            exit()

        timeEnd = time.time() - timeStart


        ########### LOG INFO ###########

        # calculate full gradient for convergence plot
        gfull, dfull = dObj.grad(x)
        if mode == 'RED':
            g_robj, pfull = rObj.red(x, step, pfull, useNoise=useNoise, extend_p=None)
            Px = x - step*(gfull + g_robj)
            # Gx
            diff = np.linalg.norm(gfull.flatten('F') + g_robj.flatten('F')) ** 2
            obj = dfull + rObj.eval(x)
        elif mode == 'PROX':
            Px, pfull = rObj.prox(np.clip(x-step*gfull,0,np.inf), step, pfull)
            # x-Px
            diff = np.linalg.norm(x.flatten('F') - Px.flatten('F')) ** 2
            obj  = dfull + rObj.eval(x)
        elif mode == 'GRAD':
            # x-Px
            Px = x-step*g
            diff = np.linalg.norm(x.flatten('F') - Px.flatten('F')) ** 2
            obj  = dfull
        else:
            print("No such mode option")
            exit()

        # acceleration
        if accelerate:
            tnext = 0.5*(1+np.sqrt(1+4*t*t))
        else:
            tnext = 1
        s = xnext + ((t-1)/tnext)*(xnext-x)
        
        # output info
        # cost[indIter] = data
        loss.append(obj)
        dist.append(diff)
        timer.append(timeEnd)
        # evaluateTol(x, xnext)
        if xtrueSet:
            snr.append(evaluateSnr(xtrue, x))

        # update
        t = tnext
        x = xnext

        # save & print
        if is_save:
            util.save_mat(xnext, abs_save_path+'/iter_{}_mat.mat'.format(indIter+1))
            util.save_img(xnext, abs_save_path+'/iter_{}_img.tif'.format(indIter+1))
        
        if verbose:
            if xtrueSet:
                print('[redEst: '+str(indIter+1)+'/'+str(numIter)+']'+' [||Gx_k||^2/||Gx_0||^2: %.5e]'%(dist[indIter]/dist[0])+' [snr: %.2f]'%(snr[indIter]))
            else:
                print('[redEst: '+str(indIter+1)+'/'+str(numIter)+']'+' [||Gx_k||^2/||Gx_0||^2: %.5e]'%(dist[indIter]/dist[0]))

        # summarize outs
        outs = {
            'dist': dist/dist[0],
            'snr': snr,
            'time': timer
        }

    return x, outs


def bcredEst(dObj, rObj, 
             num_patch=16, patch_size=40, pad=None, numIter=100, step=1, useNoise=True,
             verbose=False, is_save=False, save_path='bcred_intermediate_results', xtrue=None, xinit=None):
    """
    Block Coordinate Regularization by Denoising (BCRED)
    
    ### INPUT:
    dObj       ~ the data fidelity term, measurement/forward model
    rObj       ~ the regularizer term
    num_patch  ~ the number of blocks assigned (Patches should not overlap with each other)
    patch_size ~ the spatial size of a patch (block)
    pad        ~ the pad size for block-wise denoising / set to 'None' if you want to use the full denoiser 
    numIter    ~ the total number of iterations
    step       ~ the step-size
    verbose    ~ if true print info of each iteration
    is_save    ~ if true save the reconstruction of each iteration
    save_path  ~ the save path for is_save
    xtrue      ~ the ground truth of the image, for tracking purpose
    xinit      ~ the initial value of x 

    ### OUTPUT:
    x     ~ reconstruction of the algorithm
    outs  ~ detailed information including cost, snr, step-size and time of each iteration
    
    """
    
    ########### HELPER FUNCTION ###########

    evaluateSnr = lambda xtrue, x: 20*np.log10(np.linalg.norm(xtrue.flatten('F'))/np.linalg.norm(xtrue.flatten('F')-x.flatten('F')))

    ########### INITIALIZATION ###########
    
    # initialize save foler
    if is_save:
        abs_save_path = os.path.abspath(save_path)
        if os.path.exists(save_path):
            print("Removing '{:}'".format(abs_save_path))
            shutil.rmtree(abs_save_path, ignore_errors=True)
        # make new path
        print("Allocating '{:}'".format(abs_save_path))
        os.makedirs(abs_save_path)

    #initialize info data
    if xtrue is not None:
        xtrueSet = True
        snr = []
    else:
        xtrueSet = False

    loss = []
    dist = []
    timer = []
    
    # initialize variables
    if xinit is not None:
        pass
    else:    
        xinit = np.zeros(dObj.sigSize, dtype=np.float32) 
    x = xinit
    xnext = x
    x_patches = util.extract_nonoverlap_patches(x, num_patch, patch_size)
    xnext_patches = x_patches

    # helper variable
    p,pfull = rObj.init(num_patch, patch_size+2*pad)  # dual variable for TV
    res = dObj.res(x) # compute the residual Ax-y for xinit

    
    ########### BC-RED (EPOCH) ############

    for indIter in range(numIter):
        
        # randomize order of patches
        patchInd = np.random.permutation(num_patch)

        # calculate full gradient (g = Sx)
        gfull_data, dcost = dObj.grad(x)
        gfull_robj, pfull = rObj.red(x, step, pfull, useNoise=useNoise, extend_p=None)
        gfull_tot = gfull_data + gfull_robj

        # calculate the loss for showing back-compatibility of PROX-TV
        obj = dcost + rObj.eval(x)

        # cost[indIter] = data
        loss.append(obj)
        dist.append(np.linalg.norm(gfull_tot.flatten('F'))**2)
        if xtrueSet:
            snr.append(evaluateSnr(xtrue, x))

        # set up a timer
        timeStart = time.time()

        ## Inner Loop ##
        for i in range(num_patch):

            # extract patch
            patch_idx = patchInd[i]
            cur_patch = x_patches[patch_idx,:,:]

            # get gradient of data-fit for the extracted block
            g_data = dObj.gradBloc(res, patch_idx)

            # denoise the block with padding & get the full gradient G
            if pad is None:
                g_robj, p[patch_idx,...] = rObj.red(x, step, p[patch_idx,...], useNoise=useNoise, extend_p=None)
                g_robj_patch = util.extract_padding_patches(g_robj, patch_idx, extend_p=0)
            else:
                padded_patch = util.extract_padding_patches(x, patch_idx, extend_p=pad)
                g_robj_patch, p[patch_idx,...] = rObj.red(padded_patch, step, p[patch_idx,...], useNoise=useNoise, extend_p=pad)
            
            g_tot = g_data + g_robj_patch

            # update the selected block
            xnext_patches[patch_idx,:,:] = cur_patch - step*g_tot
            xnext = util.putback_nonoverlap_patches(xnext_patches)

            # update
            res = res - step*dObj.fmultPatch(g_tot, patch_idx)
            x = xnext
            x_patches = xnext_patches
       
        # end of the timer 
        timeEnd = time.time() - timeStart 
        timer.append(timeEnd)

        ########### LOG INFO ###########

        # save & print
        if is_save:
            util.save_mat(xnext, abs_save_path+'/iter_{}_mat.mat'.format(indIter+1))
            util.save_img(xnext, abs_save_path+'/iter_{}_img.tif'.format(indIter+1))
        
        if verbose:
            if xtrueSet:
                print('[bcredEst: '+str(indIter+1)+'/'+str(numIter)+']'+' [||Gx_k||^2/||Gx_0||^2: %.5e]'%(dist[indIter]/dist[0])+' [snr: %.2f]'%(snr[indIter]))
            else:
                print('[bcredEst: '+str(indIter+1)+'/'+str(numIter)+']'+' [||Gx_k||^2/||Gx_0||: %.5e]'%(dist[indIter]/dist[0]))

        # summarize outs
        outs = {
            'dist': dist/dist[0],
            'snr': snr,
            'time': timer
        }


    return x, outs

