"""Multiple Kernel Learning methods  """

import numpy as np 
import sys
import costs as cst 


"""MKL KRR from Cortes 2009 : 'L2 regularization for learning kernels'."""

def TrainKRRCortes(kernels,prop,verbose=False,**KRRCortesParam):
    """kernels is a list of numpy nxn kernel matrices associated to the training set.
    prop is a numpy array containing the properties corresponding to the training set.
    KRRCortesParam is a dictionary of the parameters of the algorithm."""
    # unpack algortihm parameters
    Lambda = KRRCortesParam['Lambda']
    eta = KRRCortesParam['eta']
    sigma = KRRCortesParam['sigma']
    epsilon = KRRCortesParam['epsilon']
    Nmax = KRRCortesParam['maxIter']
    mu0 = KRRCortesParam['mu0']
    
    nbOfKernels = len(kernels)

    n,m = kernels[0].shape
    propVar = np.var(prop)
    # Initialize vectors 
    Id = np.eye(n, M=n, k=0, dtype=np.float64) 
    alphaNew = np.zeros((n,1), dtype=np.float64)
    alphaOld = np.zeros((n,1), dtype=np.float64)
    v = np.zeros(nbOfKernels,dtype=np.float64)
    # mu = np.zeros(nbOfKernels,dtype=np.float64)
    mu = mu0

    # Initialize the algorithm tr(tk)/(N vp)
    kernelMat = setKernelMat(kernels,mu)
    
    regParam = sigma**2 * kernelMat.trace() / (n * propVar)
    alphaNew = np.dot(np.linalg.inv(kernelMat + regParam*Id),prop)
    MaeInit = cst.mae(np.dot(alphaNew,kernelMat)-prop)
    if verbose is True:
        print 'Initial MAE : {:.4e}'.format(MaeInit)
    N = 0
    while(np.linalg.norm(alphaNew-alphaOld) > epsilon and N <= Nmax):
        # print 'ENter ##################'
        alphaOld = alphaNew
        
        # update search direction for mu
        for it,kernel in enumerate(kernels):
            v[it] = np.dot(alphaOld.T,np.dot(kernel,alphaOld))

        # update mu
        mu = mu0 + Lambda * v / np.linalg.norm(v)
        
        # update ktot
        kernelMat = setKernelMat(kernels,mu)

        # update alpha
        regParam = sigma**2 * kernelMat.trace() / (n * propVar)
        #print regParam, kernelMat.trace(), mu      
        alphaNew = eta * alphaOld + (1-eta) * np.dot(np.linalg.inv(kernelMat+ regParam*Id),prop)
        N += 1

        if verbose is True:
            Mae = cst.mae(np.dot(alphaNew,kernelMat)-prop)
            print 'N = {:.0f} / alpha diff = {:.3e} / MAE = {:.4e}'.format(N,np.linalg.norm(alphaNew-alphaOld),Mae)
        

    print 'Training the weights with Cortes algorithm has ended in {:.0f} iterrations \n  alpha diff = {:.3e} / Initial Mae={:.4e} / Final Mae={:.4e}'\
            .format(N,np.linalg.norm(alphaNew-alphaOld),MaeInit,cst.mae(np.dot(alphaNew,kernelMat)-prop))
    propTr = np.dot(alphaNew,kernelMat)
    return alphaNew, mu, propTr


def PredictKRRCortes(kernels,alpha,mu):
    """kernels is a list of numpy nxm kernel matrices associated to the K(x_tr,x_te),
    n is the number of training elements and m is the number of testing elements.
    alpha is the optimal weight vector for the KRR step.
    mu is the optimal weight for linearly combining the kernels"""
    kernelMat = setKernelMat(kernels,mu)
    return np.dot(alpha,kernelMat)

def setKernelMat(kernels,mu):
    """Combine linarly the kernels with weights mu."""
    n,m = kernels[0].shape
    kernelMat = np.zeros((n,m),dtype=np.float64)

    for it,kernel in enumerate(kernels):
        kernelMat += mu[it] * kernel
    return kernelMat
