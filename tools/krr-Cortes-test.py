#!/usr/bin/python
import sys
import argparse
import numpy as np
import math

bufsize = 100
def main(kernelFilenames, propFilename, weightFilenames, csi):
        
    # Unpack the alpha's and  their corresponding lines/columns in the kernel matrices
    wvec = np.loadtxt(weightFilenames[0])
    alpha = np.asarray(wvec[:,0], float)
    icols = np.asarray(wvec[:,1], int)
    irows = np.asarray(wvec[:,2], int)
    # Unpack the mu's 
    mu = np.loadtxt(weightFilenames[1])
    
    p = np.loadtxt(propFilename, dtype=np.float64, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
    
    kernelFiles = []
    for it,kernelFilename in enumerate(kernelFilenames):
        kernelFiles.append(open(kernelFilename, "r"))
        # skips comments
        fstart = kernelFiles[it].tell()
        fline = kernelFiles[it].readline()
        while fline[0]=='#': 
            fstart = kernelFiles[it].tell()
            fline = kernelFiles[it].readline()
        nref = len(fline.split())
        kernelFiles[it].seek(fstart)
    
    # average counters
    testmae = 0
    trainmae = 0
    testrms = 0
    trainrms = 0
    testsup = 0
    trainsup = 0
    ntest = 0
    ntrain = 0 
    ktot = 0
    chunks = [[] for it in range(len(kernelFiles))]
    kernels = [[] for it in range(len(kernelFiles))]

    while True:
        # Read chunks of the several kernel matrices
        for it,kernelFile in enumerate(kernelFiles):
            chunks[it] = np.fromfile(kernelFile, dtype="float",count=(nref)*bufsize, sep=" ")
            nk = len(chunks[it])/(nref) 
            kernels[it] = chunks[it].reshape((nk,nref))[:,icols]
            kernels[it] = kernels[it]**csi[it]
        n,m = kernels[0].shape
        # Condition to leave the loop. if the chunk has no lines
        if n == 0: break
        # Make the composite kernel matrix using the mu
        Kernel = np.zeros((n,m),dtype=np.float64)
        for it,kernel in enumerate(kernels):
            Kernel += mu[it] * kernel
        # Create the predictions
        krp = np.dot(Kernel,alpha)
        
        # output the different errors on the training set and the testing set
        for k in xrange(nk):
            if  ktot+k in irows:
                lab = "TRAIN"
                trainmae += abs(krp[k] - p[ktot+k])
                trainrms += (krp[k] - p[ktot+k])**2
                trainsup = max(trainsup, abs(krp[k] - p[ktot+k]))
                ntrain += 1
            else:
                lab = "TEST"
                testmae += abs(krp[k] - p[ktot+k])
                testrms += (krp[k] - p[ktot+k])**2
                testsup = max(testsup, abs(krp[k] - p[ktot+k]))
                ntest +=1 
            print k+ktot, p[ktot+k], krp[k], lab
        ktot += nk
    print "# Train points MAE={:.4e}  RMSE={:.4e}  SUP={:.4e}".format(trainmae/ntrain, np.sqrt(trainrms/ntrain), trainsup)
    print "# Test points  MAE={:.4e}  RMSE={:.4e}  SUP={:.4e} ".format(testmae/ntest, np.sqrt(testrms/ntest), testsup)
    # print "# Train points MAE=%f  RMSE=%f  SUP=%f" % (trainmae/ntrain, np.sqrt(trainrms/ntrain), trainsup)
    # print "# Test points  MAE=%f  RMSE=%f  SUP=%f " % (testmae/ntest, np.sqrt(testrms/ntest), testsup)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes KRR predictions from a weights vector obtained from a previous run of krr.py with --saveweights.""")
    parser.add_argument("kernels", nargs=1, help="Kernel matrices. List of coma separated file names.")            
    parser.add_argument("weights", nargs=1, help="Weights vector") 
    parser.add_argument("props", nargs=1, help="Property file name (for cross-check)")
    parser.add_argument("--csi", type=str, default='1.0', help="Kernel scaling")
    
    args = parser.parse_args()
    kernelFilenames = args.kernels[0].split(',')
    weightsFilenames = args.weights[0].split(',')
    a = args.csi.split(',')
    if len(a) != len(kernelFilenames):
        raise ValueError("The number of kernel file names and elements of csi must be equal.")
    csi = np.zeros(len(a),dtype=np.float64)
    for it,item in enumerate(a):
        csi[it] = float(item)
    
    main(kernelFilenames=kernelFilenames, propFilename=args.props[0], weightFilenames=weightsFilenames, csi=csi)

