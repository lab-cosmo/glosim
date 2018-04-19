#!/usr/bin/python
import sys
import argparse
import numpy as np
import math

bufsize = 1000
def main(kernels, props, weights, kweights, csi, noidx=False):
    if kweights == "":
        kweights = np.ones(len(kernels))
    else:
        kweights = np.asarray(kweights.split(","),float)
    kweights /= kweights.sum()
    # reads kernel(s)
    csi=float(csi)    
    wvec = np.loadtxt(weights)
    tc = np.asarray(wvec[:,0], float)
    icols = np.asarray(wvec[:,1], int)
    irows = np.asarray(wvec[:,2], int)
    
    p=np.loadtxt(props)
    
    
    print "# Using kernels ", kernels, " with weights ", kweights
    fk=[]
    for i in xrange(0,len(kernels)):
        fk.append(open(kernels[i], "r")    )
        # skips comments
        f=fk[i]
        # determines size of reference set
        fstart = f.tell()
        fline = f.readline()
        while fline[0]=='#': 
            fstart = f.tell()
            fline=f.readline()
        nref = len(fline.split())
        f.seek(fstart)
    
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
    while True:
        for i in xrange(0,len(kernels)):
            chunk = np.fromfile(fk[i], dtype="float",count=(nref)*bufsize, sep=" ")
            if len(chunk) ==0: break
            nk = len(chunk)/(nref)
            if i==0:
                kij = chunk.reshape((nk,nref))[:,icols]*kweights[i]
            else:
                kij += chunk.reshape((nk,nref))[:,icols]*kweights[i]
        if len(chunk) ==0: break
        kij = kij**csi
        krp = np.dot(kij,tc)
        for k in xrange(nk):
            if (not noidx) and (ktot+k in irows):
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
    if ntrain >0:
        print "# Train points MAE=%f  RMSE=%f  SUP=%f" % (trainmae/ntrain, np.sqrt(trainrms/ntrain), trainsup)
    if ntest>0:
        print "# Test points  MAE=%f  RMSE=%f  SUP=%f " % (testmae/ntest, np.sqrt(testrms/ntest), testsup)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes KRR predictions from a weights vector obtained from a previous run of krr.py with --saveweights.""")
    parser.add_argument("--kernels", nargs='+', type=str, help="Kernel matrix (more than one can be read!)")  
    parser.add_argument("--props", default="", type=str, help="Property file (for cross-check)")
    parser.add_argument("--kweights", default="1", type=str, help="Comma-separated list of kernel weights (when multiple kernels are provided)")
    parser.add_argument("--weights", default="", type=str, help="KRR weights corresponding to the reference fit")
    
    #parser.add_argument("kernel", nargs=1, help="Kernel matrix")      
    #parser.add_argument("weights", nargs=1, help="Weights vector") 
    #parser.add_argument("props", nargs=1, help="Property file (for cross-check)")
    parser.add_argument("--csi", type=float, default='1.0', help="Kernel scaling")
    parser.add_argument("--noidx", action="store_true", help="Ignores indices and treats all points as testing points")
    
    args = parser.parse_args()
    
    main(kernels=args.kernels, props=args.props, weights=args.weights, kweights=args.kweights, csi=args.csi, noidx=args.noidx)

