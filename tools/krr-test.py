#!/usr/bin/python
import sys
import argparse
import numpy as np
import math

bufsize = 1000
def main(kernel, props, weights, csi):
    csi=float(csi)    
    wvec = np.loadtxt(weights)
    tc = np.asarray(wvec[:,0], float)
    icols = np.asarray(wvec[:,1], int)
    irows = np.asarray(wvec[:,2], int)
    
    p=np.loadtxt(props)
    
    f = open(kernel, "r")
    
    # skips comments
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
        chunk = np.fromfile(f, dtype="float",count=(nref)*bufsize, sep=" ")
        if len(chunk) ==0: break
        nk = len(chunk)/(nref)
        kij = chunk.reshape((nk,nref))[:,icols]
        kij = kij**csi
        krp = np.dot(kij,tc)
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
    print "# Train points MAE=%f  RMSE=%f  SUP=%f" % (trainmae/ntrain, np.sqrt(trainrms/ntrain), trainsup)
    print "# Test points  MAE=%f  RMSE=%f  SUP=%f " % (testmae/ntest, np.sqrt(testrms/ntest), testsup)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes KRR predictions from a weights vector obtained from a previous run of krr.py with --saveweights.""")
    parser.add_argument("kernel", nargs=1, help="Kernel matrix")      
    parser.add_argument("weights", nargs=1, help="Weights vector") 
    parser.add_argument("props", nargs=1, help="Property file (for cross-check)")
    parser.add_argument("--csi", type=float, default='1.0', help="Kernel scaling")
    
    args = parser.parse_args()
    
    main(kernel=args.kernel[0], props=args.props[0], weights=args.weights[0], csi=args.csi)

