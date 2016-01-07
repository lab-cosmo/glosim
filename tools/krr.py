#!/usr/bin/python

# compute kernel ridge regression for a data set given a kernel matrix
# and a vector of the observed properties. syntax:
# $  krr.py <kernel.dat> <properties.dat> [ testfrac, csi, sigma ]

import numpy as np
import sys

def main(kernel, prop,  testfrac="0.25", csi="2", sigma="1e-3", ntests=10):
    testfrac=float(testfrac) 
    csi = float(csi)
    sigma = float(sigma)
    ntests = int(ntests)
     
    # reads kernel
    fkernel=open(kernel, "r")
    fline = fkernel.readline()
    while fline[0]=='#': fline=fkernel.readline()
    sline=map(float,fline.split())
    nel = len(sline)
    kij = np.zeros((nel,nel), float)
    ik = 0
    while (len(sline)==nel):
        kij[ik]=np.asarray(sline)
        fline = fkernel.readline()
        sline=map(float,fline.split())
        ik+=1
    # heuristics to see if this is a kernel or a similarity matrix!!
    if kij[0,0]<1e-5:
        kij = (1-0.5*kij*kij)
    
    # first hyperparameter - we raise the kernel to a positive exponent to make it sharper or smoother
    kij = kij**csi
    #print >> sys.stderr, kij[0]
    
    # reads properties
    p = np.zeros(nel)
    fprop=open(prop,"r")
    fline = fprop.readline()
    while fline[0]=='#': fline=fprop.readline()
    ik=0
    while (len(fline)>0):
        p[ik]=float(fline)
        fline = fprop.readline()
        ik+=1
    if ik<nel : 
        print "ERROR"
        exit()
    
    # chooses test set randomly
    testmae=0
    trainmae=0
    testrms=0
    trainrms=0
    ntrain=0
    ntest=0
    
    vp = np.var(p)    
#    kij *= vp        
    for itest in xrange(ntests):
        ltest=[]
        ltrain=[]
        for i in xrange(nel):
            if np.random.uniform()<testfrac:  ltest.append(i)
            else: ltrain.append(i)
        ltest = np.asarray(ltest, int)
        ltrain = np.asarray(ltrain, int)
    
        tp = p[ltrain]
        tk = kij[ltrain][:,ltrain].copy()
        for i in xrange(len(ltrain)):
            tk[i,i]+=sigma  # diagonal regularization
        tc = np.linalg.solve(tk, tp)
        krp = np.dot(kij[:,ltrain],tc)   

        mae=abs(krp[ltest]-p[ltest]).sum()/len(ltest)
        rms=np.sqrt(((krp[ltest]-p[ltest])**2).sum()/len(ltest))
        print "# run: %d TEST MAE: %f RMS: %f" % (itest, mae, rms)

        testmae += abs(krp[ltest]-p[ltest]).sum()/len(ltest)
        trainmae += abs(krp[ltrain]-p[ltrain]).sum()/len(ltrain)
        testrms += np.sqrt(((krp[ltest]-p[ltest])**2).sum()/len(ltest))
        trainrms += np.sqrt(((krp[ltrain]-p[ltrain])**2).sum()/len(ltrain))
        ntrain+=len(ltrain)
        ntest+=len(ltest)

    print "# KRR results (%d tests, %f training p., %f test p.): csi=%f  sigma=%f" % (ntests, ntrain/ntests, ntest/ntests, csi, sigma) 
    print "# Train points MAE=%f  RMSE=%f" % (trainmae/ntests, trainrms/ntests)
    print "# Test points  MAE=%f  RMSE=%f" % (testmae/ntests, testrms/ntests)
    print "# Full KRR predictions: index target krr"
    for i in xrange(nel):
        print i, p[i], krp[i]

    
    
    
if __name__ == '__main__':
   main(*sys.argv[1:])
