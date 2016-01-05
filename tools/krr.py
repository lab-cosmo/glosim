#!/usr/bin/python

# compute kernel ridge regression for a data set given a kernel matrix
# and a vector of the observed properties. syntax:
# $  krr.py <kernel.dat> <properties.dat> [ testfrac, csi, sigma ]

import numpy as np
import sys

def main(kernel, prop,  testfrac="0.25", csi="2", sigma="1e-3"):
    testfrac=float(testfrac) 
    csi = float(csi)
    sigma = float(sigma)
     
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
    ltest=[]
    ltrain=[]
    for i in xrange(nel):
        if np.random.uniform()<testfrac:  ltest.append(i)
        else: ltrain.append(i)
    ltest = np.asarray(ltest)
    ltrain = np.asarray(ltrain)
    
    tp = p[ltrain]
    vp = np.var(tp)
    
    kij *= vp
    tk = kij[ltrain][:,ltrain].copy()
    for i in xrange(len(ltrain)):
        tk[i,i]+=vp*sigma  # diagonal regularization
    tc = np.linalg.solve(tk, tp)
   
    print "# KRR results: csi=%f  sigma=%f" % (csi, sigma)  
    krp = np.dot(kij[:,ltrain],tc)
    print "# Train points (%d). RMSE=%f  MAE=%f" % (len(ltrain),np.sqrt(((krp[ltrain]-p[ltrain])**2).sum()/len(ltrain)),abs(krp[ltrain]-p[ltrain]).sum()/len(ltrain))
    for t in ltrain:
        print p[t], krp[t]
    print "# Test points  (%d). RMSE=%f  MAE=%f" % (len(ltest),np.sqrt(((krp[ltest]-p[ltest])**2).sum()/len(ltest)),abs(krp[ltest]-p[ltest]).sum()/len(ltest))
    for t in ltest:
        print p[t], krp[t]
        
    
    
    
if __name__ == '__main__':
   main(*sys.argv[1:])
