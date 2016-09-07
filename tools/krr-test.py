#!/usr/bin/python
import sys
import argparse
import numpy as np
import math

bufsize = 10000
def main(kernel, props, weights, csi):
    csi=float(csi)    
    tc=np.loadtxt(weights)
    p=np.loadtxt(props)
    
    f = open(kernel, "r")
    fstart = f.tell()
    fline = f.readline()
    while fline[0]=='#': 
        fstart = f.tell()
        fline=f.readline()
    nref = len(fline.split())
    f.seek(fstart)
    
    chunk = np.fromfile(f, dtype="float",count=(nref)*bufsize, sep=" ")        
        
    kij = kij**csi
    #print lweight
    if (len(kij[0]) != len(tc)):
        print "inconsistent kernel and train vector file"
        return
    if (len(kij) != len(p)):
        print "inconsistent kernel and property file"
        return
    krp = np.dot(kij,tc)
    nconf=len(p)
    mae=abs(krp[:]-p[:]).sum()/nconf
    rms=np.sqrt(((krp[:]-p[:])**2).sum()/nconf)
    diff=abs(krp[:]-p[:])
    sup=max(diff)
    print "# test-set MAE: %f RMS: %f SUP %f" % (mae, rms, sup)
    comment= "# test-set MAE: "+str(mae)+ " RMSE: "+ str(rms)+" SUP: " +str(sup)
    fname=proplist+".predict"
    f=open(fname,'w')
    np.savetxt(f,krp,header=comment)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes KRR predictions from a weights vector obtained from a previous run of krr.py with --saveweights.""")
    parser.add_argument("kernel", nargs=1, help="Kernel matrix")      
    parser.add_argument("weights", nargs=1, help="Weights vector") 
    parser.add_argument("props", nargs=1, help="Property file (for cross-check)")
    parser.add_argument("--csi", type=float, default='1.0', help="Kernel scaling")
    
                           
    
   main(kernel=args.kernel[0], props=args.props[0], weights=args.weights[0], csi=args.csi)

