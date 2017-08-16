#!/usr/bin/env python

# select landmarks based on a kernel matrix 
# given a  vector of the observed properties it also output only properties of landmarks.
# $  select-landmarks.py <kernel.dat> [ options ]

import argparse
import numpy as np
import sys

def segfind(cp, cs):
    a = 0
    b = len(cp)    
    while (b-a)>1:
        c = int((b+a)/2)
        if cs<cp[c]:
            b = c
        else:
            a = c
    if cs < cp[a]:
        return a
    else:
        return b
        
def cur(kernel,tol=1.0e-4):

    U, S, V = np.linalg.svd(kernel)
    rank = list(S > tol).count(True) / 2
    S[rank:] = 0.0
    S = np.diag(S)
    Ap = np.dot(np.dot(U,S),V)
    p = np.sum(V[0:rank,:]**2, axis=0) / rank
    return p

def randomsubset(ndata, nsel, plist=None):    
    if nsel > ndata:
        raise ValueError("Cannot select data out of thin air")
    if nsel == ndata: 
        return np.asarray(range(ndata))
    cplist = np.zeros(ndata)
    if plist is None:
        plist = np.ones(ndata, float)
        
    # computes initial cumulative probability distr.
    cplist[0]=plist[0] 
    for i in xrange(1,ndata):
        cplist[i]=cplist[i-1]+plist[i]
    
    rdata = np.zeros(nsel, int)
    for i in xrange(nsel):
        csel = np.random.uniform() * cplist[-1]
        isel = segfind(cplist, csel)
        rdata[i] = isel
        psel = plist[isel]
        for j in xrange(isel,ndata):
            cplist[j] -= psel
    return rdata


def farthestPointSampling(kernel, nbOfFrames, nbOfLandmarks, seed=10, initalLandmark=None, listOfDiscardedPoints=None,
                          verbose=False):
    np.random.seed(seed)
    LandmarksIdx = -1 * np.ones(nbOfLandmarks, int)
    if listOfDiscardedPoints is None:
        listOfDiscardedPoints = []
    if initalLandmark is None:
        isel = int(np.random.uniform() * nbOfFrames)
        while isel in listOfDiscardedPoints:
            newIdx = int(np.random.uniform() * nbOfFrames)
    else:
        newIdx = initalLandmark

    diag = np.diag(kernel)

    ldist = 1e100 * np.ones(nbOfFrames, float)

    LandmarksIdx[0] = newIdx
    nontrue = np.setdiff1d(range(nbOfFrames), listOfDiscardedPoints)

    for nsel in xrange(1, nbOfLandmarks):

        distLine = np.sqrt(kernel[newIdx, newIdx] + diag - 2 * kernel[newIdx, :])

        dsel = distLine[nontrue]

        ldist[dsel < ldist] = dsel[dsel < ldist]

        # select indices that are the farther away
        selIds = np.argwhere(ldist == np.max(ldist)).reshape((-1,))
        # select the indices that have not already been selected
        newIds = list(np.setdiff1d(selIds, LandmarksIdx).reshape((-1,)))
        # get the first one
        newIdx = newIds[0]
        LandmarksIdx[nsel] = newIdx

        if verbose is True:
            print "selected ", newIdx, " distance ", ldist[newIdx]

    return LandmarksIdx

def main(kernel, props, mode, nland,output="distance", prefix=""):

    if prefix=="" : prefix=kernel[0:-2]     
    # reads kernel
    kij=np.loadtxt(kernel)
    nel=len(kij) 
    # reads properties if given
    if props!="":
       p = np.loadtxt(props)
       if len(p)!=nel : 
         print "ERROR ! incomplete set of properties"
         exit()
    
    np.set_printoptions(threshold=1000)
    nland = int(nland)
    
    psel = np.ones(nel,float)
    if mode == "random":
        LandmarksIdx = randomsubset(nel, nland, psel)
    elif mode == "fps":  
        LandmarksIdx = farthestPointSampling(kij,nel,nland)
        

    filand=prefix+"-landmark"+str(nland)+".index"
    np.savetxt(filand,LandmarksIdx,fmt='%1.1i')
    if props != "":
        lp = p[LandmarksIdx]
        fpland=prefix+"-landmark"+str(nland)+".prop"
        np.savetxt(fpland,lp)

    if output=="kernel":    
        print "Writing Kernels"
        lk = kij[LandmarksIdx][:,LandmarksIdx].copy()
        fkland=prefix+"-landmark"+str(nland)+".k"
        np.savetxt(fkland,lk)
        foos=prefix+"-landmark"+str(nland)+"-OOS.k"
        koos=kij[:,LandmarksIdx]
        np.savetxt(foos,koos)

    if output=="distance" :
        print "Writing Distances"
        sim=np.zeros((nel,nel))
        for i in range(nel):
            for j in range(i):
                sim[i,j]=sim[j,i]=np.sqrt(kij[i,i]+kij[j,j]-2*kij[i,j])
        ld=sim[LandmarksIdx][:,LandmarksIdx].copy()   
        fsimland=prefix+"-landmark"+str(nland)+".sim"
        np.savetxt(fsimland,ld)
        foos=prefix+"-landmark"+str(nland)+"-OOS.sim"
        simoos=sim[:,LandmarksIdx]
        np.savetxt(foos,simoos)
         
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Do landmarks selction based on a kernel matrix and export square matrix for landmarks and rectangular matrix.""")
                           
    parser.add_argument("kernel", nargs=1, help="Kernel matrix")      
    parser.add_argument("--props",type=str, default="", help="Property file")
    parser.add_argument("--mode", type=str, default="random", help="landmark selection (e.g. --mode  random / fps ")      
    parser.add_argument("--output", type=str, default="distance", help="what to output kernel/distance ")      
    parser.add_argument("--nland", type=int, default=1, help="number of landmarks")
    parser.add_argument("--prefix",  type=str, default="", help="prefix of the output files")    
    
    args = parser.parse_args()
    
    main(kernel=args.kernel[0], props=args.props, mode=args.mode,nland=args.nland,output=args.output, prefix=args.prefix)
