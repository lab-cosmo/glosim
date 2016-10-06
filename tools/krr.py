#!/usr/bin/env python

# compute kernel ridge regression for a data set given a kernel matrix
# and a vector of the observed properties. syntax:
# $  krr.py <kernel.dat> <properties.dat> [ options ]

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

def main(kernel, props, mode, trainfrac, csi, sigma, ntests, ttest, savevector="", refindex=""):

    trainfrac=float(trainfrac) 
    csi = float(csi)
    sigma = float(sigma)
    ntests = int(ntests)
    ttest=float(ttest)
    if (mode == "sequential" or mode == "all") and ntests>1:
        raise ValueError("No point in having multiple tests when using determininstic train set selection")
     
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
    if kij[0,0]<1e-8:
        kij = (1-0.5*kij*kij)
        
    
    # reads index, if available
    if refindex == "":
        rlabs = np.asarray(range(nel), int)
    else:
        rlabs = np.loadtxt(refindex,dtype=int)
        if len(rlabs) != nel:
            raise ValueError("Reference index size mismatch")
    
    # first hyperparameter - we raise the kernel to a positive exponent to make it sharper or smoother
    kij = kij**csi
    
    # reads properties
    p = np.zeros(nel)
    fprop=open(props,"r")
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
    
    # chooses test
    testmae=0
    trainmae=0
    truemae=0
    testrms=0
    trainrms=0
    truerms=0
    testsup=0
    trainsup=0
    truesup=0
    ctrain=0
    ctest=0
    ctrue=0
    
    vp = np.var(p)    
    
    if mode=="manual":
        mtrain = np.loadtxt("train.idx")
#    kij *= vp    
    if mode == "all" :
            tp = p[:]
            tk = kij[:][:].copy()
            #print lweight
            for i in xrange(len(tp)):
                tk[i,i]+=sigma #/ lweight[i]  # diagonal regularization times weight!
            tc = np.linalg.solve(tk, tp)
            krp = np.dot(kij[:,:],tc)
            mae=abs(krp[:]-p[:]).sum()/len(p)
            rms=np.sqrt(((krp[:]-p[:])**2).sum()/len(p))
            sup=abs(krp[:]-p[:]).max()
            print "# train-set MAE: %f RMS: %f SUP: %f" % (mae, rms, sup)
            ltrain = range(nel)            
    else: 
        np.set_printoptions(threshold=10000)
        ntrain = int(trainfrac*nel)
        if mode == "manual": ntrain=len(mtrain)
        ntrue = int(ttest*nel)        

        for itest in xrange(ntests):        
            ltest = np.zeros(nel-ntrain-ntrue,int)
            ltrain = np.zeros(ntrain,int)
            
            # if specified, select some elements that are completely ignored from both selection and training
            ltrue = np.zeros(ntrue, int)
            psel = np.ones(nel,float)
            if ntrue > 0:
                ltrue = randomsubset(nel, ntrue)
                psel[ltrue] = 0.0
            if mode == "random":
                ltrain[:] = randomsubset(nel, ntrain, psel)
            elif mode == "manual":
                ltrain[:] = mtrain
            elif mode == "sequential":
                ltrain[:] = range(ntrain)
            elif mode == "fps":            
                isel=int(np.random.uniform()*nel)
                while isel in ltrue:
                    isel=int(np.random.uniform()*nel)
                    
                ldist = 1e100*np.ones(nel,float)
                imin = np.zeros(nel,int) # index of the closest FPS grid point
                ltrain[0]=isel
                nontrue = np.setdiff1d(range(nel), ltrue)
                for nsel in xrange(1,ntrain):
                    dmax = 0
                    imax = 0       
                    for i in nontrue:
                        dsel = np.sqrt(kij[i,i]+kij[isel,isel]-2*kij[i,isel]) #don't assume kernel is normalised
                        if dsel < ldist[i]:
                           imin[i] = nsel-1                    
                           ldist[i] = dsel
                        if ldist[i] > dmax:
                            dmax = ldist[i]; imax = i
                    print "selected ", isel, " distance ", dmax
                    isel = imax
                    ltrain[nsel] = isel
                
                for i in xrange(nel):
                    if i in ltrue: continue                    
                    dsel = np.sqrt(kij[i,i]+kij[isel,isel]-2*kij[i, isel])
                  #  dsel = np.sqrt(1.0-kij[i, isel])
                    if dsel < ldist[i]:
                        imin[i] = nsel-1
                        ldist[i] = dsel
                    if ldist[i] > dmax:
                        dmax = ldist[i]; imax = i
            
            k = 0
            for i in xrange(nel):
                if not i in ltrain and not i in ltrue: 
                    ltest[k] = i
                    k += 1
                
            tp = p[ltrain]
            tk = kij[ltrain][:,ltrain].copy()
            #print lweight
            for i in xrange(len(ltrain)):
                tk[i,i]+=sigma #/ lweight[i]  # diagonal regularization times weight!
            tc = np.linalg.solve(tk, tp)
            krp = np.dot(kij[:,ltrain],tc)   

            mae=abs(krp[ltest]-p[ltest]).sum()/len(ltest)
            rms=np.sqrt(((krp[ltest]-p[ltest])**2).sum()/len(ltest))
            sup=abs(krp[ltest]-p[ltest]).max()
            print "# run: %d test-set MAE: %f RMS: %f SUP: %f" % (itest, mae, rms, sup)
            

            testmae += abs(krp[ltest]-p[ltest]).sum()/len(ltest)
            trainmae += abs(krp[ltrain]-p[ltrain]).sum()/len(ltrain)
            if ntrue>0: truemae += abs(krp[ltrue]-p[ltrue]).sum()/len(ltrue)
            testrms += np.sqrt(((krp[ltest]-p[ltest])**2).sum()/len(ltest))
            trainrms += np.sqrt(((krp[ltrain]-p[ltrain])**2).sum()/len(ltrain))
            if ntrue>0: truerms += np.sqrt(((krp[ltrue]-p[ltrue])**2).sum()/len(ltrue))
            testsup+=abs(krp[ltest]-p[ltest]).max()
            trainsup+=abs(krp[ltrain]-p[ltrain]).max()
            if ntrue>0: truesup+=abs(krp[ltrue]-p[ltrue]).max()
            ctrain+=len(ltrain)
            ctest+=len(ltest)
            ctrue+=len(ltrue)
                    
            for i in xrange(nel):
               if i in ltrain: 
                   lab = "TRAIN "
               elif i in ltrue:
                   lab = "TRUE "
               else: lab = "TEST"
               print i, p[i], krp[i], lab 

        print "# KRR results (%d tests, %f training p., %f test p.): csi=%f  sigma=%f" % (ntests, ctrain/ntests, ctest/ntests, csi, sigma) 
        print "# Train points MAE=%f  RMSE=%f  SUP=%f" % (trainmae/ntests, trainrms/ntests, trainsup/ntests)
        print "# Test points  MAE=%f  RMSE=%f  SUP=%f " % (testmae/ntests, testrms/ntests, testsup/ntests)
        if len(ltrue) > 0: 
            print "# True test points  MAE=%f  RMSE=%f  SUP=%f " % (truemae/ntests, truerms/ntests, truesup/ntests)
    
    if savevector:
        fname=open(savevector,'w')
        commentline=' Train Vector from kernel matrix: '+kernel+' and properties from '+ props + ' selection mode: '+mode+' : Csi, sigma = ' + str(csi) +' , '+ str(sigma)
        np.savetxt(fname,np.asarray([tc, ltrain, rlabs[ltrain]]).T,fmt=("%15.7e", "%10d", "%10d"),header=commentline)
        fname.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes KRR and analytics based on a kernel matrix and a property vector.""")
                           
    parser.add_argument("kernel", nargs=1, help="Kernel matrix")      
    parser.add_argument("props", nargs=1, help="Property file")
    parser.add_argument("--mode", type=str, default="random", help="Train point selection (e.g. --mode all / sequential / random / fps / cur / manual")      
    parser.add_argument("-f", type=float, default='0.5', help="Train fraction")
    parser.add_argument("--truetest", type=float, default='0.0', help="Take these points out from the selection procedure")
    parser.add_argument("--csi", type=float, default='1.0', help="Kernel scaling")
    parser.add_argument("--sigma", type=float, default='1e-3', help="Sigma")
    parser.add_argument("--ntests", type=int, default='1', help="Number of tests")
    parser.add_argument("--refindex",  type=str, default="", help="Structure indices of the kernel matrix (useful when dealing with a subset of a larger structures file)")        
    parser.add_argument("--saveweights",  type=str, default="", help="Save the train-set weights vector in file")    
    
    args = parser.parse_args()
    
    main(kernel=args.kernel[0], props=args.props[0], mode=args.mode, trainfrac=args.f, csi=args.csi, 
         sigma=args.sigma, ntests=args.ntests, ttest=args.truetest,savevector=args.saveweights, refindex=args.refindex)
