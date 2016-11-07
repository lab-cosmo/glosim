#!/usr/bin/env python

# compute kernel ridge regression for a data set given a kernel matrix
# and a vector of the observed properties. syntax:
# $  krr.py <kernel.dat> <properties.dat> [ options ]

import argparse
import numpy as np
import sys
import MultipleKernelLearning as mkl 
import costs as cst
from select_landmarks import farthestPointSampling,randomsubset,cur,segfind


def main(kernelFilenames, propFilename, mode, trainfrac, csi,  ntests, ttest, savevector="", refindex="",**KRRCortesParam):

    trainfrac=float(trainfrac) 
    #csi = float(csi)
    ntests = int(ntests)
    ttest=float(ttest)
    if (mode == "sequential" or mode == "all") and ntests>1:
        raise ValueError("No point in having multiple tests when using determininstic train set selection")
     
    # Reads kernels
    nbOfKernels = len(kernelFilenames)
    kernels = []
    for it,kernelFilename in enumerate(kernelFilenames):
        kernels.append(np.loadtxt(kernelFilename, dtype=np.float64, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0))
        # heuristics to see if this is a kernel or a similarity matrix!!
        if kernels[it][0,0]<1e-8:
            kernels[it] = (1-0.5*kernels[it]*kernels[it])
        # first hyperparameter - we raise the kernel to a positive exponent to make it sharper or smoother
        kernels[it] = kernels[it]**csi[it]
    
    
    # reads properties
    prop = np.loadtxt(propFilename, dtype=np.float64, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
    
    # check if size of input of the kernels and property is consistant
    for it,kernel in enumerate(kernels):
        if len(prop) != len(kernel):
            raise ValueError("Dimention mismatch between kernel {} and prop".format(kernelFilenames[it]))
    for it,kernel1 in enumerate(kernels):
        for jt,kernel2 in enumerate(kernels):
            if kernel1.shape != kernel2.shape:
                raise ValueError("Dimention mismatch between kernel {} and kernel {}".format(kernelFilenames[it]),kernelFilenames[jt])

    # Kernel matrices should be square and of the same size
    nel = len(kernels[0])
    
    # reads index, if available
    if refindex == "":
        rlabs = np.asarray(range(nel), int)
    else:
        rlabs = np.loadtxt(refindex,dtype=int)
        if len(rlabs) != nel:
            raise ValueError("Reference index size mismatch")   


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
    
    if mode=="manual":
        mtrain = np.loadtxt("train.idx")
    if mode == "all" :
        raise NotImplementedError("")
            # tp = p[:]
            # tk = kij[:][:].copy()
            # vp = np.var(tp) # variance of the property subset (to be size consistent!)            
            # vk = np.trace(tk)/len(tp)
            # print >> sys.stderr, "Regularization shift ", sigma**2 * vk/vp
            # #print lweight
            # for i in xrange(len(tp)):
            #     tk[i,i]+=sigma**2 * vk/vp  #/ lweight[i]  # diagonal regularization times weight!
            # tc = np.linalg.solve(tk, tp)
            # krp = np.dot(kij[:,:],tc)
            # mae=abs(krp[:]-p[:]).sum()/len(p)
            # rms=np.sqrt(((krp[:]-p[:])**2).sum()/len(p))
            # sup=abs(krp[:]-p[:]).max()             
            # print "# train-set MAE: %f RMS: %f SUP: %f" % (mae, rms, sup)
            # ltrain = range(nel)            
    else: 
        np.set_printoptions(threshold=10000)
        ntrain = int(trainfrac*nel)
        if mode == "manual": ntrain=len(mtrain)
        ntrue = int(ttest*nel)        
        seeds = np.random.randint(0,5000,ntests)
        alphas = []
        mus = []
        testMAEs = []
        ltrains = []
        rlabss = []
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
                # do farthest point sampling on the uniform combination of the kernels
                kij = np.zeros((nel,nel),dtype=np.float64)
                for kernel in kernels:
                    kij += kernel

                isel=int(np.random.uniform()*nel)
                while isel in ltrue:
                    isel=int(np.random.uniform()*nel)
                
                ltrain = farthestPointSampling(kij,nel,ntrain,initalLandmark=isel,listOfDiscardedPoints=ltrue,seed=seeds[itest])    
            
            k = 0
            for i in xrange(nel):
                if not i in ltrain and not i in ltrue: 
                    ltest[k] = i
                    k += 1
                

            # # the kernel should represent the variance of the energy (in a GAP interpretation) 
            # # and sigma^2 the estimate of the noise variance. However we want to keep a "naked kernel" so
            # # we can then estimate without bringing around the variance. So the problem would be
            # # (vp*N/Tr(tk) tk + sigma^2 I )^-1 p = w
            # # but equivalently we can write 
            # # ( tk + sigma^2 *tr(tk)/(N vp) I )^-1 p = w            
            

            # get prop of reference for training and testing
            propTeRef = prop[ltest]
            propTrRef = prop[ltrain]

            # Train your model and get the optimal weights out 
            kernelsTr = []
            for it,kernel  in enumerate(kernels):
                kernelsTr.append(kernel[np.ix_(ltrain,ltrain)])
            

            alpha, mu, propTr = mkl.TrainKRRCortes(kernelsTr,propTrRef,**KRRCortesParam)
            
            # Predict property using the optimal weights
            kernelsTe = []
            for it,kernel  in enumerate(kernels):
                kernelsTe.append(kernel[np.ix_(ltrain,ltest)])

            propTe = mkl.PredictKRRCortes(kernelsTe,alpha,mu)
            
            
            mae = cst.mae(propTe-propTeRef)
            rms = cst.rmse(propTe-propTeRef)
            sup = cst.sup_e(propTe-propTeRef)
            print "# run: {} test-set MAE: {:.4e} RMS: {:.4e} SUP: {:.4e}".format(itest, mae, rms, sup)
            
            # accumulate output to select the weigths corresponding to the lowest MAE
            alphas.append(alpha)
            mus.append(mu)
            testMAEs.append(mae)
            ltrains.append(ltrain)
            rlabss.append(rlabs)
            

            testmae += cst.mae(propTe-propTeRef)
            trainmae += cst.mae(propTr-propTrRef)
            #if ntrue>0: truemae += abs(krp[ltrue]-prop[ltrue]).sum()/len(ltrue)
            testrms += cst.rmse(propTe-propTeRef)
            trainrms += cst.rmse(propTr-propTrRef)
            #if ntrue>0: truerms += np.sqrt(((krp[ltrue]-prop[ltrue])**2).sum()/len(ltrue))
            testsup += cst.sup_e(propTe-propTeRef)
            trainsup += cst.sup_e(propTr-propTrRef)
            #if ntrue>0: truesup+=abs(krp[ltrue]-prop[ltrue]).max()
            ctrain+=len(ltrain)
            ctest+=len(ltest)
            ctrue+=len(ltrue)
            

            # for it,jt  in enumerate(ltrain):
            #     print jt, propTrRef[it], propTr[it], "TRAIN" 
            # for it,jt  in enumerate(ltest):
            #     print jt, propTeRef[it], propTe[it], "TEST" 
            
            # print alpha
            print 'Mu = {}'.format(mu)
        
        print "# KRR results ({:d} tests, {:f} training p., {:f} test p.): csi={}  sigma={:.2e} mu0={} Lambda={:.1f} epsilon={:.1e} eta={:.1e} "\
        .format(ntests, ctrain/ntests, ctest/ntests, csi, KRRCortesParam['sigma'],KRRCortesParam['mu0'],KRRCortesParam['Lambda'],KRRCortesParam['epsilon'],KRRCortesParam['eta']) 
        print "# Train points averages: MAE={:.4e}  RMSE={:.4e}  SUP={:.4e}".format(trainmae/ntests, trainrms/ntests, trainsup/ntests)
        print "# Test points averages: MAE={:.4e}  RMSE={:.4e}  SUP={:.4e} ".format(testmae/ntests, testrms/ntests, testsup/ntests)
        if len(ltrue) > 0: 
            print "# True test points  MAE=%f  RMSE=%f  SUP=%f " % (truemae/ntests, truerms/ntests, truesup/ntests)
    
    if savevector:
        bestRunIdx = np.argmin(testMAEs)
        falpha = open(savevector+'.alpha','w')
        fmu = open(savevector+'.mu','w')
        kernelFilenamesStr = '';
        for it,kernelFilename in enumerate(kernelFilenames): 
            kernelFilenamesStr+=kernelFilename+' '

        commentline=' Train Vector from kernel matrix with the best MAE test score ('+str(np.min(testMAEs))+'): '+ kernelFilenamesStr +', and properties from '+ propFilename + ' selection mode: '+mode+' : Csi, sigma, mu0, Lambda, epsilon, eta = ' + str(csi) +' , '+ str(KRRCortesParam['sigma']) \
                    +' , '+ str(mu0)+' , '+ str(KRRCortesParam['Lambda'])+' , '+ str(KRRCortesParam['epsilon'])+' , '+ str(KRRCortesParam['eta'])
        np.savetxt(falpha,np.asarray([alphas[bestRunIdx], ltrains[bestRunIdx], rlabss[bestRunIdx][ltrains[bestRunIdx]]]).T,fmt=("%24.15e", "%10d", "%10d"),header=commentline)
        np.savetxt(fmu,mus[bestRunIdx],fmt=("%24.15e"),header=commentline)
        
        # commentline=' Train Vector from kernel matrix with the best MAE test score ('+str(testMAEs[-1])+'): '+ kernelFilenamesStr +', and properties from '+ propFilename + ' selection mode: '+mode+' : Csi, sigma, mu0, Lambda, epsilon, eta = ' + str(csi) +' , '+ str(KRRCortesParam['sigma']) \
        #             +' , '+ str(mu0)+' , '+ str(KRRCortesParam['Lambda'])+' , '+ str(KRRCortesParam['epsilon'])+' , '+ str(KRRCortesParam['eta'])
        
        # np.savetxt(falpha,np.asarray([alpha, ltrain, rlabs[ltrain]]).T,fmt=("%24.15e", "%10d", "%10d"),header=commentline)
        # np.savetxt(fmu,mu,fmt=("%24.15e"),header=commentline)
        falpha.close()
        fmu.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes Multiple Kernel Learning KRR from Cortes and analytics based on a kernel matrix and a property vector.""")
                           
    parser.add_argument("kernels", nargs=1, help="Kernel matrices. List of coma separated file names.")      
    parser.add_argument("props", nargs=1, help="Property file name.")
    parser.add_argument("--mode", type=str, default="random", help="Train point selection (e.g. --mode all / sequential / random / fps / cur / manual")      
    parser.add_argument("-f", type=float, default='0.5', help="Train fraction")
    parser.add_argument("--truetest", type=float, default='0.0', help="Take these points out from the selection procedure")
    parser.add_argument("--csi", type=str, default='', help="Kernel scaling. list of coma separated positive values (e.g. 1,1,1 )")
    parser.add_argument("--sigma", type=float, default='1e-3', help="KRR regularization. In units of the properties. ")
    parser.add_argument("--epsilon", type=float, default='2e-3', help="KRR-Mkl param. convergence tolerance on alpha weights absolute difference.")    
    parser.add_argument("--Lambda", type=float, default='1', help="KRR-Mkl param. Radius of the ball containing the possible weights of the kernel combination, positive value")
    parser.add_argument("--mu0", type=str, default='', help="KRR-Mkl param. Center of the ball containing the possible weights of the kernel combination, list of coma separated positive values (e.g. 1,1,1 )")
    parser.add_argument("--maxIter", type=float, default='1e2', help="KRR-Mkl param. Maximal number of iteration. ")
    parser.add_argument("--eta", type=float, default='0.5', help="KRR-Mkl param. Interpolation parameter for the update of alpha, belongs to ]0,1[. ")
    parser.add_argument("--ntests", type=int, default='1', help="Number of tests")
    parser.add_argument("--refindex",  type=str, default="", help="Structure indices of the kernel matrix (useful when dealing with a subset of a larger structures file)")        
    parser.add_argument("--saveweights",  type=str, default="", help="Save the train-set weights vector in file")    
    
    args = parser.parse_args()

    kernelFilenames = args.kernels[0].split(',')

    a = args.mu0.split(',')
    if len(a) != len(kernelFilenames):
        raise ValueError("The number of kernel file names and elements of mu0 must be equal.")
    mu0 = np.zeros(len(a),dtype=np.float64)
    for it,item in enumerate(a):
        mu0[it] = float(item)

    a = args.csi.split(',')
    if len(a) != len(kernelFilenames):
        raise ValueError("The number of kernel file names and elements of csi must be equal.")
    csi = np.zeros(len(a),dtype=np.float64)
    for it,item in enumerate(a):
        csi[it] = float(item)

    KRRCortesParam = {'mu0':mu0,'epsilon':args.epsilon,'Lambda':args.Lambda,'eta':args.eta,
            'maxIter':args.maxIter,'sigma':args.sigma}  
    
    
    main(kernelFilenames=kernelFilenames, propFilename=args.props[0], mode=args.mode, 
        trainfrac=args.f, csi=csi, ntests=args.ntests, refindex=args.refindex,
        ttest=args.truetest,savevector=args.saveweights, **KRRCortesParam)

