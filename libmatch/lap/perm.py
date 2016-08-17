import numpy as np    
import sys    
__all__ = [ "xperm", "rndperm","mcperm" ]

def _mcperm(mtx, eps = 1e-3, ntry=None, seed=None):
    sz = len(mtx[0])
    idx = np.asarray(xrange(sz),int)
    
    prm = 0
    prm2 = 0
    pstride = 100*sz
    i=0
    if not seed is None: 
        np.random.seed(seed)
    while True:
        np.random.shuffle(idx)
        pi = 1.
        for j in xrange(sz):
            pi *= mtx[j, idx[j]]
        prm += pi
        prm2 += pi*pi
        i+=1
        if (not ntry is None) and i >= ntry: break
        if ntry is None and (i)%pstride==0:
            err=np.sqrt( (prm2-prm*prm/i)/i/(i-1) ) / (prm/i)
            if err<eps: break
            
    return prm/i*np.math.factorial(sz)    


# Monte Carlo evaluation of the permanent
try:
    from permanent import permanent_mc, permanent_ryser, rematch
    def mcperm(mtx, eps=1e-3, ntry=None, seed=None):  # , ntry=100000, seed=12345): #
        return permanent_mc(mtx,eps,0 if (ntry is None) else ntry, 0 if (seed is None) else seed)
    def xperm(mtx):
        return permanent_ryser(mtx)
except:
    print >> sys.stderr, "Cannot find mcpermanent.so module in pythonpath. Permanent evaluations will be very slow and approximate."
    print >> sys.stderr, "Get it from https://github.com/sandipde/MCpermanent "
    def mcperm(mtx, eps=1e-2, ntry=None, seed=None):
        return _mcperm(mtx,eps,ntry,seed)
    def xperm(mtx, eps=1e-6):
        return _mcperm(mtx,eps)
    def rematch(mtx, gamma, eps):
        raise ValueError("No Python equivalent to rematch function...")
   
import time, sys
if __name__ == "__main__":
    
    filename = sys.argv[1]
    mtx=np.loadtxt(filename)
    #mtx=np.random.rand(10,10)
    st=time.time()
    new=_mcperm(mtx, eps=1e-2)
    tnew = time.time()-st    
    st=time.time()
    cnew=mcperm(mtx,1e-3)
    ctnew = time.time() -st
    st=time.time()
    if len(mtx[0])<30: 
        ref=xperm(mtx)
    else: ref=0
    tref = time.time()-st
    
    print "Reference:          ", ref, " time: ", tref
    print "New_method:         ", new, " time: ", tnew
    print "New_method C++:         ", cnew, " time: ",ctnew
