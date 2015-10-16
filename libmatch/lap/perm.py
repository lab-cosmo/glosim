import numpy as np    
    
__all__ = [ "xperm", "rndperm","mcperm" ]

def _mcperm(mtx, eps = 1e-3, maxtry=None):
    sz = len(mtx[0])
    idx = np.asarray(xrange(sz),int)
    
    prm = 0
    prm2 = 0
    pstride=100*sz
    if not maxtry is None:  maxtry *= pstride
    i=0
    while True:
        np.random.shuffle(idx)
        pi = 1.
        for j in xrange(sz):
            pi *= mtx[j, idx[j]]
        prm += pi
        prm2 += pi*pi
        i+=1
        if (i)%pstride==0:
            err=np.sqrt( (prm2-prm*prm/i)/i/(i-1) ) / (prm/i)
            print i/pstride, prm/i, err; 
            if ((not maxtry is None) and i>maxtry) or err<eps: break
            
    return prm/i*np.math.factorial(sz)    


# Monte Carlo evaluation of the permanent
try:
    from permanent import permanent_mc, permanent_ryser
    def mcperm(mtx, eps=1e-3):
        return permanent_mc(mtx,eps)
    def xperm(mtx, eps=1e-3):
        return permanent_ryser(mtx)
except:
    print >> sys.stderr, "Cannot find mcpermanent.so module in pythonpath. Permanent evaluations will be very slow and approximate."
    print >> sys.stderr, "Get it from https://github.com/sandipde/MCpermanent "
    def mcperm(mtx, eps=1e-2):
        return _mcperm(mtx,eps)
    def xperm(mtx, eps=1e-6):
        return _mcperm(mtx,eps)
   
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
