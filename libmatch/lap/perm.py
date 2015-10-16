try:
    from permanent import permanent
    from mypermanent import mypermanent
    
except:
    print >> sys.stderr, "Cannot compute permanent kernel without a permanent module installed in pythonpath"
    print >> sys.stderr, "Get it from https://github.com/peteshadbolt/permanent "
    exit()

import numpy as np    
    
__all__ = [ "xperm", "rndperm" ]

def xperm(mtx):
     return permanent(np.array(mtx,dtype=complex)).real
   
def rndperm(mtx, eps = 1e-3, maxtry=None):
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
    return 0

import time, sys
if __name__ == "__main__":
    
    filename = sys.argv[1]
    mtx=np.loadtxt(filename)
    st=time.time()
    new=rndperm(mtx, eps=1e-3)
    tnew = time.time()-st    
    st=time.time()
    cnew=mypermanent(mtx)
    ctnew = time.time() -st
    st=time.time()
    if len(mtx[0])<30: ref=xperm(mtx)
    else: ref=0
    tref = time.time()-st
    
    print "Reference:          ", ref, " time: ", tref
    print "New_method:         ", new, " time: ", tnew
    print "New_method C++:         ", cnew, " time: ",ctnew
