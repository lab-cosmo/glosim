#!/usr/bin/python
import numpy as np
import sys, time
from copy import copy

__all__ = [ "best_cost", "lcm_best_cost", "lcm_best_cost1", "lcm_best_cost2", "best_pairs" ]

try: 
    import hungarian
    def linear_assignment(matrix):
        m=copy(matrix)
        assign=hungarian.lap(m)
        pair=[]
        for x in range(len(assign[0])):
            pair.append([x,assign[0][x]])
        return pair
except:
    from munkres import linear_assignment
    print >> sys.stderr, "WARNING! fast hungarian library is not available \n"  
def best_pairs(matrix):
    return linear_assignment(matrix)

def best_cost(matrix):
  hun=linear_assignment(matrix)
  cost=0.0
  print hun
  for pair in hun:
     cost+=matrix[pair[0],pair[1]]
  return cost

def gcd(a,b):
   if (b>a): a,b = b, a
   
   while (b):  a, b = b, a%b
   
   return a
   
def lcm(a,b):
   return a*b/gcd(b,a)

def lcm_index(x, y):
    z=lcm(x,y)
    
    ix = np.asarray(xrange(x),int)
    iy = np.asarray(xrange(y),int)
    
    lx = np.zeros(z,int)
    ly = np.zeros(z,int)
    for i in xrange(z/x):
        lx[i*x:(i+1)*x]=ix
    for i in xrange(z/y):
        ly[i*y:(i+1)*y]=iy
    return (lx,ly)
       
def lcm_matrix(m):
    x,y = m.shape
    if x==y: return m   # no-op
    
    lx, ly = lcm_index(x,y)
    return m[np.ix_(lx, ly)]

myinf = 1e2

def lcm_best_cost(mtx):
    
    # heuristics
    if (lcm(mtx.shape[0],mtx.shape[1]) < 1000):
        return lcm_best_cost1(mtx)
    else:
        return lcm_best_cost2(mtx, 1e-5)

def lcm_best_cost1(mtx):    
    nmtx = lcm_matrix(mtx)
          
    bp = best_pairs(nmtx)
    tc = 0
    
    for p in bp:
        pc = nmtx[p[0],p[1]]        
        if pc<myinf: tc += nmtx[p[0],p[1]]    
    return tc

def lcm_best_cost2(gmtx, thresh = 1e-10):
    # tresh: threshold for getting into the merging of two blocks
    mtx = gmtx
    xm, ym = mtx.shape 
    if xm==ym:
        return best_cost(mtx)
    if xm > ym:
        mtx = gmtx.T
        xm, ym = mtx.shape 
    mm = lcm(xm, ym)
    lx, ly = lcm_index(xm,ym)    
    
    # print xm, ym
    sm = min(xm, ym)
    np.set_printoptions(linewidth=1000,threshold=10000)
    tc = 0    
    bcl = []
    blocks = []    
    for i in range(mm/sm):
        blx=list(lx[range(i*sm,(i+1)*sm)])
        bly=list(ly[range(i*sm,(i+1)*sm)])        
        blocks.append((blx,bly))
        subm = mtx[np.ix_(blx,bly)]
        
        #print subm
        bp = best_pairs(subm)
      
        bc = 0
        for p in bp:
            pc = subm[p[0],p[1]]
            bc += pc  
        bcl.append(bc)
    
    nxc = 0 
    ntry = 0
        
    merged = True
    nb = len(blocks)
    tainted = np.ones((nb,nb), int)
    tainted[1,0]=1 # make sure we get in once!
    
    while merged and np.triu(tainted,1).sum()>0:     
        #print "estimate cost ", sum(bcl), "n. exchanges", nxc, "/", ntry     
        merged = False
        
        for i in xrange(nb):                  
            blxi = blocks[i][0]
            blyi = blocks[i][1]
            ni = len(blxi)                
            for j in range(i+1,nb):
                if tainted[i,j]==0: continue  
                # print np.tril(tainted,-1)              
                ntry += 1                        
                blxj = blocks[j][0]
                blyj = blocks[j][1]
                nj = len(blxj)                
                blx = blxi+blxj
                bly = blyi+blyj
                subm = mtx[np.ix_(blx,bly)]
                bp = best_pairs(subm)
          
                bc = 0
                for p in bp:
                    pc = subm[p[0],p[1]]
                    bc += pc  
                    
                if (bcl[i]+bcl[j])/bc -1  > thresh:
                    # print "MERGING %d,%d: %f+%f=%f >%f\n" %(i,j,bcl[i],bcl[j],bcl[i]+bcl[j],bc)
                    blyi = []
                    blyj = []                    
                    ci = 0
                    for ti in xrange(ni):
                        blyi.append(bly[bp[ti][1]])
                        ci += subm[bp[ti][0],bp[ti][1]]                        
                    cj = 0
                    for tj in xrange(nj):
                        blyj.append(bly[bp[ni+tj][1]])
                        cj += subm[bp[ni+tj][0],bp[ni+tj][1]]                        
                    #print "after", nblyi, nblyj
                    #~ subm = mtx[np.ix_(blxi,nblyi)]
                    #~ ci = best_cost(subm)
                    #~ subm = mtx[np.ix_(blxj,nblyj)]
                    #~ cj = best_cost(subm)
                    #~ subm = mtx[np.ix_(blxi+blxj,nblyi+nblyj)]
                    #~ ncc = best_cost(subm)
                    # print "new cost", ci+cj, bcl[i]+bcl[j], bc, "n. exchanges", nxc, "/", ntry
                    nxc+=1
                    blocks[i] = (blxi,blyi)
                    blocks[j] = (blxj,blyj)
                    bcl[i] = ci
                    bcl[j] = cj  
                    tainted[i,:] = 1
                    tainted[:,j] = 1
                    tainted[i,j] = 0
                    merged = True
                    break # it is more efficient to restart since this i has been tainted anyway
                tainted[i,j] = 0
            #if merged: break  
    
    #print "final cost", sum(bcl), len(bcl), "total exchanges: ", nxc, "/", ntry
    return sum(bcl)
    
if __name__ == "__main__":
     
    filename = sys.argv[1]
    mtx=1-np.loadtxt(filename)
    np.set_printoptions(linewidth=1000)
    st=time.time()
    new=lcm_best_cost2(mtx)
    tnew = time.time()-st    
    
    st=time.time()
    apx=lcm_best_cost2(mtx,1e-5)
    tapx = time.time()-st    
    
    st=time.time()
    ref=0
    ref=best_cost(mtx)
    tref = time.time()-st
    
    print "Reference:          ", 1-ref/len(mtx), " time: ", tref
    print "New_method:         ", 1-new/len(mtx), " time: ", tnew
    print "New_method(approx): ", 1-apx/len(mtx), " time: ", tapx
    
