#!/usr/bin/python
import numpy as np
import sys, time
from copy import copy

__all__ = [ "best_cost", "best_pairs" ]

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
    
def best_pairs(matrix):
    return linear_assignment(matrix)

def best_cost(matrix):
  hun=linear_assignment(matrix)
  cost=0.0
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
def lcm_best_cost1(mtx):    
    nmtx = lcm_matrix(mtx)
          
    bp = best_pairs(nmtx)
    tc = 0
    
    for p in bp:
        pc = nmtx[p[0],p[1]]        
        if pc<myinf: tc += nmtx[p[0],p[1]]    
    return tc

def lcm_best_cost2(gmtx):
        
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
        # sort blocks according to cost (so we always try to merge blocks with high cost)
        #~ skey = [ bcl[i]/len(blocks[i][0]) for i in range(len(bcl)) ]
        #~ print skey
        #~ sind = np.argsort(np.asarray(skey))[::-1] 
        #~ bcl = [ bcl[i] for i in sind]
        #~ blocks = [ blocks[i] for i in sind]        
        print "estimate cost ", sum(bcl), "n. exchanges", nxc, "/", ntry     
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
                    
                if bc*(1+1e-8)< (bcl[i]+bcl[j]):
                    # print "MERGING %d,%d: %f+%f=%f >%f\n" %(i,j,bcl[i],bcl[j],bcl[i]+bcl[j],bc)
                    #print "before", blyi, blyj
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
    
    print "final cost", sum(bcl), len(bcl), "total exchanges: ", nxc, "/", ntry
    return sum(bcl)



def pad(mtx, inf):
    nAB = max(mtx.shape)
    nmtx = np.ones((nAB, nAB)) * inf
    nmtx[0:mtx.shape[0],0:mtx.shape[1]] = mtx
    return nmtx
    
def main(filename):
    mbase=np.loadtxt(filename)
    notlast=True
    final_cost=0.0
    nA, nB = mbase.shape
    nAB = lcm(nA, nB)
    
    # how many times each element will be repeated
    elA = np.asarray(range(nA))
    elB = np.asarray(range(nB))
    cA = np.ones(nA) * (nAB/nA-1)
    cB = np.ones(nB) * (nAB/nB-1)

    np.set_printoptions(linewidth=1000)
    tcost = 0
    while cA.sum()>0 or cB.sum()>0:      
        print "outer loop"
        prevbp = []        
        while True:
            addA = []
            addB = []
            while len(elA)+len(addA)<len(elB) and cA.sum()>0:
                for i in xrange(nA):
                    # print len(elA)+len(addA), len(elB), cA.min()
                    if cA[i]==cA[cA.nonzero()].min():
                        addA.append(i)                        
                        cA[i]-=1
                        if len(elA)+len(addA)==len(elB): 
                            break
            while len(elB)+len(addB)<len(elA) and cB.sum()>0:
                for i in xrange(nB):
                    if cB[i]==cB[cB.nonzero()].min():
                        addB.append(i)                        
                        cB[i]-=1
                        if len(elB)+len(addB)==len(elA): 
                            break
            elA = np.asarray(list(elA)+addA)
            elB = np.asarray(list(elB)+addB)        
            mtx = mbase[np.ix_(elA, elB)]
            nmtx = pad(mtx, myinf)
            print nmtx
            print " Running an iteration ", len(elA)
            print elA
            print elB
          
            bp = best_pairs(nmtx)
            rmA = []
            rmB = [] 
            pcost = 0
            rbp = []
            for p in bp:
              pc = nmtx[p[0],p[1]]
              if pc<myinf: 
                 rbp.append((elA[p[0]], elB[p[1]]))
                 # print "MATCH2", p,rbp[-1], pc
                 pcost += nmtx[p[0],p[1]]
                 rmA.append(p[0])
                 rmB.append(p[1])
            
            #~ fsame = True
            #~ nsame = 0
            #~ for p in prevbp:
                #~ if not p in rbp:
                    #~ fsame = False
                    #~ break
                #~ nsame+=1
            #~ print rbp
            #~ print "matching assignments", nsame, " out of ", len(prevbp)
            #~ if fsame and prevbp!=[]: 
                #~ print "*** ASSIGNMENTS DID NOT CHANGE!"
                #~ # roll back and update stuff
                
            prevbp = rbp 
             
            nelA = np.delete(elA, rmA)
            nelB = np.delete(elB, rmB)
            addA = []
            addB = []
            for i in xrange(nA):
                if not i in nelA and cA[i]>0:
                    addA.append(i)
                    cA[i]-=1
            for i in xrange(nB):
                if not i in nelB and cB[i]>0:
                    addB.append(i)
                    cB[i]-=1
            #~ tcost+=pcost
            #~ if addA == [] and addB == []:
                #~ break
            #~ else:
                #~ elA = np.asarray(list(nelA)+addA)
                #~ elB = np.asarray(list(nelB)+addB)
            if addA == [] and addB == []:
                elA = nelA
                elB = nelB
                tcost += pcost
                break
            else:
                elA = np.asarray(list(elA)+addA)
                elB = np.asarray(list(elB)+addB)
            print "new selections: ", len(elA), len(elB), elA, elB
            
        print "total cost: ", pcost
      
    print "***** Final Cost= ", tcost
    return tcost
       
    
if __name__ == "__main__":
    
    filename = sys.argv[1]
    mtx=1-np.loadtxt(filename)
    np.set_printoptions(linewidth=1000)
    st=time.time()
    new=lcm_best_cost2(mtx)
    tnew = time.time()-st    
    st=time.time()
    ref=lcm_best_cost1(mtx)
    tref = time.time()-st
    
    print "Reference:  ", ref, " time: ", tref
    print "New_method: ", new, " time: ", tnew
    
