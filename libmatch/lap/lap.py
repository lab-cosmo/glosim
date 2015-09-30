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
  hun=lap(matrix)
  cost=0.0
  for pair in hun:
     cost+=matrix[pair[0],pair[1]]
  return cost

myinf = 1e2
def main1(filename):
    mtx=np.loadtxt(filename)
    
    
    if mtx.shape[0] != mtx.shape[1]:
        mm = lcm(mtx.shape[0], mtx.shape[1])
        nmtx = np.zeros((mm,mm), float)        
        for i in range(mm/mtx.shape[0]):
            for j in range(mm/mtx.shape[1]):
                nmtx[i*mtx.shape[0]:(i+1)*mtx.shape[0],j*mtx.shape[1]:(j+1)*mtx.shape[1]] = mtx
    else: nmtx = mtx
 
    print nmtx                  
    bp = best_pairs(nmtx)
    tc = 0
    
    for p in bp:
        pc = nmtx[p[0],p[1]]
        print "MATCH1: ", p, p[0]%mtx.shape[0], p[1]%mtx.shape[1], pc
        if pc<myinf: tc += nmtx[p[0],p[1]]
    print "***** reference cost: ", tc
    return tc

def gcd(a,b):
   if (b>a): a,b = b, a
   
   while (b):  a, b = b, a%b
   
   return a
   
def lcm(a,b):
   return a*b/gcd(b,a)

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

    tcost = 0
    while cA.sum()>0 or cB.sum()>0:      
        print "outer loop"
        prevbp = []
        while True:
            addA = []
            addB = []
            while len(elA)+len(addA)<len(elB) and cA.sum()>0:
                for i in xrange(nA):
                    if cA[i]==max(cA):
                        addA.append(i)                        
                        cA[i]-=1
                        if len(elA)+len(addA)==len(elB): 
                            break
            while len(elB)+len(addB)<len(elA) and cB.sum()>0:
                for i in xrange(nB):
                    if cB[i]==max(cB):
                        addB.append(i)                        
                        cB[i]-=1
                        if len(elB)+len(addB)==len(elA): 
                            break
            elA = np.asarray(list(elA)+addA)
            elB = np.asarray(list(elB)+addB)        
            mtx = mbase[np.ix_(elA, elB)]
            nmtx = pad(mtx, myinf)
            print nmtx
            print " Running an iteration "
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
                 print "MATCH2", p,rbp[-1], pc
                 pcost += nmtx[p[0],p[1]]
                 rmA.append(p[0])
                 rmB.append(p[1])
            
            fsame = True
            nsame = 0
            for p in prevbp:
                if not p in rbp:
                    fsame = False
                    break
                nsame+=1
            print rbp
            print "matching assignments", nsame, " out of ", len(prevbp)
            if fsame and prevbp!=[]: 
                print "*** ASSIGNMENTS DID NOT CHANGE!"
                # roll back and update stuff
                
                
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
    np.set_printoptions(linewidth=1000)
    st=time.time()
    ref=main1(*sys.argv[1:])
    tref = time.time()-st
    st=time.time()
    new=main(*sys.argv[1:])
    tnew = time.time()-st
    
    print "Reference:  ", ref, " time: ", tref
    print "New method: ", new, " time: ", tnew
    
