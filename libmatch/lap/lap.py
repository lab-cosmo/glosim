#!/usr/bin/python
import numpy as np
import sys
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

def main1(filename):
    mtx=np.loadtxt(filename)
    myinf= 1e100
    
    
    if mtx.shape[0] != mtx.shape[1]:
        mm = lcm(mtx.shape[0], mtx.shape[1])
        nmtx = np.zeros((mm,mm), float)        
        for i in range(mm/mtx.shape[0]):
            for j in range(mm/mtx.shape[1]):
                nmtx[i*mtx.shape[0]:(i+1)*mtx.shape[0],j*mtx.shape[1]:(j+1)*mtx.shape[1]] = mtx
    
    print nmtx                  
    bp = best_pairs(nmtx)
    tc = 0
    
    for p in bp:
        pc = nmtx[p[0],p[1]]
        print p, pc
        if pc<myinf: tc += nmtx[p[0],p[1]]
    print "reference cost: ", tc

def gcd(a,b):
   if (b>a): a,b = b, a
   
   while (b):  a, b = b, a%b
   
   return a
   
def lcm(a,b):
   return a*b/gcd(b,a)
   
def main(filename):
    mtx=np.loadtxt(filename)
    myinf= 1e100
    notlast=True
    final_cost=0.0
    nA, nB = mtx.shape
    elA = np.asarray(range(nA))
    elB = np.asarray(range(nB))
    while notlast:
      if mtx.shape[0] > mtx.shape[1]:
          mm = mtx.shape[0]
          nbuf=mtx.shape[0] - mtx.shape[1]
          nmtx = np.zeros((mm,mm), float)        
          for i in range(mtx.shape[0]):
              for j in range(mtx.shape[1]):
                  nmtx[i,j]=mtx[i,j]
              for j in range(mtx.shape[1],mtx.shape[1]+nbuf):
                  nmtx[i,j] = myinf
      else:
          mm = mtx.shape[1]
          nbuf=mtx.shape[1] - mtx.shape[0]
          if nbuf==0: notlast=False
          nmtx = np.zeros((mm,mm), float)
          nmtx[:]=myinf        
          for i in xrange(mtx.shape[0]):
              for j in xrange(mtx.shape[1]):
                  nmtx[i,j] = mtx[i,j]
         
      
      print nmtx                  
      bp = best_pairs(nmtx)
      tc = 0
      rowlist=[]
      collist=[]
      for p in bp:
          pc = nmtx[p[0],p[1]]
          print p, pc
          if pc<myinf: tc += nmtx[p[0],p[1]]
          else:
             rowlist.append(p[0])
             collist.append(p[1])
      print "total cost: ", tc
      
      if mtx.shape[0] > mtx.shape[1]:
        nold=mtx.shape[1]
        nmtx=np.zeros((len(rowlist),nold),float)
        for i in range(nbuf):
          for j in range(nold):
            nmtx[i,j]=mtx[rowlist[i],j]
        mtx=np.zeros((len(rowlist),nold))
        mtx=nmtx
         
      else:
        nold=mtx.shape[0]
        nmtx=np.zeros((nold,nbuf),float)
        for i in range(nold):
          for j in range(nbuf):
            nmtx[i,j]=mtx[i,collist[j]]
        mtx=np.zeros((nold,nbuf))
        mtx=nmtx
      final_cost+=tc
    print "Final Cost= ", final_cost
       
    
if __name__ == "__main__":
    np.set_printoptions(linewidth=1000)
    main1(*sys.argv[1:])
    main(*sys.argv[1:])
