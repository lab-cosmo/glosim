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
        nmtx = np.zeros((max(mtx.shape), max(mtx.shape)), float)
        nmtx[:] = myinf
        for i in xrange(mtx.shape[0]):
            for j in xrange(mtx.shape[1]):
                nmtx[i,j] = mtx[i,j]
    bp = best_pairs(nmtx)
    tc = 0
    
    for p in bp:
        pc = nmtx[p[0],p[1]]
        print p, pc
        if pc<myinf: tc += nmtx[p[0],p[1]]
    print "total cost: ", tc

def gcd(a,b):
   if (b>a): a,b = b, a
   
   while (b):  a, b = b, a%b
   
   return a
   
def lcm(a,b):
   return a*b/gcd(b,a)
   
def main(filename):
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
    print "total cost: ", tc
        
    
if __name__ == "__main__":
    main(*sys.argv[1:])
