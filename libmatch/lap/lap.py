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

def gcd(a,b):
   if (b>a): a,b = b, a
   
   while (b):  a, b = b, a%b
   
   return a
   
def lcm(a,b):
   return a*b/gcd(b,a)
   
def main(filename):
    mbase=np.loadtxt(filename)
    notlast=True
    final_cost=0.0
    nA, nB = mbase.shape
    elA = np.asarray(range(nA))
    elB = np.asarray(range(nB))
    while (len(elA)+nA) <= len(elB): elA = np.asarray(list(elA)+range(nA))
    while (len(elB)+nB) <= len(elA): elB = np.asarray(list(elB)+range(nB))
    npairs = lcm(nA,nB)
    selpairs = 0
    tc = 0
    while len(elA)>0 or len(elB)>0:
      if len(elA)==0 : 
          elA = np.asarray(range(nA))
          if (selpairs+nB<=npairs): elB = np.asarray(list(elB)+range(nB))
      if len(elB)==0 : 
          elB = np.asarray(range(nB))
          if (selpairs+nA<=npairs): elA = np.asarray(list(elA)+range(nA))
      while (len(elA)+nA) <= len(elB): elA = np.asarray(list(elA)+range(nA))
      while (len(elB)+nB) <= len(elA): elB = np.asarray(list(elB)+range(nB))

      mtx = mbase[np.ix_(elA, elB)]

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
      print " Running an iteration "

      print elA
      print elB
      bp = best_pairs(nmtx)
      rmA = []
      rmB = [] 
      partc = 0
      for p in bp:
          pc = nmtx[p[0],p[1]]
          if pc<myinf: 
             print "MATCH2", p, elA[p[0]], elB[p[1]], pc
             selpairs += 1
             partc += nmtx[p[0],p[1]]
             rmA.append(p[0])
             rmB.append(p[1])

      elA = np.delete(elA, rmA)
      elB = np.delete(elB, rmB)      
      print "new selections: ", elA, elB
      print "total cost: ", partc
      
      tc+=partc
    print "***** Final Cost= ", tc
       
    
if __name__ == "__main__":
    np.set_printoptions(linewidth=1000)
    main1(*sys.argv[1:])
    main(*sys.argv[1:])
