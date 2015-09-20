#!/usr/bin/python

import numpy as np
import sys, glob
from ipi.utils.io import read_file, print_file
from ipi.engine.atoms import Atoms
from ipi.utils.depend import *
from ipi.utils.units import *

def main(filename, natoms):

   ipos=open(filename,"r")
   imode=filename[-3:]
   natoms = int(natoms)
   
   ifr = 0
   nn = 2.5
   while True:
      try:
         ret = read_file(imode,ipos,readcell=True)
         pos = ret["atoms"]
         cell = ret["cell"]
         q=depstrip(pos.q).copy()
         cell.array_pbc(q)
         
         natin = pos.natoms
         q.shape=(natin,3)
         s=np.dot(depstrip(cell.ih),q.T).T
         
         # now replicate in scaled coordinates
         nrep  = int((natoms/natin*nn)**(1./3.))
         
         natrep = natin*(2*nrep+1)**3
         
         ns = np.zeros((natrep,3))
         ik = 0
         for ix in range(-nrep,nrep+1):
          for iy in range(-nrep,nrep+1):
		   for iz in range(-nrep,nrep+1):
			for i in range(natin):
			 ns[ik] = s[i]+[ix,iy,iz]
			 ik+=1
		
         ns = np.dot(depstrip(cell.h),ns.T).T          
         
         # now removes atoms until we only have natoms
         d = np.zeros(natrep)
         for i in range(natrep):
           d[i] = np.sqrt(np.dot(ns[i],ns[i]))
         di = np.argsort(d)
		 
         npos = Atoms(natoms)
         for i in range(natoms):           
           npos.q[3*i:3*(i+1)]=ns[di[i]]
         
      except EOFError: # finished reading files
         sys.exit(0)

      print_file("pdb",npos, cell)
      ifr+=1


if __name__ == '__main__':
   main(*sys.argv[1:])
